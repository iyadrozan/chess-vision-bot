"""
Eksperimen 5: Bot main sendiri (KLIK BENERAN via ydotool).

Alur = suggest.py + eksekusi: lacak -> giliran bot -> Stockfish mikir ->
konfirmasi -> klik kotak asal -> klik kotak tujuan -> verifikasi papan.

Failsafe:
- Tanpa --yes: TIAP langkah minta konfirmasi via TOMBOL di jendela
  (KLIK = jalan, SKIP = lewati, QUIT = berhenti; keyboard Enter/s/q juga
  bisa). Tanpa jendela (--no-window): konfirmasi via terminal.
- --dry-run: loop penuh tanpa menyentuh mouse (buat testing).
- Klik SELALU terverifikasi closed-loop (clicker.py): meleset = batal.
- Promosi: dialog pilihan bidak tidak bisa ditebak -> bot klik dari->ke,
  lalu BERHENTI dan minta lu klik bidak promosi manual di dialog,
  loop lanjut otomatis setelah cocok.
- Q / Esc / Ctrl+C kapan saja untuk berhenti.
- Game over (menang/kalah/remis): jendela tampilkan tombol MAIN LAGI
  yang klik tombol Play again browser + reset otomatis ke game baru.

Prasyarat: ./ydo.sh on  +  papan terlihat di layar.

Cara pakai:
  .venv/bin/python bot.py --dry-run --once
      # test sekali, tanpa klik
  .venv/bin/python bot.py
      # main, konfirmasi per langkah (disarankan untuk game pertama)
  .venv/bin/python bot.py --yes --no-window
      # full-auto tanpa konfirmasi & tanpa jendela (hati-hati!)
  .venv/bin/python bot.py --turbo --tc 180+2
      # turnamen cepat: tanpa konfirmasi, mikir <=0.3s, jam adaptif
      # (darurat <10s = jalan instan 0.12s). Contoh tc: 180+2, 3m+2, 60+0
"""

import argparse
import random
import subprocess
import sys
import time

import chess
import chess.engine
import cv2

from board_tracker import (capture_ppm, load_board, maybe_recalibrate)
from clicker import click_at, click_square, find_ui_button
from highlights import detect_highlights
from square_scan import (detect_orientation, grab_live_warp, load_templates)
from suggest import (TOTAL_W, board_from_scan, describe_move,
                     grid_to_placement, init_board_from_scan,
                     match_move_sequence, placement_of, render_window,
                     stable_scan, think)

ENGINE_PATH = "./engines/stockfish-bin"
WIN = "Chess Bot (klik tombol / Q keluar)"


def parse_tc(s):
    """'180+2' / '3m+2' / '60' -> (base_detik, inc_detik)."""
    import re
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(m?)\s*(?:\+\s*(\d+(?:\.\d+)?))?\s*", s)
    if not m:
        raise ValueError(f"format --tc salah: {s!r} (contoh: 180+2, 3m+2, 60+0)")
    base = float(m.group(1)) * (60 if m.group(2) == "m" else 1)
    return base, float(m.group(3) or 0)


class ChessClock:
    """Jam catur versi bot: lacak WAKTU KITA dari durasi giliran kita
    yang terukur (akurasi ~0.5 dtk, cukup untuk adaptasi)."""

    def __init__(self, base, inc):
        self.base, self.inc = base, inc
        self.used, self.moves, self.turn_t0 = 0.0, 0, None

    def start_turn(self):
        if self.turn_t0 is None:
            self.turn_t0 = time.time()

    def reset(self):
        self.used, self.moves, self.turn_t0 = 0.0, 0, None

    def end_turn(self):
        if self.turn_t0 is not None:
            self.used += time.time() - self.turn_t0
            self.moves += 1
            self.turn_t0 = None

    @property
    def remaining(self):
        return self.base + self.inc * self.moves - self.used

    def movetime(self, base_think):
        """Return (detik_mikir, boleh_delay, trouble)."""
        r = self.remaining
        if r <= 10:
            return 0.12, False, True
        if r <= 30:
            return min(0.3, base_think), False, False
        return min(base_think, max(0.4, r / 25)), True, False

    def fmt(self):
        r = max(0, self.remaining)
        return f"{int(r // 60)}:{int(r % 60):02d} (+{self.inc:g})"


def main():
    ap = argparse.ArgumentParser(description="Bot catur klik-sendiri")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="jangan sentuh mouse (test)")
    ap.add_argument("--yes", action="store_true",
                    help="klik tanpa konfirmasi per langkah")
    ap.add_argument("--color", choices=["white", "black", "auto"],
                    default="auto")
    ap.add_argument("--think-time", type=float, default=1.0)
    ap.add_argument("--elo", type=int, default=1400,
                    help="cap kekuatan Stockfish (1320-3190, 0 = full power). "
                         "~1400 = akurasi 85-90 persen ala pemain klub")
    ap.add_argument("--delay-max", type=float, default=2.0,
                    help="jeda acak 0.5-detik ini sebelum klik biar timing "
                         "manusiawi (0 = mati)")
    ap.add_argument("--turbo", action="store_true",
                    help="preset turnamen cepat: poll 0.15, stabil 2x, mikir "
                         "0.3s, delay kecil, tanpa konfirmasi")
    ap.add_argument("--tc", type=str, default=None, metavar="BASE+INC",
                    help="kontrol waktu, misal 180+2 / 3m+2 / 60+0. Bot "
                         "lacak jamnya sendiri + mikir adaptif "
                         "(darurat <10s = jalan instan)")
    ap.add_argument("--poll", type=float, default=0.3)
    ap.add_argument("--stable", type=int, default=3)
    ap.add_argument("--engine", type=str, default=ENGINE_PATH)
    ap.add_argument("--fen", type=str, default=None)
    ap.add_argument("--board", type=str, default="board.json")
    ap.add_argument("--templates", type=str, default="templates")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    live_clicks = not args.dry_run
    if args.turbo:
        args.poll, args.stable = 0.15, 2
        args.think_time = min(args.think_time, 0.3)
        args.delay_max = min(args.delay_max, 0.3)
        args.yes = True
        print("Mode TURBO: poll 0.15s, stabil 2x, mikir <=0.3s, "
              "tanpa konfirmasi.")
    clock = None
    if args.tc:
        try:
            base, inc = parse_tc(args.tc)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        clock = ChessClock(base, inc)
        print(f"Jam aktif: {base:g}s +{inc:g}s per langkah (adaptif).")
    if live_clicks and not args.yes:
        print("Mode KONFIRMASI: tiap langkah ditanya dulu. "
              "Pakai --yes untuk full-auto.")
    if args.dry_run:
        print("Mode DRY-RUN: mouse tidak disentuh.")

    templates = load_templates(args.templates)
    import functools
    grab = functools.partial(grab_live_warp, args.board)
    corners = load_board(args.board)

    ui = {"status": "mulai...", "last_move": None, "suggestion": None,
          "board": None, "warped": None, "quit": False, "shown": False,
          "awaiting": False, "confirm": None, "rects": {}, "last_key": -1,
          "gameover": False, "acted_for": None, "placed": False,
          "red": [], "grid": None, "red_hold": 0, "eval": None,
          "moves_san": []}

    def on_mouse(event, x, y, flags, param):
        """Tombol jendela diklik -> catat jawaban."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not (ui["awaiting"] or ui["gameover"]):
            return
        for name, (x0, y0, x1, y1) in ui["rects"].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                ui["confirm"] = name
                break

    def place_top_right():
        """Taruh jendela di kanan atas layar (sekali saja, biar tidak
        mengganggu papan di kiri)."""
        import json
        win_w = TOTAL_W
        x, y = 1920 - win_w, 0
        try:
            r = subprocess.run(["hyprctl", "monitors", "-j"],
                               capture_output=True, text=True, timeout=5)
            m = json.loads(r.stdout)[0]
            x, y = m["x"] + m["width"] - win_w, m["y"]
        except Exception:
            pass
        try:
            cv2.moveWindow(WIN, int(x), int(y))
        except Exception:
            pass

    def pump_window(warped=None):
        if args.no_window:
            return False
        if warped is not None:
            ui["warped"] = warped
        if ui["warped"] is None:
            return False
        try:
            img = render_window(ui["warped"], white_bottom, ui["status"],
                                board=ui["board"], last_move=ui["last_move"],
                                suggestion=ui["suggestion"],
                                confirm_buttons=ui["awaiting"],
                                gameover_buttons=ui["gameover"],
                                button_rects=ui["rects"],
                                bot_color=bot_color, eval_info=ui["eval"],
                                clock_str=clock.fmt() if clock else None,
                                moves_hist=ui["moves_san"])
            cv2.imshow(WIN, img)
            cv2.setMouseCallback(WIN, on_mouse)
            if not ui["placed"]:
                place_top_right()
                ui["placed"] = True
            ui["shown"] = True
        except Exception as e:
            if not ui["shown"]:
                print(f"(jendela nonaktif: {e})")
                args.no_window = True
            return False
        ui["last_key"] = cv2.waitKey(1) & 0xFF
        return ui["last_key"] in (ord("q"), 27)

    def ask_confirm(prompt):
        """Minta jawaban: tombol jendela (KLIK/SKIP/QUIT) atau keyboard
        Enter/s/q. Fallback input() terminal bila --no-window.
        Return 'klik' / 'skip' / 'quit'."""
        print(f"   {prompt}")
        print("   [klik KLIK di jendela / Enter=yes, SKIP / s=skip, QUIT / q=quit]")
        if args.no_window:
            try:
                ans = input("   Klik? [Enter]=ya, s=skip, q=quit: ").strip().lower()
            except EOFError:
                return "quit"
            if ans == "q":
                return "quit"
            if ans in ("s", "skip"):
                return "skip"
            return "klik"
        ui["awaiting"], ui["confirm"] = True, None
        try:
            while ui["confirm"] is None:
                if pump_window():
                    return "quit"
                key = ui["last_key"]
                if key == 13:
                    return "klik"
                if key in (ord("s"), ord("S")):
                    return "skip"
                time.sleep(0.05)
            return ui["confirm"]
        finally:
            ui["awaiting"] = False

    def on_frame(warped, fen_preview):
        ui["status"] = "animasi / belum stabil..."
        if pump_window(warped):
            ui["quit"] = True

    white_bottom, _ = detect_orientation(grab(), templates)
    print(f"Orientasi: {'putih di bawah' if white_bottom else 'HITAM di bawah'}")
    bot_color = (chess.WHITE if white_bottom else chess.BLACK) \
        if args.color == "auto" else chess.WHITE \
        if args.color == "white" else chess.BLACK
    print(f"Bot pegang: {'putih' if bot_color == chess.WHITE else 'hitam'}")

    if args.fen:
        board = chess.Board(args.fen)
    else:
        print("Scan posisi...")
        ui["status"] = "scan posisi..."
        board = None
        while board is None:
            _, grid, warped0 = stable_scan(grab, templates, white_bottom,
                                           args.stable, args.poll, tag="init",
                                           on_frame=on_frame)
            if grid is None:
                return 1
            hl0 = detect_highlights(warped0, white_bottom)
            board, last0, info0 = init_board_from_scan(grid, hl0["pair"],
                                                       white_bottom)
            if board is None:
                print(f"   init {info0}, ulangi...")
                time.sleep(args.poll)
        print(f"Highlight: hijau={hl0['pair']} merah={hl0['red']}")
        print(f"Posisi: {board.fen()} ({info0})")
        ui["last_move"] = last0
        ui["red"] = hl0["red"]
        ui["grid"] = grid
    ui["board"] = board

    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    except FileNotFoundError:
        print(f"Engine tidak ketemu: {args.engine}", file=sys.stderr)
        return 2
    engine.configure({"Threads": 2, "Hash": 128})
    if args.elo > 0:
        if not 1320 <= args.elo <= 3190:
            print(f"⚠️  --elo {args.elo} di luar 1320-3190, pakai 1400.",
                  file=sys.stderr)
            args.elo = 1400
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
        print(f"Kekuatan dibatasi: Elo ~{args.elo} (biar kayak manusia)")
    else:
        print("Kekuatan PENUH (mode 99%, gampang dicurigai!)")

    def do_turn():
        """Giliran bot: mikir -> konfirmasi -> klik.
        Return 'quit'/'skip'/'tahan'/ok."""
        nonlocal board
        if board.turn != bot_color or board.fen() == ui["acted_for"]:
            return "bukan-giliran"
        if board.is_game_over():
            return "game-over"
        # Pengaman skakmat: raja kita MERAH di layar tapi internal tidak
        # merasa skak -> JANGAN jalan (posisi internal pasti ngaco).
        # Resync dengan giliran yang sama; bergerak lagi hanya bila jelas.
        our_king = board.king(bot_color)
        if our_king is not None and not board.is_check() \
                and chess.square_name(our_king) in (ui["red"] or []):
            ui["red_hold"] += 1
            print(f"🛑 raja MERAH di layar tapi internal tidak skak "
                  f"({ui['red_hold']}x) -> resync, TIDAK jalan.")
            nb = board_from_scan(ui["grid"], white_bottom, board.turn) \
                if ui["grid"] is not None else None
            if nb is not None:
                board = nb
                ui["board"] = board
            ui["status"] = "raja merah, verifikasi..."
            if ui["red_hold"] >= 3:
                print("   ⚠️  3x berturut-turut: kemungkinan game sudah "
                      "selesai / papan salah baca. Menunggu...")
            return "tahan"
        ui["red_hold"] = 0
        think_t, allow_delay = args.think_time, args.delay_max > 0
        if clock is not None:
            clock.start_turn()
            think_t, allow_delay, trouble = clock.movetime(args.think_time)
            ui["status"] = (f"⏱ {clock.fmt()} DARURAT - jalan instan!"
                            if trouble else
                            f"mikir... (⏱ {clock.fmt()})")
        else:
            ui["status"] = "Stockfish mikir..."
        pump_window()
        move, s, cp, mate = think(engine, board, think_t)
        san = board.san(move)
        ramah, extra = describe_move(board, move)
        ui["suggestion"] = (move.uci(), san, s, ramah, extra)
        ui["eval"] = (cp, mate)
        ui["acted_for"] = board.fen()
        print(f"🤖 MAU JALAN: {ramah}  [{san}, {s}]")
        print(f"   {extra}")
        if not args.yes and live_clicks:
            ans = ask_confirm(f"Jalankan {ramah}?")
            if ans == "quit":
                return "quit"
            if ans == "skip":
                print("   di-skip, menunggu...")
                return "skip"
        if move.promotion:
            print("   ⚠️  PROMOSI: bot klik dari->tujuan, lalu KLIK MANUAL "
                  "bidak promosi di dialog!")
            if live_clicks and not args.yes:
                ans = ask_confirm("Klik dari->tujuan dulu?")
                if ans == "quit":
                    return "quit"
                if ans == "skip":
                    return "skip"
        fr = chess.square_name(move.from_square)
        to = chess.square_name(move.to_square)
        if live_clicks and allow_delay:
            jeda = random.uniform(0.5, max(0.5, args.delay_max))
            ui["status"] = f"mikir ala manusia ({jeda:.1f}s)..."
            pump_window()
            time.sleep(jeda)
        ui["status"] = f"klik {fr} -> {to}..."
        pump_window()
        try:
            click_square(fr, corners, white_bottom, live=live_clicks)
            click_square(to, corners, white_bottom, live=live_clicks)
        except RuntimeError as e:
            print(f"   🛑 {e}")
            ui["status"] = "klik batal (meleset)"
            acted_for = None  # coba lagi lain waktu
            return "batal"
        print(f"   diklik: {fr} -> {to}")
        if move.promotion and live_clicks:
            ui["status"] = "pilih promosi MANUAL di dialog..."
            print("   👆 pilih bidak promosi di layar, bot menunggu cocok...")
            pump_window()
            return "promosi-manual"
        ui["status"] = "tunggu papan berubah..."
        return "ok"

    def game_over_screen():
        """Tampilkan hasil + tombol MAIN LAGI / QUIT.
        Return 'main_lagi' / 'quit'."""
        outcome = board.outcome()
        result = board.result()
        term = str(outcome.termination).split(".")[-1] if outcome else "?"
        if result == "1-0":
            label = "MENANG! Bot (putih) menang" \
                if bot_color == chess.WHITE else "KALAH (hitam menang)"
        elif result == "0-1":
            label = "MENANG! Bot (hitam) menang" \
                if bot_color == chess.BLACK else "KALAH (putih menang)"
        else:
            label = f"REMIS {result}"
        print(f"🏁 GAME OVER: {label} [{term}]")
        print("   [klik MAIN LAGI di jendela / Enter=mulai lagi, QUIT / q=quit]")
        if args.no_window:
            try:
                ans = input("   Main lagi? [Enter]=ya, q=quit: ").strip().lower()
            except EOFError:
                return "quit"
            return "quit" if ans == "q" else "main_lagi"
        ui["status"] = f"{label} - main lagi?"
        ui["gameover"], ui["confirm"] = True, None
        try:
            while ui["confirm"] not in ("main_lagi", "quit"):
                if pump_window():
                    return "quit"
                if ui["last_key"] == 13:
                    return "main_lagi"
                time.sleep(0.05)
            return ui["confirm"]
        finally:
            ui["gameover"] = False

    def click_play_again():
        """Cari tombol Play again browser (kanan papan dulu, lalu full)
        dan klik. Return True bila diklik."""
        ui["status"] = "cari tombol Play again..."
        pump_window()
        full = capture_ppm()
        sh, sw = full.shape[:2]
        bx1 = int(corners.max(axis=0)[0])
        center = find_ui_button(full, "play_again",
                                region=(max(0, bx1 - 20), 0, sw, sh))
        if center is None:
            center = find_ui_button(full, "play_again")
        if center is None:
            return False
        print(f"   tombol Play again di {center}")
        ui["status"] = "klik Play again..."
        pump_window()
        click_at(center[0], center[1], live=live_clicks)
        return True

    def wait_new_game():
        """Tunggu posisi awal yang fresh (stabil). Juga menangkap bila
        komputer sudah jalan duluan. Return True / False bila quit."""
        nonlocal board
        fresh = chess.Board()
        fresh_pl = placement_of(fresh)
        while True:
            if ui["quit"]:
                return False
            ui["status"] = "tunggu game baru..."
            _, grid, warped = stable_scan(
                grab, templates, white_bottom, args.stable, args.poll,
                tag="newgame", on_frame=on_frame)
            if ui["quit"]:
                return False
            if grid is None:
                time.sleep(1.0)
                continue
            placement = grid_to_placement(grid, white_bottom)
            if placement is None:
                if pump_window(warped):
                    return False
                time.sleep(args.poll)
                continue
            if placement == fresh_pl:
                first_moves = None
            else:
                first_moves, _ = match_move_sequence(
                    fresh, placement, wild_grid=grid,
                    white_bottom=white_bottom)
                if first_moves is None:
                    ui["status"] = "tunggu game baru " \
                                   "(klik Play again manual?)..."
                    if pump_window(warped):
                        return False
                    time.sleep(args.poll)
                    continue
            board = board_from_scan(grid, white_bottom, chess.WHITE)
            if board is None:
                ui["status"] = "tunggu game baru (scan ragu)..."
                if pump_window(warped):
                    return False
                time.sleep(args.poll)
                continue
            ui["board"] = board
            ui["last_move"] = None
            ui["suggestion"] = None
            ui["eval"] = None
            ui["moves_san"] = []
            ui["acted_for"] = None
            if first_moves:
                for m in first_moves:
                    san = board.san(m)
                    ramah, _ = describe_move(board, m)
                    ui["moves_san"].append(
                        (board.fullmove_number,
                         board.turn == chess.WHITE, san))
                    board.push(m)
                    print(f"♟️  Lawan jalan duluan: {ramah}  [{san}]")
                ui["last_move"] = first_moves[-1]
            print(f"✨ Game baru: {board.fen()}")
            ui["status"] = "game baru!"
            ui["red"] = []
            ui["red_hold"] = 0
            if clock is not None:
                clock.reset()
            return True

    try:
        if args.once:
            r = do_turn()
            pump_window()
            if not args.no_window:
                print("Tekan tombol di jendela untuk keluar.")
                cv2.waitKey(0)
            return 0 if r not in ("quit",) else 0
        print("Bot jalan (Q / Ctrl+C berhenti)...")
        promo_wait = False
        lost_count = 0  # resync beruntun -> curiga papan pindah
        while True:
            if ui["quit"]:
                return 0
            if board.is_game_over():
                if game_over_screen() == "quit":
                    return 0
                if not click_play_again():
                    print("   ⚠️  tombol Play again tidak ketemu - "
                          "KLIK MANUAL di browser, bot menunggu...")
                if not wait_new_game():
                    return 0
                promo_wait = False
                continue
            r = do_turn()
            if r == "quit":
                return 0
            if r in ("game-over", "tahan"):
                time.sleep(args.poll)
                continue  # game-over diproses di cek atas loop
            if r == "promosi-manual":
                promo_wait = True
            fen, grid, warped = stable_scan(
                grab, templates, white_bottom, args.stable, args.poll,
                tag="loop", on_frame=on_frame)
            if ui["quit"]:
                return 0
            if grid is None:
                time.sleep(1.0)
                continue
            hl = detect_highlights(warped, white_bottom)
            if hl["red"] != ui["red"]:
                print(f"🔴 kotak merah: {hl['red'] or '-'} "
                      f"(hijau: {hl['pair']})")
            ui["red"], ui["grid"] = hl["red"], grid
            hint = hl["pair"]
            placement = grid_to_placement(grid, white_bottom)
            if placement is None or placement == placement_of(board):
                ui["status"] = "stabil"
                ui["board"] = board
                if pump_window(warped):
                    return 0
                time.sleep(args.poll)
                continue
            moves, _ = match_move_sequence(board, placement, hint=hint,
                                               wild_grid=grid,
                                               white_bottom=white_bottom)
            if moves is None:
                if promo_wait:
                    # dialog promosi masih terbuka / belum cocok: tunggu saja
                    ui["status"] = "tunggu pilihan promosi..."
                    if pump_window(warped):
                        return 0
                    time.sleep(args.poll)
                    continue
                lost_count += 1
                if lost_count >= 5:
                    # window digeser/resize? kalibrasi ulang otomatis
                    print("⚠️  5x resync beruntun, cek papan pindah...")
                    _, moved = maybe_recalibrate(capture_ppm, args.board)
                    if moved:
                        corners = load_board(args.board)
                    lost_count = 0
                    continue
                print("⚠️  perubahan tidak cocok 1-2 langkah -> resync.")
                nb = board_from_scan(grid, white_bottom, not board.turn)
                if nb is None:
                    continue
                board = nb
                ui["last_move"] = None
                ui["status"] = "resync!"
            else:
                for m in moves:
                    san = board.san(m)
                    ramah, _ = describe_move(board, m)
                    mover_is_bot = board.turn == bot_color
                    ui["moves_san"].append(
                        (board.fullmove_number,
                         board.turn == chess.WHITE, san))
                    board.push(m)
                    line = (f"♟️  {'Bot' if mover_is_bot else 'Lawan'} "
                            f"jalan: {ramah}  [{san}]")
                    if mover_is_bot and clock is not None:
                        clock.end_turn()
                        line += f"  ⏱ {clock.fmt()}"
                    print(line)
                promo_wait = False
                lost_count = 0
                ui["last_move"] = moves[-1]
                ui["status"] = "stabil"
            ui["board"] = board
            if pump_window(warped):
                return 0
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nBerhenti.")
    finally:
        engine.quit()
        if ui["shown"]:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
