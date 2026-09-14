"""
Eksperimen 4: Pelacak + penasihat (TANPA klik).
- Polling papan live, tunggu posisi STABIL (N scan identik beruntun,
  biar animasi geser bidak tidak kebaca sebagai posisi).
- Langkah lawan dikenali dengan mencocokkan hasil scan ke SEMUA langkah
  legal python-chess (otomatis benar untuk rokade, en passant, promosi).
- Giliran bot -> Stockfish mikir (batas waktu) -> print saran SAN + eval.
- User yang eksekusi manual di browser; bot ikut melacak.

Cara pakai:
  .venv/bin/python suggest.py --once
      # scan sekali + saran (buat posisi sekarang)
  .venv/bin/python suggest.py
      # loop penuh: lacak semua langkah, sarankan saat giliran bot
  .venv/bin/python suggest.py --color black --think-time 2.0
  .venv/bin/python suggest.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
      # mulai dari posisi tengah (tetap butuh scan cocok dulu)
"""

import argparse
import sys
import time

import chess
import chess.engine
import cv2
import numpy as np

from board_tracker import (WARP_SIZE, capture_ppm, draw_overlay,
                            maybe_recalibrate)
from highlights import detect_highlights
from square_scan import (FILES, detect_orientation, grab_live_warp,
                         load_templates, scan_to_fen)

ENGINE_PATH = "./engines/stockfish-bin"
# Geometri jendela: bar eval | papan | panel (lihat render_window)


def grid_to_placement(grid, white_bottom=True):
    """Grid char 8x8 -> dict {square: chess.Piece}. Return None bila ada '?'.
    grid[r][c]: r=0 baris atas layar."""
    if any("?" in (ch or "") for row in grid for ch in row):
        return None
    placement = {}
    order_r = range(8) if white_bottom else range(7, -1, -1)
    order_c = range(8) if white_bottom else range(7, -1, -1)
    for r in order_r:
        rank = 8 - r if white_bottom else r + 1
        for c in order_c:
            ch = grid[r][c]
            if ch is None:
                continue
            file = FILES[c] if white_bottom else FILES[7 - c]
            placement[chess.parse_square(f"{file}{rank}")] = \
                chess.Piece.from_symbol(ch)
    return placement


def placement_of(board):
    return dict(board.piece_map())


def match_move(board, placement):
    """Cari langkah legal yang hasilnya == placement. Return move / None."""
    moves, _ = match_move_sequence(board, placement, max_plies=1)
    return moves[0] if moves else None


def grid_compatible(result_placement, grid, white_bottom=True):
    """Cocok longgar: semua sel PASTI harus sama (termasuk yang KOSONG!);
    hanya '?' yang bebas apa saja. Untuk panah/highlight yang menutup
    1-3 kotak."""
    order_r = range(8) if white_bottom else range(7, -1, -1)
    order_c = range(8) if white_bottom else range(7, -1, -1)
    for r in order_r:
        rank = 8 - r if white_bottom else r + 1
        for c in order_c:
            ch = grid[r][c]
            if ch == "?":
                continue
            file = FILES[c] if white_bottom else FILES[7 - c]
            sq = chess.parse_square(f"{file}{rank}")
            if ch is None:
                if sq in result_placement:
                    return False  # harusnya kosong
            elif result_placement.get(sq) != chess.Piece.from_symbol(ch):
                return False
    return True


def match_move_sequence(board, placement, max_plies=2, hint=None,
                        wild_grid=None, white_bottom=True):
    """Cari sekuens 1-2 langkah legal yang hasilnya == placement.

    Perlu karena komputer lawan sering balas instan: loop hanya sempat
    melihat posisi awal -> posisi sesudah 2 langkah (kita + lawan).
    hint = pasangan kotak highlight hijau (langkah terakhir), misal
    ('e2', 'e4'): kandidat yang cocok dicoba DULUAN (tetap fallback ke
    pencarian penuh bila tidak cocok).
    wild_grid = grid scan mentah (boleh ada '?'): bila diberi, pencocokan
    longgar (sel '?' bebas) dipakai; bila ambigu -> None (resync).
    Return ([moves], board_sesudah) / (None, board).
    """
    hint_sq = None
    if hint:
        try:
            hint_sq = {chess.parse_square(s) for s in hint}
        except ValueError:
            hint_sq = None

    def rank_1(moves):
        moves = list(moves)
        if hint_sq is None:
            return moves
        return sorted(moves, key=lambda m:
                      0 if {m.from_square, m.to_square} == hint_sq else 1)

    strict = wild_grid is None

    def fits(b):
        if strict:
            return placement_of(b) == placement
        return grid_compatible(placement_of(b), wild_grid, white_bottom)

    cands_1 = [m for m in rank_1(board.legal_moves)
               if fits(_pushed(board, m))]
    if len(cands_1) == 1 or (strict and cands_1):
        b = _pushed(board, cands_1[0])
        return [cands_1[0]], b
    if max_plies >= 2:
        cands_2 = []
        for m1 in board.legal_moves:
            b1 = _pushed(board, m1)
            for m2 in rank_1(b1.legal_moves):
                if fits(_pushed(b1, m2)):
                    cands_2.append((m1, m2))
                    if len(cands_2) > 4:
                        break
            if len(cands_2) > 4:
                break
        if len(cands_2) == 1 or (strict and cands_2):
            m1, m2 = cands_2[0]
            b2 = _pushed(_pushed(board, m1), m2)
            return [m1, m2], b2
    return None, board


def _pushed(board, move):
    b = board.copy(stack=False)
    b.push(move)
    return b


def infer_start_turn(grid, pair, white_bottom=True):
    """Dari highlight last-move + isi kotak: tebak (from, to, giliran).

    Aturan: kotak yang kosong = asal; bidak di kotak tujuan = yang jalan;
    giliran = lawan dari warna bidak itu. Return (from_sq, to_sq, turn)
    / None bila ambigu (tidak ada highlight / '?' / dua-duanya isi).
    """
    if not pair or any("?" in (ch or "") for row in grid for ch in row):
        return None
    occ = {}
    order_r = range(8) if white_bottom else range(7, -1, -1)
    order_c = range(8) if white_bottom else range(7, -1, -1)
    for r in order_r:
        rank = 8 - r if white_bottom else r + 1
        for c in order_c:
            file = FILES[c] if white_bottom else FILES[7 - c]
            occ[f"{file}{rank}"] = grid[r][c]
    a, b = pair
    if a not in occ or b not in occ:
        return None
    ca, cb = occ[a], occ[b]
    if (ca is None) == (cb is None):
        return None  # ambigu: dua-duanya kosong / isi
    from_sq = a if ca is None else b
    mover = chess.Piece.from_symbol(cb if ca is None else ca)
    return from_sq, (b if ca is None else a), not mover.color


def init_board_from_scan(grid, pair, white_bottom=True,
                         default_turn=chess.WHITE):
    """Bangun board awal. Kalau mulai TENGAH game, highlight hijau dipakai
    untuk tahu langkah terakhir + siapa yang jalan (giliran = lawannya).
    Return (board, last_move_display|None, info_str)."""
    placement = grid_to_placement(grid, white_bottom)
    if placement is None:
        return None, None, "grid ragu-ragu"
    inf = infer_start_turn(grid, pair, white_bottom)
    if inf is None:
        board = board_from_scan(grid, white_bottom, default_turn)
        note = ("posisi awal, giliran putih" if placement == placement_of(chess.Board())
                else "tanpa highlight -> asumsi giliran putih")
        return board, None, note
    from_sq, to_sq, turn = inf
    board = board_from_scan(grid, white_bottom, turn)
    disp = chess.Move(chess.parse_square(from_sq), chess.parse_square(to_sq))
    side = "putih" if turn == chess.WHITE else "hitam"
    return board, disp, f"langkah terakhir {from_sq}->{to_sq}, giliran {side}"


def derive_castling(board):
    """Hak rokade dari posisi: raja + benteng masih di rumah = boleh.
    (Heuristik eksperimen; cukup untuk game utuh dari awal.)"""
    rights = ""
    b = board.piece_map()
    home = [(chess.E1, chess.A1, chess.H1, chess.WHITE, "K", "Q"),
            (chess.E8, chess.A8, chess.H8, chess.BLACK, "k", "q")]
    for ke, qa, kh, color, ks, qs in home:
        if b.get(ke) != chess.Piece(chess.KING, color):
            continue
        if b.get(kh) == chess.Piece(chess.ROOK, color):
            rights += ks
        if b.get(qa) == chess.Piece(chess.ROOK, color):
            rights += qs
    board.set_castling_fen(rights or "-")


def board_from_scan(grid, white_bottom, turn):
    """Bangun board dari scan. EP dikosongkan, rokade diturunkan."""
    placement = grid_to_placement(grid, white_bottom)
    if placement is None:
        return None
    board = chess.Board.empty()
    for sq, piece in placement.items():
        board.set_piece_at(sq, piece)
    board.turn = turn
    derive_castling(board)
    return board


def think(engine, board, think_time):
    """Return (move, skor_str, cp_putih|None, mate_putih|None)."""
    info = engine.analyse(board, chess.engine.Limit(time=think_time))
    move = info["pv"][0]
    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        return move, (f"#{m}" if m > 0 else f"-#{-m}"), None, m
    cp = score.score()
    return move, f"{cp / 100:+.2f}", cp, None


PIECE_ID = {chess.PAWN: "Pion", chess.KNIGHT: "Kuda", chess.BISHOP: "Gajah",
            chess.ROOK: "Benteng", chess.QUEEN: "Menteri", chess.KING: "Raja"}


def describe_move(board, move):
    """Jelaskan langkah dengan bahasa sederhana (Indonesia).

    Return (baris_utama, baris_tambahan). Contoh:
    ('Kuda g1 -> f3', 'mengembangkan kuda ke tengah')
    ('Pion e4 -> d5', 'makan Pion! Skak!')
    """
    uci = move.uci()
    if board.is_castling(move):
        side = "pendek (sisi raja)" if board.is_kingside_castling(move) \
            else "panjang (sisi ratu)"
        main = f"Rokade {side}"
        extra = "amankan raja, hubungkan benteng"
    elif board.is_en_passant(move):
        main = f"Pion {uci[:2]} -> {uci[2:4]}, makan en passant!"
        extra = "pion lawan yang maju 2 langkah ikut dimakan"
    else:
        piece = board.piece_at(move.from_square)
        main = f"{PIECE_ID[piece.piece_type]} {uci[:2]} -> {uci[2:4]}"
        extra = ""
        if move.promotion:
            main += f", jadi {PIECE_ID[move.promotion]}!"
            extra = "pion promosi, pilih bidak promosi di layar"
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square)
            main += f", makan {PIECE_ID[victim.piece_type]}!"

    after = board.copy(stack=False)
    after.push(move)
    tags = []
    if after.is_checkmate():
        tags.append("SKAKMAT! Menang!")
    elif after.is_check():
        tags.append("Skak!")
    # ancaman baru oleh bidak yang jalan (maks 2 disebut)
    if not board.is_castling(move):
        mover = board.turn
        threats = []
        for sq in after.attacks(move.to_square):
            t = after.piece_at(sq)
            if t is not None and t.color != mover \
                    and t.piece_type != chess.KING:
                threats.append(f"{PIECE_ID[t.piece_type]} {chess.square_name(sq)}")
                if len(threats) >= 2:
                    break
        if threats and not after.is_checkmate():
            tags.append("ancam: " + ", ".join(threats))
    if tags:
        extra = (extra + ". " if extra else "") + " ".join(tags)
    if not extra:  # langkah biasa: tips umum per bidak
        piece = board.piece_at(move.from_square)
        extra = {"P": "dorong pion, kuasai ruang",
                 "N": "kuda ke tengah, kuasai petak",
                 "B": "gajah diagonal panjang",
                 "R": "benteng ke lajur terbuka",
                 "Q": "menteri aktif (jangan keluar terlalu awal!)",
                 "K": "amankan raja"}.get(piece.symbol().upper(),
                                          "perkuat posisi")
    return main, extra


def grid_key(fen, grid):
    """Kunci stabilitas: '?' dinormalisasi (posisi unknown ikut kunci,
    max 6; lebih dari itu = terlalu gelap, selalu belum stabil)."""
    unknowns = tuple(sorted(
        (r, c) for r in range(8) for c in range(8) if grid[r][c] == "?"))
    if len(unknowns) > 6:
        return None
    return fen.replace("?", "."), unknowns


def stable_scan(grab, templates, white_bottom, need, poll, tag="",
                on_frame=None):
    """Tunggu sampai `need` scan beruntun stabil. '?' (panah/highlight
    menutup bidak) DITOLERANSI max 6 sel: kuncinya posisi '?' + isi
    selebihnya. Return (fen, grid, warped) / (None, None, None) bila
    capture gagal. `on_frame` dipanggil tiap poll biar jendela hidup."""
    last, count, warped = None, 0, None
    episode_t0, seen, warned = time.time(), [], False
    while True:
        try:
            warped = grab()
        except Exception as e:
            print(f"[{tag}] gagal capture: {e}")
            return None, None, None
        fen, grid, _, _ = scan_to_fen(warped, templates, white_bottom)
        if on_frame is not None:
            on_frame(warped, fen)
        key = grid_key(fen, grid)
        if key is None:
            last, count = None, 0
            if fen not in seen:
                seen.append(fen)
            seen = seen[-5:]
        elif key == last:
            count += 1
        else:
            last, count = key, 1
            if fen not in seen:
                seen.append(fen)
            seen = seen[-5:]
        if count >= need:
            return fen, grid, warped
        if not warned and time.time() - episode_t0 > 30:
            warned = True
            print(f"[{tag}] ⚠️  30 detik tanpa posisi stabil "
                  f"(terakhir lihat {len(seen)} varian, misal: "
                  f"{seen[-1][:60] if seen else '-'}...). "
                  f"Kemungkinan: animasi belum selesai / dialog promosi / "
                  f"papan tertutup. Tetap menunggu...")
        time.sleep(poll)


def move_to_warp_cells(move, white_bottom=True):
    """Kotak from/to langkah -> (row, col) warp. row 0 = atas layar."""
    cells = []
    for sq in (move.from_square, move.to_square):
        f, r = chess.square_file(sq), chess.square_rank(sq) + 1
        row = (8 - r) if white_bottom else (r - 1)
        col = f if white_bottom else 7 - f
        cells.append((row, col))
    return cells


def wrap(text, width=32):
    """Potong teks panjang jadi beberapa baris."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + 1 + len(w) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def wrap(text, width=32):
    """Potong teks panjang jadi beberapa baris."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + 1 + len(w) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def sanitize(s):
    """Hershey OpenCV cuma ASCII: buang emoji/simbol (misal jam pasir)."""
    return str(s).encode("ascii", "ignore").decode()


# Geometri jendela: bar eval | papan | panel
VIEW_BAR, VIEW_BOARD, VIEW_PANEL, VIEW_H = 28, 560, 400, 560
TOTAL_W = VIEW_BAR + VIEW_BOARD + VIEW_PANEL


def status_color(status):
    """Warna pil status (bg, teks)."""
    s = status.lower()
    if any(k in s for k in ("darurat", "game over", "kalah", "batal",
                            "resync", "hilang", "menunggu", "ragu")):
        return (60, 60, 210), (255, 255, 255)
    if any(k in s for k in ("mikir", "animasi", "klik", "cari", "pilih",
                            "scan", "tunggu")):
        return (0, 170, 210), (0, 0, 0)
    return (40, 140, 40), (255, 255, 255)


def board_stats(board):
    """Ringkasan ramah-pemula: riwayat, materi, fase, skak."""
    out = {"moves": "-", "material": "Materi imbang", "phase": "-",
           "check": False, "check_side": ""}
    if board is None:
        return out
    if board.move_stack:
        try:
            toks = board.variation_san(board.move_stack).split()
            out["moves"] = " ".join(toks[-7:])
        except Exception:
            pass
    val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
           chess.QUEEN: 9, chess.KING: 0}
    pcs = list(board.piece_map().values())
    w = sum(val[p.piece_type] for p in pcs if p.color == chess.WHITE)
    b = sum(val[p.piece_type] for p in pcs if p.color == chess.BLACK)
    d = w - b
    out["material"] = "Materi imbang" if d == 0 else \
        f"Putih +{d}" if d > 0 else f"Hitam +{-d}"
    n = board.fullmove_number
    queens = sum(1 for p in pcs if p.piece_type == chess.QUEEN)
    if n <= 10:
        phase = "Pembukaan"
    elif queens == 0 or len(pcs) <= 10:
        phase = "Akhir"
    else:
        phase = "Tengah"
    out["phase"] = f"{phase} (langkah {n})"
    out["check"] = board.is_check()
    if out["check"]:
        out["check_side"] = "Putih" if board.turn == chess.WHITE else "Hitam"
    return out


def draw_eval_bar(h, eval_info, bottom_white):
    """Bar eval vertikal: penuh dari bawah = warna bawah unggul."""
    bar = np.zeros((h, VIEW_BAR, 3), dtype=np.uint8)
    bar[:] = (45, 45, 45)
    share = 0.5
    if eval_info is not None:
        cp, mate = eval_info
        if mate is not None:
            share = 1.0 if mate > 0 else 0.0
        else:
            share = 1.0 / (1.0 + 10.0 ** (-(cp or 0) / 400.0))
    wh = int(round(share * h))
    if bottom_white:
        cv2.rectangle(bar, (0, h - wh), (VIEW_BAR - 1, h - 1),
                      (240, 240, 240), -1)
    else:
        cv2.rectangle(bar, (0, 0), (VIEW_BAR - 1, wh - 1),
                      (240, 240, 240), -1)
    cv2.rectangle(bar, (0, 0), (VIEW_BAR - 1, h - 1), (200, 200, 200), 1)
    return bar


def eval_text(eval_info):
    if eval_info is None:
        return "Eval: - (tunggu Engine mikir)"
    cp, mate = eval_info
    if mate is not None:
        return f"Eval: Mat {abs(mate)} langkah " \
            f"({'Putih' if mate > 0 else 'Hitam'} menang)"
    v = (cp or 0) / 100
    if abs(v) < 0.15:
        return "Eval: 0.0 (Seimbang)"
    return f"Eval: {v:+.1f} ({'Putih' if v > 0 else 'Hitam'} unggul)"


def fmt_moves(hist, last_pairs=3):
    """[(nomor, putih?, san)] -> '10.e4 e5 11.Nf3' (terbaru saja)."""
    if not hist:
        return "-"
    pairs = {}
    for num, white, san in hist:
        pairs.setdefault(num, {})[white] = san
    out = []
    for num in sorted(pairs)[-last_pairs:]:
        w = pairs[num].get(True, "...")
        b = pairs[num].get(False, "")
        out.append(f"{num}.{w} {b}".strip())
    return "  ".join(out)
    if eval_info is None:
        return "Eval: - (tunggu Engine mikir)"
    cp, mate = eval_info
    if mate is not None:
        return f"Eval: Mat {abs(mate)} langkah " \
            f"({'Putih' if mate > 0 else 'Hitam'} menang)"
    v = (cp or 0) / 100
    if abs(v) < 0.15:
        return "Eval: 0.0 (Seimbang)"
    return f"Eval: {v:+.1f} ({'Putih' if v > 0 else 'Hitam'} unggul)"


def render_window(warped, white_bottom, status, board=None, last_move=None,
                  suggestion=None, confirm_buttons=False, button_rects=None,
                  gameover_buttons=False, bot_color=None, eval_info=None,
                  clock_str=None, moves_hist=None):
    """Susun citra jendela: bar eval | papan + highlight | panel info.

    status: teks pendek. suggestion: (uci, san, eval_str, kalimat,
      tambahan) / None. bot_color: warna bot (untuk label kamu/lawan).
    eval_info: (cp_putih, mate_putih) / None. clock_str: "2:41 (+2)".
    button_rects (dict, diisi bila diberi): nama -> (x0,y0,x1,y1) dalam
    koordinat CITRA JENDELA penuh.
    """
    vis = draw_overlay(warped, white_bottom, coords=False)
    cell = WARP_SIZE // 8

    def arrow_for(uci_move, color):
        f = FILES.index(uci_move[0])
        r = int(uci_move[1])
        t = FILES.index(uci_move[2])
        r2 = int(uci_move[3])
        if white_bottom:
            c1, rw1, c2, rw2 = f, 8 - r, t, 8 - r2
        else:
            c1, rw1, c2, rw2 = 7 - f, r - 1, 7 - t, r2 - 1
        p1 = (int((c1 + 0.5) * cell), int((rw1 + 0.5) * cell))
        p2 = (int((c2 + 0.5) * cell), int((rw2 + 0.5) * cell))
        cv2.arrowedLine(vis, p1, p2, color, 4, cv2.LINE_AA, tipLength=0.3)

    if last_move is not None:
        (r1, c1), (r2, c2) = move_to_warp_cells(last_move, white_bottom)
        overlay = vis.copy()
        cv2.rectangle(overlay, (c1 * cell, r1 * cell),
                      ((c1 + 1) * cell, (r1 + 1) * cell), (0, 255, 255), -1)
        cv2.rectangle(overlay, (c2 * cell, r2 * cell),
                      ((c2 + 1) * cell, (r2 + 1) * cell), (0, 255, 0), -1)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    if suggestion is not None:
        arrow_for(suggestion[0], (255, 120, 0))  # panah biru = saran
    vis_big = cv2.resize(vis, (VIEW_BOARD, VIEW_BOARD),
                         interpolation=cv2.INTER_LINEAR)

    stats = board_stats(board)
    bar = draw_eval_bar(VIEW_H, eval_info, bottom_white=white_bottom)

    panel = np.zeros((VIEW_H, VIEW_PANEL, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    y = 12

    def text(line, color=(220, 220, 220), scale=0.5, dy=24, x=12, bold=1):
        nonlocal y
        cv2.putText(panel, sanitize(line), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold,
                    cv2.LINE_AA)
        y += dy

    def sep():
        nonlocal y
        cv2.line(panel, (12, y), (VIEW_PANEL - 12, y), (70, 70, 70), 1)
        y += 12

    # 1. pil status
    bg, fg = status_color(status)
    cv2.rectangle(panel, (8, y - 20), (VIEW_PANEL - 8, y + 8), bg, -1)
    text(f" {status[:34]}", fg, 0.55, 34, 14, 2)
    # 2. giliran
    if board is not None:
        side = "PUTIH" if board.turn == chess.WHITE else "HITAM"
        merk = ""
        if bot_color is not None:
            merk = "kamu (Bot)" if board.turn == bot_color else "lawan"
            merk = f" -- {merk}"
        text(f"GILIRAN: {side}{merk}", (255, 255, 255), 0.62, 28, bold=2)
        if stats["check"]:
            text(f"SKAK! Raja {stats['check_side']} diserang",
                 (80, 80, 255), 0.62, 28, bold=2)
    sep()
    # 3. eval
    text(eval_text(eval_info), (150, 220, 255), 0.52)
    # 4. saran
    if suggestion is not None:
        _, san, ev, kalimat, tambahan = suggestion
        text("SARAN:", (0, 255, 0), 0.6, 26, bold=2)
        for ln in wrap(kalimat, 34):
            text(f"  {ln}", (0, 255, 0), 0.58, 24, bold=2)
        text(f"  ({san}, {ev})", (140, 255, 140), 0.45, 22)
        for ln in wrap(tambahan, 38):
            text(f"  {ln}", (200, 200, 200), 0.45, 20)
    else:
        text("SARAN: - (bukan giliran bot)", (130, 130, 130), 0.5)
    sep()
    # 5. riwayat + materi
    if board is not None:
        if last_move is not None:
            text(f"Langkah terakhir: {last_move.uci()}", (220, 220, 220),
                 0.5)
        for ln in wrap(f"Riwayat: {fmt_moves(moves_hist)}", 40):
            text(ln, (170, 170, 170), 0.45, 20)
        text(f"{stats['material']}  |  {stats['phase']}", (220, 200, 150),
             0.5)
    # 6. jam
    if clock_str:
        text(f"Jam: {clock_str}", (0, 255, 255), 0.75, 32, bold=2)
    y += 4
    buttons = confirm_buttons or gameover_buttons
    if not buttons:
        text("FEN:", (150, 200, 255), 0.5)
        if board is not None:
            for i in range(0, len(board.fen()), 38):
                text(board.fen()[i:i + 38], (200, 200, 200), 0.38, 18)
        y += 4
        text("Kuning=asal  Hijau=tujuan", (160, 160, 160), 0.42, 20)
        text("Biru=saran   Merah=skak", (160, 160, 160), 0.42, 20)
    text("Q / Esc keluar", (130, 130, 130), 0.45)
    full = np.hstack([bar, vis_big, panel])
    if buttons:
        def button(label, x0, y0, x1, y1, color):
            cv2.rectangle(full, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(full, (x0, y0), (x1, y1), (255, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.8, 2)
            cv2.putText(full, label, (x0 + (x1 - x0 - tw) // 2,
                                      y0 + (y1 - y0 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
                        cv2.LINE_AA)
            return (x0, y0, x1, y1)

        px = VIEW_BAR + VIEW_BOARD  # panel mulai di x ini
        if gameover_buttons:
            text_atas = "Main lagi?"
            (tw, _), _ = cv2.getTextSize(
                text_atas, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(full, text_atas, (px + (VIEW_PANEL - tw) // 2, 452),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            rects = {
                "main_lagi": button("MAIN LAGI !", px + 12, 462, px + 388,
                                    504, (0, 220, 0)),
                "quit": button("QUIT", px + 12, 510, px + 388, 548,
                               (60, 60, 220)),
            }
        else:
            text_atas = "Jalankan saran ini?"
            (tw, _), _ = cv2.getTextSize(
                text_atas, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(full, text_atas, (px + (VIEW_PANEL - tw) // 2, 442),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            rects = {
                "klik": button("KLIK !", px + 12, 452, px + 388, 490,
                               (0, 220, 0)),
                "skip": button("SKIP", px + 12, 496, px + 194, 530,
                               (0, 220, 220)),
                "quit": button("QUIT", px + 206, 496, px + 388, 530,
                               (60, 60, 220)),
            }
        if button_rects is not None:
            button_rects.update(rects)
    return full


def main():
    ap = argparse.ArgumentParser(description="Pelacak + penasihat catur")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--color", choices=["white", "black", "auto"],
                    default="auto", help="warna bot (auto = ikut orientasi)")
    ap.add_argument("--think-time", type=float, default=1.0)
    ap.add_argument("--elo", type=int, default=1400,
                    help="cap kekuatan Stockfish (1320-3190, 0 = full power)")
    ap.add_argument("--poll", type=float, default=0.3)
    ap.add_argument("--stable", type=int, default=3)
    ap.add_argument("--engine", type=str, default=ENGINE_PATH)
    ap.add_argument("--fen", type=str, default=None)
    ap.add_argument("--board", type=str, default="board.json")
    ap.add_argument("--templates", type=str, default="templates")
    ap.add_argument("--no-window", action="store_true",
                    help="tanpa jendela live (mode teks saja)")
    args = ap.parse_args()

    templates = load_templates(args.templates)
    import functools
    grab = functools.partial(grab_live_warp, args.board)

    # state jendela live
    ui = {"status": "mulai...", "last_move": None, "suggestion": None,
          "board": None, "warped": None, "quit": False, "shown": False,
          "red": [], "eval": None, "moves_san": []}

    def pump_window(warped=None):
        """Gambar ulang jendela; return True bila user minta keluar (Q)."""
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
                                bot_color=bot_color, eval_info=ui["eval"],
                                moves_hist=ui["moves_san"])
            cv2.imshow("Chess Suggest (Q keluar)", img)
            ui["shown"] = True
        except Exception as e:
            if not ui["shown"]:
                print(f"(jendela nonaktif: {e})")
                args.no_window = True
            return False
        return cv2.waitKey(1) & 0xFF in (ord("q"), 27)

    def on_frame(warped, fen_preview):
        ui["status"] = "animasi / belum stabil..."
        if pump_window(warped):
            ui["quit"] = True

    # orientasi cukup sekali di awal (papan tidak dibalik tengah game)
    white_bottom, detail = detect_orientation(grab(), templates)
    print(f"Orientasi: {'putih di bawah' if white_bottom else 'HITAM di bawah'}")
    bot_color = (chess.WHITE if white_bottom else chess.BLACK) \
        if args.color == "auto" else chess.WHITE \
        if args.color == "white" else chess.BLACK
    print(f"Bot pegang: {'putih' if bot_color == chess.WHITE else 'hitam'}")

    if args.fen:
        board = chess.Board(args.fen)
        print(f"Start dari FEN: {board.fen()}")
    else:
        print("Scan posisi...")
        ui["status"] = "scan posisi..."
        board = None
        while board is None:
            fen, grid, warped = stable_scan(grab, templates, white_bottom,
                                            args.stable, args.poll, tag="init",
                                            on_frame=on_frame)
            if grid is None:
                return 1
            hl = detect_highlights(warped, white_bottom)
            board, last0, info0 = init_board_from_scan(grid, hl["pair"],
                                                       white_bottom)
            if board is None:
                print(f"   init {info0}, ulangi...")
                time.sleep(args.poll)
        print(f"Highlight: hijau={hl['pair']} merah={hl['red']}")
        print(f"Posisi: {board.fen()} ({info0})")
        ui["last_move"] = last0
        ui["red"] = hl["red"]

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
        print(f"Kekuatan dibatasi: Elo ~{args.elo}")
    else:
        print("Kekuatan PENUH (mode 99%, gampang dicurigai!)")

    suggested_for = None  # fen terakhir yang sudah disarankan (anti-spam)

    def maybe_suggest():
        nonlocal suggested_for
        if board.turn != bot_color or board.fen() == suggested_for:
            return
        if board.is_game_over():
            print(f"Game over: {board.result()} ({board.outcome().termination})")
            return
        # raja kita merah di layar tapi internal tidak skak -> saran ditahan
        our_king = board.king(bot_color)
        if our_king is not None and not board.is_check() \
                and chess.square_name(our_king) in (ui.get("red") or []):
            print("🛑 raja MERAH di layar tapi internal tidak skak -> "
                  "saran ditahan, menunggu verifikasi.")
            return
        ui["status"] = "Stockfish mikir..."
        pump_window()
        move, s, cp, mate = think(engine, board, args.think_time)
        suggested_for = board.fen()
        san = board.san(move)
        main_id, extra_id = describe_move(board, move)
        ui["suggestion"] = (move.uci(), san, s, main_id, extra_id)
        ui["eval"] = (cp, mate)
        print(f"💡 SARAN: {main_id}  [{san}, {s}]")
        print(f"   {extra_id}")

    try:
        ui["board"] = board
        ui["status"] = "stabil"
        maybe_suggest()
        pump_window()
        if args.once:
            if not args.no_window:
                print("Tekan tombol apa saja di jendela untuk keluar.")
                cv2.waitKey(0)
            return 0
        print("Loop lacak jalan (Q / Ctrl+C berhenti). "
              "Mainkan langkah di browser...")
        lost_count = 0
        while True:
            if ui["quit"]:
                print("Keluar via jendela.")
                return 0
            if board.is_game_over():
                ui["status"] = f"game over {board.result()}"
                pump_window()
                print(f"Game over: {board.result()}")
                return 0
            fen, grid, warped = stable_scan(grab, templates, white_bottom,
                                            args.stable, args.poll, tag="loop",
                                            on_frame=on_frame)
            if ui["quit"]:
                print("Keluar via jendela.")
                return 0
            if grid is None:
                ui["status"] = "papan hilang, tunggu..."
                print("Papan hilang, tunggu...")
                time.sleep(1.0)
                continue
            hl = detect_highlights(warped, white_bottom)
            if hl["red"] != ui.get("red"):
                print(f"🔴 kotak merah: {hl['red'] or '-'} "
                      f"(hijau: {hl['pair']})")
            ui["red"] = hl["red"]
            hint = hl["pair"]
            placement = grid_to_placement(grid, white_bottom)
            if placement is None or placement == placement_of(board):
                ui["status"] = "stabil"
                ui["board"] = board
                if pump_window(warped):
                    return 0
                time.sleep(args.poll)
                continue
            move = match_move_sequence(board, placement, hint=hint,
                                           wild_grid=grid,
                                           white_bottom=white_bottom)
            if move[0] is None:
                lost_count += 1
                if lost_count >= 5:
                    print("⚠️  5x resync beruntun, cek papan pindah...")
                    maybe_recalibrate(capture_ppm, args.board)
                    lost_count = 0
                    continue
                print("⚠️  perubahan tidak cocok 1-2 langkah legal -> resync.")
                nb = board_from_scan(grid, white_bottom, not board.turn)
                if nb is None:
                    continue
                board = nb
                ui["last_move"] = None
                ui["status"] = "resync!"
                print(f"   posisi sekarang: {board.fen()}")
            else:
                moves, new_board = move
                for m in moves:
                    san = board.san(m)
                    ramah, _ = describe_move(board, m)
                    mover_is_bot = board.turn == bot_color
                    ui["moves_san"].append(
                        (board.fullmove_number,
                         board.turn == chess.WHITE, san))
                    board.push(m)
                    print(f"♟️  {'Bot' if mover_is_bot else 'Lawan'} "
                          f"jalan: {ramah}  [{san}]")
                assert board.fen() == new_board.fen()
                lost_count = 0
                ui["last_move"] = moves[-1]
                ui["status"] = "stabil"
            ui["board"] = board
            maybe_suggest()
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
