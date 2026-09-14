"""
Klik mouse via ydotool (Wayland/Hyprland).

Skala monitor = 1.0 (cek: `hyprctl monitors -j`), jadi koordinat grim
(piksel fisik) == koordinat klik. Kalau ganti monitor berskala, sesuaikan
--scale (koordinat_klik = koordinat_grim / scale).

Aman secara default: SEMUA aksi mouse nyata wajib flag --live.
Tanpa --live hanya print (dry-run). Daemon: ./ydo.sh on

Cara pakai:
  .venv/bin/python clicker.py --cursor
      # baca posisi kursor sekarang (aman)
  .venv/bin/python clicker.py --move 960 540 --live
      # geser kursor TANPA klik (buat kalibrasi visual)
  .venv/bin/python clicker.py --square e4 --live
      # klik tengah kotak e4 (papan dari board.json)
  .venv/bin/python clicker.py --square e4
      # dry-run: cuma print koordinatnya
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from board_tracker import load_board, square_centers_screen

MOVE_SETTLE = 0.08   # jeda gerak -> klik (detik)
CLICK_SETTLE = 0.15  # jeda habis klik
CLICK_TOL = 8        # toleransi posisi klik (px)
# ydotool absolut di mesin ini bergain 2x (perintah X mendarat di 2X,
# diukur 4 titik, error <=1px). Jangan hapus: tanpa ini kursor terbang
# ke pojok dan klik bisa mendarat sembarangan.
ABS_GAIN = 0.5
LEFT = "0xC0"        # tombol kiri ydotool


def need_daemon():
    if shutil.which("ydotool") is None:
        raise RuntimeError("ydotool tidak kepasang")
    r = subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ydotoold mati (nyalakan: ./ydo.sh on)")


def ydotool(*args):
    need_daemon()
    r = subprocess.run(["ydotool", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ydotool gagal: {r.stderr.strip()}")
    return r.stdout.strip()


def move(x, y, live=False, scale=1.0, verify=True):
    """Geser kursor ke absolut (x, y) koordinat grim, CLOSED-LOOP.

    ydotool absolut tidak akurat bila akselerasi mouse aktif (kursor
    bisa ngaco ke pojok), jadi posisi dibaca balik via hyprctl dan
    dikoreksi relatif sampai dalam toleransi. Tanpa --live cuma print.
    """
    cx, cy = float(x) / scale, float(y) / scale
    if not live:
        print(f"[dry-run] move -> ({cx:.0f}, {cy:.0f})")
        return cx, cy
    cmd_x, cmd_y = cx * ABS_GAIN, cy * ABS_GAIN
    for _ in range(5):
        ydotool("mousemove", "-a", "-x", str(int(round(cmd_x))),
                "-y", str(int(round(cmd_y))))
        time.sleep(MOVE_SETTLE)
        if not verify:
            break
        cur = get_cursor()
        if cur is None:
            break
        dx, dy = cx - cur[0], cy - cur[1]
        if abs(dx) <= CLICK_TOL and abs(dy) <= CLICK_TOL:
            break
        cmd_x += dx * ABS_GAIN
        cmd_y += dy * ABS_GAIN
    if verify:
        cur = get_cursor()
        if cur is None or abs(cur[0] - cx) > CLICK_TOL \
                or abs(cur[1] - cy) > CLICK_TOL:
            raise RuntimeError(f"kursor tidak mencapai target "
                               f"({cx:.0f},{cy:.0f}), terakhir di {cur}. "
                               f"Klik DIBATALKAN (aman).")
    return cx, cy


def click(live=False):
    if not live:
        print("[dry-run] click kiri")
        return
    ydotool("click", LEFT)
    time.sleep(CLICK_SETTLE)


def click_at(x, y, live=False, scale=1.0):
    move(x, y, live, scale, verify=True)  # raise bila meleset -> no klik
    click(live)


def click_square(sq, corners, white_bottom=True, live=False, scale=1.0):
    """Klik tengah kotak ('e4'). Return koordinat (klik)."""
    centers = square_centers_screen(np.asarray(corners, dtype=np.float32),
                                    white_bottom)
    if sq not in centers:
        raise ValueError(f"kotak {sq} tidak dikenal")
    x, y = centers[sq]
    print(f"klik {sq} di ({x:.0f}, {y:.0f})"
          f"{'' if live else ' [dry-run]'}")
    click_at(x, y, live, scale)
    return x, y


def get_cursor():
    """Posisi kursor via hyprctl. Return (x, y) / None."""
    try:
        r = subprocess.run(["hyprctl", "cursorpos"], capture_output=True,
                           text=True, timeout=5)
        x, y = r.stdout.strip().split(",")
        return int(x.strip()), int(y.strip())
    except Exception:
        return None


UI_TEMPLATE_DIR = "templates_ui"


def find_ui_button(full_bgr, name, threshold=0.8,
                   scales=(1.0, 0.9, 1.1, 0.8, 1.2), region=None):
    """Cari tombol UI browser (misal 'play_again') di screenshot.

    Multi-skala biar tahan zoom browser. region=(x0,y0,x1,y1) opsional
    untuk batasi area (misal kanan papan). Return (x, y) tengah tombol
    (koordinat grim == koordinat klik, scale 1) / None."""
    path = os.path.join(UI_TEMPLATE_DIR, f"{name}.png")
    tpl0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl0 is None:
        raise FileNotFoundError(f"template UI hilang: {path}")
    gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    ox, oy = 0, 0
    if region is not None:
        x0, y0, x1, y1 = [int(v) for v in region]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
        ox, oy = x0, y0
        gray = gray[y0:y1, x0:x1]
    best = (0.0, None)
    for s in scales:
        tw = max(8, int(tpl0.shape[1] * s))
        th = max(8, int(tpl0.shape[0] * s))
        tpl = cv2.resize(tpl0, (tw, th))
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        r = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), (ox + ml[0] + tw // 2, oy + ml[1] + th // 2))
    score, center = best
    return center if score >= threshold else None


def main():
    ap = argparse.ArgumentParser(description="Klik via ydotool")
    ap.add_argument("--live", action="store_true",
                    help="WAJIB untuk aksi mouse nyata (tanpa ini = dry-run)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cursor", action="store_true")
    ap.add_argument("--move", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--click-at", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--square", type=str, default=None)
    ap.add_argument("--board", type=str, default="board.json")
    ap.add_argument("--black-bottom", action="store_true")
    args = ap.parse_args()

    if args.cursor:
        print("kursor:", get_cursor())
        return 0
    if args.move:
        move(*args.move, live=args.live, scale=args.scale)
        print("kursor sekarang:", get_cursor())
        return 0
    if args.click_at:
        click_at(*args.click_at, live=args.live, scale=args.scale)
        return 0
    if args.square:
        corners = load_board(args.board)
        click_square(args.square, corners,
                     white_bottom=not args.black_bottom,
                     live=args.live, scale=args.scale)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
