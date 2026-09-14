"""
Eksperimen 3: Papan -> FEN dari piksel (pure vision).
- Ekstrak 12 template bidak + 2 template kotak kosong dari posisi awal
  (isinya diketahui pasti -> auto-label gratis, tema CrazyGames fix).
- Scan: tiap kotak dicocokkan ke 14 template (NCC, toleransi geser),
  pemenang = isi kotak. Orientasi papan dideteksi otomatis.
- Output: FEN + confidence.

Syarat ekstrak: papan live HARUS posisi awal (game baru, belum jalan).
Cara pakai:
  .venv/bin/python square_scan.py --extract
      # ambil template dari papan live -> templates/
  .venv/bin/python square_scan.py --scan
      # scan papan live -> print FEN
  .venv/bin/python square_scan.py --image foto.png
      # scan dari file (tidak usum, tidak usik layar)
  .venv/bin/python square_scan.py --extract --image foto.png
      # ekstrak template dari file posisi awal
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from board_tracker import (capture_ppm, draw_overlay, load_board, roi_for,
                           warp_board, WARP_SIZE)

TEMPLATE_DIR = "templates"
CELL = WARP_SIZE // 8          # 60
BORDER = 6                     # buang garis grid di tepi kotak
TSIZE = 44                     # ukuran template (px)
CONF_MIN = 0.5                 # skor NCC tepi minimum (bidak vs bidak)
EDGE_MIN = 30                  # piksel tepi minimum (kotak kosong ~0)

# template -> (kotak sumber di posisi awal, char FEN)
SOURCES = {
    "bR": ("a8", "r"), "bN": ("b8", "n"), "bB": ("c8", "b"),
    "bQ": ("d8", "q"), "bK": ("e8", "k"), "bP": ("a7", "p"),
    "wR": ("a1", "R"), "wN": ("b1", "N"), "wB": ("c1", "B"),
    "wQ": ("d1", "Q"), "wK": ("e1", "K"), "wP": ("a2", "P"),
    "empty_L": ("d5", None), "empty_D": ("e5", None),
}
FILES = "abcdefgh"


def sq_to_rc(sq, white_bottom=True):
    """'e2' -> (row, col) dalam citra warp (row 0 = atas layar)."""
    f = FILES.index(sq[0])
    r = int(sq[1:])
    row = (8 - r) if white_bottom else (r - 1)
    col = f if white_bottom else 7 - f
    return row, col


def cell_inner(warped, row, col):
    """Isi kotak tanpa garis grid tepi, grayscale."""
    y0, x0 = row * CELL + BORDER, col * CELL + BORDER
    cell = warped[y0:y0 + CELL - 2 * BORDER, x0:x0 + CELL - 2 * BORDER]
    return cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)


def cell_edges(warped_or_gray, row=None, col=None):
    """Peta tepi (Canny) sel 48x48. Bisa dipanggil dengan
    (warped, row, col) atau langsung citra gray seukuran sel/template.

    Panah merah situs (garis + tint skak) di-mask + inpaint dulu supaya
    tidak merusak klasifikasi. Template (gray, tanpa panah) lewat apa adanya.
    Syarat ekstrak template: papan bersih tanpa panah."""
    if row is None:
        gray = warped_or_gray
    else:
        y0, x0 = row * CELL + BORDER, col * CELL + BORDER
        s = CELL - 2 * BORDER
        bgr = warped_or_gray[y0:y0 + s, x0:x0 + s]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = (bgr[:, :, i].astype(int) for i in range(3))
        red = ((r - b > 50) & (r - g > 50)).astype(np.uint8) * 255
        if red.any():
            red = cv2.dilate(red, np.ones((3, 3), np.uint8), iterations=1)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)
            return cv2.bitwise_and(edges, edges, mask=255 - red)
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)


def extract_templates(warped, out_dir=TEMPLATE_DIR, white_bottom=True):
    """Potong template dari warp posisi awal. Return dict nama -> gambar."""
    os.makedirs(out_dir, exist_ok=True)
    tpl, meta = {}, {}
    m = (CELL - 2 * BORDER - TSIZE) // 2  # 48-44 -> margin 2
    for name, (sq, _) in SOURCES.items():
        row, col = sq_to_rc(sq, white_bottom)
        inner = cell_inner(warped, row, col)
        t = inner[m:m + TSIZE, m:m + TSIZE]
        tpl[name] = t
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), t)
        meta[name] = {"square": sq}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"✅ {len(tpl)} template disimpan di {out_dir}/ "
          f"(dari kotak: {[s for s, _ in SOURCES.values()]})")
    return tpl


def load_templates(dir_path=TEMPLATE_DIR):
    """Load template sebagai PETA TEPI (bukan piksel mentah).

    Alasan: NCC piksel didominasi warna background kotak (bidak kecil di
    tengah), sedangkan template kotak kosong yang datar bikin NCC
    degenerasi (=1.0 di mana-mana). Peta tepi hanya berisi bentuk bidak.
    Template empty_* tetap diekstrak (referensi) tapi tidak dipakai match.
    """
    tpl = {}
    for name in SOURCES:
        if name.startswith("empty"):
            continue
        p = os.path.join(dir_path, f"{name}.png")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"template hilang: {p} "
                                    f"(jalankan --extract dulu)")
        tpl[name] = cell_edges(img)
    return tpl


def classify_cell(warped, row, col, templates):
    """Return (nama/None, skor, skor_runner_up, edgepx). Bandingkan peta
    tepi sel dengan peta tepi tiap template (NCC + toleransi geser +-2px).
    Sel kosong (tepi < EDGE_MIN) langsung None tanpa matching (cepat)."""
    e = cell_edges(warped, row, col)
    edgepx = int((e > 0).sum())
    if edgepx < EDGE_MIN:
        return None, 0.0, 0.0, edgepx
    ranked = sorted(
        ((name, float(cv2.matchTemplate(e, t, cv2.TM_CCOEFF_NORMED).max()))
         for name, t in templates.items()),
        key=lambda kv: -kv[1])
    (best, best_score), second = ranked[0], ranked[1][1]
    return best, best_score, second, edgepx


def detect_orientation(warped, templates):
    """Tebak putih-di-bawah atau hitam-di-bawah via vote 4 sudut.
    Tiap sudut: skor template putih terbaik vs hitam terbaik."""
    corners = [("a1", 7, 0), ("h1", 7, 7), ("a8", 0, 0), ("h8", 0, 7)]
    piece_names = list(templates)
    whites = [n for n in piece_names if n.startswith("w")]
    blacks = [n for n in piece_names if n.startswith("b")]

    def side_score(rows):
        """Rata2 skor tepi template putih vs hitam di baris warp tertentu."""
        sws, sbs = [], []
        for _, row, col in [c for c in corners if c[1] in rows]:
            e = cell_edges(warped, row, col)
            sws.append(max(float(cv2.matchTemplate(e, templates[n],
                                                   cv2.TM_CCOEFF_NORMED).max())
                           for n in whites))
            sbs.append(max(float(cv2.matchTemplate(e, templates[n],
                                                   cv2.TM_CCOEFF_NORMED).max())
                           for n in blacks))
        return float(np.mean(sws)), float(np.mean(sbs))

    sw_low, sb_low = side_score((7,))    # baris bawah layar
    sw_high, sb_high = side_score((0,))  # baris atas layar
    if max(sw_low, sb_low, sw_high, sb_high) < CONF_MIN:
        return True, "default (semua sudut kosong/tidak yakin)"
    white_bottom = (sw_low - sb_low) > (sw_high - sb_high)
    detail = (f"bawah: W={sw_low:.2f}/B={sb_low:.2f} "
              f"atas: W={sw_high:.2f}/B={sb_high:.2f}")
    return white_bottom, detail


def scan_to_fen(warped, templates, white_bottom=True):
    """Klasifikasi 64 kotak -> (fen_board, grid_char, min_conf, lows).

    Aturan: tepi sedikit (EDGE_MIN) = kosong. Skor < CONF_MIN padahal
    banyak tepi = tidak yakin ('?'), misal highlight/seleksi. Pengecualian:
    oklusi panah (skor 0.4+ tapi runner-up jauh di bawah) tetap diterima.
    Warna kotak kosong tidak perlu template (tak ada di FEN)."""
    chars = {name: fen for name, (_, fen) in SOURCES.items() if fen}
    grid, min_conf, lows = [], 1.0, []
    for r in range(8):
        row_chars = []
        for c in range(8):
            name, score, second, edgepx = classify_cell(warped, r, c,
                                                       templates)
            if edgepx < EDGE_MIN:
                row_chars.append(None)  # kosong, yakin
                continue
            sure = score >= CONF_MIN or (score >= 0.4 and second <= 0.2)
            if not sure:
                row_chars.append("?")
                file = FILES[c] if white_bottom else FILES[7 - c]
                lname = f"{file}{8 - r if white_bottom else r + 1}"
                lows.append((lname, name, round(score, 2), edgepx))
            else:
                min_conf = min(min_conf, score)
                row_chars.append(chars[name])
        grid.append(row_chars)

    # susun FEN dari rank 8 ke 1
    ranks = []
    order = range(8) if white_bottom else range(7, -1, -1)
    for r in order:
        rank, empty = "", 0
        cols = range(8) if white_bottom else range(7, -1, -1)
        for c in cols:
            ch = grid[r][c]
            if ch is None:
                empty += 1
            else:
                if empty:
                    rank += str(empty)
                    empty = 0
                rank += ch
        if empty:
            rank += str(empty)
        ranks.append(rank)
    return "/".join(ranks), grid, min_conf, lows


def print_board(grid, white_bottom=True):
    order = range(8) if white_bottom else range(7, -1, -1)
    for r in order:
        rank = 8 - r if white_bottom else r + 1
        print(f"{rank} " + " ".join(ch if ch else "." for ch in grid[r]))


def grab_live_warp(board_path="board.json"):
    corners = load_board(board_path)
    full = capture_ppm()
    sh, sw = full.shape[:2]
    roi = roi_for(corners, sw=sw, sh=sh)
    crop = capture_ppm(region=roi)
    rel = corners - np.array([roi[0], roi[1]], dtype=np.float32)
    warped, _ = warp_board(crop, rel)
    return warped


def main():
    ap = argparse.ArgumentParser(description="Papan -> FEN (pure vision)")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--board", type=str, default="board.json")
    ap.add_argument("--templates", type=str, default=TEMPLATE_DIR)
    args = ap.parse_args()

    if args.extract:
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"Gagal baca {args.image}", file=sys.stderr)
                return 1
            from detect_chessboard import find_board_corners
            corners = find_board_corners(img)
            if corners is None:
                print("❌ papan tidak ketemu di gambar.")
                return 1
            warped, _ = warp_board(img, corners)
        else:
            warped = grab_live_warp(args.board)
        extract_templates(warped, args.templates)
        cv2.imwrite("/tmp/opencode/extract_src.png", draw_overlay(warped))
        print("Sumber warp disimpan: /tmp/opencode/extract_src.png "
              "(cek: harus posisi awal!)")
        return 0

    # default: scan
    templates = load_templates(args.templates)
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Gagal baca {args.image}", file=sys.stderr)
            return 1
        from detect_chessboard import find_board_corners
        corners = find_board_corners(img)
        if corners is None:
            print("❌ papan tidak ketemu di gambar.")
            return 1
        warped, _ = warp_board(img, corners)
    else:
        warped = grab_live_warp(args.board)

    white_bottom, detail = detect_orientation(warped, templates)
    print(f"Orientasi: {'putih di bawah' if white_bottom else 'HITAM di bawah'} "
          f"({detail})")
    fen, grid, min_conf, lows = scan_to_fen(warped, templates, white_bottom)
    print_board(grid, white_bottom)
    print(f"FEN: {fen}")
    print(f"confidence terendah: {min_conf:.2f} "
          f"(ambang {CONF_MIN})")
    if lows:
        print(f"⚠️  {len(lows)} kotak tidak yakin: {lows}")
        return 2
    if "?" in fen:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
