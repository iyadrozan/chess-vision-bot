"""
Eksperimen 1: Deteksi papan catur di layar.
- Screenshot layar pake mss
- Deteksi pake OpenCV (findChessboardCorners + deteksi grid kotak)
- Kalau ketemu: print "PAPAN CATUR TERDETEKSI!"

Cara pakai:
  .venv/bin/python detect_chessboard.py --once        # cek sekali
  .venv/bin/python detect_chessboard.py --loop        # cek terus tiap 1 detik
  .venv/bin/python detect_chessboard.py --test        # test pakai gambar sintetis
  .venv/bin/python detect_chessboard.py --image foto.png  # test pakai file gambar
  .venv/bin/python detect_chessboard.py --loop --show # tampilkan jendela preview
"""

import argparse
import time
import sys

import cv2
import numpy as np


# Pola yang umum untuk papan kalibrasi / papan catur polos.
# (7,7) = papan 8x8 kotak -> 7x7 sudut dalam. Ini yang paling standar.
CHESSBOARD_PATTERNS = [(7, 7), (8, 8), (7, 5), (5, 5), (6, 6), (9, 6)]


def detect_classic_pattern(gray):
    """Deteksi pola papan catur klasik (kotak hitam-putih polos, tanpa bidak).

    Return: (found: bool, pattern: tuple|None)
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for pattern in CHESSBOARD_PATTERNS:
        found, _ = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            return True, pattern
    # Coba juga versi SB yang lebih robust (kalau OpenCV support)
    try:
        for pattern in CHESSBOARD_PATTERNS:
            found, _ = cv2.findChessboardCornersSB(gray, pattern)
            if found:
                return True, pattern
    except Exception:
        pass
    return False, None


def _cluster_lines(positions, tol=8):
    """Kelompokkan posisi garis yang berdekatan (satu garis fisik
    sering terdeteksi sebagai beberapa segmen)."""
    positions = sorted(positions)
    groups = []
    for p in positions:
        if groups and p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [sum(g) / len(g) for g in groups]


def _find_grid_lines(gray):
    """Ekstraksi garis grid horizontal/vertikal. Return (h_lines, v_lines)
    berupa posisi piksel yang sudah di-cluster. Dipakai detect_grid_board
    dan board_tracker (biar tidak duplikasi logika)."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(50, min(h, w) // 8),
        minLineLength=max(80, int(min(h, w) * 0.2)),
        maxLineGap=15,
    )
    if lines is None:
        return [], []

    hors, vers = [], []  # posisi y garis horizontal / posisi x garis vertikal
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        length = (dx ** 2 + dy ** 2) ** 0.5
        if dx > dy * 5 and length > w * 0.15:
            hors.append((int(y1) + int(y2)) / 2)
        elif dy > dx * 5 and length > h * 0.15:
            vers.append((int(x1) + int(x2)) / 2)

    return _cluster_lines(hors), _cluster_lines(vers)


def _run_step(positions, tol_ratio=0.3):
    """Median spasi dari rangkaian reguler terpanjang. Return (step, n_garis).
    n_garis = jumlah garis dalam run (9 = papan utuh)."""
    pos = sorted(positions)
    if len(pos) < 2:
        return 0.0, len(pos)
    diffs = np.diff(pos)
    med = float(np.median(diffs))
    if med <= 0:
        return 0.0, len(pos)
    ok = [abs(d - med) <= tol_ratio * med for d in diffs]
    best, cur = 1, 1
    for good in ok:
        cur = cur + 1 if good else 1
        best = max(best, cur)
    # step dihitung ulang dari run terpanjang biar tidak kepengaruh outlier
    return med, best


def _fit_grid(lines, step, tol=8, need=7):
    """Cocokkan 9 garis grid (8*step) ke posisi terdeteksi.

    Tiap garis terdeteksi dicoba sebagai tepi atas maupun tepi bawah.
    Return (skor 0..9, y0) terbaik; skor = jumlah dari 9 posisi ideal
    yang ada garis dekatnya. Butuh skor >= need."""
    if step <= 0 or not lines:
        return 0, 0.0
    best = (0, 0.0)
    for anchor in lines:
        for y0 in (anchor, anchor - 8 * step):
            grid = [y0 + i * step for i in range(9)]
            score = sum(1 for g in grid
                        if any(abs(l - g) <= tol for l in lines))
            if score > best[0]:
                best = (score, y0)
    return best if best[0] >= need else (best[0], None)


def _strip_lines(bgr, axis, lo, hi, margin=12):
    """Deteksi ulang garis sumbu lawan di dalam strip [lo, hi] + margin.
    axis='v' -> garis vertikal papan diketahui, cari garis horizontal
    di strip vertikal x∈[lo,hi]. Return (cross_lines, cross_step, cross_n)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if axis == "v":
        x0, x1 = max(0, int(lo - margin)), min(w, int(hi + margin))
        strip = gray[:, x0:x1]
    else:
        y0, y1 = max(0, int(lo - margin)), min(h, int(hi + margin))
        strip = gray[y0:y1, :]
    lines_h, lines_v = _find_grid_lines(strip)
    cross = lines_h if axis == "v" else lines_v
    step, n = _run_step(cross)
    return cross, step, n


def find_board_corners(bgr):
    """Cari sudut papan (tl, tr, br, bl) sebagai float32 (4,2).
    Return None kalau tidak ada grid 8x8 yang valid.

    Strateginya: pakai sumbu terkuat sebagai jangkar (butuh run >=7
    garis), lalu deteksi ulang sumbu lawan di dalam strip + cocokkan
    grid 9 garis (toleransi 2 garis hilang karena oklusi).
    Terakhir wajib kotak persegi (aspek 0.88..1.12)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _find_grid_lines(gray)
    h_step, h_n = _run_step(h_lines)
    v_step, v_n = _run_step(v_lines)

    cands = []
    if v_n >= 7:
        cands.append(("v", v_lines, v_step))
    if h_n >= 7:
        cands.append(("h", h_lines, h_step))
    # coba jangkar terkuat dulu
    cands.sort(key=lambda c: _run_step(c[1])[1], reverse=True)

    for axis, anchor_lines, step in cands:
        score_a, edge_a = _fit_grid(anchor_lines, step)
        if edge_a is None:
            continue
        lo_a, hi_a = edge_a, edge_a + 8 * step
        cross, _, _ = _strip_lines(bgr, axis, lo_a, hi_a)
        score_c, edge_c = _fit_grid(cross, step)
        if edge_c is None:
            continue
        lo_c, hi_c = edge_c, edge_c + 8 * step
        # wajib persegi (toleransi 12%)
        wa, wb = hi_a - lo_a, hi_c - lo_c
        if not (0.88 <= wa / max(wb, 1e-9) <= 1.12):
            continue
        if wa * wb < h * w * 0.03:
            continue
        if axis == "v":
            x0, x1, y0, y1 = lo_a, hi_a, lo_c, hi_c
        else:
            x0, x1, y0, y1 = lo_c, hi_c, lo_a, hi_a
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        dtype=np.float32)
    return None


def detect_grid_board(bgr, debug_path=None):
    """Deteksi papan catur ala game (chess.com / lichess / CrazyGames,
    ada bidak) via grid fitting Hough.

    Idenya: papan catur = 9 garis horizontal + 9 garis vertikal yang
    panjang dan berjarak reguler (boleh 2 garis hilang karena oklusi
    bidak/popup). Hasilnya harus persegi. Bohongan umum (teks terminal,
    border jendela) gugur di fitting / syarat persegi.

    Return: (found: bool, info: dict)
    """
    h, w = bgr.shape[:2]
    corners = find_board_corners(bgr)

    if debug_path:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h_lines, v_lines = _find_grid_lines(gray)
        vis = bgr.copy()
        for y in h_lines:
            cv2.line(vis, (0, int(y)), (w - 1, int(y)), (0, 255, 0), 2)
        for x in v_lines:
            cv2.line(vis, (int(x), 0), (int(x), h - 1), (255, 0, 0), 2)
        if corners is not None:
            box = corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [box], True, (0, 255, 255), 3)
        cv2.imwrite(debug_path, vis)

    if corners is None:
        return False, {"reason": "tidak ada grid 8x8 persegi yang valid"}

    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    board_area = float((x1 - x0) * (y1 - y0))
    return True, {"box": [round(float(v), 1) for v in (x0, y0, x1, y1)],
                  "board_area_pct": round(100 * board_area / (h * w), 1)}


def is_chessboard(bgr, debug_path=None):
    """Gabungan kedua metode. Return (found, metode, detail)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    found, pattern = detect_classic_pattern(gray)
    if found:
        return True, "pola-klasik", f"pattern {pattern}"

    found, info = detect_grid_board(bgr, debug_path=debug_path)
    if found:
        return True, "grid-garis", str(info)

    return False, "-", str(info)


def grab_screen():
    """Screenshot layar. Return gambar BGR (numpy).

    Di Wayland (Hyprland dsb.) mss hanya dapat gambar hitam karena
    protokolnya tidak mengizinkan screen-grab via XSHM, jadi pakai
    `grim` dulu, baru fallback ke mss (X11).
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("grim"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(["grim", tmp], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            img = cv2.imread(tmp)
            if img is None:
                raise RuntimeError("grim gagal membaca hasil screenshot")
            return img
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    import mss
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # 1 = monitor utama
        shot = sct.grab(monitor)
        img = np.array(shot)  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def make_synthetic_board(squares=8, cell=60):
    """Bikin gambar papan catur sintetis buat testing (tanpa perlu buka gambar)."""
    size = squares * cell
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(squares):
        for c in range(squares):
            color = 255 if (r + c) % 2 == 0 else 0
            board[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = color
    # Kasih border putih biar mirip screenshot
    board = cv2.copyMakeBorder(board, 40, 40, 40, 40,
                               cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return board


def make_random_image(w=640, h=480):
    """Gambar acak (bukan papan catur) buat negative test."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def check_and_report(img, show=False, debug_path=None):
    found, metode, detail = is_chessboard(img, debug_path=debug_path)
    if found:
        print(f"✅ PAPAN CATUR TERDETEKSI! (metode={metode}, {detail})")
    else:
        print(f"❌ tidak ada papan catur. ({detail})")
    if show:
        preview = img.copy()
        h, w = preview.shape[:2]
        scale = min(1.0, 900 / max(h, w))
        if scale < 1.0:
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)))
        label = "TERDETEKSI" if found else "tidak ada"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(preview, f"Papan catur: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Deteksi Papan Catur (tekan Q untuk keluar)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return found


def main():
    ap = argparse.ArgumentParser(description="Deteksi papan catur di layar")
    ap.add_argument("--once", action="store_true", help="cek layar sekali")
    ap.add_argument("--loop", action="store_true", help="cek layar terus-menerus")
    ap.add_argument("--interval", type=float, default=1.0, help="jeda antar cek (detik)")
    ap.add_argument("--test", action="store_true", help="test pakai gambar sintetis")
    ap.add_argument("--image", type=str, default=None, help="test pakai file gambar")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela preview")
    ap.add_argument("--debug", type=str, default=None, metavar="OUT.PNG",
                    help="simpan gambar debug berisi garis grid yang terdeteksi")
    args = ap.parse_args()

    # Default: cek sekali kalau tidak ada flag
    if not (args.once or args.loop or args.test or args.image):
        args.once = True

    if args.test:
        print("--- Test 1: gambar papan catur sintetis (HARUS terdeteksi) ---")
        board = make_synthetic_board()
        r1 = check_and_report(board, show=args.show)
        print("--- Test 2: gambar acak (HARUS tidak terdeteksi) ---")
        noise = make_random_image()
        r2 = check_and_report(noise, show=args.show)
        print()
        if r1 and not r2:
            print("🎉 TEST LOLOS: detektor bekerja dengan benar.")
            return 0
        print("⚠️  TEST GAGAL: cek logika deteksi.")
        return 1

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Gagal baca gambar: {args.image}", file=sys.stderr)
            return 2
        check_and_report(img, show=args.show, debug_path=args.debug)
        return 0

    if args.loop:
        print("Mengecek layar tiap "
              f"{args.interval} detik... tekan Ctrl+C untuk berhenti.\n")
        try:
            while True:
                try:
                    img = grab_screen()
                except Exception as e:
                    print(f"Gagal screenshot: {e}")
                    print("Kalau di Wayland, pastikan portal screenshare jalan "
                          "atau pakai --image / --test dulu.")
                    return 3
                check_and_report(img, show=args.show, debug_path=args.debug)
                if args.show:
                    # kalau preview ditutup, lanjut loop
                    pass
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nBerhenti.")
            return 0

    # --once
    try:
        img = grab_screen()
    except Exception as e:
        print(f"Gagal screenshot: {e}")
        return 3
    check_and_report(img, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
