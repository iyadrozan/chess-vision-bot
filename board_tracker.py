"""
Eksperimen 2: Board tracker realtime.
- Deteksi papan SEKALI (pakai Hough dari detect_chessboard)
- Sesudahnya capture CROP area papan via grim (cepat, ~50ms)
- Warp ke top-down 8x8 + overlay grid + label kotak + FPS

Cara pakai:
  .venv/bin/python board_tracker.py --calibrate
      # deteksi papan di layar, print sudut + simpan ke board.json
  .venv/bin/python board_tracker.py --show
      # live preview overlay (Q / Ctrl+C untuk keluar)
  .venv/bin/python board_tracker.py --show --seconds 10
      # live preview 10 detik (buat ukur FPS stabil)
  .venv/bin/python board_tracker.py --image foto.png --save-warp warp.png
      # test pipeline di file statis (tanpa butuh papan terbuka)
"""

import argparse
import json
import subprocess
import sys
import time

import cv2
import numpy as np

from detect_chessboard import find_board_corners

WARP_SIZE = 480
BOARD_FILE = "board.json"


def capture_ppm(region=None):
    """Capture layar via `grim -t ppm`. region = (x, y, w, h) atau None
    (full screen). Return BGR numpy. PPM dipilih karena 5x lebih cepat
    dari PNG (tanpa encode/decode)."""
    cmd = ["grim", "-t", "ppm", "-"]
    if region is not None:
        x, y, w, h = [int(v) for v in region]
        cmd = ["grim", "-g", f"{x},{y} {w}x{h}", "-t", "ppm", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    if raw[:2] != b"P6":
        raise RuntimeError(f"output grim bukan PPM ({raw[:20]!r})")
    # parse header P6 robust (lewati komentar)
    parts, i = [], 2
    while len(parts) < 3:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            while raw[i:i + 1] != b"\n":
                i += 1
        else:
            j = i
            while not raw[j:j + 1].isspace():
                j += 1
            parts.append(raw[i:j])
            i = j
    w, h, _ = map(int, parts)
    i += 1
    rgb = np.frombuffer(raw[i:i + w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)


def order_corners(corners):
    """Urutkan 4 titik jadi (tl, tr, br, bl)."""
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    s, d = c.sum(axis=1), np.diff(c, axis=1).ravel()
    return np.array([c[np.argmin(s)], c[np.argmin(d)],
                     c[np.argmax(s)], c[np.argmax(d)]], dtype=np.float32)


def warp_board(frame, corners, size=WARP_SIZE):
    """Warp area papan ke tampak atas size x size."""
    src = order_corners(corners)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, m, (size, size)), m


def square_centers_screen(corners, white_bottom=True):
    """Pusat 64 kotak dalam koordinat layar. Return dict 'e2' -> (x, y).

    white_bottom=True: baris 1 di bawah (sudut bl). Kalau papan
    dibalik (hitam di bawah), pakai white_bottom=False.
    """
    tl, tr, br, bl = order_corners(corners)
    centers = {}
    files = "abcdefgh" if white_bottom else "hgfedcba"
    for r in range(8):      # r=0 -> baris atas papan di layar
        for f in range(8):  # f=0 -> kolom kiri papan di layar
            fr, fc = (r + 0.5) / 8, (f + 0.5) / 8
            top = tl + (tr - tl) * fc
            bot = bl + (br - bl) * fc
            x, y = top + (bot - top) * fr
            rank = 8 - r if white_bottom else r + 1
            centers[f"{files[f]}{rank}"] = (float(x), float(y))
    return centers


def draw_overlay(warped, white_bottom=True, coords=True):
    """Gambar grid + label file/rank di atas citra warp.
    coords=False bila situsnya sudah mencetak koordinat sendiri
    (biar tidak dobel, misal dipakai render_window)."""
    s = warped.shape[0]
    cell = s / 8
    vis = warped.copy()
    files = "abcdefgh" if white_bottom else "hgfedcba"
    for i in range(9):
        p = int(round(i * cell))
        cv2.line(vis, (p, 0), (p, s), (0, 255, 0), 1)
        cv2.line(vis, (0, p), (s, p), (0, 255, 0), 1)
    if not coords:
        return vis
    for r in range(8):
        rank = 8 - r if white_bottom else r + 1
        cv2.putText(vis, str(rank), (4, int((r + 0.5) * cell) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    for f in range(8):
        cv2.putText(vis, files[f], (int((f + 0.5) * cell) - 5, s - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return vis


def calibrate(save_path=BOARD_FILE):
    """Deteksi papan di layar sekarang, simpan sudut ke JSON."""
    print("Capture full screen untuk kalibrasi...")
    frame = capture_ppm()
    corners = find_board_corners(frame)
    if corners is None:
        print("❌ papan tidak ketemu di layar. Buka dulu game caturnya, "
              "terus ulangi.")
        return None
    data = {"corners": [[float(x), float(y)] for x, y in corners],
            "screen": {"w": frame.shape[1], "h": frame.shape[0]}}
    with open(save_path, "w") as f:
        json.dump(data, f)
    x0, y0 = corners.min(axis=0).astype(int)
    x1, y1 = corners.max(axis=0).astype(int)
    print(f"✅ papan ketemu: x0={x0} y0={y0} x1={x1} y1={y1} "
          f"({x1-x0}x{y1-y0}px) -> {save_path}")
    return np.array(data["corners"], dtype=np.float32)


def load_board(path=BOARD_FILE):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["corners"], dtype=np.float32)


def maybe_recalibrate(grab_full, board_path=BOARD_FILE, thresh=15,
                      verbose=True):
    """Cek apakah papan pindah (window digeser/resize). Kalau sudut baru
    ketemu dan bergeser > thresh px (dan ukurannya wajar), board.json
    di-update. Return (corners, pindah: bool).

    Pengaman botuiu nggak macet diam-diam: panggil berkala + saat scan
    berkali-kali gagal. grab_live_warp baca board.json tiap capture,
    jadi ROI otomatis ikut yang baru.
    """
    try:
        old = load_board(board_path)
    except (FileNotFoundError, KeyError, ValueError):
        old = None
    try:
        frame = grab_full()
    except Exception as e:
        if verbose:
            print(f"[recal] gagal capture full: {e}")
        return old, False
    new = find_board_corners(frame)
    if new is None:
        if verbose:
            print("[recal] papan tidak ketemu di layar penuh.")
        return old, False
    if old is not None:
        shift = float(np.abs(new - old).max())
        old_area = float((old[2][0] - old[0][0]) * (old[2][1] - old[0][1]))
        new_area = float((new[2][0] - new[0][0]) * (new[2][1] - new[0][1]))
        if shift <= thresh:
            return old, False
        if not (0.25 * old_area <= new_area <= 4.0 * old_area):
            if verbose:
                print(f"[recal] ukuran aneh ({new_area:.0f} vs {old_area:.0f}), "
                      f"diabaikan (bukan papan kita?).")
            return old, False
    data = {"corners": [[float(x), float(y)] for x, y in new],
            "screen": {"w": frame.shape[1], "h": frame.shape[0]}}
    with open(board_path, "w") as f:
        json.dump(data, f)
    if verbose:
        x0, y0 = new.min(axis=0).astype(int)
        x1, y1 = new.max(axis=0).astype(int)
        print(f"[recal] papan PINDAH -> kalibrasi ulang: "
              f"({x0},{y0})-({x1},{y1})")
    return new, True


def roi_for(corners, margin=30, sw=None, sh=None):
    """Kotak crop (x, y, w, h) di sekitar papan + margin, di-clip ke layar."""
    x0, y0 = corners.min(axis=0).astype(int) - margin
    x1, y1 = corners.max(axis=0).astype(int) + margin
    x0, y0 = max(0, x0), max(0, y0)
    if sw is not None:
        x1 = min(sw, x1)
    if sh is not None:
        y1 = min(sh, y1)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def run_live(board_path=BOARD_FILE, seconds=0, white_bottom=True):
    corners = load_board(board_path)
    full = capture_ppm()
    sh, sw = full.shape[:2]
    roi = roi_for(corners, sw=sw, sh=sh)
    # sudut papan relatif terhadap ROI (karena capture-nya crop)
    rel = corners - np.array([roi[0], roi[1]], dtype=np.float32)

    print(f"Tracking ROI {roi[2]}x{roi[3]}px. "
          f"{'Jalan ' + str(seconds) + ' detik.' if seconds else 'Q / Ctrl+C untuk keluar.'}")
    prev = None
    n, t0, diffs = 0, time.time(), []
    try:
        while True:
            t = time.time()
            crop = capture_ppm(region=roi)
            warped, _ = warp_board(crop, rel)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.abs(gray.astype(int) - prev.astype(int)).mean()))
            prev = gray
            vis = draw_overlay(warped, white_bottom)
            fps = (n + 1) / (time.time() - t0 + 1e-9)
            cv2.putText(vis, f"{fps:.1f} fps diff:{diffs[-1] if diffs else 0:.1f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Board Tracker (Q keluar)", vis)
            n += 1
            if seconds and time.time() - t0 >= seconds:
                break
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"Selesai: {n} frame dalam {dt:.1f}s = {n / max(dt, 1e-9):.1f} fps "
          f"(termasuk warp+overlay).")
    if diffs:
        print(f"Rata2 perubahan antar-frame: {sum(diffs)/len(diffs):.2f} "
              f"(kecil = papan diam, besar = ada gerakan/animasi).")


def run_image(path, save_warp=None, white_bottom=True):
    img = cv2.imread(path)
    if img is None:
        print(f"Gagal baca {path}", file=sys.stderr)
        return 1
    corners = find_board_corners(img)
    if corners is None:
        print("❌ papan tidak ketemu di gambar.")
        return 1
    print(f"✅ corners: {corners.tolist()}")
    warped, _ = warp_board(img, corners)
    vis = draw_overlay(warped, white_bottom)
    centers = square_centers_screen(corners, white_bottom)
    print(f"Pusat e2: {centers['e2']}, e7: {centers['e7']}")
    if save_warp:
        cv2.imwrite(save_warp, vis)
        print(f"Warp + overlay disimpan: {save_warp}")
    else:
        cv2.imshow("Warp test (tutup untuk keluar)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Board tracker realtime")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--save-warp", type=str, default=None)
    ap.add_argument("--board", type=str, default=BOARD_FILE)
    ap.add_argument("--black-bottom", action="store_true",
                    help="papan dibalik (hitam di bawah)")
    args = ap.parse_args()
    white_bottom = not args.black_bottom

    if args.calibrate:
        return 0 if calibrate(args.board) is not None else 1
    if args.image:
        return run_image(args.image, args.save_warp, white_bottom)
    # default live
    try:
        load_board(args.board)
    except FileNotFoundError:
        print(f"Belum ada {args.board}, kalibrasi dulu...")
        if calibrate(args.board) is None:
            return 1
    run_live(args.board, args.seconds, white_bottom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
