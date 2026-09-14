"""
Deteksi highlight papan: hijau = langkah terakhir, merah = raja diskak.

Prinsip: self-calibrating, tanpa warna bawaan tema.
- Warna dasar = median ring tepi (5px terluar isi kotak) per paritas.
  Ring tepi dipakai biar kebal bidak (bidak di tengah, highlight di
  seluruh kotak).
- Highlight = outlier: sel yang menyimpang jauh dari median paritasnya
  ke arah hijau / merah.
- Hijau harus tepat 2 sel (kotak asal + tujuan) -> pasangan last-move.
- Merah -> list kotak (biasanya 1: raja yang skak).

Return square names, jadi butuh white_bottom.
"""

import cv2
import numpy as np

from square_scan import CELL, BORDER, FILES

RING = 5          # lebar ring tepi (px) di dalam isi kotak
GREEN_G = 12.0    # margin skor-hijau di atas median
GREEN_D = 10.0    # jarak warna minimum sel hijau
RED_R = 25.0      # margin skor-merah di atas median
RED_D = 15.0      # jarak warna minimum sel merah


def ring_stat(warped, row, col):
    """Median BGR ring tepi sel (48x48 -> pinggir 5px). Median (bukan mean)
    biar kebal bidak besar yang kakinya nyentuh pinggir ring."""
    y0, x0 = row * CELL + BORDER, col * CELL + BORDER
    s = CELL - 2 * BORDER  # 48
    cell = warped[y0:y0 + s, x0:x0 + s].astype(float)
    mask = np.zeros((s, s), bool)
    mask[:RING, :] = True
    mask[-RING:, :] = True
    mask[:, :RING] = True
    mask[:, -RING:] = True
    return np.median(cell[mask], axis=0)  # BGR


# alias lama
def ring_mean(warped, row, col):
    return ring_stat(warped, row, col)


def rc_to_sq(row, col, white_bottom=True):
    rank = 8 - row if white_bottom else row + 1
    files = FILES if white_bottom else FILES[::-1]
    return f"{files[col]}{rank}"


def detect_highlights(warped, white_bottom=True):
    """Return {'pair': (sqA, sqB) | None, 'red': [sq...],
    'green_cells': [...], 'debug': {...}}."""
    rings, green_score, red_score, dist = {}, {}, {}, {}
    light_rings, dark_rings = [], []
    for r in range(8):
        for c in range(8):
            m = ring_stat(warped, r, c)
            rings[(r, c)] = m
            (light_rings if (r + c) % 2 == 0 else dark_rings).append(m)
    base_l = np.median(np.array(light_rings), axis=0)
    base_d = np.median(np.array(dark_rings), axis=0)
    g_med = np.median([m[1] - (m[0] + m[2]) / 2 for m in rings.values()])
    r_med = np.median([m[2] - (m[0] + m[1]) / 2 for m in rings.values()])

    green, red = [], []
    for (r, c), m in rings.items():
        base = base_l if (r + c) % 2 == 0 else base_d
        d = float(np.abs(m - base).mean())
        dist[(r, c)] = d
        g = float(m[1] - (m[0] + m[2]) / 2)
        rr = float(m[2] - (m[0] + m[1]) / 2)
        green_score[(r, c)] = g
        red_score[(r, c)] = rr
        if rr > r_med + RED_R and d > RED_D:
            red.append((r, c))
        elif g > g_med + GREEN_G and d > GREEN_D:
            green.append((r, c))

    pair = None
    if len(green) == 2:
        (r1, c1), (r2, c2) = green
        pair = (rc_to_sq(r1, c1, white_bottom),
                rc_to_sq(r2, c2, white_bottom))
    return {
        "pair": pair,
        "red": [rc_to_sq(r, c, white_bottom) for r, c in red],
        "green_cells": [rc_to_sq(r, c, white_bottom) for r, c in green],
        "debug": {"g_med": round(float(g_med), 1),
                  "r_med": round(float(r_med), 1),
                  "n_green": len(green), "n_red": len(red)},
    }
