# ♟️ Chess Vision Bot (DewaKipasASLI)

Bot catur yang main **murni dari piksel layar** — melihat papan, membaca
posisi, mikir pakai Stockfish, lalu klik langkahnya sendiri. Dibuat dan
teruji di **Hyprland/Wayland (Arch/CachyOS)** melawan komputer
SimpleChess/CrazyGames.

```
grim (crop papan) → warp top-down → klasifikasi 64 kotak → FEN
  → python-chess (lacak legalitas) → Stockfish (mikir)
  → ydotool (klik) → verifikasi papan berubah
```

## Fitur

- 👁️ **Pure vision** — tidak baca DOM/memory game, cuma screenshot
- 🧲 Deteksi papan tahan bidak (grid-fitting Hough + syarat persegi)
- ♟️ Klasifikasi bidak via edge-template matching (tahan highlight & panah)
- 🟩 Baca **highlight hijau** (langkah terakhir) & 🟥 **kotak merah** (skak)
- 🤖 Klik closed-loop terverifikasi (meleset = batal, tidak klik ngawur)
- 🧠 Stockfish 17 + cap Elo biar main kayak manusia (~1400)
- ⏱️ Mode turbo + jam adaptif buat blitz (darurat <10 dtk = jalan instan)
- 🪟 Jendela live: eval bar, riwayat, materi, jam, tombol klik/MAIN LAGI
- 🔄 Auto-rematch + kalibrasi ulang otomatis kalau window pindah

## Prasyarat

| Kebutuhan  | Keterangan                                   |
|------------|----------------------------------------------|
| OS         | Linux + Wayland (Hyprland teruji)            |
| Python     | 3.14 + venv                                  |
| grim       | `sudo pacman -S grim` (screenshot Wayland)   |
| ydotool    | `sudo pacman -S ydotool` (klik mouse)        |
| hyprctl    | bawaan Hyprland (baca posisi kursor/monitor) |
| Stockfish  | download manual (lihat instalasi)            |

## Instalasi

```bash
# 1. Clone + venv
git clone https://github.com/iyadrozan/chess-vision-bot.git
cd chess-vision-bot
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Engine Stockfish 17.1 (tidak ikut repo, ~78MB)
mkdir -p engines
curl -sL -o /tmp/sf.tar "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar"
tar -xf /tmp/sf.tar -C engines/ stockfish/stockfish-ubuntu-x86-64-avx2
mv engines/stockfish/stockfish-ubuntu-x86-64-avx2 engines/stockfish-bin
chmod +x engines/stockfish-bin && rm -rf /tmp/sf.tar engines/stockfish

# 3. Daemon klik (JANGAN di-enable, on/off manual)
sudo usermod -aG input $USER   # lalu logout/login sekali
./ydo.sh on                    # ./ydo.sh off | status
```

## Kalibrasi (wajib sekali per mesin)

```bash
# 1. Buka game catur di browser, pastikan papan terlihat
# 2. Cari papan + simpan koordinat ke board.json
.venv/bin/python board_tracker.py --calibrate

# 3. Dari posisi AWAL (buah lengkap, belum jalan), ekstrak template bidak
.venv/bin/python square_scan.py --extract
```

> `board.json` milik mesin masing-masing (tidak di-commit).
> Kalau window browser digeser, bot kalibrasi ulang otomatis.

## Cara pakai

```bash
# Lihat papan realtime + overlay grid (verifikasi tracking)
.venv/bin/python board_tracker.py --show

# Scan papan jadi FEN
.venv/bin/python square_scan.py --scan

# Mode penasihat (TANPA klik): lacak + saran bahasa Indonesia
.venv/bin/python suggest.py

# BOT MAIN (konfirmasi via tombol KLIK di jendela per langkah)
.venv/bin/python bot.py

# Turnamen cepat: tanpa konfirmasi + jam adaptif
.venv/bin/python bot.py --turbo --tc 180+2

# Test tanpa sentuh mouse
.venv/bin/python bot.py --dry-run --once --no-window
```

## Opsi penting `bot.py`

| Flag | Default | Arti |
|------|---------|------|
| `--turbo` | off | preset blitz: poll 0.15 dtk, mikir ≤0.3 dtk, tanpa konfirmasi |
| `--tc 180+2` | - | kontrol waktu (`180+2`, `3m+2`, `60+0`); mikir adaptif, darurat <10 dtk jalan instan |
| `--elo 1400` | 1400 | cap Stockfish (1320–3190, `0` = full power 99%) |
| `--delay-max 2.0` | 2.0 | jeda acak sebelum klik biar timing manusiawi (`0` = mati) |
| `--yes` | off | klik tanpa konfirmasi |
| `--color` | auto | warna bot (`white`/`black`, auto = ikut posisi papan) |
| `--no-window` | off | mode teks (konfirmasi via terminal) |

## Struktur proyek

| File | Peran |
|------|-------|
| `detect_chessboard.py` | Deteksi papan: pola klasik + grid-fitting Hough |
| `board_tracker.py` | Track ROI, warp top-down, overlay, auto-rekalibrasi |
| `square_scan.py` | 64 kotak → FEN (edge-template matching) |
| `highlights.py` | Deteksi highlight hijau (last move) & merah (skak) |
| `suggest.py` | Lacak langkah + saran Stockfish + jendela live |
| `bot.py` | Loop penuh: mikir → konfirmasi → klik → verifikasi → rematch |
| `clicker.py` | Wrapper ydotool closed-loop + cari tombol UI |
| `ydo.sh` | on/off/status daemon ydotoold |
| `templates/` | Template 12 bidak (hasil `--extract`) |
| `templates_ui/` | Template tombol UI browser (Play again) |
| `engines/` | Binary Stockfish (**download manual**, tidak di-commit) |

## Cara kerja pengaman (Failsafe)

- Klik selalu diverifikasi closed-loop via `hyprctl cursorpos`; meleset >8px = **batal klik**.
- Raja merah di layar tapi internal tidak skak = **dilarang jalan**, resync dulu.
- Panah & highlight situs di-mask/ditoleransi (`?` + wildcard matching).
- Promosi = klik manual di dialog (bot menunggu cocok).
- 5× resync beruntun = cek papan pindah + kalibrasi ulang otomatis.

## Troubleshooting

| Gejala | Solusi |
|--------|--------|
| Screenshot hitam (`mss`) | Normal di Wayland — sudah otomatis pakai `grim` |
| Kursor terbang ke pojok | `ydotool` absolut gain 2× di mesin ini; sudah dikompensasi (`ABS_GAIN`) + verify |
| `ydotool` gagal connect | `./ydo.sh on` dulu |
| Warning Qt `wayland plugin` | Tidak berbahaya (fallback XWayland), abaikan |
| Bot diam di layar review/analisis | Wajar — bukan game aktif; mulai game baru |
| Saran 99% mencurigakan | Default sudah cap `--elo 1400`; turunkan ke `1320` |

## Catatan

- Bot ini main **melawan komputer**, murni eksperimen computer vision.
  Jangan dipakai di game rating lawan manusia (melanggar ToS situs catur).
- Rating akun bereksperimen naik 1115 → 1464 selama pengembangan. 😄
