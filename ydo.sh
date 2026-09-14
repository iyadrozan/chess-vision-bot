#!/usr/bin/env bash
# Kontrol manual daemon ydotool (sengaja TIDAK di-enable via systemd).
# Pakai: ./ydo.sh on | off | status
set -u
SOCK="/run/user/1000/.ydotool_socket"
LOG="/tmp/ydotoold.log"

case "${1:-status}" in
  on)
    if pgrep -x ydotoold >/dev/null; then
      echo "ydotoold sudah jalan."
    else
      : > "$LOG"
      nohup ydotoold >"$LOG" 2>&1 &
      for _ in $(seq 1 20); do
        [ -S "$SOCK" ] && break
        sleep 0.25
      done
      if [ -S "$SOCK" ]; then
        echo "ydotoold ON (socket: $SOCK)"
      else
        echo "GAGAL start, lihat $LOG"
        exit 1
      fi
    fi
    ;;
  off)
    if pgrep -x ydotoold >/dev/null; then
      pkill -x ydotoold
      echo "ydotoold OFF."
    else
      echo "ydotoold memang lagi mati."
    fi
    ;;
  status)
    if pgrep -x ydotoold >/dev/null; then
      echo "ydotoold JALAN (socket: $([ -S "$SOCK" ] && echo ada || echo HILANG))"
    else
      echo "ydotoold MATI."
    fi
    ;;
  *)
    echo "Pakai: $0 on | off | status"
    exit 2
    ;;
esac
