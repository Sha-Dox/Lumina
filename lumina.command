#!/bin/bash
# Lumina launcher — forced-native arch + logging
LOG="$HOME/.lumina/launch.log"
cd "$(dirname "$0")"
echo "=== $(date) launch | $(uname -m) ===" >> "$LOG"
INTERP=$(python3 -c "import sys; print(sys.executable)" 2>/dev/null || true)
if [ -z "$INTERP" ]; then INTERP=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; fi
if [ "$(uname -m)" = "arm64" ]; then
  exec /usr/bin/arch -arm64 "$INTERP" -u main.py "$@" >> "$LOG" 2>&1
else
  exec "$INTERP" -u main.py "$@" >> "$LOG" 2>&1
fi
