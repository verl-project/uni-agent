#!/bin/sh
set -e
Xvfb "${DISPLAY:-:0}" -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do
  xdpyinfo -display "${DISPLAY:-:0}" >/dev/null 2>&1 && break
  sleep 0.2
done
exec "$@"
