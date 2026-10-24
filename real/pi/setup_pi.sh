#!/usr/bin/env bash
# Set up the SO-101 arm server on a Raspberry Pi (or any Linux box): Ubuntu / Debian / Raspberry Pi OS,
# aarch64 or x86_64, Python 3.11 - 3.14. Installs ONLY pyserial (+ numpy, optional) into real/.venv-pi.
#
# Run on the Pi, in the unpacked bundle (or repo) root:
#     bash real/pi/setup_pi.sh                    # online, or offline if real/pi/wheelhouse exists
#     bash real/pi/setup_pi.sh --service          # + systemd user service "printpick-arm" (starts at boot)
#     bash real/pi/setup_pi.sh --no-udev          # skip the udev rule (needs sudo)
#     bash real/pi/setup_pi.sh --no-root          # never call sudo (prints the root commands instead)
# The bundle comes from the Mac: bash real/pi/make_wheelhouse.sh  (wheels for the Pi's Python + this code).
set -euo pipefail
USER="${USER:-$(id -un)}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/real/.venv-pi"
WHEELHOUSE="$ROOT/real/pi/wheelhouse"
PY="${PYTHON:-python3}"
SERVICE=0
UDEV=1
SERIAL_PORT=""
NOROOT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --service) SERVICE=1; shift ;;
    --no-udev) UDEV=0; shift ;;
    --no-root) NOROOT=1; shift ;;
    --port) SERIAL_PORT="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
SUDO=""
if [ "$NOROOT" = 1 ]; then SUDO="none"
elif [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO="none"; fi
fi
asroot() {  # run a command as root, or print it if that is impossible
  if [ "$SUDO" = "none" ]; then echo "   (no sudo) run as root: $*"; return 1; fi
  $SUDO "$@"
}

# ---------------------------------------------------------------- 1. python
say "python"
command -v "$PY" >/dev/null || { echo "no $PY: sudo apt install python3 python3-venv"; exit 1; }
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "   $PY = $PYV on $(uname -m), $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
case "$PYV" in 3.11|3.12|3.13|3.14) ;; *) echo "   WARNING: tested for Python 3.11 - 3.14 (stdlib + pyserial only, so newer is probably fine)";; esac

# ---------------------------------------------------------------- 2. venv
say "virtual environment $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  if ! "$PY" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    echo "   'python3 -m venv' failed (Ubuntu/Debian ship ensurepip separately)."
    if asroot apt-get install -y "python${PYV}-venv" 2>/dev/null || asroot apt-get install -y python3-venv 2>/dev/null; then
      "$PY" -m venv "$VENV"
    else
      echo "   no apt access: venv without pip, then pip from the wheelhouse"
      "$PY" -m venv --without-pip "$VENV"
      PIPWHL="$(ls "$WHEELHOUSE"/pip-*.whl 2>/dev/null | head -1 || true)"
      [ -n "$PIPWHL" ] || { echo "   no pip wheel in $WHEELHOUSE: make the wheelhouse on the Mac (make_wheelhouse.sh)"; exit 1; }
      "$VENV/bin/python" "$PIPWHL/pip" install --no-index "$PIPWHL"     # pip runs from its own wheel
    fi
  fi
fi
VPY="$VENV/bin/python"

# ---------------------------------------------------------------- 3. packages
say "packages (pyserial required, numpy optional)"
install() {   # wheelhouse first (lab networks often block PyPI), then PyPI
  if [ -d "$WHEELHOUSE" ] && "$VPY" -m pip install -q --no-index --find-links "$WHEELHOUSE" "$@" 2>/dev/null; then
    echo "   $* from the wheelhouse"; return 0; fi
  if "$VPY" -m pip install -q --timeout 15 "$@"; then echo "   $* from PyPI"; return 0; fi
  return 1
}
install pyserial || {
  echo "   pyserial not installable. Last resort: sudo apt install python3-serial, then rerun with --python /usr/bin/python3"
  echo "   (or copy pyserial-*.whl here and: PYTHONPATH=pyserial-3.5-py2.py3-none-any.whl python3 -m real.arm_server)"
  exit 1; }
install numpy || echo "   numpy not installed (fine: the arm server does not need it)"
"$VPY" -c "import serial; print('   pyserial', serial.__version__)"

# ---------------------------------------------------------------- 4. serial port access
say "serial port access"
if ! id -nG "$USER" | grep -qw dialout; then
  if asroot usermod -aG dialout "$USER"; then
    echo "   added $USER to 'dialout': log out and back in (or: newgrp dialout) before using the port"
  fi
else
  echo "   $USER is in dialout"
fi
if [ "$UDEV" = 1 ]; then
  RULE=/etc/udev/rules.d/99-printpick-so101.rules
  TMP="$(mktemp)"
  cat > "$TMP" <<'EOF'
# SO-101 servo bus adapters: readable by everyone, stable name /dev/so101, and keep ModemManager away
# (it probes new /dev/ttyACM* devices with AT commands, which garbles the servo bus).
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", SYMLINK+="so101", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", KERNEL=="ttyACM[0-9]*", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", KERNEL=="ttyUSB[0-9]*", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
# FTDI-based adapters: 16 ms default USB latency -> 1 ms (every servo read waits for it)
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF
  if asroot install -m 0644 "$TMP" "$RULE"; then
    asroot udevadm control --reload-rules && asroot udevadm trigger --subsystem-match=tty || true
    echo "   udev rule $RULE installed (/dev/so101 appears for the CH343 adapter 1a86:55d3)"
  fi
  rm -f "$TMP"
fi
if systemctl is-active --quiet ModemManager 2>/dev/null; then
  echo "   note: ModemManager is running; the udev rule tells it to ignore the servo adapter"
fi

# ---------------------------------------------------------------- 5. self-test (no hardware)
say "self-test with simulated servos"
cd "$ROOT"
"$VPY" -m real.feetech --port mock scan
"$VPY" -m real.hello_arm --port mock --go --speed 90 --amp 5 | tail -2

say "serial devices now"
ls -l /dev/so101 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "   none (plug the arm's USB adapter into the Pi; power the servos)"
ls -l /dev/serial/by-id/ 2>/dev/null || true
if [ -z "$SERIAL_PORT" ]; then
  for p in /dev/so101 /dev/ttyACM0 /dev/ttyUSB0; do [ -e "$p" ] && { SERIAL_PORT="$p"; break; }; done
fi
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"

# ---------------------------------------------------------------- 6. optional systemd user service
if [ "$SERVICE" = 1 ]; then
  say "systemd user service printpick-arm"
  UNIT_DIR="$HOME/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  sed -e "s#@ROOT@#$ROOT#g" -e "s#@VPY@#$VPY#g" -e "s#@PORT@#$SERIAL_PORT#g" \
      "$ROOT/real/pi/printpick-arm.service" > "$UNIT_DIR/printpick-arm.service"
  systemctl --user daemon-reload
  systemctl --user enable printpick-arm.service
  loginctl enable-linger "$USER" 2>/dev/null || asroot loginctl enable-linger "$USER" || true
  echo "   start: systemctl --user start printpick-arm ; logs: journalctl --user -u printpick-arm -f"
fi

say "next"
TSIP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
echo "   tailscale: ${TSIP:-not up (sudo tailscale up)}"
echo "   1. bus check:   $VPY -m real.feetech --port $SERIAL_PORT scan"
echo "   2. calibration: $VPY -m real.joint_calib from-servos --port $SERIAL_PORT   (or: guided)"
echo "   3. hello:       $VPY -m real.hello_arm --port $SERIAL_PORT [--go]"
echo "   4. server:      $VPY -m real.arm_server --port $SERIAL_PORT"
echo "      Mac:         python -m real.arm_client remote://${TSIP:-<pi-ip>}:8765 status"
