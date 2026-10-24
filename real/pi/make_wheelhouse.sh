#!/usr/bin/env bash
# On the Mac (or any machine with internet): build the offline Pi bundle for the arm server.
#   1. downloads Linux wheels (pyserial, numpy, pip) for the Pi's Python + CPU into real/pi/wheelhouse
#   2. packs the Pi-side code + wheels into runs/printpick_pi_bundle.tar.gz (no MuJoCo, no simulator)
# Usage:   bash real/pi/make_wheelhouse.sh [PYTHON_VERSION] [ARCH]      defaults: 3.14 aarch64
# Then:    scp runs/printpick_pi_bundle.tar.gz <pi>:   and on the Pi:
#          mkdir -p ~/printpick && tar xzf printpick_pi_bundle.tar.gz -C ~/printpick && cd ~/printpick \
#            && bash real/pi/setup_pi.sh
# Several Python versions:  for v in 3.12 3.13 3.14; do bash real/pi/make_wheelhouse.sh $v; done
# Slow link to the Pi:       NO_NUMPY=1 bash real/pi/make_wheelhouse.sh   (bundle ~2 MB; numpy is optional)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYV="${1:-3.14}"
ARCH="${2:-aarch64}"
WH="$ROOT/real/pi/wheelhouse"
OUT="$ROOT/runs/printpick_pi_bundle.tar.gz"
mkdir -p "$WH" "$ROOT/runs"

PIP=""
for c in "python3 -m pip" "pip3" "pip"; do
  if $c --version >/dev/null 2>&1; then PIP="$c"; break; fi
done
[ -n "$PIP" ] || { echo "no pip on this machine (python3 -m pip)"; exit 1; }
ABI="cp${PYV/./}"
echo "== wheels for CPython $PYV ($ABI) on linux $ARCH into $WH (pip: $PIP)"
PLATS=""
for p in manylinux_2_17 manylinux2014 manylinux_2_28 manylinux_2_31 manylinux_2_34 musllinux_1_2; do
  PLATS="$PLATS --platform ${p}_${ARCH}"
done
# shellcheck disable=SC2086
$PIP download -q --disable-pip-version-check --dest "$WH" --only-binary=:all: --python-version "$PYV" --implementation cp --abi "$ABI" \
     --abi abi3 --abi none $PLATS --platform any pyserial pip \
  || { echo "pyserial/pip download failed (no internet?)"; exit 1; }
EXCL=""
if [ "${NO_NUMPY:-0}" = 1 ]; then
  EXCL="--exclude=numpy-*"
else
  # shellcheck disable=SC2086
  $PIP download -q --disable-pip-version-check --dest "$WH" --only-binary=:all: --python-version "$PYV" --implementation cp \
       --abi "$ABI" $PLATS numpy \
    || echo "   numpy has no $ABI $ARCH wheel (optional: the arm server does not need numpy)"
fi
ls -1 "$WH"

echo "== bundle $OUT"
cd "$ROOT"
FILES="real/__init__.py real/feetech.py real/joint_calib.py real/kin_lite.py real/arm_server.py real/arm_client.py
       real/hello_arm.py real/replay.py real/so101_chain.json real/pi real/DEMO_DAY.md real/printer real/discover.py"
[ -f real/joint_calib.json ] && FILES="$FILES real/joint_calib.json"
# shellcheck disable=SC2086
XA=""; tar --no-xattrs -cf /dev/null -T /dev/null 2>/dev/null && XA="--no-xattrs"   # macOS bsdtar: no Apple xattrs
COPYFILE_DISABLE=1 tar $XA $EXCL -czf "$OUT" --exclude='__pycache__' --exclude='.venv-pi' --exclude='._*' $FILES   # no macOS ._ files
ls -lh "$OUT"
echo "   scp $OUT <pi>:  then on the Pi: mkdir -p ~/printpick && tar xzf printpick_pi_bundle.tar.gz -C ~/printpick"
echo "   && cd ~/printpick && bash real/pi/setup_pi.sh"
