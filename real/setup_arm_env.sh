#!/usr/bin/env bash
# Separate Python 3.12 venv for LIVE SO-101 work: lerobot 0.6.1 needs Python >= 3.12 and would downgrade
# numpy/opencv (and pull torch) in the shared simulation venv. Creates real/.venv-arm.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv-arm --python 3.12
uv pip install --python .venv-arm/bin/python 'lerobot[feetech]==0.6.1' mujoco numpy scipy opencv-python-headless \
    websockets pyserial pytest
.venv-arm/bin/python -c "from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig; print('lerobot OK')"
echo "use: cd .. && PYTHONPATH=. real/.venv-arm/bin/python -m real.run_demo ... --live"
