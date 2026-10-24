#!/usr/bin/env bash
# Fetch the SO-101 MuJoCo model into real/assets/so101 (the repo .gitignore excludes assets/).
# Pinned to the mujoco_menagerie commit listed in THIRD_PARTY_NOTICES.md (same pin as CI).
set -euo pipefail
cd "$(dirname "$0")"
SO101_REF="${SO101_REF:-cda5236f98383f6fbaa4820c6b18dba892cd087f}"
[ -f assets/so101/so101.xml ] && { echo "real/assets/so101 already present"; exit 0; }
tmp="$(mktemp -d)"
git -C "$tmp" init -q
git -C "$tmp" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$tmp" sparse-checkout set robotstudio_so101
git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$SO101_REF"
git -C "$tmp" checkout -q FETCH_HEAD
mkdir -p assets
cp -R "$tmp/robotstudio_so101" assets/so101
echo "mujoco_menagerie robotstudio_so101 @ $SO101_REF" > assets/so101/SOURCE.txt
echo "fetched into real/assets/so101 (temp clone left in $tmp)"
