#!/usr/bin/env bash
# Run the simulation on a remote Linux box (e.g. a GPU workstation) while editing locally.
#
#   scripts/remote.sh sync HOST DIR             rsync this checkout to HOST:DIR
#   scripts/remote.sh run  HOST DIR ARGS...     sync, then run `python ARGS...` there (MUJOCO_GL=egl, nice 10)
#
#   scripts/remote.sh run gpu-box '~/printfarm' scripts/eval_unload.py --n 40 --procs 8
#
# HOST is anything ssh accepts (an alias from ~/.ssh/config works). DIR is relative to the remote home unless
# absolute; quote a leading ~ so that it expands on the remote side. The remote side needs its own venv at
# DIR/.venv (or set PF_REMOTE_PY) and the assets that are not in git: real/assets/so101 via
# real/fetch_assets.sh; assets/lab is regenerated on demand.
# .venv (any top-level dot entry), assets, real/assets, runs and build are never copied and never deleted there.
#
# The sync mirrors the code (rsync --delete), so it refuses to write into an existing non-empty directory that
# it did not create, unless PF_REMOTE_FORCE=1 (which also adopts the directory for later syncs).
set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 3 ] || usage
cmd=$1 host=$2 raw=$3 dir=${3%/}
shift 3

case "$dir" in
  ""|"~"|"."|'$HOME'|*..*) echo "remote.sh: refusing DIR='$raw'" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
marker=.printfarm-remote
force=${PF_REMOTE_FORCE:-0}
py=${PF_REMOTE_PY:-.venv/bin/python}

sync() {
  # $dir is expanded by the remote shell, so a leading ~ means the remote home.
  ssh "$host" "d=$dir
    if [ -e \"\$d\" ] && [ ! -e \"\$d/$marker\" ] && [ -n \"\$(ls -A \"\$d\" 2>/dev/null)\" ] && [ $force != 1 ]; then
      echo \"remote.sh: \$d exists, is not empty and was not created by remote.sh (PF_REMOTE_FORCE=1 to adopt it)\" >&2
      exit 3
    fi
    mkdir -p \"\$d\" && touch \"\$d/$marker\""
  # top-level dot entries (.git, .venv, caches, the marker) are neither copied nor deleted
  rsync -az --delete \
    --exclude '/.*' --exclude /assets --exclude /real/assets --exclude /runs --exclude /build \
    --exclude __pycache__ --exclude MUJOCO_LOG.TXT --exclude .DS_Store \
    "$root/" "$host:$dir/"
}

case "$cmd" in
  sync) sync ;;
  run)
    [ $# -ge 1 ] || usage
    sync
    args=$(printf '%q ' "$@")
    ssh "$host" "cd $dir && mkdir -p runs && MUJOCO_GL=egl PYTHONPATH=. nice -n 10 $py $args"
    ;;
  *) usage ;;
esac
