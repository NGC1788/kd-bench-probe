#!/usr/bin/env bash
# kd-bench-probe: MaskedKD label probe on the KSHS x AIM Lab benchmark.
#   ./run.sh setup                       # uv, Python 3.11, torch 2.5.1+cu124, benchmark @ pinned commit
#   ./run.sh prepare  --dataset coco     # download + validate data (benchmark scripts), cache DeiT weights
#   ./run.sh teacher  --dataset coco     # DeiT-S teacher, 30 epochs (reference protocol)
#   ./run.sh probe    --dataset coco     # arms x seeds, one job at a time by default
#   ./run.sh all      --dataset coco     # prepare -> teacher -> probe -> report
#   ./run.sh status   --dataset coco
#   ./run.sh report
#   ./run.sh smoke                       # CPU, tiny fake data, debug models: checks the pipeline
#   ./run.sh size                        # disk used by this folder
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$KDB_HOME"
BENCH_URL=https://github.com/jeehoo0507/kshs-aimlab-benchmarks
BENCH_COMMIT=e1c39e7e1cc48e57305a2c12b0302e1df3349396   # branch codex/maskedkd-reference-runs
BENCH="$KDB_HOME/external/kshs-aimlab-benchmarks"
TORCH_PIN=${TORCH_PIN:-"torch==2.5.1 torchvision==0.20.1"}          # same as the reference runs
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}   # use .../whl/cpu for CPU-only checks
PY="$KDB_HOME/.venv/bin/python"

ensure_uv() {
  command -v uv >/dev/null 2>&1 && return
  echo "uv not found: installing it into $KDB_HOME/.tools/uv (this folder only, no PATH or profile changes)"
  mkdir -p "$KDB_HOME/.tools/uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$KDB_HOME/.tools/uv" INSTALLER_NO_MODIFY_PATH=1 sh
}

setup() {
  ensure_uv
  if [ ! -d "$BENCH/.git" ]; then
    mkdir -p "$KDB_HOME/external"
    git clone -q "$BENCH_URL" "$BENCH"
  fi
  git -C "$BENCH" fetch -q origin codex/maskedkd-reference-runs
  git -C "$BENCH" checkout -q "$BENCH_COMMIT"
  uv venv --python 3.11 --allow-existing "$KDB_HOME/.venv"
  "$PY" -c "import torch" 2>/dev/null || uv pip install --python "$PY" $TORCH_PIN --index-url "$TORCH_INDEX"
  uv pip install --python "$PY" -r "$BENCH/requirements-reference.txt"
  "$PY" -c "import sys, torch; print('python', sys.version.split()[0], sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
  echo "benchmark at $(git -C "$BENCH" rev-parse --short HEAD)"
}

cmd=${1:-help}; shift || true
case "$cmd" in
  setup) setup ;;
  prepare|teacher|probe|all|status|worker) exec "$PY" -m probe.runner "$cmd" "$@" ;;
  report) exec "$PY" -m probe.report "$@" ;;
  smoke)
    "$PY" tests/make_fake_coco.py outputs/smoke/data/coco_single
    exec "$PY" -m probe.runner all --dataset coco --data-root outputs/smoke/data/coco_single \
      --output-root outputs/smoke --config configs/smoke.json --device cpu --debug --no-prepare "$@" ;;
  size) du -sh "$KDB_HOME" "$KDB_HOME"/.venv "$KDB_HOME"/.cache "$KDB_HOME"/external "$KDB_HOME"/outputs 2>/dev/null || true ;;
  *) sed -n 2,13p "$0" ;;
esac
