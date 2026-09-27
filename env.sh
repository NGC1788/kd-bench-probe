# Source this file (run.sh does it for you). Every file the project creates stays
# inside this folder, so `rm -rf <this folder>` removes all of it:
# Python (uv-managed), .venv, package caches, model weights, CUDA kernel cache,
# the benchmark clone, datasets and outputs. Nothing is written to ~/.cache,
# ~/.local, shell profiles or the system Python.
KDB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KDB_HOME
export UV_CACHE_DIR="$KDB_HOME/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$KDB_HOME/.cache/uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_TOOL_DIR="$KDB_HOME/.cache/uv-tools"
export UV_NO_CONFIG=1
export TORCH_HOME="$KDB_HOME/.cache/torch"
export XDG_CACHE_HOME="$KDB_HOME/.cache/xdg"
export PIP_CACHE_DIR="$KDB_HOME/.cache/pip"
export CUDA_CACHE_PATH="$KDB_HOME/.cache/nv"
export MPLCONFIGDIR="$KDB_HOME/.cache/mpl"
export HF_HOME="$KDB_HOME/.cache/hf"
export PYTHONNOUSERSITE=1
export PATH="$KDB_HOME/.tools/uv:$PATH"
