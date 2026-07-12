#!/usr/bin/env sh
# Launch the VQ OpenAI-compatible server on Apple Silicon.
# usage: sh run_mac.sh <model_dir> [port]   (needs: pip install "mlx>=0.31" "mlx-lm>=0.31")
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR="${1:?usage: run_mac.sh <model_dir> [port]}"
PORT="${2:-8090}"
PY="${PYTHON:-python3}"
exec "$PY" "$SCRIPT_DIR/vq_serve.py" --model "$MODEL_DIR" --port "$PORT"
