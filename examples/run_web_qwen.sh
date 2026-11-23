#!/usr/bin/env bash
# Usage: bash examples/run_web_qwen.sh /path/to/WEB [output-directory]
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/WEB [output-directory]" >&2
  exit 2
fi
DATA_ROOT="$1"
OUTPUT="${2:-$ROOT/outputs/qwen_web}"
uv run --extra harness python -m torch.distributed.run \
  --standalone --nproc_per_node="${NUM_GPUS:-1}" \
  --module inference.run \
  --config "$ROOT/configs/qwen_web.json" \
  --annotations "$DATA_ROOT/anno.json" --videos "$DATA_ROOT/videos" \
  --output "$OUTPUT"
uv run --extra harness python -m evaluation.run \
  --run "$OUTPUT" --annotations "$DATA_ROOT/anno.json"
