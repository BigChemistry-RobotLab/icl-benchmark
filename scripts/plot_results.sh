#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RESULTS_DIR="${1:-results}"
OUTPUT_DIR="${2:-reports}"

exec uv run --extra experiments python -m experiments plot \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"
