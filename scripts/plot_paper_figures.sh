#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RESULTS_DIR="${1:-results/paper}"
OUTPUT_DIR="${2:-reports/paper}"

exec uv run --extra experiments python -m experiments.analysis.paper_figures \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"
