#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

exec uv run --extra experiments --extra tabfm python -m experiments run --model-type tabfm "$@"
