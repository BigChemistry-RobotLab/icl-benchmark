#!/usr/bin/env bash
# Run the paper campaign stages from the flat prepared data.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

DATA_ROOT="data"
LOCAL_CONFIG="experiments/configs/paper_local.json"
ENDPOINT_CONFIG="experiments/configs/paper_llm.json"
RESULTS_ROOT="results/paper"
REPORT_ROOT="reports/paper"
LOG_ROOT="$RESULTS_ROOT/logs/campaign"

ALL_DATASETS=(surface_tension beta_sheet_formation self_assembly cancer_breast cancer_lung)
DEFAULT_MODE_DATASETS=(surface_tension self_assembly)
LOCAL_MODELS=(xgboost tabpfn tabfm)
LOCAL_VARIANTS=(full balanced)
ENDPOINT_VARIANTS=(full balanced corrupted_full corrupted_balanced zero_shot)
MODE_VARIANTS=(full balanced)

STAGE="all"
MODE=""
DATASET_FILTER=()
MODEL_FILTER=()
MODE_DATASETS=("${DEFAULT_MODE_DATASETS[@]}")
RETRY_FAILED=0
FAIL_FAST=0
DRY_RUN=0
VERBOSE=0

PROGRESS=0
COMPLETED=0
TOTAL=0
FAILED=()
JOBS=4
STOPPED=0
JOB_LOGS=()
JOB_LABELS=()
JOB_ORDINALS=()

usage() {
  cat <<'EOF'
Run the paper campaign stages.

Usage: scripts/run_campaign.sh [options]

Stages:
  local  3 local models x 5 datasets x full/balanced x 100 repeats
  mode   all endpoint models x surface_tension and self_assembly
         x full/balanced x one_by_one and all_at_once x 5 repeats
  main   all endpoint models x 5 datasets x one mode
         x full/balanced/corrupted_full/corrupted_balanced/zero_shot x 5 repeats
  plot   write reports from saved records
  all    local, mode, main, then plot; requires --mode

Options:
  --stage STAGE            local, mode, main, plot, or all (default: all)
  --mode MODE              one_by_one or all_at_once; required for main/all
  --datasets "A B"         run a subset of datasets
  --models "A B"           run a subset of endpoint model identifiers
  --mode-datasets "A B"    mode-comparison datasets
                           (default: surface_tension self_assembly)
  --retry-failed           retry failed records only
  --jobs N                 run up to N endpoint commands at once (default: 4).
                           All commands share one request budget through the
                           results folder, so the total load does not grow.
  --fail-fast              stop at the first failed command
  --dry-run                print commands and counts; run nothing
  --verbose                stream each command log to this terminal
  -h, --help               show this help

Replicates come from the configs. The script passes --max-length for local
models, computed from full.csv (or train.csv plus test.csv).
EOF
}

fail() {
  printf 'run_campaign: %s\n' "$*" >&2
  exit 2
}

contains() {
  local needle=$1 item
  shift
  for item in "$@"; do
    if [[ $item == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

while (($#)); do
  case "$1" in
    --stage)
      STAGE="${2:?--stage needs a value}"
      shift 2
      ;;
    --mode)
      MODE="${2:?--mode needs a value}"
      shift 2
      ;;
    --datasets)
      read -r -a DATASET_FILTER <<< "${2:?--datasets needs a value}"
      shift 2
      ;;
    --models)
      read -r -a MODEL_FILTER <<< "${2:?--models needs a value}"
      shift 2
      ;;
    --mode-datasets)
      read -r -a MODE_DATASETS <<< "${2:?--mode-datasets needs a value}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?--jobs needs a value}"
      shift 2
      ;;
    --retry-failed)
      RETRY_FAILED=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

case "$STAGE" in
  local|mode|main|plot|all) ;;
  *) fail "unknown stage: $STAGE" ;;
esac

if [[ -n $MODE && $MODE != one_by_one && $MODE != all_at_once ]]; then
  fail "unknown mode: $MODE"
fi
if [[ $STAGE == main || $STAGE == all ]] && [[ -z $MODE ]]; then
  fail "--mode is required for --stage $STAGE; choose it after the mode comparison"
fi
if [[ $STAGE == mode && -n $MODE ]]; then
  printf 'run_campaign: --mode is ignored for --stage mode\n' >&2
fi
if ! [[ $JOBS =~ ^[1-9][0-9]*$ ]]; then
  fail "--jobs must be a positive integer"
fi

SELECTED_DATASETS=()
for dataset in "${ALL_DATASETS[@]}"; do
  if ((${#DATASET_FILTER[@]})) && ! contains "$dataset" "${DATASET_FILTER[@]}"; then
    continue
  fi
  SELECTED_DATASETS+=("$dataset")
done
if ((${#SELECTED_DATASETS[@]} == 0)); then
  fail "no dataset selected"
fi

SELECTED_MODE_DATASETS=()
for dataset in "${MODE_DATASETS[@]}"; do
  if ((${#DATASET_FILTER[@]})) && ! contains "$dataset" "${DATASET_FILTER[@]}"; then
    continue
  fi
  SELECTED_MODE_DATASETS+=("$dataset")
done

ENDPOINT_SPECS=()
if [[ -f $ENDPOINT_CONFIG ]]; then
  while IFS=$'\t' read -r model_type model_id; do
    if [[ -n ${model_type:-} ]]; then
      ENDPOINT_SPECS+=("$model_type|$model_id")
    fi
  done < <(python3 - "$ENDPOINT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
for name, settings in config.get("models", {}).items():
    print(f"{settings['type']}\t{name}")
PY
  )
fi
if ((${#ENDPOINT_SPECS[@]} == 0)); then
  fail "no endpoint models registered in $ENDPOINT_CONFIG"
fi

if ((${#MODEL_FILTER[@]})); then
  for requested in "${MODEL_FILTER[@]}"; do
    found=0
    for spec in "${ENDPOINT_SPECS[@]}"; do
      if [[ ${spec#*|} == "$requested" ]]; then
        found=1
        break
      fi
    done
    if ((!found)); then
      fail "model is not registered in $ENDPOINT_CONFIG: $requested"
    fi
  done
fi

safe_name() {
  printf '%s' "${1//[^A-Za-z0-9._-]/_}"
}

log_path() {
  printf '%s/%s-%s.log' "$LOG_ROOT" "$STAGE" "$(safe_name "$1")"
}

reference_path() {
  case "$2" in
    full) printf '%s/%s/train.csv' "$DATA_ROOT" "$1" ;;
    balanced) printf '%s/%s/train_balanced.csv' "$DATA_ROOT" "$1" ;;
    corrupted_full) printf '%s/%s/train_corrupted.csv' "$DATA_ROOT" "$1" ;;
    corrupted_balanced) printf '%s/%s/train_balanced_corrupted.csv' "$DATA_ROOT" "$1" ;;
    *) return 1 ;;
  esac
}

check_inputs() {
  local dataset=$1 variant=$2 directory="$DATA_ROOT/$1"
  local files=("$directory/test.csv" "$directory/task.json") file
  if [[ $variant != zero_shot ]]; then
    files+=("$(reference_path "$dataset" "$variant")")
  fi
  for file in "${files[@]}"; do
    if [[ ! -f $file ]]; then
      printf 'run_campaign: missing input: %s\n' "$file" >&2
      return 1
    fi
  done
}

dataset_max_length() {
  local dataset=$1
  python3 - "$DATA_ROOT/$dataset" <<'PY'
import csv
import sys
from pathlib import Path

folder = Path(sys.argv[1])
files = [folder / "full.csv"]
if not files[0].is_file():
    files = [folder / "train.csv", folder / "test.csv"]

longest = 0
for path in files:
    if not path.is_file():
        continue
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "sequence" not in (reader.fieldnames or []):
            raise SystemExit(f"missing sequence column: {path}")
        for row in reader:
            sequence = row.get("sequence") or ""
            count = 0
            index = 0
            while index < len(sequence):
                index += 2 if sequence[index:index + 2] == "P'" else 1
                count += 1
            longest = max(longest, count)
if longest <= 0:
    raise SystemExit(f"cannot compute max_length for {folder}")
print(longest)
PY
}

check_local_options() {
  python3 - "$LOCAL_CONFIG" "${LOCAL_MODELS[@]}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
options = config.get("baseline_options") or {}
missing = []
for model in sys.argv[2:]:
    if model == "xgboost":
        continue
    model_options = options.get(model) or {}
    if not model_options.get("checkpoint_path") and not model_options.get("allow_download", False):
        missing.append(model)
if missing:
    print(
        "run_campaign: set checkpoint_path or allow_download in "
        f"{sys.argv[1]} for: {', '.join(missing)}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

record_result() {
  local status=$1 label=$2 ordinal=$3 log=$4
  if ((status == 0)); then
    printf '[ok %d/%d] %s\n' "$ordinal" "$TOTAL" "$label"
    COMPLETED=$((COMPLETED + 1))
    return 0
  fi
  printf '[fail %d/%d] %s\n' "$ordinal" "$TOTAL" "$label" >&2
  tail -n 20 "$log" >&2 || true
  FAILED+=("$log")
  if ((FAIL_FAST)); then
    STOPPED=1
  fi
}

run_command() {
  local log=$1
  shift
  local label="${log#"$RESULTS_ROOT"/}"
  local status
  PROGRESS=$((PROGRESS + 1))
  if ((DRY_RUN)); then
    printf 'dry-run %d/%d:' "$PROGRESS" "$TOTAL"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  printf '[run %d/%d] %s\n' "$PROGRESS" "$TOTAL" "$label"
  if ((VERBOSE)); then
    if "$@" 2>&1 | tee "$log"; then
      status=0
    else
      status=${PIPESTATUS[0]}
    fi
  else
    if "$@" >"$log" 2>&1; then
      status=0
    else
      status=$?
    fi
  fi
  record_result "$status" "$label" "$PROGRESS" "$log"
  if ((STOPPED)); then
    printf 'run_campaign: stopped after the first failure\n' >&2
    exit 1
  fi
}

drain_one() {
  # Collect one finished background command and record its result.
  local index log status
  wait -n || true
  for index in "${!JOB_LOGS[@]}"; do
    log=${JOB_LOGS[$index]}
    [[ -f $log.status ]] || continue
    status=$(<"$log.status")
    rm -f "$log.status"
    record_result "$status" "${JOB_LABELS[$index]}" "${JOB_ORDINALS[$index]}" "$log"
    unset 'JOB_LOGS[index]' 'JOB_LABELS[index]' 'JOB_ORDINALS[index]'
    JOB_LOGS=("${JOB_LOGS[@]}")
    JOB_LABELS=("${JOB_LABELS[@]}")
    JOB_ORDINALS=("${JOB_ORDINALS[@]}")
    return 0
  done
}

drain_jobs() {
  # Wait for every background command and record its result.
  while ((${#JOB_LOGS[@]})); do
    drain_one
  done
}

finish_jobs() {
  # Drain the pool, then apply the fail-fast result.
  drain_jobs
  if ((STOPPED)); then
    printf 'run_campaign: stopped after the first failure\n' >&2
    exit 1
  fi
}

start_command() {
  # Start one endpoint command in the background, or wait for a free slot.
  local log=$1
  shift
  local label="${log#"$RESULTS_ROOT"/}"
  local ordinal
  PROGRESS=$((PROGRESS + 1))
  ordinal=$PROGRESS
  if ((DRY_RUN)); then
    printf 'dry-run %d/%d:' "$ordinal" "$TOTAL"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  if ((STOPPED)); then
    return 0
  fi
  while ((${#JOB_LOGS[@]} >= JOBS)); do
    drain_one
    if ((STOPPED)); then
      return 0
    fi
  done
  mkdir -p "$(dirname "$log")"
  printf '[run %d/%d] %s\n' "$ordinal" "$TOTAL" "$label"
  (
    local status
    if ((VERBOSE)); then
      if "$@" 2>&1 | tee "$log"; then
        status=0
      else
        status=${PIPESTATUS[0]}
      fi
    else
      if "$@" >"$log" 2>&1; then
        status=0
      else
        status=$?
      fi
    fi
    printf '%s\n' "$status" >"$log.status"
  ) &
  JOB_LOGS+=("$log")
  JOB_LABELS+=("$label")
  JOB_ORDINALS+=("$ordinal")
}

select_endpoints() {
  local wanted=$1 spec model_type model_id
  for spec in "${ENDPOINT_SPECS[@]}"; do
    model_type=${spec%%|*}
    model_id=${spec#*|}
    if [[ $wanted != any && $model_type != "$wanted" ]]; then
      continue
    fi
    if ((${#MODEL_FILTER[@]})) && ! contains "$model_id" "${MODEL_FILTER[@]}"; then
      continue
    fi
    printf '%s|%s\n' "$model_type" "$model_id"
  done
}

selected_endpoint_count() {
  select_endpoints "$1" | wc -l | tr -d ' '
}

command_total() {
  local local_count mode_count main_count
  local_count=$((
    ${#SELECTED_DATASETS[@]} * ${#LOCAL_VARIANTS[@]} * ${#LOCAL_MODELS[@]}
  ))
  mode_count=$((
    ${#SELECTED_MODE_DATASETS[@]} * ${#MODE_VARIANTS[@]} * 2
    * $(selected_endpoint_count any)
  ))
  main_count=$((
    ${#SELECTED_DATASETS[@]} * ${#ENDPOINT_VARIANTS[@]}
    * $(selected_endpoint_count any)
  ))
  case "$STAGE" in
    local) printf '%s\n' "$local_count" ;;
    mode) printf '%s\n' "$mode_count" ;;
    main) printf '%s\n' "$main_count" ;;
    plot) printf '1\n' ;;
    all) printf '%s\n' "$((local_count + mode_count + main_count + 1))" ;;
  esac
}

run_endpoint() {
  local dataset=$1 variant=$2 mode=$3 model_type=$4 model_id=$5
  local wrapper args log
  if [[ $model_type == llm ]]; then
    wrapper="scripts/run_llm.sh"
  else
    wrapper="scripts/run_decision.sh"
  fi
  args=(
    --config "$ENDPOINT_CONFIG"
    --model "$model_id"
    --mode "$mode"
    --predict-data "$DATA_ROOT/$dataset/test.csv"
    --task "$DATA_ROOT/$dataset/task.json"
  )
  if [[ $variant != zero_shot ]]; then
    args+=(--reference-data "$(reference_path "$dataset" "$variant")")
  fi
  if ((RETRY_FAILED)); then
    args+=(--retry-failed)
  fi
  log=$(log_path "endpoint-$dataset-$variant-$mode-$model_id")
  start_command "$log" "$wrapper" "${args[@]}"
}

run_local() {
  local model dataset variant reference max_length log
  local args
  if ! check_local_options; then
    if ((DRY_RUN)); then
      printf 'run_campaign: dry-run continues without configured local checkpoints\n' >&2
    else
      fail "configure the local checkpoints before the local stage"
    fi
  fi
  for model in "${LOCAL_MODELS[@]}"; do
    for dataset in "${SELECTED_DATASETS[@]}"; do
      if ! max_length=$(dataset_max_length "$dataset"); then
        fail "cannot compute max_length for $dataset"
      fi
      for variant in "${LOCAL_VARIANTS[@]}"; do
        if ! check_inputs "$dataset" "$variant"; then
          fail "missing inputs for $dataset/$variant"
        fi
        reference=$(reference_path "$dataset" "$variant")
        log=$(log_path "local-$dataset-$variant-$model")
        args=(
          --config "$LOCAL_CONFIG"
          --reference-data "$reference"
          --predict-data "$DATA_ROOT/$dataset/test.csv"
          --task "$DATA_ROOT/$dataset/task.json"
          --max-length "$max_length"
        )
        if ((RETRY_FAILED)); then
          args+=(--retry-failed)
        fi
        run_command "$log" "scripts/run_${model}.sh" "${args[@]}"
      done
    done
  done
}

run_mode() {
  local spec dataset variant mode model_type model_id
  local specs=()
  if ((${#SELECTED_MODE_DATASETS[@]} == 0)); then
    printf 'run_campaign: no mode-comparison dataset selected; skipping mode stage\n' >&2
    return 0
  fi
  mapfile -t specs < <(select_endpoints any)
  if ((${#specs[@]} == 0)); then
    fail "no endpoint model selected for the mode comparison"
  fi
  for spec in "${specs[@]}"; do
    model_type=${spec%%|*}
    model_id=${spec#*|}
    for dataset in "${SELECTED_MODE_DATASETS[@]}"; do
      for variant in "${MODE_VARIANTS[@]}"; do
        if ! check_inputs "$dataset" "$variant"; then
          fail "missing inputs for $dataset/$variant"
        fi
        for mode in one_by_one all_at_once; do
          run_endpoint "$dataset" "$variant" "$mode" "$model_type" "$model_id"
        done
      done
    done
  done
  finish_jobs
}

run_main() {
  local spec dataset variant model_type model_id
  local specs=()
  mapfile -t specs < <(select_endpoints any)
  if ((${#specs[@]} == 0)); then
    fail "no endpoint model selected"
  fi
  for spec in "${specs[@]}"; do
    model_type=${spec%%|*}
    model_id=${spec#*|}
    for dataset in "${SELECTED_DATASETS[@]}"; do
      for variant in "${ENDPOINT_VARIANTS[@]}"; do
        if ! check_inputs "$dataset" "$variant"; then
          fail "missing inputs for $dataset/$variant"
        fi
        run_endpoint "$dataset" "$variant" "$MODE" "$model_type" "$model_id"
      done
    done
  done
  finish_jobs
}

run_plot() {
  run_command "$LOG_ROOT/plot.log" scripts/plot_results.sh "$RESULTS_ROOT" "$REPORT_ROOT"
}

TOTAL=$(command_total)
if [[ $STAGE == mode ]]; then
  STAGE_DATASETS=("${SELECTED_MODE_DATASETS[@]}")
else
  STAGE_DATASETS=("${SELECTED_DATASETS[@]}")
fi

printf 'run_campaign: stage=%s mode=%s datasets=%s commands=%s\n' \
  "$STAGE" "${MODE:-both}" "${STAGE_DATASETS[*]:-none}" "$TOTAL"

case "$STAGE" in
  local) run_local ;;
  mode) run_mode ;;
  main) run_main ;;
  plot) run_plot ;;
  all)
    run_local
    run_mode
    run_main
    run_plot
    ;;
esac

if ((${#FAILED[@]})); then
  printf 'run_campaign: %d failed command(s)\n' "${#FAILED[@]}" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi

if ((DRY_RUN)); then
  printf 'run_campaign: planned %d command(s)\n' "$TOTAL"
else
  printf 'run_campaign: completed %d/%d command(s)\n' "$COMPLETED" "$TOTAL"
fi
