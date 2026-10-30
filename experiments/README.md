# Experiments

The `experiments/` package runs one model on one dataset split and writes reports
from saved records. All commands in this guide run from the repository root.

## Setup

```sh
uv sync --extra experiments
cp .env.example .env
```

Set `OPENAI_API_KEY`. The decision endpoint also accepts `OPENROUTER_API_KEY`
and prefers it when both are set. `OPENAI_API_BASE` or `OPENAI_BASE_URL` sets
the endpoint origin. Install the local baseline extras when needed:

```sh
uv sync --extra experiments --extra tabpfn --extra tabfm
```

## Data

The dataset files are expected to be under `data/<dataset>/`. Please obtain the processed datasets from our [Zenodo repository](TODO) and place them in the `data/` folder.

| Dataset | Source | Target | Rows (train/test) | Classes | Max length |
|---|---|---|---:|---:|---:|
| Surface activity | [Houx et al.](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15010226/v1) | `surface_active` | 409 (328/81) | 2 | 3 |
| Beta-sheet formation | [Nissi et al.](https://doi.org/10.5281/zenodo.15003785) | `beta_sheet_formation_class` | 268 (214/54) | 2 | 5 |
| Self-assembly | [Njirjak et al.](https://doi.org/10.1038/s42256-024-00928-1) | `self_assembly_label` | 368 (294/74) | 2 | 24 |
| Breast-cancer activity | [Grisoni et al.](https://doi.org/10.1007/s00894-019-4007-6) | `activity_class` | 199 (159/40) | 3 | 38 |
| Lung-cancer activity | [Grisoni et al.](https://doi.org/10.1007/s00894-019-4007-6) | `activity_class` | 151 (120/31) | 3 | 38 |

Surface activity uses `sequence`, `pH`, and `tail_length`; the other tasks use
only `sequence`. Binary labels are `false` and `true`. Ternary labels are
`experimentally_inactive`, `moderately_active`, and `very_active`.

All tasks use a fixed holdout with seed 42. The four single-sequence tasks use a
stratified 80/20 split. Surface activity uses grouped five-fold selection by
peptide sequence; the fold closest to a 20% test fraction and the overall class
proportions becomes the holdout.

```text
data/<dataset>/
  full.csv                       all cleaned rows (audit only)
  test.csv                       fixed holdout
  task.json                      task contract and instructions
  train.csv                      full training split
  train_balanced.csv             class-balanced training subset
  train_corrupted.csv            full split with every label changed
  train_balanced_corrupted.csv   balanced subset with every label changed
```

| Variant | Reference file |
|---|---|
| `full` | `train.csv` |
| `balanced` | `train_balanced.csv` |
| `corrupted_full` | `train_corrupted.csv` |
| `corrupted_balanced` | `train_balanced_corrupted.csv` |
| `zero_shot` | none |

The `balanced` variant undersamples each training class to the smallest class
count. The corrupted variants are endpoint controls. Local baselines use only
`full` and `balanced`.

## Models

| Family | Models | Config |
|---|---|---|
| LLM | `openai/gpt-6.1-sol`, `deepseek/deepseek-v4.1-flash-1017`, `google/gemma-4-31b-it` | `paper_llm.json` |
| Decision | `typesafe/jev-1.14`, `openai/gpt-6-luna-decisions` | `paper_llm.json` |
| Local | XGBoost, TabPFN v2, TabFM | `paper_local.json` |

The paper configs use 5 endpoint replicates and 100 local replicates.

Endpoint models receive peptide sequences as text; surface activity also includes
`pH` and `tail_length`. TabPFN and TabFM receive positional integer tokens.
XGBoost receives positional one-hot tokens. Non-sequence inputs are appended as
numeric features. XGBoost uses fixed library defaults with 0.8 row and column
subsampling; TabPFN and TabFM use their defaults.

Endpoint runs use one of two request modes:

- `one_by_one` sends one target row per request.
- `all_at_once` sends all target rows in one request.

Local models require `--max-length`, given in the data table. The campaign
computes it from the prepared sequences.

## Configuration

| Config | Use | Results folder |
|---|---|---|
| `configs/paper.json` | Base template. No endpoint registry. | `results/paper` |
| `configs/paper_local.json` | Local baselines. | `results/paper/local` |
| `configs/paper_llm.json` | Five endpoint models. | `results/paper/llm` |
| `configs/xgboost.json` | XGBoost only. | `results/xgboost` |

A config sets the seed, replicate counts, device, model registry, and optional
local checkpoint paths. Each `models` entry has a `type` (`llm` or `decision`);
endpoint runs require an explicit `context_window`. A model that is not
registered fails validation before any request. `--results-dir` overrides the
configured folder.

## Run

### Campaign

```sh
scripts/run_campaign.sh --stage local
scripts/run_campaign.sh --stage mode
scripts/run_campaign.sh --stage main --mode all_at_once
scripts/run_campaign.sh --stage plot
```

| Stage | Commands | Runs | Groups |
|---|---:|---:|---:|
| Local | 30 | 3,000 | 30 |
| Mode comparison | 40 | 200 | 40 |
| Main endpoints | 125 | 625 | 125 |
| Unique total with mode comparison | — | 3,725 | 175 |

The mode stage runs both modes on `surface_tension` and `self_assembly`. The
chosen-mode runs are reused by the main stage, so the unique total is smaller
than the sum of the stages. Choose the mode after the comparison; there is no
automatic selection.

### Single request

```sh
uv run --extra experiments python -m experiments run \
  --config experiments/configs/paper_llm.json \
  --model-type llm \
  --model openai/gpt-6.1-sol \
  --mode one_by_one \
  --reference-data data/surface_tension/train.csv \
  --predict-data data/surface_tension/test.csv \
  --task data/surface_tension/task.json
```

```sh
uv run --extra experiments python -m experiments run \
  --config experiments/configs/paper_local.json \
  --model-type xgboost \
  --reference-data data/self_assembly/train.csv \
  --predict-data data/self_assembly/test.csv \
  --task data/self_assembly/task.json \
  --max-length 24
```

Omit `--reference-data` for zero-shot. Pass a corrupted table as the reference
for a corrupted control. See `python -m experiments run --help` for all options.

## Outputs

Each run writes `<results_dir>/runs/<run_id>.json`; request checkpoints go to
`<results_dir>/checkpoints/`. Completed records are reused. `--retry-failed`
retries failed records only. Changed inputs, settings, mode, or seed create new
identities.

```sh
uv run --extra experiments python -m experiments plot \
  --results-dir results/paper \
  --output-dir reports/paper
```

| File | Content |
|---|---|
| `metrics_summary.csv` | Per-group macro-F1 and, for binary tasks, positive-class F1 with sample SD. |
| `per_class_metrics.csv` | Per-class precision, recall, and F1. |
| `confusion_matrices.csv` | True and predicted label counts. |
| `run_f1_scores.csv` | One macro-F1 row per evaluation. |
| `resources.csv` | Latency, cost, usage, device, and profile fields. |
| `performance_<variant>.*` | Mean macro-F1 bars with replicate SD. |
| `run_f1_boxplot_<variant>.*` | One macro-F1 point per evaluation. |
| `summary.json` | Groups, run count, figures, and failure count. |

Figures use complete replicate groups only. Tables keep incomplete groups and
failures. Figures are written as PNG, SVG, and PDF.

## Compare the inference modes

```sh
uv run --extra experiments python -m experiments.analysis.mode_comparison \
  --results-dir results/paper
```

The command pairs completed `one_by_one` and `all_at_once` records with the same
dataset, variant, model, and split. Both modes need the replicate identifiers
`0..N-1`; the default is `N = 5`. It averages replicate macro-F1 scores in each
stratum, then applies a paired t-test and a Wilcoxon signed-rank test to the
stratum deltas. The command prints one Markdown table and makes no model calls.
Add `--csv PATH` to write the rows as CSV.

## Verification

```sh
uv run --extra experiments pytest
uvx ty check && uv run ruff check .
bash -n scripts/*.sh
```

The tests mock endpoint calls and use small local fixtures. They make no paid
calls and download no checkpoints.
