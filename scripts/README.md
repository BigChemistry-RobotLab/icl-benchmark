# Scripts

These wrappers change to the repository root, add the required uv extras, and
forward all arguments. Run them from any directory.

| Script | Purpose | Extras |
|---|---|---|
| `run_campaign.sh` | Run the paper campaign stages. | Per model type. |
| `run_llm.sh` | Run one LLM endpoint request. | `experiments` |
| `run_decision.sh` | Run one decision endpoint request. | `experiments` |
| `run_xgboost.sh` | Run one XGBoost request. | `experiments` |
| `run_tabpfn.sh` | Run one TabPFN request. | `experiments`, `tabpfn` |
| `run_tabfm.sh` | Run one TabFM request. | `experiments`, `tabfm` |
| `plot_results.sh` | Write reports from saved records. | `experiments` |
| `plot_paper_figures.sh` | Write Figures 2–4 for the full and balanced reference sets. | `experiments` |
| `mode_comparison.sh` | Compare `one_by_one` and `all_at_once`. | `experiments` |

## Campaign

```sh
scripts/run_campaign.sh --stage local
scripts/run_campaign.sh --stage mode
scripts/run_campaign.sh --stage main --mode one_by_one
scripts/run_campaign.sh --stage plot
```

`--stage all --mode one_by_one` runs local, mode, main, and plot in that order.
The mode stage compares `one_by_one` and `all_at_once`; choose the mode for the
main stage from that comparison.

| Option | Use |
|---|---|
| `--stage STAGE` | `local`, `mode`, `main`, `plot`, or `all`. |
| `--mode MODE` | `one_by_one` or `all_at_once`; required for `main` and `all`. |
| `--datasets "A B"` | Run a subset of datasets. |
| `--models "A B"` | Run a subset of registered endpoint models. |
| `--jobs N` | Run up to N endpoint commands at once. Default: 4. |
| `--retry-failed` | Retry failed records only. |
| `--dry-run` | Print commands and counts; run nothing. |
| `--verbose` | Stream each command log to the terminal. |
| `--fail-fast` | Stop at the first failed command. |

The campaign runs local commands one at a time. Endpoint commands share one
request budget through the results folder, so the total load does not grow with
`--jobs`. Logs go to `results/paper/logs/campaign/`. A rerun reuses completed
records.

Counts with the five registered endpoint models:

| Stage | Commands | Runs | Groups |
|---|---:|---:|---:|
| Local | 30 | 3,000 | 30 |
| Mode comparison | 40 | 200 | 40 |
| Main endpoints | 125 | 625 | 125 |
| Unique total with mode comparison | — | 3,725 | 175 |

## Manual runs

```sh
scripts/run_llm.sh \
  --config experiments/configs/paper_llm.json \
  --model openai/gpt-6.1-sol \
  --mode one_by_one \
  --reference-data data/surface_tension/train.csv \
  --predict-data data/surface_tension/test.csv \
  --task data/surface_tension/task.json
```

```sh
scripts/run_xgboost.sh \
  --config experiments/configs/paper_local.json \
  --reference-data data/self_assembly/train.csv \
  --predict-data data/self_assembly/test.csv \
  --task data/self_assembly/task.json \
  --max-length 24
```

Omit `--reference-data` for zero-shot. Use `run_decision.sh` for a registered
decision model. Add `--replicates` to override the config count, `--results-dir`
to override the results folder, and `--retry-failed` to retry failures. Local
models need `--max-length`; add `--profile` to record wall time and peak memory.
See [experiments/README.md](../experiments/README.md) for the model registry,
data variants, and output files.

## Reports

```sh
scripts/plot_results.sh results/paper reports/paper
scripts/plot_paper_figures.sh results/paper reports/paper
scripts/mode_comparison.sh --results-dir results/paper --expected-replicates 5
```

`plot_results.sh` and `plot_paper_figures.sh` take the results and output
folders as positional arguments. `plot_results.sh` writes the tables and the
analysis figures. `plot_paper_figures.sh` writes only Figures 2–4 for the
manuscript, including balanced versions of Figures 3 and 4.
`mode_comparison.sh` prints the paired mode-comparison table and
accepts `--csv PATH`. Neither script makes model calls.

## Check the scripts

```sh
bash -n scripts/*.sh
```
