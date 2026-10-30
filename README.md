# In-Context Learning for Predicting Peptide Properties in Low-Data Regimes

In-context learning for predicting peptide properties in low-data regimes. The
repository compares general-purpose LLMs, typed decision models, and tabular
foundation models with classical baselines on five peptide classification tasks.
It also provides `icl_agent`, a small package that predicts from labeled
reference rows.

## Layout

| Path | Content |
|---|---|
| `src/icl_agent/` | Prediction package and CLI. |
| `experiments/` | Benchmark runner, configs, metrics, and figures. |
| `scripts/` | Shell wrappers for the paper campaign. |
| `examples/` | Example task files. |

## Install

Use Python 3.12 or later and uv:

```sh
uv sync --extra experiments
cp .env.example .env
```

Omit `--extra experiments` to use only the package.

Set `OPENAI_API_KEY`. `OPENAI_DEFAULT_MODEL` sets the default package model and
`OPENAI_API_BASE` sets an OpenAI-compatible endpoint. `OPENROUTER_API_KEY` is
the preferred key for decision models in the experiment workflow. Existing
environment variables take precedence over `.env`.

For the local baselines, add the optional extras:

```sh
uv sync --extra experiments --extra tabpfn --extra tabfm
```

## Task file

A task selects the input and output columns:

```json
{
  "inputs": [
    {"column": "tail_length", "alias": "Tail length"},
    {"column": "sequence", "alias": "Peptide sequence"},
    {"column": "pH"}
  ],
  "outputs": [{"column": "surface_active", "alias": "Surface active"}],
  "additional_context": "Predict surface activity from the supplied examples.",
  "decision_question": "Is the peptide C{tail_length}-{sequence} surface active at pH {pH}?",
  "allowed_labels": {"surface_active": ["false", "true"]}
}
```

Set `type` to `"context"` for an input that appears only in the reference data.
`allowed_labels` restricts classification outputs. `decision_question` is
required for decision runs and must use at least one non-context input; it may
omit the others. See [`examples/`](examples) and
[the experiment guide](experiments/README.md).

## Python API

```python
import asyncio
from icl_agent import load_task, run_agent

async def main():
    result = await run_agent(
        task=load_task("examples/surface_activity.json"),
        reference_data="data/surface_tension/train.csv",
        predict_data="data/surface_tension/test.csv",
    )
    print(result.model_dump())

asyncio.run(main())
```

The model receives all reference rows and all target rows in one chat run.
Results hold one prediction per target row, in input order, plus a short
reasoning summary and run metadata. `reference_data` and `predict_data` also
accept lists of string-valued dictionaries. Omit `reference_data` for zero-shot
prediction.

Decision tasks use the OpenAI Decisions API:

```python
from icl_agent import load_task, run_decision

result = await run_decision(
    task=load_task("examples/surface_activity.json"),
    reference_data="data/surface_tension/train.csv",
    predict_data="data/surface_tension/test.csv",
    model="gpt-6-luna",
)
```

A decision task needs exactly one output with two or three allowed labels. One
request supports at most 200 target rows.

## Command line

```sh
uv run icl-agent \
  --task examples/surface_activity.json \
  --reference-data data/surface_tension/train.csv \
  --predict-data data/surface_tension/test.csv
```

The CLI prints one JSON object. Omit `--reference-data` for zero-shot prediction.
Use `--model`, `--temperature`, `--max-output-tokens`, `--reasoning-effort`, and
`--seed` to override request settings.

## Reproduce the paper

Prepare the tables under `data/<dataset>/`, then run the campaign
stages:

```sh
scripts/run_campaign.sh --stage local
scripts/run_campaign.sh --stage mode
scripts/run_campaign.sh --stage main --mode one_by_one
scripts/run_campaign.sh --stage plot
```

The mode stage compares `one_by_one` and `all_at_once`; choose the mode for the
main stage from that comparison. See [experiments/README.md](experiments/README.md)
for datasets, configs, outputs, and the paired mode comparison, and
[scripts/README.md](scripts/README.md) for all wrappers.

## Development

Tests use mocked API calls and small local fixtures:

```sh
uv run --extra experiments pytest
uvx ty check && uv run ruff check .
```
