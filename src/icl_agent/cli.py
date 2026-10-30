"""Run predictions from the command line."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from .config import InferenceSettings, load_task
from .llm_agent import run_agent


def main(argv: Sequence[str] | None = None) -> None:
    """Read task and data paths and print predictions as JSON."""
    parser = argparse.ArgumentParser(description="Predict from labeled reference rows.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--reference-data", help="Labeled CSV; omit for zero-shot prediction")
    parser.add_argument("--predict-data", required=True)
    parser.add_argument("--model", help="Override OPENAI_DEFAULT_MODEL from the environment or .env")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--seed", type=int, help="Request seed; requires endpoint support")
    args = parser.parse_args(argv)
    try:
        results = asyncio.run(
            run_agent(
                task=load_task(args.task),
                reference_data=args.reference_data if args.reference_data is not None else [],
                predict_data=args.predict_data,
                model=args.model,
                settings=InferenceSettings(
                    temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                    seed=args.seed, reasoning_effort=args.reasoning_effort
                ),
            )
        )
    except Exception as exc:
        print(f"icl-agent: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(results.model_dump(), ensure_ascii=False))
