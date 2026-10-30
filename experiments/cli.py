"""Run one experiment or write reports from saved results."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import MODEL_TYPES, MODES, REASONING_EFFORTS, load_config


def _parser() -> argparse.ArgumentParser:
    """Define the experiment commands and their arguments."""
    parser = argparse.ArgumentParser(description="Run peptide classification experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one model on one dataset split.")
    run.add_argument("--config", type=Path, required=True, help="Experiment JSON file.")
    run.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    run.add_argument("--model", help="Endpoint provider identifier or local model type.")
    run.add_argument("--reference-data", type=Path, help="Labeled reference CSV. Omit for zero-shot.")
    run.add_argument("--predict-data", type=Path, required=True, help="Target CSV.")
    run.add_argument("--task", type=Path, required=True, help="Task JSON file.")
    run.add_argument("--mode", choices=MODES, help="Endpoint request grouping.")
    run.add_argument("--max-length", type=int, help="Maximum sequence length in residues.")
    run.add_argument("--results-dir", type=Path, help="Override the configured results folder.")
    run.add_argument("--seed", type=int, help="Override the configured seed.")
    run.add_argument("--replicates", type=int, help="Override the configured replicate count.")
    run.add_argument("--context-window", type=int, help="Override the configured context window.")
    run.add_argument("--temperature", type=float, help="Override the configured temperature.")
    run.add_argument("--max-output-tokens", type=int, help="Override the configured output limit.")
    run.add_argument("--reasoning-effort", choices=REASONING_EFFORTS)
    run.add_argument("--max-questions", type=int, help="Override the decision question limit.")
    run.add_argument("--profile", action="store_true", help="Profile a local model run.")
    run.add_argument("--retry-failed", action="store_true", help="Retry failed records only.")
    plot = commands.add_parser("plot", help="Write reports from saved run records.")
    plot.add_argument("--results-dir", type=Path, default=Path("results"))
    plot.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one experiment request."""
    from .runner import RunRequest, run_experiments

    request = RunRequest(
        model_type=args.model_type,
        model=args.model,
        reference_data=args.reference_data,
        predict_data=args.predict_data,
        task=args.task,
        max_length=args.max_length,
        mode=args.mode,
        results_dir=args.results_dir,
        seed=args.seed,
        replicates=args.replicates,
        context_window=args.context_window,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_questions=args.max_questions,
        retry_failed=args.retry_failed,
        profile=args.profile,
    )
    return run_experiments(load_config(args.config), request)


def _plot(args: argparse.Namespace) -> dict[str, Any]:
    """Write reports from saved run records."""
    from .analysis.reporting import write_reports

    return write_reports(args.results_dir, args.output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute one command and print its JSON summary."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = _run(args)
        else:
            result = _plot(args)
    except Exception as exc:  # Report command failures at the CLI boundary.
        print(f"experiments: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, default=str))
    if args.command != "run" or result.get("status") == "completed":
        return
    for failure in result.get("errors", []):
        print(f"experiments: {failure}", file=sys.stderr)
    raise SystemExit(1)
