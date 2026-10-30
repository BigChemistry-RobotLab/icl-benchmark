"""Execute validated runs and summarize their records."""

import sys
from time import perf_counter
from typing import Any

from ..config import LOCAL_MODEL_TYPES, ExperimentConfig
from .endpoint import run_endpoint
from .local import run_local
from .records import (
    finalize_record,
    fingerprint,
    new_record,
    read_json,
    refresh_cached_cost,
    save_json,
)
from .request import RunPlan, RunRequest
from .validation import validate_run


def execute_run(plan: RunPlan, replicate: int, *, retry_failed: bool = False) -> dict[str, Any]:
    """Run or reuse one replicate record for the validated plan."""
    seed = plan.base_seed + replicate
    run_id = fingerprint({
        "protocol_id": plan.protocol_id,
        "dataset": plan.dataset,
        "variant": plan.variant,
        "model": plan.model,
        "mode": plan.mode,
        "replicate": replicate,
    })
    path = plan.results_dir / "runs" / f"{run_id}.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("status") == "completed":
            refresh_cached_cost(plan.results_dir, previous, path)
            return previous
        if not retry_failed:
            return previous
    record = new_record(plan, run_id, replicate, seed)
    started = perf_counter()
    try:
        if plan.model_type in LOCAL_MODEL_TYPES:
            run_local(plan, record, seed)
        else:
            run_endpoint(plan, record, run_id, seed, retry_failed)
        finalize_record(plan, record)
    except Exception as exc:  # noqa: BLE001 - Save each failed replicate for explicit retry.
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    if record["latency_seconds"] is None:
        record["latency_seconds"] = perf_counter() - started
    save_json(path, record)
    return record


def run_experiments(config: ExperimentConfig, request: RunRequest) -> dict[str, Any]:
    """Run all replicates of one atomic request and return a JSON summary.

    Args:
        config: Experiment settings and endpoint registry.
        request: One run request from the command line.

    Returns:
        The target identity, replicate counts, and error messages.

    Raises:
        ValueError: Pre-request validation fails, so no model call occurs.
    """
    plan = validate_run(config, request)
    records = []
    errors = []
    for replicate in range(plan.replicates):
        print(
            f"replicate {replicate + 1}/{plan.replicates} "
            f"{plan.dataset}/{plan.variant}/{plan.model}",
            file=sys.stderr,
            flush=True,
        )
        record = execute_run(plan, replicate, retry_failed=request.retry_failed)
        records.append(record)
        if record.get("status") != "completed":
            errors.append(f"{record['run_id']}: {record.get('error')}")
    completed = sum(record.get("status") == "completed" for record in records)
    return {
        "status": "completed" if completed == len(records) else "failed",
        "runs": len(records),
        "completed": completed,
        "failed": len(records) - completed,
        "dataset": plan.dataset,
        "variant": plan.variant,
        "model": plan.model,
        "mode": plan.mode,
        "results_dir": str(plan.results_dir),
        "errors": errors,
    }
