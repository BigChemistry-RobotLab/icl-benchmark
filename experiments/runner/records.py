"""Persist run records, request checkpoints, and measured costs."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..analysis.metrics import compute_metrics
from .request import RunPlan

SCHEMA_VERSION = 2


def fingerprint(value: Any) -> str:
    """Return a stable identifier for JSON data."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a result file after its content is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read one persisted result."""
    return json.loads(path.read_text(encoding="utf-8"))


def request_cost(metadata: dict[str, Any]) -> float | None:
    """Return the provider cost from direct usage or saved responses."""

    def amount(value: Any) -> float | None:
        """Return a finite cost value when it is present."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
            if math.isfinite(result) and result >= 0:
                return result
        return None

    for value in (metadata.get("cost"), metadata.get("usage", {}).get("cost")):
        cost = amount(value)
        if cost is not None:
            return cost
    responses = metadata.get("responses", [])
    if not responses:
        return None
    costs = [amount(response.get("raw_usage", {}).get("cost")) for response in responses]
    return sum(cost for cost in costs if cost is not None) if all(cost is not None for cost in costs) else None


def refresh_cached_cost(results_dir: Path, record: dict[str, Any], path: Path) -> None:
    """Update a completed run from measured costs in its saved requests."""
    costs = []
    for request_id in record.get("requests", []):
        checkpoint = results_dir / "checkpoints" / f"{request_id}.json"
        if not checkpoint.is_file():
            return
        request = read_json(checkpoint)
        if request.get("status") != "completed":
            return
        cost = request_cost(request.get("metadata", {}))
        if cost is None:
            return
        costs.append(cost)
    if costs and record.get("cost") != sum(costs):
        record["cost"] = sum(costs)
        save_json(path, record)


def new_record(
    plan: RunPlan, run_id: str, replicate: int, seed: int,
) -> dict[str, Any]:
    """Create the complete record skeleton for one replicate."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "protocol_id": plan.protocol_id,
        "split_id": plan.split_id,
        "dataset": plan.dataset,
        "variant": plan.variant,
        "model_type": plan.model_type,
        "model": plan.model,
        "mode": plan.mode,
        "replicate": replicate,
        "seed": seed,
        "expected_replicates": plan.replicates,
        "reference_data": str(plan.reference_data) if plan.reference_data is not None else None,
        "predict_data": str(plan.predict_data),
        "task": str(plan.task_path),
        "status": "failed",
        "labels": list(plan.labels),
        "target_inputs": [dict(row) for row in plan.targets],
        "predictions": [],
        "metrics": {},
        "usage": {},
        "cost": None,
        "latency_seconds": None,
        "error": None,
        "requests": [],
        "checkpoint_path": None,
        "metadata": {},
        "max_length": plan.max_length,
    }


def finalize_record(plan: RunPlan, record: dict[str, Any]) -> None:
    """Validate predictions, compute metrics, and mark a completed record."""
    predictions = [str(item) for item in record["predictions"]]
    record["predictions"] = predictions
    if len(predictions) != len(plan.labels):
        raise ValueError("Predictions must match the target count")
    allowed = set(plan.spec.labels)
    invalid = [label for label in predictions if label not in allowed]
    if invalid:
        raise ValueError(f"Predictions must use the allowed labels: {invalid[0]}")
    record["metrics"] = compute_metrics(record["labels"], predictions, list(plan.spec.labels))
    record["status"] = "completed"

