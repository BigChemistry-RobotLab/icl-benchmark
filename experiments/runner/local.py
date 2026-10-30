"""Run one local model replicate and store its predictions."""

from time import perf_counter
from typing import Any

from ..datasets import select_columns
from ..models.local import predict_baseline
from ..models.profile import profile_model
from .request import RunPlan


def run_local(plan: RunPlan, record: dict[str, Any], seed: int) -> None:
    """Run one local classifier replicate and store its predictions."""
    columns = (*plan.spec.inputs, plan.spec.target)
    references = select_columns(plan.reference_rows, columns)
    started = perf_counter()
    if plan.profile:
        result = profile_model(
            plan.model, plan.spec, references, plan.targets, seed=seed,
            device=plan.device, options=plan.options,
        )
        profile = result.get("profile")
        record["metadata"] = {**result["metadata"], "profile": profile}
        record["latency_seconds"] = (
            float(profile["wall_seconds"])
            if isinstance(profile, dict) and "wall_seconds" in profile
            else perf_counter() - started
        )
    else:
        result = predict_baseline(
            plan.model, plan.spec, references, plan.targets, seed,
            device=plan.device, options=plan.options,
        )
        record["metadata"] = dict(result["metadata"])
        record["latency_seconds"] = perf_counter() - started
    record["predictions"] = list(result["predictions"])

