"""Run endpoint request groups and save their checkpoints."""

import asyncio
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from icl_agent.results import PredictionResult

from ..endpoints import chat, decisions
from .budget import RequestBudget
from .records import fingerprint, read_json, request_cost, save_json
from .request import RunPlan


def request_payload(
    plan: RunPlan, group: list[dict[str, str]], seed: int, run_id: str, index: int,
) -> dict[str, Any]:
    """Build the exact request payload and its admitted request value."""
    settings = plan.settings
    if settings is None:
        raise ValueError("Endpoint models require registry settings")
    if plan.model_type == "llm":
        payload = chat.build_payload(
            plan.task, plan.reference_rows, group, plan.model, settings, seed,
        )
    else:
        payload = {
            "task": plan.task.model_dump(mode="json"),
            "references": plan.reference_rows,
            "targets": group,
            "model": plan.model,
            "settings": settings.model_dump(mode="json"),
            "seed": seed,
            "instructions": decisions.INSTRUCTIONS,
            "request": decisions.admission(
                plan.task, plan.reference_rows, group, plan.model, settings,
            ),
        }
    payload.update(run_id=run_id, request_index=index)
    return payload


async def send_request_async(
    plan: RunPlan, group: list[dict[str, str]], seed: int,
) -> PredictionResult:
    """Send one admitted endpoint request and return its prediction result."""
    settings = plan.settings
    if settings is None:
        raise ValueError("Endpoint models require registry settings")
    if plan.model_type == "llm":
        return await chat.send_request(
            task=plan.task,
            references=plan.reference_rows,
            targets=group,
            model=plan.model,
            settings=settings,
            seed=seed,
        )
    return await decisions.send_request(
        task=plan.task,
        reference_data=plan.reference_rows,
        predict_data=group,
        model=plan.model,
        settings=settings,
    )


MAX_CONCURRENT_REQUESTS = 32
REQUESTS_PER_SECOND = 32.0
BUDGET_FOLDER = ".budget"


def request_budget(plan: RunPlan) -> RequestBudget:
    """Return the shared budget for this results folder.

    Every command that writes to the same results folder uses the same folder
    for the budget, so concurrent commands share one limit.
    """
    return RequestBudget(
        plan.results_dir / BUDGET_FOLDER, MAX_CONCURRENT_REQUESTS, REQUESTS_PER_SECOND,
    )


def run_endpoint(
    plan: RunPlan, record: dict[str, Any], run_id: str, seed: int, retry_failed: bool,
) -> None:
    """Run all endpoint request groups and store checkpoints and predictions."""
    asyncio.run(_run_endpoint_async(plan, record, run_id, seed, retry_failed))


async def _run_endpoint_async(
    plan: RunPlan, record: dict[str, Any], run_id: str, seed: int, retry_failed: bool,
) -> None:
    """Send the admitted request groups with bounded concurrency."""
    settings = plan.settings
    if settings is None:
        raise ValueError("Endpoint models require registry settings")
    groups = [[row] for row in plan.targets] if plan.mode == "one_by_one" else [plan.targets]
    prepared: list[tuple[int, list[dict[str, str]], dict[str, Any], str, Path, dict[str, Any]]] = []
    for index, group in enumerate(groups):
        payload = request_payload(plan, group, seed, run_id, index)
        request_id = fingerprint(payload)
        request_path = plan.results_dir / "checkpoints" / f"{request_id}.json"
        record["checkpoint_path"] = str(request_path)
        checkpoint: dict[str, Any] = (
            read_json(request_path) if request_path.exists() else {"status": "pending"}
        )
        if checkpoint.get("status") == "failed" and not retry_failed:
            raise RuntimeError(
                f"Request {request_id} failed; use --retry-failed to retry it"
            )
        prepared.append((index, group, payload, request_id, request_path, checkpoint))

    results: list[dict[str, Any] | None] = [None] * len(groups)
    budget = request_budget(plan)

    async def execute(
        index: int,
        group: list[dict[str, str]],
        payload: dict[str, Any],
        request_id: str,
        request_path: Path,
        checkpoint: dict[str, Any],
    ) -> None:
        if checkpoint.get("status") == "completed":
            results[index] = checkpoint
            print(
                f"request {index + 1}/{len(groups)} {request_id} completed",
                file=sys.stderr,
                flush=True,
            )
            return
        async with budget.slot():
            await budget.wait_turn()
            print(
                f"request {index + 1}/{len(groups)} {request_id} pending",
                file=sys.stderr,
                flush=True,
            )
            request_record: dict[str, Any] = {
                "request_id": request_id, "payload": payload, "status": "failed",
            }
            started = perf_counter()
            try:
                response = await send_request_async(plan, group, seed)
                request_record.update(
                    status="completed",
                    predictions=[dict(item) for item in response.predictions],
                    rationale=response.reasoning,
                    metadata=response.metadata,
                )
            except Exception as exc:  # noqa: BLE001 - Save each request failure for explicit retry.
                request_record["error"] = f"{type(exc).__name__}: {exc}"
            latency = perf_counter() - started
            request_record["latency_seconds"] = latency
            save_json(request_path, request_record)
            results[index] = request_record
            print(
                f"request {index + 1}/{len(groups)} {request_id} "
                f"{request_record['status']} {latency:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    wall_started = perf_counter()
    await asyncio.gather(*(execute(*item) for item in prepared))
    record["latency_seconds"] = perf_counter() - wall_started

    usage: dict[str, int | float] = {}
    predictions: list[str] = []
    request_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, (_, _, _, request_id, _, _) in enumerate(prepared):
        record["requests"].append(request_id)
        request_record = results[index]
        if request_record is None:
            errors.append("Endpoint request did not complete")
            continue
        if request_record.get("status") != "completed":
            errors.append(str(request_record.get("error") or "Endpoint request failed"))
            continue
        metadata: dict[str, Any] = request_record.get("metadata") or {}
        for key, value in (metadata.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
        request_records.append(request_record)
        predicted_rows: list[dict[str, str]] = request_record.get("predictions") or []
        predictions.extend(str(row[plan.spec.target]) for row in predicted_rows)
    if errors:
        raise RuntimeError(errors[0])

    record["predictions"] = predictions
    record["usage"] = usage
    native_costs = [request_cost(item.get("metadata", {})) for item in request_records]
    record["cost"] = (
        sum(cost for cost in native_costs if cost is not None)
        if all(cost is not None for cost in native_costs)
        else None
    )
    instructions = chat.INSTRUCTIONS if plan.model_type == "llm" else decisions.INSTRUCTIONS
    record["metadata"] = {
        "model": plan.model,
        "model_type": plan.model_type,
        "settings": settings.model_dump(mode="json"),
        "instructions": instructions,
    }
    input_price = settings.input_usd_per_million
    output_price = settings.output_usd_per_million
    if (
        record["cost"] is None
        and input_price is not None
        and output_price is not None
        and "input_tokens" in usage
        and "output_tokens" in usage
    ):
        record["cost"] = (
            usage["input_tokens"] * input_price + usage["output_tokens"] * output_price
        ) / 1_000_000
