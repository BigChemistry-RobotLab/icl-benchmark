"""Predict fixed task labels with the Decisions endpoint."""

import json
import os
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from icl_agent.config import REQUEST_TIMEOUT_SECONDS, Task
from icl_agent.data import DataSource, load_rows
from icl_agent.decision_agent import (
    DECISION_INSTRUCTIONS as INSTRUCTIONS,
)
from icl_agent.decision_agent import (
    decision_input,
    decision_instructions,
)
from icl_agent.results import PredictionResult

from ..config import ModelConfig, endpoint_identity

__all__ = ["INSTRUCTIONS", "admission", "build_payload", "endpoint_url", "send_request"]


def endpoint_url() -> str:
    """Return the Decisions URL at the configured endpoint origin."""
    parts = urlsplit(endpoint_identity())
    if parts.scheme not in {"https", "http"} or not parts.netloc:
        raise ValueError("Configure an HTTP endpoint origin for Decisions")
    return urlunsplit((parts.scheme, parts.netloc, "/api/alpha/decisions", "", ""))


def build_payload(
    task: Task, references: list[dict[str, str]], targets: list[dict[str, str]], model: str,
) -> dict[str, Any]:
    """Build the compact endpoint state and one choice question per target.

    The state holds the instructions and task context as text, and the labeled
    reference rows as a column header and row arrays. Each question uses the task
    ``decision_question`` template with the target input values. The endpoint
    accepts only choice questions, so each question lists its allowed labels as
    criteria.

    Args:
        task: Task with one output, two or three allowed labels, and a question.
        references: Complete selected training rows.
        targets: Input rows in prediction order.
        model: Explicit endpoint model identifier.

    Returns:
        A Decisions request without chat inference parameters.

    Raises:
        ValueError: The model, task, or label contract is unsupported.
    """
    if not model.strip():
        raise ValueError("A nonempty decision model is required")
    if len(task.outputs) != 1:
        raise ValueError("Decision tasks require exactly one output")
    output = task.outputs[0]
    labels = task.allowed_labels.get(output.column, [])
    if len(labels) not in {2, 3}:
        raise ValueError("Decision tasks require exactly two or three allowed labels")
    inputs = [item.column for item in task.inputs]
    target_inputs = [item.column for item in task.inputs if not item.is_context]
    references = load_rows(references, [*inputs, output.column], "Reference data")
    targets = load_rows(targets, target_inputs, "Prediction data")
    if not targets:
        raise ValueError("Decision requests require at least one target")
    return {
        "model": model,
        "state": decision_input(task, references),
        "questions": {
            f"target_{index}": {
                "type": "choice",
                "instructions": decision_instructions(task, row),
                "criteria": {label: label for label in labels},
            }
            for index, row in enumerate(targets)
        },
    }


def admission(
    task: Task, references: list[dict[str, str]], targets: list[dict[str, str]],
    model: str, settings: ModelConfig,
) -> dict[str, Any]:
    """Check the complete request against explicit endpoint limits.

    Args:
        task: Task with one output and its allowed labels.
        references: Complete selected training rows.
        targets: Input rows in prediction order.
        model: Explicit endpoint model identifier.
        settings: Decision model capacity and question limit.

    Returns:
        The admitted Decisions payload.

    Raises:
        ValueError: The model type, capacity, or request size is invalid.
    """
    if settings.type != "decision":
        raise ValueError("Decision admission requires a decision model")
    if settings.context_window is None:
        raise ValueError("Configure the decision context_window")
    if len(targets) > settings.max_questions:
        raise ValueError("Target count exceeds the decision endpoint question limit")
    payload = build_payload(task, references, targets, model)
    bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024
    if bound > settings.context_window:
        raise ValueError("Complete context exceeds conservative decision admission bound")
    return payload


def _map_answers(
    task: Task, targets: list[dict[str, str]], payload: dict[str, Any], response: Any,
) -> PredictionResult:
    """Validate answer coverage and restore the original target order."""
    if not isinstance(response, dict):
        raise TypeError("Decision response must be an object")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise TypeError("Decision answers must be an object")
    expected = list(payload["questions"])
    if set(answers) != set(expected):
        raise ValueError("Decision answers must cover exactly the requested targets")
    output = task.outputs[0].column
    labels = task.allowed_labels[output]
    predictions = []
    probabilities: dict[str, Any] = {}
    for index, target in enumerate(targets):
        name = f"target_{index}"
        answer = answers[name]
        if not isinstance(answer, dict):
            raise TypeError(f"Decision answer {name} must be an object")
        if answer.get("type") != "choice":
            raise ValueError(f"Decision answer {name} must have type choice")
        label = answer.get("choice")
        if not isinstance(label, str) or label not in labels:
            raise ValueError(f"Decision answer {name} has an invalid label")
        probabilities[name] = answer.get("probabilities", {})
        predictions.append({**target, output: label})
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise TypeError("Decision usage must be an object")
    return PredictionResult(predictions=predictions, reasoning="", metadata={
        "model": response.get("model", payload["model"]),
        "provider": response.get("provider"), "response_id": response.get("id"),
        "usage": usage, "cost": usage.get("cost"), "answers": answers,
        "probabilities": probabilities, "rationale_available": False,
    })


async def send_request(
    *, task: Task, reference_data: DataSource, predict_data: DataSource,
    model: str, settings: ModelConfig,
) -> PredictionResult:
    """Send one admitted request and return predictions in target order.

    Args:
        task: Validated task and label contract.
        reference_data: Complete selected reference rows or CSV path.
        predict_data: Target rows or CSV path.
        model: Explicit endpoint model identifier.
        settings: Model capacity, question limit, and request prices.

    Returns:
        Predictions, endpoint metadata, and an empty reasoning summary.

    Raises:
        ValueError: The model type, settings, or datasets are invalid.
        RuntimeError: The endpoint returns an error status.
    """
    inputs = [item.column for item in task.inputs]
    target_inputs = [item.column for item in task.inputs if not item.is_context]
    outputs = [item.column for item in task.outputs]
    references = load_rows(reference_data, [*inputs, *outputs], "Reference data")
    targets = load_rows(predict_data, target_inputs, "Prediction data")
    if not targets:
        return PredictionResult(predictions=[], reasoning="", metadata={"rationale_available": False})
    payload = admission(task, references, targets, model, settings)
    endpoint = endpoint_url()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY for Decisions")
    started = perf_counter()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(endpoint, json=payload, headers={"Authorization": f"Bearer {key}"})
    if response.status_code != 200:
        raise RuntimeError(f"Decision endpoint returned HTTP {response.status_code}")
    result = _map_answers(task, targets, payload, response.json())
    result.metadata.update(endpoint=endpoint, elapsed_seconds=perf_counter() - started)
    return result
