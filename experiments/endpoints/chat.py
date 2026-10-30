"""Build and send chat endpoint requests for one task group."""

import json
from typing import Any

from icl_agent import InferenceSettings
from icl_agent.config import Task
from icl_agent.llm_agent import INSTRUCTIONS, output_schema, run_agent
from icl_agent.prompt_builder import reference_prompt, target_prompt
from icl_agent.results import PredictionResult

from ..config import ModelConfig


def admission(
    task: Task,
    references: list[dict[str, str]],
    targets: list[dict[str, str]],
    settings: ModelConfig,
) -> str:
    """Check the complete chat prompt and any configured output budget."""
    if settings.context_window is None:
        raise ValueError("Explicit context_window is required")
    prompt = target_prompt(task, reference_prompt(task, references), targets)
    content = INSTRUCTIONS + prompt + json.dumps(output_schema(task, len(targets)).model_json_schema())
    prompt_bound = len(content.encode("utf-8")) + 1024
    if settings.max_output_tokens is not None and settings.max_output_tokens < len(targets) + 16:
        raise ValueError("Output budget is too small for the requested target count")
    if prompt_bound + (settings.max_output_tokens or 0) > settings.context_window:
        raise ValueError("Complete context exceeds conservative token admission bound")
    return prompt


def build_payload(
    task: Task,
    references: list[dict[str, str]],
    targets: list[dict[str, str]],
    model: str,
    settings: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Build one admitted chat request and its saved request value."""
    return {
        "task": task.model_dump(mode="json"),
        "references": references,
        "targets": targets,
        "model": model,
        "settings": settings.model_dump(mode="json"),
        "seed": seed,
        "instructions": INSTRUCTIONS,
        "output_schema": output_schema(task, len(targets)).model_json_schema(),
        "request": admission(task, references, targets, settings),
    }


async def send_request(
    *,
    task: Task,
    references: list[dict[str, str]],
    targets: list[dict[str, str]],
    model: str,
    settings: ModelConfig,
    seed: int,
) -> PredictionResult:
    """Send one admitted chat request and return predictions in target order.

    Args:
        task: Validated task and output contract.
        references: Complete selected reference rows.
        targets: Target rows in prediction order.
        model: Explicit endpoint model identifier.
        settings: Chat model capacity and request parameters.
        seed: Request seed for this replicate.

    Returns:
        Predictions and request metadata from the chat agent.
    """
    return await run_agent(
        task=task,
        reference_data=references,
        predict_data=targets,
        model=model,
        settings=InferenceSettings(
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort,
            seed=seed if settings.send_seed else None,
        ),
    )
