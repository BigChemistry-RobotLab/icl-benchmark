"""Create the agent and predict all target rows in one run."""

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from agents import Agent, ModelSettings, OpenAIProvider, RunConfig, Runner
from agents.models.default_models import get_default_model
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, Field, create_model

from .config import (
    REQUEST_MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    InferenceSettings,
    Task,
)
from .data import DataSource, load_rows
from .prompt_builder import reference_prompt, target_prompt
from .results import PredictionError, PredictionResult

INSTRUCTIONS = """Predict the outputs for every target row, in the same order.
Use the reference rows to infer the input to output mapping. If no reference rows are provided, use the task context for zero-shot prediction.
Follow all task-specific output labels, units, formats, headers, and schema. Treat empty cells as missing values, not zero. Missing reference outputs are not examples of valid labels or predictions.
Return exactly one prediction per target row in the required structured format.
After the predictions, briefly state the basis for the predictions and any relevant uncertainty. Use task context and relevant general knowledge to interpret the inputs. When reference rows are available, prefer patterns supported by the examples. State relevant uncertainty. Do not invent evidence or claim validation that was not performed."""


def _create_agent(
    *, model: str | None, output_type: type[BaseModel], settings: InferenceSettings
) -> Agent[None]:
    """Create an agent ready for prediction.

    Args:
        model: Model identifier, or None to use the SDK default.
        output_type: Required structured output schema.
        settings: Optional model request parameters.

    Returns:
        An agent with the prediction instructions and output schema.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
    return Agent(
        name="ICL predictor",
        instructions=INSTRUCTIONS,
        model=model,
        output_type=output_type,
        model_settings=ModelSettings(
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            reasoning=Reasoning(effort=settings.reasoning_effort) if settings.reasoning_effort else None,
            extra_args={"seed": settings.seed} if settings.seed is not None else None,
            preserve_raw_usage=True,
        ),
    )


def output_schema(task: Task, target_count: int) -> type[BaseModel]:
    """Create the structured output schema for the target rows.

    Each target row uses a ``target_<n>`` alias. Each output value uses its
    output column name as the alias.
    """
    if target_count < 0:
        raise ValueError("Target count must not be negative")
    fields: dict[str, Any] = {}
    for index, item in enumerate(task.outputs):
        labels = task.allowed_labels.get(item.column)
        value_type = Literal.__getitem__(tuple(labels)) if labels else str
        fields[f"output_{index}"] = (value_type, Field(alias=item.column, description=item.label))
    config = ConfigDict(extra="forbid", strict=True)
    values = create_model("OutputValues", __config__=config, **fields)
    target_fields: dict[str, Any] = {
        f"row_{index}": (values, Field(alias=f"target_{index}"))
        for index in range(target_count)
    }
    targets = create_model("TargetPredictions", __config__=config, **target_fields)
    return create_model(
        "StructuredPrediction",
        __config__=config,
        predictions=(targets, ...),
        reasoning=(str, ...),
    )


def _serialize(value: Any) -> Any:
    """Convert SDK metadata to JSON values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _run_metadata(run: Any, agent: Agent[None], settings: InferenceSettings, elapsed: float) -> dict[str, Any]:
    """Collect request details that the SDK supplies."""
    metadata: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "model": agent.model or get_default_model(),
        "settings": settings.model_dump(),
    }
    context = getattr(run, "context_wrapper", None)
    usage = getattr(context, "usage", None)
    if usage is not None:
        metadata["usage"] = _serialize(usage)
    responses = []
    for response in getattr(run, "raw_responses", []):
        details = {
            key: _serialize(value)
            for key in ("response_id", "request_id", "usage", "raw_usage")
            if (value := getattr(response, key, None)) is not None
        }
        reasoning = [
            _serialize(item) for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "reasoning"
        ]
        if reasoning:
            details["provider_reasoning"] = reasoning
        responses.append(details)
    if responses:
        metadata["responses"] = responses
    return metadata


async def run_agent(
    *,
    task: Task,
    reference_data: DataSource,
    predict_data: DataSource,
    model: str | None = None,
    settings: InferenceSettings | None = None,
) -> PredictionResult:
    """Predict all target rows in one agent run.

    Args:
        task: Validated task settings.
        reference_data: Labeled CSV path or sequence of text dictionaries.
        predict_data: Target CSV path or sequence of text dictionaries.
        model: Model identifier, or None to use OPENAI_DEFAULT_MODEL.
        settings: Request settings. Seeds require endpoint support.

    Returns:
        Inputs and predictions in target row order with shared reasoning.

    Raises:
        ValueError: Task settings or datasets are invalid.
        PredictionError: A model run or its output validation fails.
    """
    task = Task.model_validate(task.model_dump())
    settings = InferenceSettings.model_validate((settings or InferenceSettings()).model_dump())
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a nonempty string")
    inputs = [item.column for item in task.inputs]
    prediction_inputs = [item.column for item in task.inputs if not item.is_context]
    outputs = [item.column for item in task.outputs]
    references = load_rows(reference_data, inputs + outputs, "Reference data")
    targets = load_rows(predict_data, prediction_inputs, "Prediction data")
    if not targets:
        return PredictionResult(predictions=[], reasoning="")

    schema = output_schema(task, len(targets))
    reference = reference_prompt(task, references)
    agent = _create_agent(model=model, output_type=schema, settings=settings)
    try:
        started = perf_counter()
        async with AsyncOpenAI(
            timeout=REQUEST_TIMEOUT_SECONDS, max_retries=REQUEST_MAX_RETRIES,
        ) as client:
            run = await Runner.run(
                agent,
                target_prompt(task, reference, targets),
                max_turns=2,
                run_config=RunConfig(
                    tracing_disabled=True,
                    model_provider=OpenAIProvider(openai_client=client),
                ),
            )
        raw = run.final_output
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(by_alias=True)
        validated = schema.model_validate(raw).model_dump(by_alias=True)
        predictions = [validated["predictions"][f"target_{index}"] for index in range(len(targets))]
        return PredictionResult(
            predictions=[
                {**{column: row[column] for column in prediction_inputs}, **prediction}
                for row, prediction in zip(targets, predictions, strict=True)
            ],
            reasoning=validated["reasoning"],
            metadata=_run_metadata(run, agent, settings, perf_counter() - started),
        )
    except Exception as exc:
        raise PredictionError(exc) from exc
