"""Validate run requests and build the resolved run plan."""

import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from icl_agent import load_task
from icl_agent.config import Task
from icl_agent.data import load_rows

from ..config import (
    LOCAL_MODEL_TYPES,
    MODEL_TYPES,
    MODES,
    ExperimentConfig,
    ModelConfig,
    load_environment,
)
from ..datasets import (
    DatasetSpec,
    dataset_identity,
    dataset_spec,
    endpoint_spec,
    select_columns,
)
from ..endpoints import chat, decisions
from ..models.local import check_options
from .identity import protocol_identity
from .records import fingerprint
from .request import RunPlan, RunRequest


def resolve_model(
    config: ExperimentConfig, request: RunRequest,
) -> tuple[str, str, ModelConfig | None, str]:
    """Resolve the model identifier, registry settings, and inference mode."""
    model_type = request.model_type
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type in LOCAL_MODEL_TYPES:
        if request.mode is not None:
            raise ValueError("mode is only valid for llm and decision models")
        if request.model is not None and request.model != model_type:
            raise ValueError(f"A local model is its model type: {model_type}")
        unused = [
            name for name, value in (
                ("context_window", request.context_window),
                ("temperature", request.temperature),
                ("max_output_tokens", request.max_output_tokens),
                ("reasoning_effort", request.reasoning_effort),
                ("max_questions", request.max_questions),
            )
            if value is not None
        ]
        if unused:
            raise ValueError(f"Endpoint request settings are not used by local models: {unused}")
        return model_type, model_type, None, "local"
    if request.max_length is not None:
        raise ValueError(f"max_length is only used by local models, not {model_type} models")
    model = (request.model or "").strip()
    if not model:
        raise ValueError("model is required for llm and decision models")
    mode = request.mode
    if mode is None:
        raise ValueError("mode is required for llm and decision models")
    if mode not in MODES:
        raise ValueError(f"Unknown inference mode: {mode}")
    settings = config.models.get(model)
    if settings is None:
        raise ValueError(f"Model is not registered in the config: {model}")
    if settings.type != model_type:
        raise ValueError(f"Registered model type does not match --model-type: {settings.type}")
    return model_type, model, settings, mode


def apply_overrides(settings: ModelConfig, request: RunRequest) -> ModelConfig:
    """Apply explicit request overrides and validate the selected model type."""
    updates: dict[str, Any] = {}
    if request.context_window is not None:
        updates["context_window"] = request.context_window
    if request.temperature is not None:
        updates["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        updates["max_output_tokens"] = request.max_output_tokens
    if request.reasoning_effort is not None:
        updates["reasoning_effort"] = request.reasoning_effort
    if request.max_questions is not None:
        updates["max_questions"] = request.max_questions
    if not updates:
        return settings
    try:
        return ModelConfig.model_validate({**settings.model_dump(), **updates})
    except ValueError as error:
        raise ValueError(f"Invalid model settings: {error}") from error


def resolve_seed(config: ExperimentConfig, request: RunRequest) -> int:
    """Resolve and validate the protocol seed."""
    seed = request.seed if request.seed is not None else config.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def replicate_count(config: ExperimentConfig, request: RunRequest, model_type: str) -> int:
    """Resolve the replicate count for the selected model type."""
    if request.replicates is not None:
        count = request.replicates
    elif model_type in LOCAL_MODEL_TYPES:
        count = config.baseline_replicates
    else:
        count = config.llm_replicates
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("replicates must be a positive integer")
    return count


def check_paths(request: RunRequest) -> None:
    """Check that every request input file exists before loading it."""
    if request.reference_data is not None and not request.reference_data.is_file():
        raise ValueError(f"Reference data is missing: {request.reference_data}")
    if not request.predict_data.is_file():
        raise ValueError(f"Prediction data is missing: {request.predict_data}")
    if not request.task.is_file():
        raise ValueError(f"Task file is missing: {request.task}")


def resolve_max_length(request: RunRequest) -> int:
    """Resolve the maximum sequence length for local encoding."""
    max_length = request.max_length
    if max_length is None:
        raise ValueError("max_length is required for local models: set --max-length")
    if isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    return int(max_length)


def check_local_inputs(task: Task) -> None:
    """Reject context input columns that local encoders would drop."""
    context_columns = [item.column for item in task.inputs if item.is_context]
    if context_columns:
        raise ValueError(
            f"Local models do not support context input columns: {context_columns}"
        )


def check_labels(
    spec: DatasetSpec, predict_rows: list[dict[str, str]], reference_rows: list[dict[str, str]],
) -> list[str]:
    """Check allowed labels and local control coverage in the input tables."""
    allowed = set(spec.labels)
    labels = []
    for index, row in enumerate(predict_rows):
        label = row[spec.target]
        if label not in allowed:
            raise ValueError(f"Prediction data row {index}: invalid label {label!r}")
        labels.append(label)
    for index, row in enumerate(reference_rows):
        if row[spec.target] not in allowed:
            raise ValueError(f"Reference data row {index}: invalid label {row[spec.target]!r}")
    return labels


def check_local_reference(
    model_type: str, spec: DatasetSpec, reference_rows: list[dict[str, str]],
) -> None:
    """Require a complete local training table with every declared class."""
    if model_type not in LOCAL_MODEL_TYPES:
        return
    if not reference_rows:
        raise ValueError("Reference data is required for local models")
    found = {row[spec.target] for row in reference_rows}
    missing = [label for label in spec.labels if label not in found]
    if missing:
        raise ValueError(f"Reference data is missing declared classes: {missing}")


def check_credentials(model_type: str) -> None:
    """Check the endpoint credentials required by the model type."""
    load_environment()
    if model_type == "llm":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not configured")
        return
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY is required for decision models")


def check_endpoint_settings(settings: ModelConfig, model: str) -> None:
    """Check explicit endpoint capacity and output limits."""
    if settings.context_window is None:
        raise ValueError(f"{model}: context_window must be explicit")
    if settings.max_output_tokens is not None and settings.max_output_tokens >= settings.context_window:
        raise ValueError(f"{model}: output limit must be smaller than context window")


def check_admission(
    model_type: str,
    task: Task,
    reference_rows: list[dict[str, str]],
    targets: list[dict[str, str]],
    model: str,
    settings: ModelConfig,
    mode: str,
) -> None:
    """Admit every exact request group before any endpoint call is made."""
    groups = [[row] for row in targets] if mode == "one_by_one" else [targets]
    for group in groups:
        try:
            if model_type == "llm":
                chat.admission(task, reference_rows, group, settings)
            else:
                decisions.admission(task, reference_rows, group, model, settings)
        except (TypeError, ValueError) as error:
            label = "LLM" if model_type == "llm" else "Decision"
            raise ValueError(f"{label} admission failed: {error}") from error


def check_local_dependencies(
    config: ExperimentConfig, model: str, options: dict[str, Any],
) -> None:
    """Check the optional package, checkpoint, and CUDA requirement."""
    if find_spec(model) is None:
        raise ValueError(f"Missing optional dependency: {model}")
    check_options(model, options)
    if model in {"tabpfn", "tabfm"}:
        checkpoint = options.get("checkpoint_path")
        if checkpoint is None and not options.get("allow_download", False):
            raise ValueError(f"{model}: set checkpoint_path or allow_download explicitly")
        if checkpoint is not None:
            path = Path(str(checkpoint))
            available = path.is_file() if model == "tabpfn" else path.is_dir()
            if not available:
                raise ValueError(f"{model}: checkpoint path is unavailable: {path}")
        if config.device.startswith("cuda"):
            if find_spec("torch") is None:
                raise ValueError("CUDA requires torch")
            if not import_module("torch").cuda.is_available():
                raise ValueError("CUDA is not available")


def validate_run(config: ExperimentConfig, request: RunRequest) -> RunPlan:
    """Check every request input before any model or endpoint call.

    Args:
        config: Experiment settings and endpoint registry.
        request: One run request from the command line.

    Returns:
        The resolved run plan with identity and prepared rows.

    Raises:
        ValueError: A setting, data table, dependency, or admission check fails.
    """
    model_type, model, settings, mode = resolve_model(config, request)
    if request.profile and model_type not in LOCAL_MODEL_TYPES:
        raise ValueError("profile is only valid for local models")
    base_seed = resolve_seed(config, request)
    replicates = replicate_count(config, request, model_type)
    results_dir = request.results_dir or config.results_dir
    check_paths(request)
    task = load_task(request.task)
    dataset, variant = dataset_identity(request.reference_data, request.predict_data)
    if model_type in LOCAL_MODEL_TYPES:
        check_local_inputs(task)
        max_length = resolve_max_length(request)
        spec = dataset_spec(task, dataset, max_length)
    else:
        max_length = None
        spec = endpoint_spec(task, dataset)
    input_columns = tuple(item.column for item in task.inputs)
    predict_rows = load_rows(request.predict_data, (*spec.inputs, spec.target), "Prediction data")
    if not predict_rows:
        raise ValueError("Prediction data must contain at least one row")
    reference_rows: list[dict[str, str]] = []
    if request.reference_data is not None:
        reference_rows = load_rows(
            request.reference_data, (*input_columns, spec.target), "Reference data",
        )
    labels = check_labels(spec, predict_rows, reference_rows)
    check_local_reference(model_type, spec, reference_rows)
    targets = select_columns(predict_rows, spec.inputs)
    options: dict[str, Any] = {}
    effective_settings: ModelConfig | None = None
    if settings is not None:
        effective_settings = apply_overrides(settings, request)
        check_endpoint_settings(effective_settings, model)
        check_credentials(model_type)
        check_admission(
            model_type, task, reference_rows, targets, model, effective_settings, mode,
        )
    else:
        options = dict(config.baseline_options.get(model) or {})
        check_local_dependencies(config, model, options)
    identity = protocol_identity(
        task=task,
        model_type=model_type,
        model=model,
        settings=effective_settings,
        device=config.device,
        options=options,
        mode=mode,
        base_seed=base_seed,
        max_length=max_length,
        reference_rows=reference_rows,
        predict_rows=predict_rows,
        spec=spec,
    )
    protocol_id = fingerprint(identity)
    split_id = fingerprint({"reference_rows": reference_rows, "predict_rows": predict_rows})
    return RunPlan(
        model_type=model_type,
        model=model,
        mode=mode,
        settings=effective_settings,
        task=task,
        spec=spec,
        device=config.device,
        options=options,
        dataset=dataset,
        variant=variant,
        reference_data=request.reference_data,
        predict_data=request.predict_data,
        task_path=request.task,
        reference_rows=reference_rows,
        targets=targets,
        labels=labels,
        base_seed=base_seed,
        replicates=replicates,
        profile=request.profile,
        results_dir=results_dir,
        max_length=max_length,
        protocol_id=protocol_id,
        split_id=split_id,
    )

