"""Build the content and settings identity of one run protocol."""

from typing import Any

from icl_agent.config import Task

from ..config import LOCAL_MODEL_TYPES, ModelConfig, endpoint_identity
from ..datasets import DatasetSpec
from ..endpoints import chat, decisions
from ..models.encoding import TOKENS


def protocol_identity(
    *,
    task: Task,
    model_type: str,
    model: str,
    settings: ModelConfig | None,
    device: str,
    options: dict[str, Any],
    mode: str,
    base_seed: int,
    max_length: int | None,
    reference_rows: list[dict[str, str]],
    predict_rows: list[dict[str, str]],
    spec: DatasetSpec,
) -> dict[str, Any]:
    """Build the content and settings identity of one run protocol."""
    identity: dict[str, Any] = {
        "task": task.model_dump(mode="json"),
        "reference_rows": reference_rows,
        "predict_rows": predict_rows,
        "model_type": model_type,
        "model": model,
        "mode": mode,
        "seed": base_seed,
        "max_length": max_length,
    }
    if model_type in LOCAL_MODEL_TYPES:
        identity["settings"] = {"device": device, "options": options}
        identity["encoding"] = list(TOKENS)
        identity["instructions"] = None
        return identity
    if settings is None:
        raise ValueError("Endpoint models require registry settings")
    identity["settings"] = settings.model_dump(mode="json")
    if model_type == "llm":
        identity["instructions"] = chat.INSTRUCTIONS
        identity["endpoint"] = endpoint_identity()
        return identity
    identity["instructions"] = decisions.INSTRUCTIONS
    identity["decision_endpoint"] = decisions.endpoint_url()
    identity["decision_payload"] = decisions.build_payload(
        task, [], [{name: "example" for name in spec.inputs}], model,
    )
    return identity
