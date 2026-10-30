"""Read experiment settings and endpoint model registry entries."""

import os
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOCAL_MODEL_TYPES = ("xgboost", "tabpfn", "tabfm")
MODEL_TYPES = ("llm", "decision", *LOCAL_MODEL_TYPES)
MODES = ("one_by_one", "all_at_once")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def load_environment() -> None:
    """Load the repository .env file without overriding existing variables."""
    load_dotenv(Path.cwd() / ".env", override=False)


def endpoint_identity() -> str:
    """Return the endpoint location without URL credentials or query data."""
    load_environment()
    endpoint = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


class ModelConfig(BaseModel):
    """Set the request limits and prices for one endpoint model."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["llm", "decision"]
    context_window: int | None = Field(default=None, gt=0, strict=True)
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    reasoning_effort: ReasoningEffort | None = None
    send_seed: bool = False
    input_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_questions: int = Field(default=200, gt=0, le=200, strict=True)

    @model_validator(mode="after")
    def check_type_settings(self) -> Self:
        """Reject settings that the selected model type does not use."""
        if self.type == "decision":
            if (self.temperature is not None or self.max_output_tokens is not None
                    or self.reasoning_effort is not None or self.send_seed):
                raise ValueError("Decision models do not accept chat inference settings")
        elif self.max_questions != 200:
            raise ValueError("LLM models require max_questions = 200")
        return self


class ExperimentConfig(BaseModel):
    """Define the replicate counts, local options, and endpoint registry."""

    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, strict=True)
    llm_replicates: int = Field(default=3, gt=0, strict=True)
    baseline_replicates: int = Field(default=100, gt=0, strict=True)
    device: str = "cpu"
    results_dir: Path = Path("results")
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    baseline_options: dict[str, dict] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def check_model_ids(cls, value: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        """Reject blank provider identifiers."""
        if any(not name.strip() for name in value):
            raise ValueError("Model registry keys must not be blank")
        return value

    @field_validator("baseline_options")
    @classmethod
    def check_baseline_options(cls, value: dict[str, dict]) -> dict[str, dict]:
        """Allow options only for local model types."""
        if set(value) - set(LOCAL_MODEL_TYPES):
            raise ValueError("Baseline options must name local model types")
        if value.get("xgboost"):
            raise ValueError("XGBoost uses fixed settings; options must be empty")
        return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a JSON experiment file."""
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
