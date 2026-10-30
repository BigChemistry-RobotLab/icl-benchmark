"""Store prediction results and report prediction failures."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    """Store target inputs, predicted outputs, and one reasoning summary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    predictions: list[dict[str, str]]
    reasoning: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionError(RuntimeError):
    """Report a failed prediction run."""

    def __init__(self, cause: Exception) -> None:
        """Store the original error message."""
        super().__init__(f"Prediction run: {cause}")
