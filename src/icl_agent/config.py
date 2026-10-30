"""Load and validate task settings."""

from pathlib import Path
from string import Formatter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# One provider request must finish inside this limit. A stalled request then
# fails in a bounded time and does not hold the run open.
REQUEST_TIMEOUT_SECONDS = 180.0
REQUEST_MAX_RETRIES = 1


class Column(BaseModel):
    """Select a column, display name, and optional context type."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    column: str = Field(min_length=1)
    alias: str | None = None
    type: Literal["context"] | None = None

    @property
    def label(self) -> str:
        """Return the alias or the column name for display."""
        return self.alias if self.alias is not None else self.column

    @property
    def is_context(self) -> bool:
        """Return whether the column is context for the agent only."""
        return self.type == "context"

    @field_validator("alias")
    @classmethod
    def check_alias(cls, value: str | None) -> str | None:
        """Reject an alias that is empty or contains only spaces."""
        if value is not None and not value.strip():
            raise ValueError("Column aliases must not be blank")
        return value

    @field_validator("column")
    @classmethod
    def check_column(cls, value: str) -> str:
        """Reject a name that contains only spaces."""
        if not value.strip():
            raise ValueError("Column names must not be blank")
        return value


class Task(BaseModel):
    """Define the input columns, output columns, additional context, and decision question."""

    model_config = ConfigDict(extra="forbid", strict=True)

    inputs: list[Column] = Field(min_length=1)
    outputs: list[Column] = Field(min_length=1)
    additional_context: str | None = None
    decision_question: str | None = None
    allowed_labels: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_selections(self) -> Self:
        """Require unique input and output columns with no overlap."""
        if any(item.is_context for item in self.outputs):
            raise ValueError("context columns must be inputs")
        columns = [item.column for item in self.inputs + self.outputs]
        if len(columns) != len(set(columns)):
            raise ValueError("Input and output columns must be unique and disjoint")
        labels = [item.label for item in self.inputs + self.outputs]
        if len(labels) != len(set(labels)):
            raise ValueError("Input and output display labels must be unique")
        output_names = {item.column for item in self.outputs}
        if not set(self.allowed_labels) <= output_names:
            raise ValueError("Allowed label keys must name output columns")
        for values in self.allowed_labels.values():
            if not values or any(not value.strip() for value in values):
                raise ValueError("Allowed labels must contain nonempty text")
            if len(values) != len(set(values)):
                raise ValueError("Allowed labels must be unique")
        return self

    @model_validator(mode="after")
    def check_decision_question(self) -> Self:
        """Require a decision question that uses at least one target input."""
        template = self.decision_question
        if template is None:
            return self
        if not template.strip():
            raise ValueError("Decision question must not be blank")
        try:
            parts = list(Formatter().parse(template))
        except ValueError as error:
            raise ValueError("Decision question must use valid braces") from error
        names: list[str] = []
        for _, name, spec, conversion in parts:
            if name is None:
                continue
            if not name:
                raise ValueError("Decision question placeholders must name an input column")
            if spec or conversion is not None:
                raise ValueError(
                    "Decision question placeholders must not use format specs or conversions")
            names.append(name)
        if not names:
            raise ValueError("Decision question must contain at least one placeholder")
        inputs = [item.column for item in self.inputs if not item.is_context]
        unknown = [name for name in names if name not in inputs]
        if unknown:
            raise ValueError(
                f"Decision question placeholder {unknown[0]!r} is not a target input column")
        return self


class InferenceSettings(BaseModel):
    """Set optional model request parameters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    seed: int | None = Field(default=None, ge=0)


def load_task(path: str | Path) -> Task:
    """Load a task from a UTF-8 JSON file.

    Args:
        path: Path to the task file.

    Returns:
        The validated task.
    """
    return Task.model_validate_json(Path(path).read_text(encoding="utf-8"))
