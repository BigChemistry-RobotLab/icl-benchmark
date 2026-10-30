"""Describe one experiment request and its validated run plan."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icl_agent.config import Task

from ..config import ModelConfig
from ..datasets import DatasetSpec


@dataclass(frozen=True)
class RunRequest:
    """Describe one run request from the command line."""

    model_type: str
    model: str | None
    reference_data: Path | None
    predict_data: Path
    task: Path
    max_length: int | None = None
    mode: str | None = None
    results_dir: Path | None = None
    seed: int | None = None
    replicates: int | None = None
    context_window: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    max_questions: int | None = None
    retry_failed: bool = False
    profile: bool = False


@dataclass(frozen=True)
class RunPlan:
    """Store the validated identity and prepared rows for one run."""

    model_type: str
    model: str
    mode: str
    settings: ModelConfig | None
    task: Task
    spec: DatasetSpec
    device: str
    options: dict[str, Any]
    dataset: str
    variant: str
    reference_data: Path | None
    predict_data: Path
    task_path: Path
    reference_rows: list[dict[str, str]]
    targets: list[dict[str, str]]
    labels: list[str]
    base_seed: int
    replicates: int
    profile: bool
    results_dir: Path
    max_length: int | None
    protocol_id: str
    split_id: str
