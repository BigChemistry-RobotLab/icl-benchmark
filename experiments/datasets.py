"""Derive the task contract and dataset identity from inputs and paths."""

from dataclasses import dataclass
from pathlib import Path

from icl_agent.config import Task


@dataclass(frozen=True)
class DatasetSpec:
    """Store the source and model input contract for a task."""

    name: str
    inputs: tuple[str, ...]
    target: str
    labels: tuple[str, ...]
    max_length: int


def _task_contract(task: Task) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return the single target, non-context inputs, and labels."""
    if len(task.outputs) != 1:
        raise ValueError("Dataset tasks require exactly one output")
    target = task.outputs[0].column
    labels = task.allowed_labels.get(target)
    if not labels:
        raise ValueError("Dataset tasks require allowed labels for the output")
    inputs = tuple(item.column for item in task.inputs if not item.is_context)
    return target, inputs, tuple(labels)


def dataset_spec(task: Task, name: str, max_length: int) -> DatasetSpec:
    """Build the local model contract from a validated task.

    Args:
        task: Task with one output and its allowed labels.
        name: Dataset name used in result records.
        max_length: Maximum sequence length in residues.

    Returns:
        The task contract with only the non-context inputs.

    Raises:
        ValueError: The name, length, or task contract is unsupported.
    """
    if not name.strip():
        raise ValueError("Dataset name must not be blank")
    if max_length <= 0:
        raise ValueError("Maximum sequence length must be positive")
    target, inputs, labels = _task_contract(task)
    if "sequence" not in inputs:
        raise ValueError("Dataset tasks require a non-context sequence input")
    return DatasetSpec(
        name=name,
        inputs=inputs,
        target=target,
        labels=labels,
        max_length=max_length,
    )


def endpoint_spec(task: Task, name: str) -> DatasetSpec:
    """Build the endpoint contract for one task without a local length limit.

    Args:
        task: Task with one output and its allowed labels.
        name: Dataset name used in result records.

    Returns:
        The task contract with the local length limit set to one.

    Raises:
        ValueError: The task contract is unsupported.
    """
    target, inputs, labels = _task_contract(task)
    if not inputs:
        raise ValueError("Dataset tasks require at least one non-context input")
    return DatasetSpec(
        name=name,
        inputs=inputs,
        target=target,
        labels=labels,
        max_length=1,
    )


VARIANT_FILES = {
    "train.csv": "full",
    "train_balanced.csv": "balanced",
    "train_corrupted.csv": "corrupted_full",
    "train_balanced_corrupted.csv": "corrupted_balanced",
}


def dataset_identity(reference_data: Path | None, predict_data: Path) -> tuple[str, str]:
    """Resolve the dataset name and training variant from the input paths.

    The variant comes from the reference file name. The four flat names are
    ``train.csv`` (full), ``train_balanced.csv`` (balanced),
    ``train_corrupted.csv`` (corrupted full), and
    ``train_balanced_corrupted.csv`` (corrupted balanced). An unknown name uses
    the ``default`` variant.
    """
    predict_parent = predict_data.resolve().parent
    if reference_data is None:
        return predict_parent.name, "zero_shot"
    reference = reference_data.resolve()
    if reference.parent == predict_parent:
        return predict_parent.name, VARIANT_FILES.get(reference.name, "default")
    return predict_parent.name, "default"


def select_columns(
    rows: list[dict[str, str]], columns: tuple[str, ...],
) -> list[dict[str, str]]:
    """Return new rows with only the requested columns."""
    return [{column: row[column] for column in columns} for row in rows]
