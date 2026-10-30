"""Build a prompt for all prediction rows."""

import json
from collections.abc import Sequence

from .config import Column, Task
from .data import Row, rows_to_csv


def _display_csv(rows: Sequence[Row], columns: Sequence[Column]) -> str:
    """Write CSV with display labels and unchanged cell text."""
    labeled_rows = [{item.label: row[item.column] for item in columns} for row in rows]
    return rows_to_csv(labeled_rows, [item.label for item in columns])


def reference_prompt(task: Task, rows: Sequence[Row]) -> str:
    """Write the context and all reference rows."""
    reference = _display_csv(rows, task.inputs + task.outputs)
    context = task.additional_context or ""
    if task.allowed_labels:
        context += "\nAllowed output labels: " + json.dumps(task.allowed_labels, ensure_ascii=False)
    if not rows:
        context += "\nZero-shot prediction: no labeled reference examples are supplied."
    return f"Context:\n{context}\n\nReference data:\n{reference}"


def target_prompt(task: Task, reference: str, rows: Sequence[Row]) -> str:
    """Add all target rows with only non-context inputs."""
    outputs = '", "'.join(item.label for item in task.outputs)
    target_inputs = [item for item in task.inputs if not item.is_context]
    column = target_id_column(task)
    target = rows_to_csv([
        {column: f"target_{index}", **{item.label: row[item.column] for item in target_inputs}}
        for index, row in enumerate(rows)
    ], [column, *[item.label for item in target_inputs]])
    return (f'{reference}\nPredict "{outputs}" for the following targets in row order. '
            f'Use each "{column}" value as its key in the predictions object:\n{target}')


def target_id_column(task: Task) -> str:
    """Return a target ID header that does not replace a task column."""
    existing = {name for item in [*task.inputs, *task.outputs] for name in [item.column, item.label]}
    column = "target_id"
    while column in existing:
        column = "_" + column
    return column
