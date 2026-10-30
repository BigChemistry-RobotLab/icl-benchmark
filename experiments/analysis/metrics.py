"""Compute classification metrics with an explicit label order."""

from collections.abc import Sequence
from typing import Any


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Compute class scores and a confusion matrix.

    Args:
        y_true: True labels in test row order.
        y_pred: Predicted labels in test row order.
        labels: All valid labels. The last binary label is positive.

    Returns:
        Macro F1, class scores, and counts in the supplied label order.

    Raises:
        ValueError: A label is invalid or the input lengths differ.
    """
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be nonempty and unique.")
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("True and predicted labels must have equal nonzero lengths.")
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual not in positions or predicted not in positions:
            raise ValueError("True and predicted labels must be in the allowed labels.")
        matrix[positions[actual]][positions[predicted]] += 1
    per_class = {}
    for index, label in enumerate(labels):
        correct = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
    result: dict[str, Any] = {
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": list(labels),
    }
    if len(labels) == 2:
        result["positive_f1"] = per_class[labels[-1]]["f1"]
    return result
