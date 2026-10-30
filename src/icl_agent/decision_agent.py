"""Answer predicate or choice questions with the OpenAI Decisions API."""

import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.decision import Answer
from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoice,
    QuestionQuestionParamPredicate,
)
from openai.types.decision_input_message_param import DecisionInputMessageParam
from openai.types.decision_input_text_param import DecisionInputTextParam
from openai.types.shared_params.decision_input_json import DecisionInputJSON

from .config import REQUEST_MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, Task
from .data import DataSource, Row, load_rows
from .results import PredictionError, PredictionResult

DECISION_INSTRUCTIONS = (
    "Use the labeled reference rows to infer how the task inputs map to the output. "
    "If no reference rows are provided, use the task context for zero-shot prediction. "
    "Empty cells are missing values, not zero. "
)

# The Decisions API accepts at most this many questions in one request.
MAX_DECISION_QUESTIONS = 200


def decision_input(
    task: Task, references: list[Row],
) -> list[DecisionInputMessageParam]:
    """Write the shared evidence for one Decisions request.

    Reference values use a column header and row arrays. This form omits the
    repeated column names from every row.

    Args:
        task: Validated task settings.
        references: Labeled reference rows.

    Returns:
        One user message with text instructions and JSON evidence.
    """
    reference_columns = [*task.inputs, *task.outputs]
    reference_data = {
        "columns": [item.label for item in reference_columns],
        "rows": [
            [row[item.column] for item in reference_columns] for row in references
        ],
    }
    context = task.additional_context or ""
    text = DECISION_INSTRUCTIONS
    if context:
        text += f"\nTask context:\n{context}"
    if not references:
        text += "\nNo labeled reference rows are supplied."
    data: dict[str, Any] = {"references": reference_data}
    return [
        DecisionInputMessageParam(
            role="user",
            content=[
                DecisionInputTextParam(type="input_text", text=text),
                DecisionInputJSON(type="input_json", data=data),
            ],
        )
    ]


def decision_instructions(task: Task, row: Row) -> str:
    """Format the task decision question for one target row.

    Args:
        task: Task with a ``decision_question`` template.
        row: Target row with the non-context input values.

    Returns:
        The template text with the row values.

    Raises:
        ValueError: The task has no decision question.
    """
    template = task.decision_question
    if template is None:
        raise ValueError("Decision tasks require a decision_question")
    target_inputs = [item for item in task.inputs if not item.is_context]
    return template.format_map(
        {item.column: row[item.column] for item in target_inputs}
    )


def decision_questions(task: Task, targets: list[Row]) -> list[Question]:
    """Create one predicate or choice question per target row.

    The task ``decision_question`` template supplies the text. Each placeholder
    is a non-context input column name, and the target row value replaces it.
    A two-label task asks one predicate question. A three-label task asks one
    choice question.

    Args:
        task: Task with exactly one output, its allowed labels, and a question.
        targets: Target rows in prediction order.

    Returns:
        Questions named ``target_0`` to ``target_<n>`` in row order.

    Raises:
        ValueError: The task contract, template, or target count is unsupported.
    """
    if len(task.outputs) != 1:
        raise ValueError("Decision tasks require exactly one output")
    output = task.outputs[0]
    labels = task.allowed_labels.get(output.column)
    if not labels:
        raise ValueError("Decision tasks require allowed labels for the output")
    if len(labels) not in {2, 3}:
        raise ValueError("Decision questions require two or three allowed labels")
    if len(targets) > MAX_DECISION_QUESTIONS:
        raise ValueError(
            f"Decision requests accept at most {MAX_DECISION_QUESTIONS} target rows"
        )
    questions: list[Question] = []
    for index, row in enumerate(targets):
        name = f"target_{index}"
        instructions = decision_instructions(task, row)
        if len(labels) == 2:
            predicate: QuestionQuestionParamPredicate = {
                "type": "predicate",
                "name": name,
                "instructions": instructions,
            }
            questions.append(predicate)
        else:
            choice: QuestionQuestionParamChoice = {
                "type": "choice",
                "name": name,
                "instructions": instructions,
                "choices": [{"value": label} for label in labels],
            }
            questions.append(choice)
    return questions


def _labels_from_answers(task: Task, target_count: int, answers: list[Answer]) -> list[str]:
    """Validate answer coverage and return one label per target in row order.

    Args:
        task: Task with exactly one output and its allowed labels.
        target_count: Number of requested target rows.
        answers: Decision answers in response order.

    Returns:
        Predicted labels indexed by target row.

    Raises:
        ValueError: An answer is missing, duplicated, or not a valid answer.
    """
    expected = [f"target_{index}" for index in range(target_count)]
    by_name: dict[str, Answer] = {}
    for answer in answers:
        name = answer.name
        if not isinstance(name, str) or name not in expected:
            raise ValueError("Decision answers must name the requested target rows")
        if name in by_name:
            raise ValueError("Decision answers must name each target row once")
        by_name[name] = answer
    labels = task.allowed_labels[task.outputs[0].column]
    predictions: list[str] = []
    for name in expected:
        answer = by_name.get(name)
        if answer is None:
            raise ValueError("Decision answers must cover exactly the requested target rows")
        if len(labels) == 2:
            if answer.type != "predicate":
                raise ValueError(f"Decision answer {name} must have type predicate")
            probability = getattr(answer, "probability", None)
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"Decision answer {name} has an invalid probability")
            predictions.append(labels[1] if probability >= 0.5 else labels[0])
            continue
        if answer.type != "choice":
            raise ValueError(f"Decision answer {name} must have type choice")
        label = getattr(answer, "choice", None)
        if not isinstance(label, str) or label not in labels:
            raise ValueError(f"Decision answer {name} has an invalid label")
        predictions.append(label)
    return predictions


async def run_decision(
    *,
    task: Task,
    reference_data: DataSource,
    predict_data: DataSource,
    model: str,
) -> PredictionResult:
    """Answer one question per target row with the OpenAI Decisions API.

    Two-label tasks use predicate questions and three-label tasks use choice
    questions. The request holds the instructions and task context as text, and
    the reference rows as JSON. Each question holds the target input values.

    The endpoint and credentials come from ``OPENAI_API_KEY`` and
    ``OPENAI_BASE_URL``. The endpoint must expose the OpenAI Decisions API.
    OpenAI currently supports only ``gpt-6-luna``.

    Args:
        task: Validated task with one output and its allowed labels.
        reference_data: Labeled CSV path or sequence of text dictionaries.
        predict_data: Target CSV path or sequence of text dictionaries.
        model: OpenAI Decisions model identifier.

    Returns:
        Target rows with predicted outputs and request metadata.

    Raises:
        ValueError: Task settings, datasets, or model name are invalid.
        PredictionError: The request or its answer validation fails.
    """
    task = Task.model_validate(task.model_dump())
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    inputs = [item.column for item in task.inputs]
    prediction_inputs = [item.column for item in task.inputs if not item.is_context]
    outputs = [item.column for item in task.outputs]
    references = load_rows(reference_data, [*inputs, *outputs], "Reference data")
    targets = load_rows(predict_data, prediction_inputs, "Prediction data")
    if not targets:
        return PredictionResult(predictions=[], reasoning="")
    questions = decision_questions(task, targets)
    payload = decision_input(task, references)
    try:
        load_dotenv(Path.cwd() / ".env", override=False)
        api_base = os.getenv("OPENAI_API_BASE")
        if api_base:
            os.environ["OPENAI_BASE_URL"] = api_base
        started = perf_counter()
        async with AsyncOpenAI(
            timeout=REQUEST_TIMEOUT_SECONDS, max_retries=REQUEST_MAX_RETRIES,
        ) as client:
            decision = await client.decisions.create(model=model, input=payload, questions=questions)
        labels = _labels_from_answers(task, len(targets), decision.answers)
        return PredictionResult(
            predictions=[
                {**{column: row[column] for column in prediction_inputs},
                 task.outputs[0].column: label}
                for row, label in zip(targets, labels, strict=True)
            ],
            reasoning="",
            metadata={
                "model": decision.model,
                "usage": decision.usage.model_dump(mode="json"),
                "answers": [answer.model_dump(mode="json") for answer in decision.answers],
                "rationale_available": False,
                "elapsed_seconds": perf_counter() - started,
            },
        )
    except Exception as exc:
        raise PredictionError(exc) from exc
