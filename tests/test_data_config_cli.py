"""Check task files, CSV cells, and CLI output."""

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from icl_agent import PredictionError, PredictionResult, Task, load_task
from icl_agent.cli import main
from icl_agent.data import load_rows, rows_to_csv


@pytest.mark.parametrize("settings", [
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}], "description": "Old key"},
    {"inputs": [], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x"}], "outputs": []},
    {"inputs": [{"column": "x"}, {"column": "x"}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}, {"column": "y"}]},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "x"}]},
    {"inputs": [{"column": ""}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "  "}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": 1}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x", "unknown": "a"}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}], "additional_context": 1},
    {"inputs": [{"column": "x", "alias": ""}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x", "alias": "  "}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x", "alias": "y"}], "outputs": [{"column": "y"}]},
    {"inputs": [{"column": "x", "alias": "label"}], "outputs": [{"column": "y", "alias": "label"}]},
])
def test_invalid_task(settings):
    """Reject invalid column selections and metadata."""
    with pytest.raises(ValidationError):
        Task.model_validate(settings)


def test_load_task(tmp_path):
    """Load optional aliases and additional context from JSON."""
    path = tmp_path / "task.json"
    path.write_text('{"inputs":[{"column":"x"}],"outputs":[{"column":"y","alias":"β"}]}')
    task = load_task(path)
    assert task.additional_context is None
    assert task.inputs[0].alias is None
    assert task.outputs[0].alias == "β"


def test_load_context_input(tmp_path):
    """Load a context input column."""
    path = tmp_path / "task.json"
    path.write_text(
        '{"inputs":[{"column":"x","type":"context"}],'
        '"outputs":[{"column":"y"}]}'
    )
    task = load_task(path)
    assert task.inputs[0].is_context


def test_valid_decision_question():
    """Accept a template with at least one target input placeholder."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}, {"column": "y"}],
        "outputs": [{"column": "label"}],
        "decision_question": "Is {x} greater than {y}?",
    })
    assert task.decision_question == "Is {x} greater than {y}?"


def test_decision_question_may_omit_target_inputs():
    """Accept a template that uses only some target inputs."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}, {"column": "z"}],
        "outputs": [{"column": "y"}],
        "decision_question": "Is {x} high?",
    })
    assert task.decision_question == "Is {x} high?"


@pytest.mark.parametrize("settings", [
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": ""},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "   "},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "Is {x high?"},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "Is it high?"},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "Is {z} high?"},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "Is {x:>3} high?"},
    {"inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
     "decision_question": "Is {x!r} high?"},
    {"inputs": [{"column": "x"}, {"column": "ctx", "type": "context"}],
     "outputs": [{"column": "y"}],
     "decision_question": "Is {ctx} high?"},
])
def test_invalid_decision_question(settings):
    """Reject decision question templates that do not fit the task inputs."""
    with pytest.raises(ValidationError):
        Task.model_validate(settings)


def test_reject_unknown_column_type():
    """Reject a column type other than context."""
    with pytest.raises(ValidationError):
        Task.model_validate({
            "inputs": [{"column": "x", "type": "other"}],
            "outputs": [{"column": "y"}],
        })


def test_reject_context_output():
    """Reject a context column in the output selection."""
    with pytest.raises(ValidationError):
        Task.model_validate({
            "inputs": [{"column": "x"}],
            "outputs": [{"column": "y", "type": "context"}],
        })


def test_csv_round_trip(tmp_path):
    """Keep empty cells, spaces, quotes, line breaks, and Unicode."""
    rows = [{"x": '  β,"quoted"\r\nnext\rlast\n', "y": ""}, {"x": "001", "y": " NaN "}]
    path = tmp_path / "data.csv"
    path.write_text(rows_to_csv(rows, ["x", "y"]), encoding="utf-8", newline="")
    assert load_rows(path, ["x", "y"], "Reference data") == rows


@pytest.mark.parametrize("text,match", [
    ("", "header"),
    ("x,x\n1,2\n", "unique"),
    ("z,y\n1,2\n", "missing"),
    ("x,y\n1\n", "row 0"),
    ("x,y\n1,2,3\n", "row 0"),
    ('x,y\n"unclosed,2', "row 0"),
    ('x,y\n"one"junk,2', "row 0"),
    ("x,y\n\n", "row 0"),
])
def test_invalid_csv(tmp_path, text, match):
    """Reject invalid headers and records."""
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_rows(path, ["x", "y"], "Prediction data")


def test_header_only_csv(tmp_path):
    """Accept a header with no target records."""
    path = tmp_path / "empty.csv"
    path.write_text("x\n")
    assert load_rows(path, ["x"], "Prediction data") == []


def test_cli_json_and_errors(tmp_path, monkeypatch, capsys):
    """Print the result or a run error with a failing exit code."""
    task_path = tmp_path / "task.json"
    task_path.write_text('{"inputs":[{"column":"x"}],"outputs":[{"column":"y"}]}')
    result = PredictionResult(predictions=[{"x": "1", "y": "  β  "}], reasoning="  Pattern.\n")
    mock = AsyncMock(return_value=result)
    monkeypatch.setattr("icl_agent.cli.run_agent", mock)
    args = [
        "--task", str(task_path), "--reference-data", "ref.csv",
        "--predict-data", "target.csv", "--model", "chosen-model",
    ]
    main(args)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result.model_dump()
    assert captured.err == ""
    assert mock.call_args.kwargs["model"] == "chosen-model"
    mock.side_effect = PredictionError(RuntimeError("context limit"))
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Prediction run: context limit" in captured.err


def test_cli_uses_default_model(tmp_path, monkeypatch, capsys):
    """Accept a command with no model override."""
    task_path = tmp_path / "task.json"
    task_path.write_text('{"inputs":[{"column":"x"}],"outputs":[{"column":"y"}]}')
    mock = AsyncMock(return_value=PredictionResult(predictions=[], reasoning=""))
    monkeypatch.setattr("icl_agent.cli.run_agent", mock)
    main(["--task", str(task_path), "--reference-data", "r", "--predict-data", "p"])
    assert mock.call_args.kwargs["model"] is None
    assert json.loads(capsys.readouterr().out) == {"predictions": [], "reasoning": "", "metadata": {}}


@pytest.mark.parametrize("labels", [
    {"unknown": ["a"]}, {"y": []}, {"y": [""]}, {"y": [" "]}, {"y": ["a", "a"]},
])
def test_invalid_allowed_labels(labels):
    """Reject output class definitions that are not valid."""
    with pytest.raises(ValidationError):
        Task.model_validate({
            "inputs": [{"column": "x"}], "outputs": [{"column": "y"}],
            "allowed_labels": labels,
        })


@pytest.mark.parametrize("settings", [
    {"temperature": -1.0}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"max_output_tokens": 0}, {"max_output_tokens": True}, {"seed": -1}, {"seed": True},
])
def test_invalid_inference_settings(settings):
    """Reject invalid request limits and seeds."""
    from icl_agent import InferenceSettings

    with pytest.raises(ValidationError):
        InferenceSettings.model_validate(settings)


def test_cli_zero_shot_settings(tmp_path, monkeypatch, capsys):
    """Allow an omitted reference CSV and explicit model settings."""
    task_path = tmp_path / "task.json"
    task_path.write_text('{"inputs":[{"column":"x"}],"outputs":[{"column":"y"}]}')
    mock = AsyncMock(return_value=PredictionResult(predictions=[], reasoning=""))
    monkeypatch.setattr("icl_agent.cli.run_agent", mock)
    main([
        "--task", str(task_path), "--predict-data", "target.csv",
        "--temperature", "0.5", "--max-output-tokens", "100", "--seed", "7",
    ])
    assert mock.call_args.kwargs["reference_data"] == []
    assert mock.call_args.kwargs["settings"].model_dump() == {
        "temperature": 0.5, "max_output_tokens": 100, "seed": 7, "reasoning_effort": None,
    }
    assert json.loads(capsys.readouterr().out)["metadata"] == {}
