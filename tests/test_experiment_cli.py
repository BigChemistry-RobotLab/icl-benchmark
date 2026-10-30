"""Check the experiment command line for the run and plot commands."""

import json
from pathlib import Path

import pytest

from experiments.cli import _parser, main
from experiments.config import ExperimentConfig
from experiments.runner import RunRequest


def test_parser_exposes_only_run_and_plot():
    """Accept the atomic run and plot commands and reject removed commands."""
    parser = _parser()
    plot = parser.parse_args(["plot"])
    assert plot.command == "plot"
    assert plot.results_dir == Path("results")
    assert plot.output_dir == Path("reports")

    run = parser.parse_args([
        "run", "--config", "config.json", "--model-type", "llm",
        "--predict-data", "test.csv", "--task", "task.json",
    ])
    assert run.command == "run"
    assert run.model_type == "llm"
    assert run.reference_data is None
    assert run.mode is None
    assert run.max_length is None
    assert run.profile is False
    assert run.retry_failed is False

    for removed in ("prepare", "preflight", "pilot", "review", "summarize", "profile"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_run_dispatches_the_complete_request(monkeypatch, capsys):
    """Build one run request from the flags and print the JSON summary."""
    captured: dict = {}

    def fake_run_experiments(config, request):
        captured["config"] = config
        captured["request"] = request
        return {"status": "completed", "runs": 1, "completed": 1, "failed": 0,
                "dataset": "peptide", "variant": "full",
                "model": "provider/chat-model", "mode": "all_at_once",
                "results_dir": "results", "errors": []}

    monkeypatch.setattr("experiments.runner.run_experiments", fake_run_experiments)
    monkeypatch.setattr("experiments.cli.load_config", lambda path: ExperimentConfig())
    main([
        "run", "--config", "unused.json", "--model-type", "llm",
        "--model", "provider/chat-model", "--reference-data", "reference.csv",
        "--predict-data", "test.csv", "--task", "task.json",
        "--mode", "all_at_once", "--results-dir", "out", "--seed", "7",
        "--replicates", "2", "--context-window", "5000", "--temperature", "0.2",
        "--max-output-tokens", "100", "--reasoning-effort", "low",
        "--max-questions", "200", "--max-length", "5", "--profile",
        "--retry-failed",
    ])
    request = captured["request"]
    assert isinstance(request, RunRequest)
    assert request.model_type == "llm"
    assert request.model == "provider/chat-model"
    assert request.reference_data == Path("reference.csv")
    assert request.predict_data == Path("test.csv")
    assert request.task == Path("task.json")
    assert request.mode == "all_at_once"
    assert request.results_dir == Path("out")
    assert request.seed == 7
    assert request.replicates == 2
    assert request.context_window == 5000
    assert request.temperature == 0.2
    assert request.max_output_tokens == 100
    assert request.reasoning_effort == "low"
    assert request.max_questions == 200
    assert request.max_length == 5
    assert request.profile is True
    assert request.retry_failed is True
    assert isinstance(captured["config"], ExperimentConfig)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["runs"] == 1


def test_failed_run_prints_errors_and_exits(monkeypatch, capsys):
    """Report failed replicates on stderr and return a nonzero status."""
    def fake_run_experiments(config, request):
        return {"status": "failed", "runs": 1, "completed": 0, "failed": 1,
                "dataset": "peptide", "variant": "full", "model": "xgboost",
                "mode": "local", "results_dir": "results",
                "errors": ["run-1: endpoint failure"]}

    monkeypatch.setattr("experiments.runner.run_experiments", fake_run_experiments)
    monkeypatch.setattr("experiments.cli.load_config", lambda path: ExperimentConfig())
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", "unused.json", "--model-type", "xgboost",
              "--predict-data", "test.csv", "--task", "task.json",
              "--max-length", "5"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "failed"
    assert "run-1: endpoint failure" in captured.err


def test_validation_error_is_reported(monkeypatch, capsys):
    """Print validation errors and return a nonzero status."""
    def fake_run_experiments(config, request):
        raise ValueError("Reference data is required for local models")

    monkeypatch.setattr("experiments.runner.run_experiments", fake_run_experiments)
    monkeypatch.setattr("experiments.cli.load_config", lambda path: ExperimentConfig())
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", "unused.json", "--model-type", "xgboost",
              "--predict-data", "test.csv", "--task", "task.json"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "experiments: Reference data is required for local models" in captured.err


def test_plot_dispatches_directories(monkeypatch, capsys, tmp_path):
    """Pass the results and output directories to the report writer."""
    captured: dict = {}

    def fake_write_reports(results_dir, output_dir):
        captured["results_dir"] = results_dir
        captured["output_dir"] = output_dir
        return {"runs": 2, "groups": [], "figures": [], "failed_runs": 0}

    monkeypatch.setattr(
        "experiments.analysis.reporting.write_reports", fake_write_reports)
    main(["plot", "--results-dir", str(tmp_path / "results"),
          "--output-dir", str(tmp_path / "reports")])
    assert captured["results_dir"] == tmp_path / "results"
    assert captured["output_dir"] == tmp_path / "reports"
    assert json.loads(capsys.readouterr().out)["runs"] == 2


def test_plot_uses_default_directories(monkeypatch, capsys):
    """Use the default results and report folders for plot."""
    captured: dict = {}

    def fake_write_reports(results_dir, output_dir):
        captured["results_dir"] = results_dir
        captured["output_dir"] = output_dir
        return {"runs": 0}

    monkeypatch.setattr(
        "experiments.analysis.reporting.write_reports", fake_write_reports)
    main(["plot"])
    assert captured["results_dir"] == Path("results")
    assert captured["output_dir"] == Path("reports")
    assert json.loads(capsys.readouterr().out) == {"runs": 0}


def test_plot_failure_exits(monkeypatch, capsys):
    """Print report failures and return a nonzero status."""
    def fake_write_reports(results_dir, output_dir):
        raise ValueError("No experiment records found in results")

    monkeypatch.setattr(
        "experiments.analysis.reporting.write_reports", fake_write_reports)
    with pytest.raises(SystemExit) as error:
        main(["plot", "--results-dir", "missing"])
    assert error.value.code == 1
    assert "experiments: No experiment records found" in capsys.readouterr().err
