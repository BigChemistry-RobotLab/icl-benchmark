"""Check atomic run validation, saved records, checkpoints, and costs."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.config import (
    ExperimentConfig,
    ModelConfig,
    ReasoningEffort,
    load_config,
)
from experiments.endpoints import chat as chat_endpoint
from experiments.endpoints import decisions as decisions_endpoint
from experiments.runner import (
    RunRequest,
    run_experiments,
    validate_run,
)
from experiments.runner import endpoint as endpoint_runner
from experiments.runner import local as local_runner
from experiments.runner import validation as validation_runner
from experiments.runner.records import request_cost

LLM_MODEL = "provider/chat-model"
DECISION_MODEL = "provider/decision-model"
OUTPUT = "outcome"
VARIANT_FILES = {
    "full": "train.csv",
    "balanced": "train_balanced.csv",
    "corrupted_full": "train_corrupted.csv",
    "corrupted_balanced": "train_balanced_corrupted.csv",
}


@pytest.fixture(autouse=True)
def isolated_endpoint(monkeypatch):
    """Keep .env values and endpoint credentials out of each test."""
    monkeypatch.setattr("experiments.config.load_dotenv", lambda *args, **kwargs: None)
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    """Write one small CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(
    tmp_path: Path,
    *,
    variants: tuple[str, ...] = ("full",),
    train_labels: tuple[str, ...] = ("0", "1", "0", "1"),
    test_labels: tuple[str, ...] = ("0", "1"),
    context_input: bool = False,
    task_changes: dict | None = None,
    root_name: str = "data",
) -> dict[str, Path]:
    """Write a synthetic dataset with a task, test table, and flat variant tables."""
    root = tmp_path / root_name / "processed" / "peptide"
    root.mkdir(parents=True)
    inputs: list[dict[str, str]] = [{"column": "sequence"}]
    if context_input:
        inputs.append({"column": "batch", "type": "context"})
    task = {
        "inputs": inputs,
        "outputs": [{"column": OUTPUT}],
        "additional_context": "Predict the outcome.",
        "decision_question": "Is the sequence {sequence} class 1?",
        "allowed_labels": {OUTPUT: ["0", "1"]},
    }
    if task_changes:
        task.update(task_changes)
    (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
    input_columns = [item["column"] for item in inputs if item.get("type") != "context"]
    write_csv(
        root / "test.csv",
        [*input_columns, OUTPUT],
        [[f"T{index}", label] for index, label in enumerate(test_labels)],
    )
    paths = {"root": root, "task": root / "task.json", "test": root / "test.csv"}
    reference_columns = [item["column"] for item in inputs]
    for variant in variants:
        path = root / VARIANT_FILES[variant]
        write_csv(
            path,
            [*reference_columns, OUTPUT],
            [[f"R{index}", *(["b"] if context_input else []), label]
             for index, label in enumerate(train_labels)],
        )
        paths[variant] = path
    return paths


def llm_config(
    tmp_path: Path,
    *,
    replicates: int | None = None,
    context_window: int | None = 100000,
    max_output_tokens: int | None = 1000,
    temperature: float | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    send_seed: bool = False,
    input_usd_per_million: float | None = None,
    output_usd_per_million: float | None = None,
) -> ExperimentConfig:
    """Return a config with one registered chat model."""
    settings = ModelConfig(
        type="llm",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        send_seed=send_seed,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    if replicates is None:
        return ExperimentConfig(
            results_dir=tmp_path / "results", models={LLM_MODEL: settings})
    return ExperimentConfig(
        results_dir=tmp_path / "results",
        llm_replicates=replicates,
        models={LLM_MODEL: settings},
    )


def decision_config(
    tmp_path: Path,
    *,
    replicates: int | None = None,
    context_window: int | None = 100000,
    max_questions: int = 200,
) -> ExperimentConfig:
    """Return a config with one registered decision model."""
    settings = ModelConfig(
        type="decision", context_window=context_window, max_questions=max_questions)
    if replicates is None:
        return ExperimentConfig(
            results_dir=tmp_path / "results", models={DECISION_MODEL: settings})
    return ExperimentConfig(
        results_dir=tmp_path / "results",
        llm_replicates=replicates,
        models={DECISION_MODEL: settings},
    )


def local_config(
    tmp_path: Path,
    *,
    baseline_replicates: int = 100,
    device: str = "cpu",
    baseline_options: dict[str, dict] | None = None,
) -> ExperimentConfig:
    """Return a config for local classifiers."""
    return ExperimentConfig(
        results_dir=tmp_path / "results",
        baseline_replicates=baseline_replicates,
        device=device,
        baseline_options=baseline_options or {},
    )


def run_request(paths: dict[str, Path], **overrides) -> RunRequest:
    """Return one chat run request for a synthetic dataset."""
    values: dict = {
        "model_type": "llm",
        "model": LLM_MODEL,
        "reference_data": paths.get("full"),
        "predict_data": paths["test"],
        "task": paths["task"],
        "mode": "all_at_once",
    }
    values.update(overrides)
    return RunRequest(**values)


def local_request(paths: dict[str, Path], **overrides) -> RunRequest:
    """Return one local classifier request for a synthetic dataset."""
    values: dict = {
        "model_type": "xgboost",
        "model": None,
        "reference_data": paths.get("full"),
        "predict_data": paths["test"],
        "task": paths["task"],
        "max_length": 5,
    }
    values.update(overrides)
    return RunRequest(**values)


def read_records(results_dir: Path) -> list[dict]:
    """Return every saved run record sorted by replicate."""
    paths = sorted((results_dir / "runs").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(records, key=lambda record: record["replicate"])


def local_predictions(test_rows: list[dict[str, str]], value: str = "0") -> dict:
    """Return one complete local prediction result."""
    return {"predictions": [value] * len(test_rows), "metadata": {"model": "xgboost"}}


def endpoint_rows(group: list[dict[str, str]], value: str = "0") -> list[dict[str, str]]:
    """Return one predicted output object per target row."""
    return [{OUTPUT: value} for _ in group]


def forbid_calls(*args, **kwargs):
    """Fail a test that calls a model."""
    pytest.fail("No model call may occur before validation passes")


# Local model runs


def test_local_run_writes_default_records_and_reuses(tmp_path, monkeypatch):
    """Write 100 default local records, strip audit cells, and reuse completed work."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    calls: list[dict] = []

    def predict(model, spec, train_rows, test_rows, seed, device="cpu", options=None):
        calls.append({"model": model, "spec": spec, "train": train_rows,
                      "test": test_rows, "seed": seed, "device": device,
                      "options": options})
        return local_predictions(test_rows)

    monkeypatch.setattr(local_runner, "predict_baseline", predict)
    request = local_request(paths)
    summary = run_experiments(config, request)
    assert summary["status"] == "completed"
    assert summary["runs"] == 100
    assert summary["completed"] == 100
    assert summary["failed"] == 0
    assert summary["dataset"] == "peptide"
    assert summary["variant"] == "full"
    assert summary["model"] == "xgboost"
    assert summary["mode"] == "local"
    assert summary["results_dir"] == str(tmp_path / "results")
    assert summary["errors"] == []

    records = read_records(tmp_path / "results")
    assert [record["replicate"] for record in records] == list(range(100))
    assert [record["seed"] for record in records] == list(range(42, 142))
    record = records[0]
    assert record["schema_version"] == 2
    assert record["expected_replicates"] == 100
    assert record["model_type"] == "xgboost"
    assert record["model"] == "xgboost"
    assert record["mode"] == "local"
    assert record["dataset"] == "peptide"
    assert record["variant"] == "full"
    assert record["max_length"] == 5
    assert record["status"] == "completed"
    assert record["labels"] == ["0", "1"]
    assert record["target_inputs"] == [{"sequence": "T0"}, {"sequence": "T1"}]
    assert record["predictions"] == ["0", "0"]
    assert record["reference_data"] == str(paths["full"])
    assert record["predict_data"] == str(paths["test"])
    assert record["task"] == str(paths["task"])
    assert record["protocol_id"] and record["split_id"]
    assert not {"phase", "condition", "control"} & set(record)
    assert record["metrics"]["macro_f1"] is not None
    assert len(calls) == 100
    assert all(set(call["train"][0]) == {"sequence", OUTPUT} for call in calls)
    assert all(set(call["test"][0]) == {"sequence"} for call in calls)
    assert calls[0]["seed"] == 42
    assert calls[-1]["seed"] == 141
    assert calls[0]["device"] == "cpu"

    first_record = records[0]
    calls.clear()
    assert run_experiments(config, request) == summary
    assert calls == []
    assert read_records(tmp_path / "results")[0] == first_record


def test_replicate_override_controls_record_count(tmp_path, monkeypatch):
    """Run only the requested number of local replicates."""
    paths = write_dataset(tmp_path, train_labels=("0", "1"))
    config = local_config(tmp_path)
    monkeypatch.setattr(
        local_runner, "predict_baseline",
        lambda model, spec, train, test, seed, device="cpu", options=None:
        local_predictions(test),
    )
    summary = run_experiments(config, local_request(paths, replicates=2))
    assert summary["runs"] == 2
    assert summary["completed"] == 2
    records = read_records(tmp_path / "results")
    assert [record["replicate"] for record in records] == [0, 1]
    assert all(record["expected_replicates"] == 2 for record in records)


@pytest.mark.parametrize(
    "variant", ["full", "balanced", "corrupted_full", "corrupted_balanced"])
def test_reference_filename_sets_the_variant(tmp_path, monkeypatch, variant):
    """Derive the variant label from the reference file name."""
    paths = write_dataset(tmp_path, train_labels=("0", "1"), variants=(variant,))
    config = local_config(tmp_path, baseline_replicates=1)
    monkeypatch.setattr(
        local_runner, "predict_baseline",
        lambda model, spec, train, test, seed, device="cpu", options=None:
        local_predictions(test),
    )
    summary = run_experiments(
        config, local_request(paths, reference_data=paths[variant]))
    assert summary["variant"] == variant
    record = read_records(tmp_path / "results")[0]
    assert record["variant"] == variant
    assert record["reference_data"] == str(paths[variant])


def test_reference_at_dataset_root_uses_default_variant(tmp_path, monkeypatch):
    """Use the default variant when the reference file name is unknown."""
    paths = write_dataset(tmp_path, variants=(), test_labels=("0",))
    write_csv(paths["root"] / "reference.csv", ["sequence", OUTPUT],
              [["R0", "0"], ["R1", "1"]])
    config = local_config(tmp_path, baseline_replicates=1)
    monkeypatch.setattr(
        local_runner, "predict_baseline",
        lambda model, spec, train, test, seed, device="cpu", options=None:
        local_predictions(test),
    )
    summary = run_experiments(
        config, local_request(paths, reference_data=paths["root"] / "reference.csv"))
    assert summary["variant"] == "default"


def test_unrelated_reference_parent_uses_default_variant(tmp_path, monkeypatch):
    """Fall back to the prediction folder when the inputs do not share a parent."""
    paths = write_dataset(tmp_path)
    other = write_dataset(tmp_path, root_name="other", train_labels=("0", "1"))
    config = local_config(tmp_path, baseline_replicates=1)
    monkeypatch.setattr(
        local_runner, "predict_baseline",
        lambda model, spec, train, test, seed, device="cpu", options=None:
        local_predictions(test),
    )
    summary = run_experiments(
        config, local_request(paths, reference_data=other["full"]))
    assert summary["dataset"] == "peptide"
    assert summary["variant"] == "default"


def test_failed_local_records_require_explicit_retry(tmp_path, monkeypatch):
    """Reuse a failed local record until an explicit retry is requested."""
    paths = write_dataset(tmp_path, train_labels=("0", "1"), test_labels=("0",))
    config = local_config(tmp_path, baseline_replicates=1)
    calls: list[int] = []

    def predict(model, spec, train, test, seed, device="cpu", options=None):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("local failure")
        return local_predictions(test)

    monkeypatch.setattr(local_runner, "predict_baseline", predict)
    request = local_request(paths)
    assert run_experiments(config, request)["status"] == "failed"
    record = read_records(tmp_path / "results")[0]
    assert "local failure" in record["error"]
    assert run_experiments(config, request)["status"] == "failed"
    assert len(calls) == 1
    assert run_experiments(config, replace(request, retry_failed=True))["status"] \
        == "completed"
    assert len(calls) == 2


def test_local_profile_records_resource_metadata(tmp_path, monkeypatch):
    """Store profile measurements for a local run when requested."""
    paths = write_dataset(tmp_path, train_labels=("0", "1"), test_labels=("0",))
    config = local_config(tmp_path, baseline_replicates=1)
    seen: dict = {}

    def profile(model, spec, train_rows, test_rows, seed=42, device="cpu", options=None):
        seen["seed"] = seed
        return {"predictions": ["0"], "metadata": {"model": model},
                "profile": {"wall_seconds": 1.5, "process_peak_rss_bytes": 10,
                            "platform": "test", "cuda_peak_allocated_bytes": None}}

    monkeypatch.setattr(local_runner, "profile_model", profile)
    summary = run_experiments(config, local_request(paths, profile=True))
    assert summary["status"] == "completed"
    record = read_records(tmp_path / "results")[0]
    assert record["metadata"]["profile"]["wall_seconds"] == 1.5
    assert record["metadata"]["profile"]["process_peak_rss_bytes"] == 10
    assert record["latency_seconds"] == 1.5
    assert seen["seed"] == 42


def test_results_dir_override_controls_record_location(tmp_path, monkeypatch):
    """Write records to the request results directory when supplied."""
    paths = write_dataset(tmp_path, train_labels=("0", "1"), test_labels=("0",))
    config = local_config(tmp_path, baseline_replicates=1)
    monkeypatch.setattr(
        local_runner, "predict_baseline",
        lambda model, spec, train, test, seed, device="cpu", options=None:
        local_predictions(test),
    )
    override = tmp_path / "override"
    summary = run_experiments(config, local_request(paths, results_dir=override))
    assert summary["results_dir"] == str(override)
    assert (override / "runs").is_dir()
    assert not (tmp_path / "results" / "runs").exists()


# Local validation


def test_local_requires_reference_data(tmp_path, monkeypatch):
    """Reject a zero-shot local model request."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    monkeypatch.setattr(local_runner, "predict_baseline", forbid_calls)
    with pytest.raises(ValueError, match="Reference data is required for local models"):
        run_experiments(config, local_request(paths, reference_data=None))


def test_local_rejects_endpoint_flags(tmp_path, monkeypatch):
    """Reject mode, endpoint settings, and profile rules for local models."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    monkeypatch.setattr(local_runner, "predict_baseline", forbid_calls)
    with pytest.raises(ValueError, match="mode is only valid for llm and decision"):
        run_experiments(config, local_request(paths, mode="one_by_one"))
    with pytest.raises(ValueError, match="local model is its model type"):
        run_experiments(config, local_request(paths, model="tabpfn"))
    with pytest.raises(ValueError, match="Endpoint request settings are not used"):
        run_experiments(config, local_request(paths, temperature=0.5))
    with pytest.raises(ValueError, match="Endpoint request settings are not used"):
        run_experiments(config, local_request(paths, context_window=1000))


def test_local_rejects_context_inputs(tmp_path):
    """Reject context input columns that local encoders cannot use."""
    paths = write_dataset(tmp_path, context_input=True)
    config = local_config(tmp_path)
    with pytest.raises(ValueError, match="context input columns"):
        validate_run(config, local_request(paths))


def test_local_max_length_resolution(tmp_path):
    """Use the explicit maximum length and reject invalid values."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    plan = validate_run(config, local_request(paths, max_length=5))
    assert plan.max_length == 5
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        validate_run(config, local_request(paths, max_length=0))
    with pytest.raises(ValueError, match="max_length is only used by local models"):
        validate_run(llm_config(tmp_path), run_request(paths, max_length=5))


def test_local_requires_max_length(tmp_path):
    """Require --max-length for local models."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    with pytest.raises(ValueError, match="max_length is required for local models"):
        validate_run(config, local_request(paths, max_length=None))
    assert validate_run(config, local_request(paths, max_length=4)).max_length == 4


def test_local_reference_requires_every_declared_class(tmp_path):
    """Reject a reference table that is missing a declared class."""
    paths = write_dataset(tmp_path, train_labels=("0", "0"))
    with pytest.raises(ValueError, match="missing declared classes"):
        validate_run(local_config(tmp_path), local_request(paths))


def test_local_dependency_and_checkpoint_checks(tmp_path, monkeypatch):
    """Check optional packages, checkpoint options, and the CUDA requirement."""
    paths = write_dataset(tmp_path)
    monkeypatch.setattr(validation_runner, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="Missing optional dependency: xgboost"):
        validate_run(local_config(tmp_path), local_request(paths))

    monkeypatch.setattr(validation_runner, "find_spec", lambda name: object())
    with pytest.raises(ValueError, match="set checkpoint_path or allow_download"):
        validate_run(local_config(tmp_path), local_request(paths, model_type="tabpfn"))
    with pytest.raises(ValueError, match="checkpoint path is unavailable"):
        validate_run(
            local_config(tmp_path, baseline_options={
                "tabpfn": {"checkpoint_path": str(tmp_path / "absent")}}),
            local_request(paths, model_type="tabpfn"),
        )
    with pytest.raises(ValueError, match="Unknown tabpfn baseline options"):
        validate_run(
            local_config(tmp_path, baseline_options={"tabpfn": {"other": True}}),
            local_request(paths, model_type="tabpfn"),
        )
    plan = validate_run(
        local_config(tmp_path, baseline_options={"tabpfn": {"allow_download": True}}),
        local_request(paths, model_type="tabpfn"),
    )
    assert plan.options == {"allow_download": True}

    monkeypatch.setattr(
        validation_runner, "find_spec",
        lambda name: None if name == "torch" else object())
    with pytest.raises(ValueError, match="CUDA requires torch"):
        validate_run(
            local_config(
                tmp_path, device="cuda:0",
                baseline_options={"tabpfn": {"allow_download": True}},
            ),
            local_request(paths, model_type="tabpfn"),
        )
    # XGBoost uses its own CUDA support and must not require torch.
    assert validate_run(
        local_config(tmp_path, device="cuda:0"), local_request(paths)).device == "cuda:0"


def test_missing_input_paths_are_reported(tmp_path):
    """Report each missing input file before loading data."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path)
    with pytest.raises(ValueError, match="Reference data is missing"):
        validate_run(config, local_request(paths, reference_data=tmp_path / "absent.csv"))
    with pytest.raises(ValueError, match="Prediction data is missing"):
        validate_run(config, local_request(paths, predict_data=tmp_path / "absent.csv"))
    with pytest.raises(ValueError, match="Task file is missing"):
        validate_run(config, local_request(paths, task=tmp_path / "absent.json"))


def test_invalid_labels_fail_before_running(tmp_path, monkeypatch):
    """Reject unknown prediction or reference labels before a model call."""
    bad_predict = write_dataset(tmp_path, root_name="bad-predict",
                                test_labels=("0", "2"))
    bad_reference = write_dataset(tmp_path, root_name="bad-reference",
                                  train_labels=("2", "1"))
    config = local_config(tmp_path)
    monkeypatch.setattr(local_runner, "predict_baseline", forbid_calls)
    with pytest.raises(ValueError, match="invalid label"):
        validate_run(config, local_request(bad_predict))
    with pytest.raises(ValueError, match="invalid label"):
        validate_run(config, local_request(bad_reference))


def test_empty_prediction_data_is_rejected(tmp_path):
    """Reject a prediction table with no rows."""
    paths = write_dataset(tmp_path, test_labels=())
    with pytest.raises(ValueError, match="at least one row"):
        validate_run(local_config(tmp_path), local_request(paths))


# Endpoint runs


def test_llm_zero_shot_writes_three_default_records(tmp_path, monkeypatch):
    """Run three zero-shot replicates by default and record the shared variant."""
    paths = write_dataset(tmp_path, variants=(), test_labels=("0", "1"))
    config = llm_config(tmp_path, input_usd_per_million=1.0, output_usd_per_million=2.0)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="reason",
                               metadata={"usage": {"input_tokens": 2, "output_tokens": 1}})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    summary = run_experiments(config, run_request(paths, reference_data=None))
    assert summary["status"] == "completed"
    assert summary["runs"] == 3
    assert summary["variant"] == "zero_shot"
    assert summary["mode"] == "all_at_once"
    assert len(calls) == 3
    assert all(call["reference_data"] == [] for call in calls)
    assert all(set(row) == {"sequence"} for call in calls for row in call["predict_data"])
    records = read_records(tmp_path / "results")
    assert len(records) == 3
    record = records[0]
    assert record["reference_data"] is None
    assert record["variant"] == "zero_shot"
    assert record["expected_replicates"] == 3
    assert record["model_type"] == "llm"
    assert record["mode"] == "all_at_once"
    assert record["cost"] == pytest.approx(4 / 1_000_000)
    assert calls[0]["model"] == LLM_MODEL
    assert calls[0]["settings"].max_output_tokens == 1000


def test_llm_one_by_one_sends_one_request_per_target(tmp_path, monkeypatch):
    """Group one target per request in one-by-one mode."""
    paths = write_dataset(tmp_path, test_labels=("0", "1"))
    config = llm_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    summary = run_experiments(config, run_request(paths, mode="one_by_one"))
    assert summary["status"] == "completed"
    assert len(calls) == 2
    assert [call["predict_data"] for call in calls] == [
        [{"sequence": "T0"}], [{"sequence": "T1"}]]
    record = read_records(tmp_path / "results")[0]
    assert len(record["requests"]) == 2


def test_llm_one_by_one_requests_run_concurrently(tmp_path, monkeypatch):
    """Keep more than one one-by-one request in flight."""
    paths = write_dataset(tmp_path, test_labels=("0", "1", "0"))
    config = llm_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    active = 0
    peak = 0

    async def fake_run_agent(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.25)
        active -= 1
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    summary = run_experiments(config, run_request(paths, mode="one_by_one"))
    assert summary["status"] == "completed"
    assert peak > 1


def test_llm_one_by_one_request_starts_are_rate_limited(tmp_path, monkeypatch):
    """Space one-by-one request starts by the configured interval."""
    paths = write_dataset(tmp_path, test_labels=("0", "1"))
    config = llm_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    starts: list[float] = []

    async def fake_run_agent(**kwargs):
        starts.append(asyncio.get_running_loop().time())
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    summary = run_experiments(config, run_request(paths, mode="one_by_one"))
    assert summary["status"] == "completed"
    assert len(starts) == 2
    interval = 1.0 / endpoint_runner.REQUESTS_PER_SECOND
    assert starts[1] - starts[0] >= interval * 0.8


def test_llm_reference_keeps_context_and_targets_omit_it(tmp_path, monkeypatch):
    """Send context columns only in the reference table."""
    paths = write_dataset(tmp_path, context_input=True, test_labels=("0",))
    config = llm_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    summary = run_experiments(config, run_request(paths))
    assert summary["status"] == "completed"
    assert calls[0]["reference_data"] == [{"sequence": "R0", "batch": "b", OUTPUT: "0"},
                                          {"sequence": "R1", "batch": "b", OUTPUT: "1"},
                                          {"sequence": "R2", "batch": "b", OUTPUT: "0"},
                                          {"sequence": "R3", "batch": "b", OUTPUT: "1"}]
    assert calls[0]["predict_data"] == [{"sequence": "T0"}]


def test_unset_output_budget_and_reasoning_are_forwarded(tmp_path, monkeypatch):
    """Forward an unset output limit, the reasoning level, and the request seed."""
    paths = write_dataset(tmp_path, test_labels=("0",))
    config = llm_config(tmp_path, replicates=1, max_output_tokens=None,
                        reasoning_effort="low", send_seed=True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    assert run_experiments(config, run_request(paths))["status"] == "completed"
    settings = calls[0]["settings"]
    assert settings.max_output_tokens is None
    assert settings.reasoning_effort == "low"
    assert settings.seed == 42


@pytest.mark.parametrize(("overrides", "message"), [
    ({"mode": None}, "mode is required for llm and decision models"),
    ({"max_length": 5}, "max_length is only used by local models"),
    ({"model": "unknown/model"}, "Model is not registered in the config"),
    ({"model": "  "}, "model is required for llm and decision models"),
    ({"model_type": "svm"}, "Unknown model type"),
    ({"mode": "mixed"}, "Unknown inference mode"),
    ({"model_type": "decision"}, "Registered model type does not match"),
])
def test_endpoint_request_rules(tmp_path, overrides, message):
    """Reject invalid model, mode, and length combinations."""
    paths = write_dataset(tmp_path)
    request = replace(run_request(paths), **overrides)
    with pytest.raises(ValueError, match=message):
        validate_run(llm_config(tmp_path), request)


def test_validation_failure_makes_no_endpoint_call(tmp_path, monkeypatch):
    """Reject invalid requests before any paid request is sent."""
    paths = write_dataset(tmp_path)
    config = llm_config(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def forbidden(**kwargs):
        pytest.fail("No request may be sent before validation passes")

    monkeypatch.setattr(chat_endpoint, "run_agent", forbidden)
    with pytest.raises(ValueError, match="max_length is only used by local models"):
        run_experiments(config, run_request(paths, max_length=5))
    bad = write_dataset(tmp_path, root_name="bad", test_labels=("0", "2"))
    with pytest.raises(ValueError, match="invalid label"):
        run_experiments(config, run_request(bad))


def test_endpoint_settings_and_credentials_fail_before_requests(tmp_path, monkeypatch):
    """Check capacity, price limits, keys, and decision settings before requests."""
    paths = write_dataset(tmp_path)
    request = run_request(paths)

    async def forbidden(**kwargs):
        pytest.fail("No request may be sent before validation passes")

    monkeypatch.setattr(chat_endpoint, "run_agent", forbidden)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        validate_run(llm_config(tmp_path), request)
    with pytest.raises(ValueError, match="context_window must be explicit"):
        validate_run(
            ExperimentConfig(results_dir=tmp_path / "results",
                             models={LLM_MODEL: ModelConfig(type="llm")}),
            request,
        )
    with pytest.raises(ValueError, match="output limit must be smaller than context"):
        validate_run(llm_config(tmp_path, context_window=100, max_output_tokens=100),
                     request)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="LLM admission failed"):
        validate_run(llm_config(tmp_path, context_window=1100), request)
    with pytest.raises(ValueError, match="LLM admission failed"):
        validate_run(llm_config(tmp_path, max_output_tokens=1), request)

    decision_request = replace(request, model=DECISION_MODEL, model_type="decision")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY or OPENAI_API_KEY"):
        validate_run(decision_config(tmp_path), decision_request)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Invalid model settings"):
        validate_run(decision_config(tmp_path),
                     replace(decision_request, temperature=0.2))
    with pytest.raises(ValueError, match="question limit"):
        validate_run(decision_config(tmp_path),
                     replace(decision_request, max_questions=1))


def test_env_file_supplies_endpoint_credentials(tmp_path, monkeypatch):
    """Load the repository .env before the credential check."""
    paths = write_dataset(tmp_path)
    calls = []

    def fake_load_dotenv(path, override=False):
        calls.append(Path(path))
        monkeypatch.setenv("OPENAI_API_KEY", "from-dotenv")

    monkeypatch.setattr("experiments.config.load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(validation_runner, "check_admission", lambda *args, **kwargs: None)
    plan = validate_run(llm_config(tmp_path), run_request(paths))
    assert plan.protocol_id
    assert calls


def test_request_checkpoint_resume_retry_and_cached_cost(tmp_path, monkeypatch):
    """Resume completed requests, retry failures, and refresh measured cost."""
    paths = write_dataset(tmp_path, test_labels=("0", "1"))
    config = llm_config(
        tmp_path, replicates=1, input_usd_per_million=1.0,
        output_usd_per_million=2.0,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("endpoint failure")
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="",
                               metadata={"usage": {"input_tokens": 2, "output_tokens": 1}})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    request = run_request(paths, mode="one_by_one")

    failed = run_experiments(config, request)
    assert failed["status"] == "failed"
    assert failed["failed"] == 1
    assert len(failed["errors"]) == 1
    assert len(calls) == 2
    record = read_records(tmp_path / "results")[0]
    assert record["status"] == "failed"
    assert "endpoint failure" in record["error"]
    assert len(record["requests"]) == 2
    assert record["checkpoint_path"]
    checkpoints = sorted((tmp_path / "results" / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 2
    statuses = sorted(json.loads(path.read_text())["status"] for path in checkpoints)
    assert statuses == ["completed", "failed"]

    assert run_experiments(config, request)["status"] == "failed"
    assert len(calls) == 2

    retried = run_experiments(config, replace(request, retry_failed=True))
    assert retried["status"] == "completed"
    assert len(calls) == 3
    record = read_records(tmp_path / "results")[0]
    assert record["status"] == "completed"
    assert len(record["requests"]) == 2
    assert record["cost"] == pytest.approx((4 * 1.0 + 2 * 2.0) / 1_000_000)

    for path in checkpoints:
        checkpoint = json.loads(path.read_text())
        checkpoint["metadata"]["responses"] = [{"raw_usage": {"cost": 0.25}}]
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
    refreshed = run_experiments(config, request)
    assert refreshed["status"] == "completed"
    assert len(calls) == 3
    record = read_records(tmp_path / "results")[0]
    assert record["cost"] == pytest.approx(0.5)
    persisted = json.loads(
        (tmp_path / "results" / "runs" / f"{record['run_id']}.json").read_text())
    assert persisted["cost"] == pytest.approx(0.5)


def test_output_schema_changes_request_id_not_run_id(tmp_path, monkeypatch):
    """Keep the run identity and change only the request identity."""
    paths = write_dataset(tmp_path, test_labels=("0",))
    config = llm_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("invalid response")
        return SimpleNamespace(predictions=endpoint_rows(kwargs["predict_data"]),
                               reasoning="", metadata={})

    monkeypatch.setattr(chat_endpoint, "run_agent", fake_run_agent)
    request = run_request(paths, mode="one_by_one")
    assert run_experiments(config, request)["status"] == "failed"
    failed = read_records(tmp_path / "results")[0]

    original = chat_endpoint.output_schema

    def revised_schema(task, target_count):
        schema = original(task, target_count)
        schema.model_config["json_schema_extra"] = {
            "description": "Exact target coverage"}
        return schema

    monkeypatch.setattr(chat_endpoint, "output_schema", revised_schema)
    completed = run_experiments(config, replace(request, retry_failed=True))
    assert completed["status"] == "completed"
    record = read_records(tmp_path / "results")[0]
    assert record["run_id"] == failed["run_id"]
    assert record["requests"][0] != failed["requests"][0]
    assert len(calls) == 2
    checkpoint = json.loads(
        (tmp_path / "results" / "checkpoints" / f"{record['requests'][0]}.json").read_text())
    assert checkpoint["payload"]["output_schema"]["description"] == "Exact target coverage"

    monkeypatch.setattr(chat_endpoint, "output_schema", original)
    assert run_experiments(config, request)["status"] == "completed"
    assert len(calls) == 2
    assert read_records(tmp_path / "results")[0]["run_id"] == record["run_id"]


def test_decision_run_records_provider_cost_and_reuses(tmp_path, monkeypatch):
    """Use decision requests, provider cost, and cached records."""
    paths = write_dataset(tmp_path)
    config = decision_config(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_decision(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            predictions=[{**row, OUTPUT: "0"} for row in kwargs["predict_data"]],
            reasoning="",
            metadata={"usage": {"input_tokens": 4, "cost": 0.125},
                      "rationale_available": False},
        )

    monkeypatch.setattr(decisions_endpoint, "send_request", fake_decision)
    request = replace(run_request(paths), model=DECISION_MODEL, model_type="decision")
    summary = run_experiments(config, request)
    assert summary["status"] == "completed"
    assert summary["runs"] == 3
    assert len(calls) == 3
    assert all(call["settings"].type == "decision" for call in calls)
    assert all(call["model"] == DECISION_MODEL for call in calls)
    record = read_records(tmp_path / "results")[0]
    assert record["model_type"] == "decision"
    assert record["model"] == DECISION_MODEL
    assert record["cost"] == 0.125
    assert record["mode"] == "all_at_once"
    checkpoint = json.loads(
        next((tmp_path / "results" / "checkpoints").glob("*.json")).read_text())
    request_body = checkpoint["payload"]["request"]
    assert set(request_body) == {"model", "state", "questions"}
    assert "temperature" not in request_body
    assert next(iter(request_body["questions"].values()))["type"] == "choice"
    assert run_experiments(config, request)["status"] == "completed"
    assert len(calls) == 3


def test_decision_zero_shot_omits_reference(tmp_path, monkeypatch):
    """Run decision models without reference rows for zero-shot."""
    paths = write_dataset(tmp_path, variants=(), test_labels=("0",))
    config = decision_config(tmp_path, replicates=1)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[dict] = []

    async def fake_decision(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            predictions=[{**row, OUTPUT: "0"} for row in kwargs["predict_data"]],
            reasoning="", metadata={"rationale_available": False})

    monkeypatch.setattr(decisions_endpoint, "send_request", fake_decision)
    request = replace(run_request(paths, reference_data=None),
                      model=DECISION_MODEL, model_type="decision", mode="one_by_one")
    summary = run_experiments(config, request)
    assert summary["status"] == "completed"
    assert summary["variant"] == "zero_shot"
    assert calls[0]["reference_data"] == []
    record = read_records(tmp_path / "results")[0]
    assert record["reference_data"] is None
    assert record["variant"] == "zero_shot"


# Protocol identity


def test_protocol_identity_tracks_content_settings_and_endpoint(tmp_path, monkeypatch):
    """Invalidate identities when content, settings, or the endpoint change."""
    paths = write_dataset(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = llm_config(tmp_path)
    request = run_request(paths)
    first = validate_run(config, request)
    assert validate_run(config, request).protocol_id == first.protocol_id
    assert validate_run(config, request).split_id == first.split_id

    changed = write_dataset(tmp_path, root_name="changed", test_labels=("1", "1"))
    other = validate_run(config, replace(
        request, reference_data=changed["full"], predict_data=changed["test"]))
    assert other.split_id != first.split_id
    assert other.protocol_id != first.protocol_id

    task_changed = write_dataset(tmp_path, root_name="task",
                                 task_changes={"additional_context": "Changed."})
    assert validate_run(config, replace(request, task=task_changed["task"])).protocol_id \
        != first.protocol_id
    assert validate_run(config, replace(request, temperature=0.5)).protocol_id \
        != first.protocol_id

    monkeypatch.setenv("OPENAI_API_BASE", "https://user:first@one.example/v1?key=first#frag")
    one = validate_run(config, request).protocol_id
    monkeypatch.setenv("OPENAI_API_BASE",
                       "https://other:second@one.example/v1?key=second")
    assert validate_run(config, request).protocol_id == one
    monkeypatch.setenv("OPENAI_API_BASE", "https://two.example/v1")
    assert validate_run(config, request).protocol_id != one


def test_protocol_identity_tracks_local_length(tmp_path):
    """Invalidate identities when the local length changes and ignore replicates."""
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path, baseline_replicates=2)
    request = local_request(paths, max_length=4)
    first = validate_run(config, request)
    assert validate_run(config, replace(request, max_length=5)).protocol_id \
        != first.protocol_id
    assert validate_run(config, replace(request, replicates=3)).protocol_id \
        == first.protocol_id


def test_replicate_count_change_reuses_completed_records(tmp_path, monkeypatch):
    """Reuse completed records when only the replicate count changes.

    Args:
        tmp_path: Temporary folder for the dataset and result records.
        monkeypatch: Pytest fixture that replaces the local model call.
    """
    paths = write_dataset(tmp_path)
    config = local_config(tmp_path, baseline_replicates=4)
    seeds: list[int] = []

    def predict(model, spec, train_rows, test_rows, seed, device="cpu", options=None):
        seeds.append(seed)
        return local_predictions(test_rows)

    monkeypatch.setattr(local_runner, "predict_baseline", predict)
    summary = run_experiments(config, local_request(paths, replicates=2))
    assert summary["completed"] == 2
    assert len(seeds) == 2

    seeds.clear()
    summary = run_experiments(config, local_request(paths, replicates=4))
    assert summary["runs"] == 4
    assert summary["completed"] == 4
    assert seeds == [44, 45]

    records = read_records(tmp_path / "results")
    assert [record["replicate"] for record in records] == [0, 1, 2, 3]
    assert len({record["protocol_id"] for record in records}) == 1
    assert [record["expected_replicates"] for record in records] == [2, 2, 4, 4]


def test_protocol_identity_tracks_decision_contract(tmp_path, monkeypatch):
    """Invalidate decision identities when the request contract changes."""
    paths = write_dataset(tmp_path)
    config = decision_config(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    request = replace(run_request(paths), model=DECISION_MODEL, model_type="decision")
    before = validate_run(config, request).protocol_id
    original = decisions_endpoint.build_payload

    def changed(*args, **kwargs):
        payload = original(*args, **kwargs)
        next(iter(payload["questions"].values()))["instructions"] += " changed"
        return payload

    monkeypatch.setattr(decisions_endpoint, "build_payload", changed)
    assert validate_run(config, request).protocol_id != before


def test_validate_run_returns_plan_with_identity(tmp_path, monkeypatch):
    """Expose the resolved rows and identity of a validated request."""
    paths = write_dataset(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = llm_config(tmp_path, replicates=2)
    plan = validate_run(config, run_request(paths, mode="one_by_one"))
    assert plan.model_type == "llm"
    assert plan.model == LLM_MODEL
    assert plan.mode == "one_by_one"
    assert plan.settings is not None and plan.settings.type == "llm"
    assert plan.dataset == "peptide"
    assert plan.variant == "full"
    assert plan.replicates == 2
    assert plan.base_seed == 42
    assert plan.results_dir == tmp_path / "results"
    assert plan.max_length is None
    assert plan.labels == ["0", "1"]
    assert plan.targets == [{"sequence": "T0"}, {"sequence": "T1"}]
    assert plan.reference_rows == [{"sequence": "R0", OUTPUT: "0"},
                                   {"sequence": "R1", OUTPUT: "1"},
                                   {"sequence": "R2", OUTPUT: "0"},
                                   {"sequence": "R3", OUTPUT: "1"}]
    assert plan.spec.target == OUTPUT
    assert plan.spec.inputs == ("sequence",)
    assert plan.protocol_id and plan.split_id


# Configuration


@pytest.mark.parametrize("settings", [
    {"type": "decision", "temperature": 0.2},
    {"type": "decision", "max_output_tokens": 10},
    {"type": "decision", "reasoning_effort": "low"},
    {"type": "decision", "send_seed": True},
    {"type": "decision", "max_questions": 0},
    {"type": "decision", "max_questions": 201},
    {"type": "llm", "max_questions": 10},
    {"type": "llm", "context_window": 0},
    {"type": "llm", "max_output_tokens": 0},
    {"type": "llm", "temperature": -1},
    {"type": "llm", "temperature": float("nan")},
    {"type": "llm", "input_usd_per_million": float("inf")},
    {"type": "llm", "reasoning_effort": "unsupported"},
    {"type": "other"},
])
def test_invalid_model_settings(settings):
    """Reject settings that the registered model type does not use."""
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_valid_model_settings():
    """Accept the documented model settings."""
    assert ModelConfig(type="llm").max_questions == 200
    decision = ModelConfig(type="decision", context_window=1000)
    assert decision.max_questions == 200
    assert decision.context_window == 1000


@pytest.mark.parametrize("settings", [
    {"seed": -1},
    {"seed": True},
    {"model": "  "},
    {"llm_replicates": True},
    {"llm_replicates": 0},
    {"baseline_replicates": 0},
    {"models": {" ": ModelConfig(type="llm")}},
    {"baseline_options": {"llm": {}}},
    {"baseline_options": {"xgboost": {"extra": 1}}},
    {"datasets": []},
    {"llms": {}},
])
def test_invalid_experiment_settings(settings):
    """Reject old keys, blank registry names, and invalid counts."""
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_experiment_config_defaults():
    """Expose the approved default replicate counts."""
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.llm_replicates == 3
    assert config.baseline_replicates == 100
    assert config.device == "cpu"
    assert config.results_dir == Path("results")
    assert config.models == {}
    assert config.baseline_options == {}


def test_load_config_reads_the_new_schema(tmp_path):
    """Read registry entries and baseline options from JSON."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "seed": 7,
        "llm_replicates": 2,
        "baseline_replicates": 4,
        "models": {LLM_MODEL: {"type": "llm", "context_window": 5000}},
        "baseline_options": {"tabpfn": {"allow_download": True}},
    }), encoding="utf-8")
    config = load_config(path)
    assert config.seed == 7
    assert config.llm_replicates == 2
    assert config.baseline_replicates == 4
    assert config.models[LLM_MODEL].context_window == 5000
    assert config.baseline_options == {"tabpfn": {"allow_download": True}}


@pytest.mark.parametrize(("metadata", "expected"), [
    ({"cost": 0.4, "usage": {"cost": 0.3}}, 0.4),
    ({"usage": {"cost": 0}}, 0.0),
    ({"responses": [{"raw_usage": {"cost": 0.1}}, {"raw_usage": {"cost": 0.2}}]}, 0.3),
    ({"cost": None, "usage": {"cost": 0.2}}, 0.2),
    ({"responses": [{"raw_usage": {"cost": 0.1}}, {}]}, None),
    ({"responses": []}, None),
    ({"cost": True}, None),
    ({"cost": float("nan")}, None),
])
def test_request_cost_uses_complete_provider_values(metadata, expected):
    """Read direct costs or the complete measured response total."""
    result = request_cost(metadata)
    assert result is None if expected is None else result == pytest.approx(expected)
