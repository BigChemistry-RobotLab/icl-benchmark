"""Check prediction boundaries without API calls."""

import asyncio
import csv
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import AgentOutputSchema, ModelBehaviorError
from pydantic import ValidationError

from icl_agent import InferenceSettings, PredictionError, Task, run_agent
from icl_agent.llm_agent import output_schema
from icl_agent.prompt_builder import reference_prompt, target_prompt


def make_task(outputs=None):
    """Create task settings for a test."""
    return Task.model_validate({
        "inputs": [{"column": "x", "alias": "Input name"}],
        "outputs": [{"column": name} for name in (outputs or ["y"])],
        "additional_context": "Estimate the output.",
    })


@pytest.fixture
def runner(monkeypatch):
    """Replace SDK execution for all prediction tests."""
    # The SDK client needs a key even when the runner is mocked.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock = AsyncMock()
    monkeypatch.setattr("icl_agent.llm_agent.Runner.run", mock)
    return mock


@pytest.mark.parametrize("names", [["y"], ["space name", "_private", "model_dump", "a.b", "温度", "output_0"]])
def test_batch_rows_and_output_aliases(runner, names):
    """Include all references and targets in one run."""
    task = make_task(names)
    references = [
        {"x": '  α,"quoted"\r\nline  ', **dict.fromkeys(names, ""), "unused": "secret"},
        {"x": "second", **dict.fromkeys(names, "  001.0  ")},
        {"x": "second", **dict.fromkeys(names, "  001.0  ")},
    ]
    targets = [{"x": "first target", names[0]: "hidden"}, {"x": "next target"}, {"x": "next target"}]
    expected = {
        "predictions": {"target_0": dict.fromkeys(names, "  63.2\nβ  "),
                        "target_1": dict.fromkeys(names, ""), "target_2": dict.fromkeys(names, "")},
        "reasoning": "  Similar inputs.  ",
    }
    schema = output_schema(task, 3)
    runner.return_value = SimpleNamespace(final_output=schema.model_validate(expected))
    results = asyncio.run(run_agent(
        task=task, reference_data=references, predict_data=targets, model="test-model"
    ))
    assert results.model_dump(exclude={"metadata"}) == {
        "predictions": [{"x": row["x"], **prediction} for row, prediction in zip(targets, expected["predictions"].values(), strict=True)],
        "reasoning": expected["reasoning"],
    }
    assert runner.await_count == 1
    for call in runner.await_args_list:
        agent, prompt = call.args
        assert agent.model == "test-model"
        assert agent.tools == []
        assert call.kwargs["max_turns"] == 2
        assert set(call.kwargs) == {"max_turns", "run_config"}
        assert call.kwargs["run_config"].tracing_disabled
        assert "hidden" not in prompt and "secret" not in prompt
        assert prompt.count("Input name") == 2
        output_labels = '", "'.join(names)
        prefix, target = prompt.split(f'\nPredict "{output_labels}" for the following targets in row order. Use each "target_id" value as its key in the predictions object:\n')
        reference_csv = prefix.split("\n\nReference data:\n")[1]
        assert list(csv.reader(io.StringIO(reference_csv, newline=""))) == [
            ["Input name", *names],
            *[[row["x"], *[row[name] for name in names]] for row in references],
        ]
        assert list(csv.reader(io.StringIO(target))) == [["target_id", "Input name"], *[[f"target_{index}", row["x"]] for index, row in enumerate(targets)]]


def test_prompt_layout_and_output_descriptions():
    """Use display labels in CSV and source names in the output schema."""
    task = Task.model_validate({
        "inputs": [{"column": "x", "alias": 'Input, "α"'}, {"column": "pH"}],
        "outputs": [{"column": "y", "alias": "Surface tension"}, {"column": "z"}],
        "additional_context": "Estimate the outputs.\nUse the reference units.",
    })
    reference = reference_prompt(task, [{"x": "001", "pH": "", "y": "63.2", "z": "low"}])
    prompt = target_prompt(task, reference, [{"x": "002", "pH": "7"}])
    assert prompt == (
        "Context:\nEstimate the outputs.\nUse the reference units.\n\nReference data:\n"
        '"Input, ""α""",pH,Surface tension,z\r\n001,,63.2,low\r\n'
        '\nPredict "Surface tension", "z" for the following targets in row order. Use each "target_id" value as its key in the predictions object:\n'
        'target_id,"Input, ""α""",pH\r\ntarget_0,002,7\r\n'
    )
    schema = AgentOutputSchema(output_schema(task, 1)).json_schema()
    properties = schema["$defs"]["OutputValues"]["properties"]
    assert set(properties) == {"y", "z"}
    assert properties["y"]["description"] == "Surface tension"
    assert properties["z"]["description"] == "z"


@pytest.mark.parametrize("model", ["openai/gpt-6-sol", "deepseek/deepseek-v4.1-flash", "google/gemma-4-31b-it"])
def test_endpoint_model_names_are_not_sdk_provider_prefixes(runner, monkeypatch, model):
    """Send complete model identifiers to the configured endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    runner.return_value = SimpleNamespace(final_output={"predictions": {"target_0": {"y": "0"}}, "reasoning": ""})
    asyncio.run(run_agent(task=make_task(), reference_data=[{"x": "a", "y": "0"}],
                          predict_data=[{"x": "b"}], model=model))
    provider = runner.await_args.kwargs["run_config"].model_provider
    endpoint_model = provider.get_model(model)
    assert endpoint_model.model == model
    assert str(endpoint_model._client.base_url) == "https://gateway.example/v1/"


def test_context_inputs_are_reference_only(runner):
    """Show context inputs in examples but not prediction targets."""
    task = Task.model_validate({
        "inputs": [
            {"column": "x"},
            {"column": "context_value", "alias": "Context value", "type": "context"},
        ],
        "outputs": [{"column": "y"}],
    })
    runner.return_value = SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "2"}}, "reasoning": "Pattern."}
    )

    result = asyncio.run(run_agent(
        task=task,
        reference_data=[{"x": "1", "context_value": "63.2", "y": "2"}],
        predict_data=[{"x": "3"}],
        model="test-model",
    ))

    assert result.model_dump(exclude={"metadata"}) == {
        "predictions": [{"x": "3", "y": "2"}],
        "reasoning": "Pattern.",
    }
    prompt = runner.await_args.args[1]
    assert "Context value" in prompt
    assert "Context value" not in prompt.split("Predict", 1)[1]


def test_prompt_without_additional_context():
    """Keep the task section empty when no additional context is supplied."""
    task = Task.model_validate({"inputs": [{"column": "x"}], "outputs": [{"column": "y"}]})
    reference = reference_prompt(task, [{"x": "1", "y": "2"}])
    assert reference == "Context:\n\n\nReference data:\nx,y\r\n1,2\r\n"


def test_sdk_schema_requires_exact_string_outputs():
    """Check the schema through the SDK output adapter."""
    adapter = AgentOutputSchema(output_schema(make_task(["a", "b"]), 1))
    schema = adapter.json_schema()
    assert schema["required"] == ["predictions", "reasoning"]
    values = schema["$defs"]["OutputValues"]
    assert values["required"] == ["a", "b"]
    assert values["additionalProperties"] is False
    valid = {"predictions": {"target_0": {"a": "", "b": "β"}}, "reasoning": "Pattern."}
    assert adapter.validate_json(json.dumps(valid)).model_dump(by_alias=True) == valid
    for bad in [
        {"predictions": {"target_0": {"a": "1"}}, "reasoning": "Missing output"},
        {"predictions": {"target_0": {"a": 1, "b": "2"}}, "reasoning": "Number"},
        {"predictions": {"target_0": {"a": "1", "b": "2", "c": "3"}}, "reasoning": "Extra"},
        {"predictions": {"target_0": {"a": "1", "b": "2"}}},
        {"predictions": {"target_0": {"a": "1", "b": "2"}}, "reasoning": None},
        {"predictions": {"a": "1", "b": "2"}, "reasoning": "Wrong shape"},
    ]:
        with pytest.raises(ModelBehaviorError):
            adapter.validate_json(json.dumps(bad))


@pytest.mark.parametrize("failure", [
    RuntimeError("context limit"),
    RuntimeError("API unavailable"),
    {"predictions": {"target_0": {"y": 3}}, "reasoning": "Invalid"},
    {"predictions": {"target_0": {"other": "3"}}, "reasoning": "Wrong column"},
])
def test_failure_reports_run_error(runner, failure):
    """Keep the source error when the run fails."""
    if isinstance(failure, Exception):
        runner.side_effect = failure
    else:
        runner.return_value = SimpleNamespace(final_output=failure)
    with pytest.raises(PredictionError, match="Prediction run") as error:
        asyncio.run(run_agent(
            task=make_task(), reference_data=[{"x": "1", "y": "2"}],
            predict_data=[{"x": "a"}, {"x": "b"}, {"x": "c"}], model="test"
        ))
    assert error.value.__cause__ is not None
    assert runner.await_count == 1


@pytest.mark.parametrize("count", [0, 1, 3])
def test_prediction_count_must_match_targets(runner, count):
    """Reject missing or extra prediction rows."""
    runner.return_value = SimpleNamespace(final_output={
        "predictions": {f"target_{index}": {"y": "2"} for index in range(count)}, "reasoning": "Pattern.",
    })
    with pytest.raises(PredictionError, match="target_"):
        asyncio.run(run_agent(
            task=make_task(), reference_data=[{"x": "1", "y": "2"}],
            predict_data=[{"x": "a"}, {"x": "b"}], model="test",
        ))
    runner.assert_awaited_once()


def test_large_batch_maps_reordered_target_ids(runner):
    """Map all 65 targets by ID when the response keys are in reverse order."""
    task = make_task()
    task.allowed_labels = {"y": ["0", "1"]}
    targets = [{"x": "same input"} for _ in range(65)]
    runner.return_value = SimpleNamespace(final_output={
        "predictions": {f"target_{index}": {"y": str(index % 2)} for index in reversed(range(65))},
        "reasoning": "Pattern.",
    })
    result = asyncio.run(run_agent(task=task, reference_data=[], predict_data=targets))
    assert len(result.predictions) == 65
    assert [row["y"] for row in result.predictions] == [str(index % 2) for index in range(65)]
    prompt = runner.await_args.args[1]
    assert "target_0,same input" in prompt
    assert "target_64,same input" in prompt
    schema = AgentOutputSchema(output_schema(task, 65)).json_schema()
    target_schema = schema["$defs"]["TargetPredictions"]
    assert target_schema["required"] == [f"target_{index}" for index in range(65)]
    assert target_schema["additionalProperties"] is False


@pytest.mark.parametrize("predictions", [
    {"target_0": {"y": "a"}, "target_2": {"y": "b"}},
    {"target_0": {"y": "a"}, "target_1": {"y": "b"}, "target_2": {"y": "c"}},
    [{"y": "a"}, {"y": "b"}],
])
def test_exact_target_identity_contract(runner, predictions):
    """Reject replaced IDs, extra IDs, and a response without target IDs."""
    runner.return_value = SimpleNamespace(final_output={"predictions": predictions, "reasoning": ""})
    with pytest.raises(PredictionError):
        asyncio.run(run_agent(task=make_task(), reference_data=[], predict_data=[{"x": "a"}, {"x": "a"}]))


def test_target_id_header_does_not_replace_input_or_output_names():
    """Keep input cells when the task uses target ID names."""
    task = Task.model_validate(
        {"inputs": [{"column": "x", "alias": "target_id"}], "outputs": [{"column": "_target_id"}]}
    )
    prompt = target_prompt(task, "Reference", [{"x": "unchanged"}, {"x": "unchanged"}])
    target_csv = prompt.split("predictions object:\n")[1]
    assert list(csv.reader(io.StringIO(target_csv))) == [
        ["__target_id", "target_id"], ["target_0", "unchanged"], ["target_1", "unchanged"],
    ]


@pytest.mark.parametrize("references,targets", [
    ([{"x": "1"}], [{"x": "2"}]),
    ([{"x": "1", "y": "2"}], [{"x": "2"}, {"wrong": "3"}]),
    ([{"x": "1", "y": 2}], [{"x": "2"}]),
])
def test_validate_all_data_before_calls(runner, references, targets):
    """Reject bad data before the first API call."""
    with pytest.raises(ValueError):
        asyncio.run(run_agent(
            task=make_task(), reference_data=references, predict_data=targets, model="test"
        ))
    runner.assert_not_called()


def test_empty_predictions(runner):
    """Return no results when there are no target rows."""
    assert asyncio.run(run_agent(
        task=make_task(), reference_data=[{"x": "", "y": ""}],
        predict_data=[], model="test",
    )).model_dump() == {"predictions": [], "reasoning": "", "metadata": {}}
    runner.assert_not_called()


def test_mutated_task_is_revalidated(runner):
    """Reject task edits that add an overlap."""
    task = make_task()
    task.outputs.append(task.inputs[0])
    with pytest.raises(ValidationError):
        asyncio.run(run_agent(task=task, reference_data=[], predict_data=[], model="test"))
    runner.assert_not_called()


@pytest.mark.parametrize("override", [None, "explicit-model"])
@pytest.mark.parametrize("environment_model", [None, "environment-model"])
def test_model_environment(tmp_path, monkeypatch, override, environment_model):
    """Load file settings and keep environment and explicit model values."""
    import os

    from agents.models.default_models import get_default_model

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    if environment_model is not None:
        monkeypatch.setenv("OPENAI_DEFAULT_MODEL", environment_model)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-key\nOPENAI_DEFAULT_MODEL=file-model\n"
    )
    mock = AsyncMock(return_value=SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "1"}}, "reasoning": "Pattern."}
    ))
    monkeypatch.setattr("icl_agent.llm_agent.Runner.run", mock)
    asyncio.run(run_agent(
        task=make_task(), reference_data=[{"x": "a", "y": "1"}],
        predict_data=[{"x": "b"}], model=override,
    ))
    agent = mock.call_args.args[0]
    assert (agent.model or get_default_model()) == (
        override or environment_model or "file-model"
    )
    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_api_base_environment_configures_sdk_base_url(tmp_path, monkeypatch):
    """Map OPENAI_API_BASE to the base URL used by the SDK."""
    import os

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-key\nOPENAI_API_BASE=https://gateway.example/v1\n"
    )
    mock = AsyncMock(return_value=SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "1"}}, "reasoning": "Pattern."}
    ))
    monkeypatch.setattr("icl_agent.llm_agent.Runner.run", mock)

    asyncio.run(run_agent(
        task=make_task(), reference_data=[{"x": "a", "y": "1"}],
        predict_data=[{"x": "b"}], model="test-model",
    ))

    assert os.environ["OPENAI_BASE_URL"] == "https://gateway.example/v1"


def test_zero_shot_labels_settings_and_metadata(runner):
    """Keep zero-shot labels and record supplied SDK metadata."""
    task = make_task()
    task.allowed_labels = {"y": ["active", "inactive"]}
    settings = InferenceSettings(temperature=0.5, max_output_tokens=200, seed=7, reasoning_effort="low")
    runner.return_value = SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "active"}}, "reasoning": "Task context."},
        context_wrapper=SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 3}),
        raw_responses=[SimpleNamespace(
            request_id="request-1", response_id="response-1", raw_usage={"cost": 0.01},
            output=[],
        )],
    )
    result = asyncio.run(run_agent(
        task=task, reference_data=[], predict_data=[{"x": "target", "y": "hidden-label"}],
        model="chosen", settings=settings,
    ))
    agent, prompt = runner.await_args.args
    assert "Zero-shot prediction" in prompt
    assert '"active", "inactive"' in prompt
    assert "hidden-label" not in prompt
    assert agent.model_settings.temperature == 0.5
    assert agent.model_settings.max_tokens == 200
    assert agent.model_settings.reasoning.effort == "low"
    assert agent.model_settings.extra_args == {"seed": 7}
    assert result.metadata["model"] == "chosen"
    assert result.metadata["settings"] == settings.model_dump()
    assert result.metadata["elapsed_seconds"] >= 0
    assert result.metadata["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert result.metadata["responses"][0]["request_id"] == "request-1"
    assert "provider_reasoning" not in result.metadata["responses"][0]
    json.dumps(result.model_dump())


def test_allowed_labels_reject_unknown_output(runner):
    """Reject a model class that the task does not permit."""
    task = make_task()
    task.allowed_labels = {"y": ["active", "inactive"]}
    runner.return_value = SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "other"}}, "reasoning": "Unsupported."}
    )
    with pytest.raises(PredictionError, match="active"):
        asyncio.run(run_agent(task=task, reference_data=[], predict_data=[{"x": "target"}]))


def test_empty_zero_shot_targets(runner):
    """Skip the API when no target rows exist."""
    result = asyncio.run(run_agent(task=make_task(), reference_data=[], predict_data=[]))
    assert result.model_dump() == {"predictions": [], "reasoning": "", "metadata": {}}
    runner.assert_not_called()


def test_provider_reasoning_is_separate(runner):
    """Keep supplied provider fields outside the prediction rationale."""
    from pydantic import BaseModel

    class ReasoningItem(BaseModel):
        type: str = "reasoning"
        summary: list[str]

    runner.return_value = SimpleNamespace(
        final_output={"predictions": {"target_0": {"y": "2"}}, "reasoning": "Brief rationale."},
        raw_responses=[SimpleNamespace(output=[ReasoningItem(summary=["Provider summary."])])],
    )
    result = asyncio.run(run_agent(
        task=make_task(), reference_data=[], predict_data=[{"x": "target"}], model="test"
    ))
    assert result.reasoning == "Brief rationale."
    assert result.metadata["responses"][0]["provider_reasoning"] == [
        {"type": "reasoning", "summary": ["Provider summary."]}
    ]
