"""Check decision requests and answer handling without API calls."""

import asyncio
import json
import os

import httpx2
import pytest
from openai import AsyncOpenAI

from icl_agent import PredictionError, Task, run_decision


def make_task(
    outputs: tuple[str, ...] = ("label",), labels: tuple[str, ...] | None = ("low", "high")
) -> Task:
    """Create a task with text labels and an output contract."""
    label_count = len(labels) if labels is not None else 0
    decision_question = (
        "Which class does {x} have?" if label_count == 3 else "Is {x} high?"
    )
    return Task.model_validate({
        "inputs": [{"column": "x"}],
        "outputs": [{"column": name} for name in outputs],
        "additional_context": "Task context.",
        "decision_question": decision_question,
        "allowed_labels": {outputs[0]: list(labels)} if labels is not None else {},
    })


def predicate_answer(name: str, probability: float) -> dict:
    """Create one typed predicate answer for a target row."""
    return {"type": "predicate", "name": name, "probability": probability}


def choice_answer(name: str, label: str) -> dict:
    """Create one typed choice answer for a three-label task."""
    labels = ["first", "second", "third"]
    probabilities = [
        {"value": value, "probability": 0.8 if value == label else 0.1}
        for value in labels
    ]
    return {
        "type": "choice",
        "name": name,
        "choice": label,
        "confidence": 0.8,
        "probabilities": probabilities,
    }


def decision_response(answers: list[dict], *, model: str = "gpt-6-luna") -> httpx2.Response:
    """Create one complete Decisions response."""
    return httpx2.Response(200, json={
        "answers": answers,
        "model": model,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })


def install_api(monkeypatch, responses: list[httpx2.Response]) -> list[httpx2.Request]:
    """Install a local OpenAI transport and capture requests."""
    calls: list[httpx2.Request] = []
    queue = list(responses)

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Capture one request and return the next response."""
        calls.append(request)
        return queue.pop(0)

    def factory(*args, **kwargs) -> AsyncOpenAI:
        """Create a client with the local transport and no retries."""
        return AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr("icl_agent.decision_agent.AsyncOpenAI", factory)
    return calls


def install_recording_api(
    monkeypatch, responses: list[httpx2.Response],
) -> tuple[list[httpx2.Request], dict[str, str | None]]:
    """Install a client factory that records the base URL environment value."""
    calls: list[httpx2.Request] = []
    observed: dict[str, str | None] = {}
    queue = list(responses)

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Capture one request and return the next response."""
        calls.append(request)
        return queue.pop(0)

    def factory(*args, **kwargs) -> AsyncOpenAI:
        """Record the environment value before the client is created."""
        observed["base_url_env"] = os.environ.get("OPENAI_BASE_URL")
        return AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr("icl_agent.decision_agent.AsyncOpenAI", factory)
    return calls, observed


def test_one_predicate_question_per_target(monkeypatch):
    """Send one predicate per target and restore the target inputs."""
    references = [{"x": "first reference", "label": "low"},
                  {"x": "second reference", "label": "high"}]
    targets = [{"x": "first target", "label": "hidden"},
               {"x": "second target", "label": "hidden"}]
    calls = install_api(monkeypatch, [decision_response([
        predicate_answer("target_0", 0.1), predicate_answer("target_1", 0.9),
    ])])
    result = asyncio.run(run_decision(task=make_task(), reference_data=references,
                                     predict_data=targets, model="gpt-6-luna"))
    assert result.predictions == [{"x": "first target", "label": "low"},
                                  {"x": "second target", "label": "high"}]
    assert result.reasoning == ""
    assert result.metadata["model"] == "gpt-6-luna"
    assert result.metadata["rationale_available"] is False
    assert result.metadata["usage"]["total_tokens"] == 15
    assert result.metadata["answers"][0]["probability"] == 0.1
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "https://api.openai.test/v1/decisions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert set(body) == {"model", "input", "questions"}
    assert body["model"] == "gpt-6-luna"
    content = body["input"][0]["content"]
    assert [part["type"] for part in content] == ["input_text", "input_json"]
    assert content[1]["data"] == {"references": {
        "columns": ["x", "label"],
        "rows": [
            ["first reference", "low"],
            ["second reference", "high"],
        ],
    }}
    assert "targets" not in content[1]["data"]
    assert "hidden" not in json.dumps(body)
    questions = body["questions"]
    assert [item["name"] for item in questions] == ["target_0", "target_1"]
    assert all(item["type"] == "predicate" for item in questions)
    assert questions[0]["instructions"] == "Is first target high?"
    assert questions[1]["instructions"] == "Is second target high?"


def test_answers_are_mapped_by_name(monkeypatch):
    """Restore target order when answers arrive out of order."""
    install_api(monkeypatch, [decision_response([
        predicate_answer("target_1", 0.9), predicate_answer("target_0", 0.1),
    ])])
    result = asyncio.run(run_decision(task=make_task(), reference_data=[],
                                     predict_data=[{"x": "first"}, {"x": "second"}],
                                     model="gpt-6-luna"))
    assert result.predictions == [{"x": "first", "label": "low"},
                                  {"x": "second", "label": "high"}]


def test_three_labels_use_choice_questions(monkeypatch):
    """Use choice questions when the task has three labels."""
    task = make_task(labels=("first", "second", "third"))
    calls = install_api(monkeypatch, [decision_response([
        choice_answer("target_0", "second"),
    ])])
    result = asyncio.run(run_decision(task=task, reference_data=[],
                                     predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert result.predictions == [{"x": "first", "label": "second"}]
    questions = json.loads(calls[0].content)["questions"]
    assert questions[0]["type"] == "choice"
    assert questions[0]["instructions"] == "Which class does first have?"
    assert [item["value"] for item in questions[0]["choices"]] == [
        "first", "second", "third"]


@pytest.mark.parametrize("answers", [
    [],
    [{"type": "refusal", "name": "target_0"}],
    [{"type": "predicate", "name": "wrong", "probability": 0.5}],
    [{"type": "predicate", "name": "target_0"}],
    [{"type": "predicate", "name": "target_0", "probability": -0.1}],
    [{"type": "predicate", "name": "target_0", "probability": 1.1}],
    [{"type": "choice", "name": "target_0", "choice": "low"}],
    [predicate_answer("target_0", 0.5)] * 2,
])
def test_invalid_answers_fail(monkeypatch, answers):
    """Reject missing, duplicated, refused, or non-contract answers."""
    install_api(monkeypatch, [decision_response(answers)])
    with pytest.raises(PredictionError):
        asyncio.run(run_decision(task=make_task(), reference_data=[],
                                 predict_data=[{"x": "first"}], model="gpt-6-luna"))


def test_empty_targets_skip_the_request(monkeypatch):
    """Return an empty result without a request for no target rows."""
    calls = install_api(monkeypatch, [])
    result = asyncio.run(run_decision(task=make_task(), reference_data=[],
                                     predict_data=[], model="gpt-6-luna"))
    assert result.predictions == []
    assert result.reasoning == ""
    assert calls == []


def test_http_failure_is_reported_without_credentials(monkeypatch):
    """Wrap endpoint failures and do not store the API key."""
    calls = install_api(monkeypatch, [httpx2.Response(400, json={"error": {"message": "bad"}})])
    with pytest.raises(PredictionError) as error:
        asyncio.run(run_decision(task=make_task(), reference_data=[],
                                 predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert "400" in str(error.value)
    assert "test-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("task", [
    make_task(outputs=("first", "second")),
    make_task(labels=None),
    make_task(labels=("low",)),
    make_task(labels=tuple(f"label{index}" for index in range(256))),
])
def test_unsupported_task_contracts_fail_before_requests(monkeypatch, task):
    """Reject task contracts that the Decisions API cannot express."""
    calls = install_api(monkeypatch, [])
    with pytest.raises(ValueError):
        asyncio.run(run_decision(task=task, reference_data=[],
                                 predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert calls == []


def test_question_limit_is_enforced(monkeypatch):
    """Reject more questions than the Decisions API accepts."""
    calls = install_api(monkeypatch, [])
    with pytest.raises(ValueError, match="at most 200"):
        asyncio.run(run_decision(task=make_task(), reference_data=[],
                                 predict_data=[{"x": str(index)} for index in range(201)],
                                 model="gpt-6-luna"))
    assert calls == []


def test_missing_decision_question_fails_before_requests(monkeypatch):
    """Reject a decision task without a question before any request."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}],
        "outputs": [{"column": "label"}],
        "allowed_labels": {"label": ["low", "high"]},
    })
    calls = install_api(monkeypatch, [])
    with pytest.raises(ValueError, match="decision_question"):
        asyncio.run(run_decision(task=task, reference_data=[],
                                 predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert calls == []


def test_context_inputs_are_excluded_from_target_results(monkeypatch):
    """Send context only as evidence and keep it out of predictions."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}, {"column": "ctx", "type": "context"}],
        "outputs": [{"column": "label"}],
        "decision_question": "Is {x} high?",
        "allowed_labels": {"label": ["low", "high"]},
    })
    calls = install_api(
        monkeypatch, [decision_response([predicate_answer("target_0", 0.9)])])
    result = asyncio.run(run_decision(
        task=task,
        reference_data=[{"x": "reference", "ctx": "reference context", "label": "low"}],
        predict_data=[{"x": "first"}],
        model="gpt-6-luna",
    ))
    assert result.predictions == [{"x": "first", "label": "high"}]
    body = json.loads(calls[0].content)
    data = body["input"][0]["content"][1]["data"]
    assert data == {"references": {
        "columns": ["x", "ctx", "label"],
        "rows": [["reference", "reference context", "low"]],
    }}
    assert body["questions"][0]["instructions"] == "Is first high?"


def test_decision_question_may_omit_some_target_inputs(monkeypatch):
    """Format a question that uses only one of several target inputs."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}, {"column": "y"}],
        "outputs": [{"column": "label"}],
        "decision_question": "Is {x} high?",
        "allowed_labels": {"label": ["low", "high"]},
    })
    calls = install_api(
        monkeypatch, [decision_response([predicate_answer("target_0", 0.9)])])
    result = asyncio.run(run_decision(
        task=task,
        reference_data=[{"x": "reference", "y": "unused", "label": "low"}],
        predict_data=[{"x": "first", "y": "unused"}],
        model="gpt-6-luna",
    ))
    assert result.predictions == [{"x": "first", "y": "unused", "label": "high"}]
    body = json.loads(calls[0].content)
    assert body["questions"][0]["instructions"] == "Is first high?"


def test_target_id_column_does_not_replace_task_inputs(monkeypatch):
    """Keep the target row key when the task already has a target_id column."""
    task = Task.model_validate({
        "inputs": [{"column": "x"}, {"column": "target_id"}],
        "outputs": [{"column": "label"}],
        "decision_question": "Is {x} ({target_id}) high?",
        "allowed_labels": {"label": ["low", "high"]},
        "additional_context": "Task context.",
    })
    calls = install_api(
        monkeypatch, [decision_response([predicate_answer("target_0", 0.9)])])
    result = asyncio.run(run_decision(
        task=task,
        reference_data=[{"x": "reference", "target_id": "row-1", "label": "low"}],
        predict_data=[{"x": "first", "target_id": "row-2"}],
        model="gpt-6-luna",
    ))
    assert result.predictions == [{"x": "first", "target_id": "row-2", "label": "high"}]
    body = json.loads(calls[0].content)
    data = body["input"][0]["content"][1]["data"]
    assert data == {"references": {
        "columns": ["x", "target_id", "label"],
        "rows": [["reference", "row-1", "low"]],
    }}
    assert body["questions"][0]["instructions"] == "Is first (row-2) high?"


def test_openai_api_base_is_mapped_to_base_url(monkeypatch):
    """Set OPENAI_BASE_URL from OPENAI_API_BASE before the client is created."""
    monkeypatch.setattr("icl_agent.decision_agent.load_dotenv",
                        lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    _, observed = install_recording_api(
        monkeypatch, [decision_response([predicate_answer("target_0", 0.1)])])
    asyncio.run(run_decision(task=make_task(), reference_data=[],
                             predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert observed["base_url_env"] == "https://gateway.example/v1"
    assert os.environ["OPENAI_BASE_URL"] == "https://gateway.example/v1"


def test_missing_api_base_leaves_base_url_unset(monkeypatch):
    """Keep the resolved base URL unset when OPENAI_API_BASE is not configured."""
    monkeypatch.setattr("icl_agent.decision_agent.load_dotenv",
                        lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _, observed = install_recording_api(
        monkeypatch, [decision_response([predicate_answer("target_0", 0.1)])])
    asyncio.run(run_decision(task=make_task(), reference_data=[],
                             predict_data=[{"x": "first"}], model="gpt-6-luna"))
    assert observed["base_url_env"] is None


def test_results_types_come_from_the_results_module():
    """Keep the prediction types in the results module and re-export them."""
    import icl_agent
    from icl_agent import results

    assert icl_agent.PredictionResult is results.PredictionResult
    assert icl_agent.PredictionError is results.PredictionError
