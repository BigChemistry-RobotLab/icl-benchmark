"""Check decision payloads, admission limits, and endpoint answer mapping."""

import asyncio
import json

import httpx
import pytest

from experiments.config import ModelConfig
from experiments.endpoints.decisions import (
    INSTRUCTIONS,
    admission,
    build_payload,
    endpoint_url,
    send_request,
)
from icl_agent.config import Column, Task


def task(labels: list[str] | None = None) -> Task:
    """Create a task with text labels and a feature allowlist."""
    labels = labels or ["low", "high"]
    decision_question = (
        "Which class does {input} have?" if len(labels) == 3 else "Is {input} high?"
    )
    return Task(inputs=[Column(column="input")], outputs=[Column(column="label")],
                allowed_labels={"label": labels},
                additional_context="Task context",
                decision_question=decision_question)


def settings(**kwargs) -> ModelConfig:
    """Create explicit decision admission settings."""
    return ModelConfig(type="decision", context_window=100000, **kwargs)


def test_state_parts_and_choice_questions():
    """Send text instructions, JSON evidence, and one compact choice per target."""
    references = [{"input": str(index), "label": "low", "audit": "secret"} for index in range(50)]
    targets = [{"input": "first", "label": "hidden"}, {"input": "second", "label": "hidden"}]
    payload = build_payload(task(), references, targets, "test/decision")
    assert set(payload) == {"model", "state", "questions"}
    message = payload["state"][0]
    assert message["role"] == "user"
    text_part, json_part = message["content"]
    assert text_part["type"] == "input_text"
    assert text_part["text"].startswith(INSTRUCTIONS)
    assert "Task context:\nTask context" in text_part["text"]
    assert json_part["type"] == "input_json"
    assert json_part["data"] == {"references": {
        "columns": ["input", "label"],
        "rows": [[str(index), "low"] for index in range(50)],
    }}
    assert "targets" not in json_part["data"]
    assert "secret" not in json.dumps(payload)
    assert "hidden" not in json.dumps(payload)
    questions = payload["questions"]
    assert list(questions) == ["target_0", "target_1"]
    assert all(question["type"] == "choice" for question in questions.values())
    assert questions["target_0"]["instructions"] == "Is first high?"
    assert questions["target_1"]["instructions"] == "Is second high?"
    assert questions["target_0"]["criteria"] == {"low": "low", "high": "high"}
    assert "temperature" not in json.dumps(payload)


def test_payload_requires_decision_model_settings():
    """Reject chat model settings for a decision request."""
    with pytest.raises(ValueError, match="decision model"):
        admission(task(), [], [{"input": "a"}], "test/decision",
                  ModelConfig(type="llm", context_window=1000))
    with pytest.raises(ValueError):
        ModelConfig(type="decision", temperature=0.2)


def test_decision_endpoint_uses_origin_without_credentials(monkeypatch):
    """Build the Decisions URL from the redacted endpoint origin."""
    monkeypatch.setattr("experiments.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_BASE", "https://user:secret@example.com/v1?token=hidden#fragment")
    assert endpoint_url() == "https://example.com/api/alpha/decisions"
    monkeypatch.setenv("OPENAI_API_BASE", "not-a-url")
    with pytest.raises(ValueError, match="HTTP endpoint origin"):
        endpoint_url()


def test_admission_checks_payload_size_and_question_limit():
    """Reject oversized requests without reducing the selected context."""
    config = settings(max_questions=1)
    with pytest.raises(ValueError, match="question limit"):
        admission(task(), [], [{"input": "a"}, {"input": "b"}], "test/decision", config)
    config.context_window = 1025
    with pytest.raises(ValueError, match="admission"):
        admission(task(), [{"input": "long" * 1000, "label": "low"}],
                  [{"input": "a"}], "test/decision", config)


def test_admission_accepts_exact_conservative_bound():
    """Use the complete serialized payload and reserve for admission."""
    config = settings()
    targets = [{"input": "é"}]
    payload = build_payload(task(), [], targets, "test/decision")
    bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024
    config.context_window = bound
    assert admission(task(), [], targets, "test/decision", config) == payload
    config.context_window -= 1
    with pytest.raises(ValueError, match="admission"):
        admission(task(), [], targets, "test/decision", config)


@pytest.mark.parametrize("labels", [["x"], ["a", "b", "c", "d"]])
def test_reject_unsupported_choice_contract(labels):
    """Require a two-class or three-class task."""
    with pytest.raises(ValueError, match="two or three"):
        build_payload(task(labels), [], [{"input": "a"}], "test/decision")


def install_endpoint(monkeypatch, body, status=200):
    """Install a local HTTP transport and capture requests."""
    calls = []
    original_client = httpx.AsyncClient

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("experiments.endpoints.decisions.httpx.AsyncClient",
                        lambda **kwargs: original_client(
                            transport=httpx.MockTransport(respond), **kwargs))
    return calls


def test_response_order_metadata_and_endpoint(monkeypatch):
    """Restore target order and keep reported costs and probabilities."""
    body = {"answers": {
        "target_1": {"type": "choice", "choice": "high",
                     "probabilities": {"low": 0.1, "high": 0.9}},
        "target_0": {"type": "choice", "choice": "low",
                     "probabilities": {"low": 0.8, "high": 0.2}},
    }, "model": "resolved/model", "provider": "test", "id": "request-1",
        "usage": {"input_tokens": 20, "output_tokens": 5, "cost": 0.002}}
    calls = install_endpoint(monkeypatch, body)
    result = asyncio.run(send_request(
        task=task(), reference_data=[],
        predict_data=[{"input": "a"}, {"input": "b"}],
        model="test/decision", settings=settings()))
    assert result.predictions == [{"input": "a", "label": "low"},
                                  {"input": "b", "label": "high"}]
    assert result.reasoning == ""
    assert result.metadata["rationale_available"] is False
    assert result.metadata["cost"] == 0.002
    assert result.metadata["model"] == "resolved/model"
    assert result.metadata["probabilities"]["target_1"]["high"] == 0.9
    assert len(calls) == 1
    assert str(calls[0].url) == "https://openrouter.ai/api/alpha/decisions"
    assert calls[0].headers["Authorization"] == "Bearer test-secret"
    assert "test-secret" not in result.model_dump_json()


@pytest.mark.parametrize("answers", [
    {},
    [],
    {"wrong_id": {"type": "choice", "choice": "low"}},
    {"target_0": {"type": "choice"}},
    {"target_0": {"type": "predicate", "probability": 0.5}},
    {"target_0": {"type": "choice", "choice": "unknown"}},
    {"target_0": {"type": "choice", "choice": 0}},
    {"target_0": {"type": "choice", "choice": "low"},
     "target_1": {"type": "choice", "choice": "high"}},
    {"target_0": []},
])
def test_malformed_answers_fail(monkeypatch, answers):
    """Reject missing targets, unknown names, and invalid answers."""
    install_endpoint(monkeypatch, {"answers": answers})
    with pytest.raises((TypeError, ValueError), match="Decision"):
        asyncio.run(send_request(
            task=task(), reference_data=[], predict_data=[{"input": "a"}],
            model="test/decision", settings=settings()))


def test_admission_failure_sends_no_request(monkeypatch):
    """Apply endpoint limits before an HTTP request."""
    calls = install_endpoint(monkeypatch, {})
    config = settings()
    config.context_window = 1
    with pytest.raises(ValueError, match="admission"):
        asyncio.run(send_request(
            task=task(), reference_data=[], predict_data=[{"input": "a"}],
            model="test/decision", settings=config))
    assert calls == []


def test_endpoint_error_has_no_retry_or_credentials(monkeypatch):
    """Report endpoint status without storing response text or credentials."""
    calls = install_endpoint(monkeypatch, {"error": "test-secret"}, status=401)
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        asyncio.run(send_request(
            task=task(), reference_data=[], predict_data=[{"input": "a"}],
            model="test/decision", settings=settings()))
    assert len(calls) == 1
    assert "test-secret" not in str(error.value)


def test_three_class_choice_answer_is_mapped(monkeypatch):
    """Restore a choice answer for a three-label task."""
    body = {"answers": {"target_0": {
        "type": "choice", "choice": "1",
        "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
    }}}
    install_endpoint(monkeypatch, body)
    result = asyncio.run(send_request(
        task=task(["0", "1", "2"]), reference_data=[],
        predict_data=[{"input": "a"}], model="test/decision",
        settings=settings()))
    assert result.predictions == [{"input": "a", "label": "1"}]
    assert result.metadata["probabilities"]["target_0"]["1"] == 0.8


def test_three_class_choices_and_empty_targets(monkeypatch):
    """Use exact criteria for three labels and skip empty targets."""
    payload = build_payload(task(["0", "1", "2"]), [], [{"input": "a"}],
                            "test/decision")
    question = payload["questions"]["target_0"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Which class does a have?"
    assert list(question["criteria"]) == ["0", "1", "2"]
    calls = install_endpoint(monkeypatch, {})
    result = asyncio.run(send_request(
        task=task(), reference_data=[], predict_data=[],
        model="test/decision", settings=settings()))
    assert result.predictions == []
    assert calls == []


def test_model_must_be_nonempty():
    """Reject a blank provider identifier before building a request."""
    with pytest.raises(ValueError, match="nonempty decision model"):
        build_payload(task(), [], [{"input": "a"}], "  ")
