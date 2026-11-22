"""Offline coverage for Aliyun transport, response contracts and durable retries."""
from __future__ import annotations

import json
import logging
import multiprocessing
import socket
import ssl
import threading
import time
from dataclasses import replace
from email.utils import formatdate

import pytest

from fallpre import decision_trace
from fallpre.adapters.aliyun_decision import (
    API_PATH,
    AliyunDecisionProvider,
    DecisionProviderError,
    HTTPResponse,
    HTTPSJsonTransport,
    parse_response,
)
from fallpre.decision_config import load_decision_config
from fallpre.decision_contracts import ProviderResult
from fallpre.decision_evidence import DecisionEnvelope, adapt_envelope, synthetic_envelope
from fallpre.decision_fixtures import synthetic_evidence
from fallpre.decision_loop import DecisionLoop, DecisionWorker
from fallpre.decision_provider import MockDecisionProvider
from fallpre.decision_state import (
    DECISION_PROMPT_VERSION,
    DECISION_STATE_SCHEMA_VERSION,
    build_decision_state,
    decision_questions,
)

FAKE_KEY = "synthetic-test-secret-only-never-use-real-key"
CHOICE_PROBABILITIES = {"sufficient": 0.92, "insufficient": 0.05, "conflicting": 0.03}
NEXT_PROBABILITIES = {"emit_result": 0.93, "observe_more": 0.04, "request_review": 0.03}


def _response(*, sufficiency="sufficient", next_step="emit_result", fall=0.72, request_id="req-ok-001"):
    return {
        "model": "decision-model-preview",
        "request_id": request_id,
        "answers": {
            "fall_occurred": {"type": "noul", "noul": fall},
            "evidence_sufficiency": {
                "type": "choice", "choice": sufficiency, "confidence": 0.92,
                "probabilities": dict(CHOICE_PROBABILITIES),
            },
            "next_step": {
                "type": "choice", "choice": next_step, "confidence": 0.91,
                "probabilities": dict(NEXT_PROBABILITIES),
            },
        },
        "usage": {"input_tokens": 119},
        "latency_ms": 46.5,
    }


class QueueTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post_json(self, *, host, path, headers, body, deadline):
        self.calls.append({"host": host, "path": path, "headers": dict(headers),
                           "body": body, "deadline": deadline})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _config(tmp_path, *, timeout=3.0, retries=1):
    return load_decision_config(
        env_file=tmp_path / "no-real-env",
        environ={
            "FALLPRE_DECISION_PROVIDER": "aliyun",
            "DASHSCOPE_API_KEY": FAKE_KEY,
            "DASHSCOPE_WORKSPACE_ID": "synthetic-test-space",
            "DASHSCOPE_REGION": "cn-beijing",
            "FALLPRE_DECISION_MODEL": "decision-model-preview",
            "FALLPRE_DECISION_TIMEOUT_S": str(timeout),
            "FALLPRE_DECISION_MAX_RETRIES": str(retries),
        },
        include_credentials=True,
    )


def _evidence(**changes):
    values = synthetic_evidence().model_dump(mode="python")
    values.update(changes)
    return type(synthetic_evidence()).model_validate(values)


def _ok_response(**kwargs):
    return HTTPResponse(200, {}, json.dumps(_response(**kwargs), allow_nan=False).encode())


def test_state_and_questions_are_versioned_whitelisted_and_non_identifying():
    evidence = _evidence(
        event_key="private-event-name-should-not-leave-local-store",
        yolo_model_version="/Users/private/model-path.pt",
        policy_version="policy-v9",
        action_label="standing",
        fall_process_label="non_fall_activity",
    )
    state = build_decision_state(evidence)
    serialized = json.dumps(state, ensure_ascii=False, sort_keys=True)
    questions = decision_questions()
    assert state["state_schema_version"] == DECISION_STATE_SCHEMA_VERSION
    assert state["prompt_version"] == DECISION_PROMPT_VERSION
    assert set(questions) == {"fall_occurred", "evidence_sufficiency", "next_step"}
    assert questions["fall_occurred"]["type"] == "noul"
    assert set(questions["evidence_sufficiency"]["criteria"]) == {
        "sufficient", "insufficient", "conflicting",
    }
    assert set(questions["next_step"]["criteria"]) == {"emit_result", "observe_more", "request_review"}
    assert "private-event-name" not in serialized
    assert "/Users/private" not in serialized
    assert "session_id" not in state and "stream_id" not in state
    assert "fixture-target-" not in serialized
    assert "fall_score" in state["process"]["temporal"]
    assert "raw_score" in state["process"]["fall_head"]
    assert "independent_sensor_confirmation" in state["evidence_relationships"]


def test_success_response_is_normalized_and_un_calibrated(tmp_path):
    transport = QueueTransport(_ok_response())
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    result = provider.assess(synthetic_evidence(), deadline=13.0)
    assert result.status == "ready"
    assert result.fall_probability == 0.72
    assert result.fall_probability_raw == 0.72
    assert result.fall_probability_calibrated is None
    assert result.probability_kind == "uncalibrated_model_estimate"
    assert result.provider_metadata.request_id == "req-ok-001"
    assert result.provider_metadata.input_tokens == 119
    assert result.provider_metadata.service_latency_ms == 46.5
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request["host"] == "synthetic-test-space.cn-beijing.maas.aliyuncs.com"
    assert request["path"] == API_PATH
    assert FAKE_KEY in request["headers"]["Authorization"]
    sent_state = json.loads(request["body"])["state"]
    assert sent_state["state_schema_version"] == DECISION_STATE_SCHEMA_VERSION
    assert "Authorization" not in request["body"].decode()


def test_trace_correlates_retry_and_zero_result_without_request_body_or_credentials(tmp_path, monkeypatch):
    logger = logging.Logger("isolated-test-trace")
    monkeypatch.setattr(decision_trace, "_log", logger)
    trace = tmp_path / "server.jsonl"
    decision_trace.enable_decision_trace(trace)
    transport = QueueTransport(HTTPResponse(503, {"retry-after": "0"}, b"private-error-body"),
                               _ok_response(fall=0.0))
    loop = DecisionLoop(tmp_path / "trace.sqlite3", run_id="trace-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport)
    try:
        receipt = loop.observe(synthetic_envelope(session_id="session", stream_id="stream",
                                                  source_time_s=5.0, suffix="trace"))
        assert loop.run_one()["status"] == "retry_pending"
        assert loop.run_one()["status"] == "accepted"
        lines = trace.read_text()
        events = [json.loads(line) for line in lines.splitlines()]
        assert [event["stage"] for event in events] == [
            "task_claimed", "provider_dispatch", "provider_http_response", "provider_failure",
            "task_claimed", "provider_dispatch", "provider_http_response", "result_commit",
        ]
        assert all(event["decision_id"] == receipt["decision_id"] for event in events)
        assert events[-1]["accepted"] and events[-1]["fall_probability"] == 0.0
        assert events[-1]["request_id"] == "req-ok-001"
        assert all(value not in lines for value in (FAKE_KEY, "Authorization", "private-error-body", "questions"))
    finally:
        loop.close()
        logger.handlers[0].close()


def test_quality_gate_returns_insufficient_without_transport_call(tmp_path):
    evidence = _evidence(event_decidable=False)
    transport = QueueTransport()
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    result = provider.assess(evidence, deadline=13.0)
    assert result.status == "insufficient"
    assert result.fall_probability is None
    assert result.reason_codes
    assert transport.calls == []


@pytest.mark.parametrize(
    ("sufficiency", "next_step", "status", "expected_sufficiency", "expected_next"),
    [
        ("insufficient", "observe_more", "insufficient", "insufficient", "observe_more"),
        ("conflicting", "request_review", "insufficient", "conflicting", "request_review"),
        ("sufficient", "observe_more", "insufficient", "conflicting", "request_review"),
    ],
)
def test_nonready_and_conflicting_answer_combinations_withhold_current_probability(
    tmp_path, sufficiency, next_step, status, expected_sufficiency, expected_next,
):
    transport = QueueTransport(_ok_response(sufficiency=sufficiency, next_step=next_step))
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    result = provider.assess(synthetic_evidence(), deadline=13.0)
    assert result.status == status
    assert result.fall_probability is None
    assert result.fall_probability_raw == 0.72
    assert result.probability_kind == "uncalibrated_model_estimate"
    assert result.evidence_sufficiency == expected_sufficiency
    assert result.next_step == expected_next
    assert result.provider_metadata.answers.evidence_sufficiency.choice == sufficiency


def test_optional_action_and_geometry_mapping_detects_source_conflicts():
    envelope = synthetic_envelope(session_id="mapping-session", stream_id="mapping", source_time_s=10.0,
                                  suffix="mapping")
    envelope["perception"]["fall_process"]["geometry"]["transition"] = "upright_to_low"
    envelope["perception"]["current_action"]["label"] = "falling"
    envelope["evidence_summary"]["geometry_transition"] = "upright_to_low"
    envelope["evidence_summary"]["action_label"] = "falling"
    item = DecisionEnvelope.model_validate(envelope)
    adaptation = adapt_envelope(item, run_id="mapping-run", snapshot_id="snapshot-map",
                                 decision_id="decision-map", request_digest="0" * 64)
    assert adaptation.evidence is not None
    assert adaptation.evidence.action_label == "falling"
    assert adaptation.evidence.geometry_transition == "upright_to_low"
    inconsistent = json.loads(json.dumps(envelope))
    inconsistent["evidence_summary"]["geometry_transition"] = "upright"
    failed = adapt_envelope(inconsistent, run_id="mapping-run", snapshot_id="snapshot-map",
                            decision_id="decision-map", request_digest="0" * 64)
    assert failed.evidence is None
    assert "SOURCE_CONFLICT:geometry_transition" in failed.missing_fields


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["answers"].pop("next_step"),
        lambda value: value["answers"]["fall_occurred"].update(type="choice"),
        lambda value: value["answers"]["fall_occurred"].update(noul=True),
        lambda value: value["answers"]["fall_occurred"].update(noul=float("nan")),
        lambda value: value["answers"]["fall_occurred"].update(noul=float("inf")),
        lambda value: value["answers"]["evidence_sufficiency"].update(choice="maybe"),
        lambda value: value["answers"]["evidence_sufficiency"]["probabilities"].pop("conflicting"),
        lambda value: value["answers"]["evidence_sufficiency"]["probabilities"].update(sufficient=0.5),
        lambda value: value["answers"]["next_step"].update(confidence=True),
        lambda value: value["answers"]["next_step"].update(type="noul"),
        lambda value: value["usage"].update(input_tokens=True),
        lambda value: value.update(model="another-model"),
        lambda value: value.update(request_id="unsafe request id with spaces"),
        lambda value: value.update(latency_ms=float("inf")),
    ],
)
def test_invalid_response_contracts_are_rejected(mutate):
    value = _response()
    mutate(value)
    with pytest.raises(DecisionProviderError, match="INVALID_RESPONSE_CONTRACT|MISMATCH|INVALID"):
        parse_response(json.dumps(value, allow_nan=True).encode(), expected_model="decision-model-preview")


def test_missing_answer_and_non_json_have_sanitized_errors():
    for body, expected in ((b"not json", "INVALID_JSON"), (b"[]", "INVALID_RESPONSE_CONTRACT")):
        with pytest.raises(DecisionProviderError) as captured:
            parse_response(body, expected_model="decision-model-preview")
        assert captured.value.code == expected
        assert "not json" not in str(captured.value)


def test_retry_after_http_date_is_parsed_as_a_bounded_delay(tmp_path):
    transport = QueueTransport(
        HTTPResponse(429, {"retry-after": formatdate(1002.0, usegmt=True)}, b""),
    )
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0,
                                      wall_clock=lambda: 1000.0, transport=transport)
    with pytest.raises(DecisionProviderError) as captured:
        provider.assess(synthetic_evidence(), deadline=13.0)
    assert captured.value.retry_after_s == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("status", "retry_after", "retryable", "code"),
    [
        (401, None, False, "HTTP_401"),
        (403, None, False, "HTTP_403"),
        (429, "3", True, "HTTP_429"),
        (500, None, True, "HTTP_500"),
        (503, None, True, "HTTP_503"),
        (302, None, False, "REDIRECT_REJECTED"),
    ],
)
def test_http_failures_are_classified_without_reading_error_body(tmp_path, status, retry_after, retryable, code):
    headers = {"x-request-id": "req-safe-10"}
    if retry_after:
        headers["retry-after"] = retry_after
    transport = QueueTransport(HTTPResponse(status, headers, FAKE_KEY.encode()))
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    with pytest.raises(DecisionProviderError) as captured:
        provider.assess(synthetic_evidence(), deadline=13.0)
    error = captured.value
    assert error.code == code
    assert error.retryable is retryable
    assert error.request_id == "req-safe-10"
    assert error.http_status == status
    assert FAKE_KEY not in str(error)
    assert FAKE_KEY not in repr(error)
    assert len(transport.calls) == 1


def test_transport_deadline_size_redirect_and_tls_guards():
    class Clock:
        value = 10.0

        def __call__(self):
            return self.value

    clock = Clock()
    constructed = []

    class Socket:
        def __init__(self):
            self.timeouts = []

        def settimeout(self, value):
            self.timeouts.append(value)

    class Response:
        status = 200

        def __init__(self, chunks, headers=None, advance=0.0):
            self.chunks = list(chunks)
            self.headers = headers or [("Content-Length", str(sum(map(len, self.chunks))))]
            self.advance = advance

        def getheaders(self):
            return self.headers

        def read1(self, amount):
            del amount
            clock.value += self.advance
            return self.chunks.pop(0) if self.chunks else b""

    class Connection:
        def __init__(self, response):
            self.response = response
            self.sock = Socket()
            self.requests = []
            self.closed = False

        def request(self, method, path, *, body, headers):
            self.requests.append((method, path, headers))

        def connect(self):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            self.closed = True

    connection = Connection(Response([b"{}", b""]))

    def factory(host, *, timeout, context):
        constructed.append((host, timeout, context))
        return connection

    transport = HTTPSJsonTransport(clock=clock, connection_factory=factory, max_response_bytes=3)
    response = transport.post_json(host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH,
                                   headers={"Authorization": "Bearer fake"}, body=b"{}", deadline=12.0)
    assert response.body == b"{}"
    assert len(connection.requests) == 1
    assert connection.requests[0][1] == API_PATH
    assert constructed[0][2].verify_mode == ssl.CERT_REQUIRED
    assert constructed[0][2].check_hostname is True
    assert connection.closed

    too_large = Connection(Response([b"1234"], [("Content-Length", "4")]))
    with pytest.raises(DecisionProviderError, match="RESPONSE_TOO_LARGE"):
        HTTPSJsonTransport(clock=clock, connection_factory=lambda *args, **kwargs: too_large,
                           max_response_bytes=3).post_json(
                               host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH,
                               headers={}, body=b"{}", deadline=12.0,
                           )
    assert too_large.requests == [("POST", API_PATH, {})]

    redirect = Connection(Response([], [("Location", "https://evil.example/"), ("Content-Length", "0")]))
    redirect.response.status = 302
    result = HTTPSJsonTransport(clock=clock, connection_factory=lambda *args, **kwargs: redirect).post_json(
        host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH, headers={"Authorization": "Bearer fake"},
        body=b"{}", deadline=12.0,
    )
    assert result.status == 302  # transport does not follow redirects; provider rejects this response
    assert len(redirect.requests) == 1
    with pytest.raises(DecisionProviderError, match="ENDPOINT_NOT_ALLOWED"):
        HTTPSJsonTransport(clock=clock).post_json(host="evil.example", path=API_PATH,
                                                   headers={}, body=b"{}", deadline=12.0)


def test_transport_timeout_and_slow_chunk_are_bounded():
    class Clock:
        value = 1.0

        def __call__(self):
            return self.value

    clock = Clock()

    def timeout_factory(*args, **kwargs):
        raise socket.timeout()

    with pytest.raises(DecisionProviderError, match="NETWORK_TIMEOUT"):
        HTTPSJsonTransport(clock=clock, connection_factory=timeout_factory).post_json(
            host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH, headers={}, body=b"{}", deadline=2.0,
        )

    class ReadTimeoutResponse:
        status = 200

        def getheaders(self):
            return [("Transfer-Encoding", "chunked")]

        def read1(self, amount):
            del amount
            raise socket.timeout()

    class ReadTimeoutConnection:
        sock = type("Socket", (), {"settimeout": lambda self, value: None})()

        def request(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def getresponse(self):
            return ReadTimeoutResponse()

        def close(self):
            pass

    with pytest.raises(DecisionProviderError, match="NETWORK_TIMEOUT"):
        HTTPSJsonTransport(clock=clock, connection_factory=lambda *a, **k: ReadTimeoutConnection()).post_json(
            host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH, headers={}, body=b"{}", deadline=2.0,
        )

    class Socket:
        def settimeout(self, value):
            self.timeout = value

    class SlowResponse:
        status = 200

        def getheaders(self):
            return [("Transfer-Encoding", "chunked")]

        def read1(self, amount):
            del amount
            clock.value += 0.6
            return b"x"

    class Connection:
        def __init__(self):
            self.sock = Socket()

        def request(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def getresponse(self):
            return SlowResponse()

        def close(self):
            pass

    with pytest.raises(DecisionProviderError, match="DEADLINE_EXCEEDED"):
        HTTPSJsonTransport(clock=clock, connection_factory=lambda *a, **k: Connection()).post_json(
            host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH, headers={}, body=b"{}", deadline=2.0,
        )

    no_connection = []
    with pytest.raises(DecisionProviderError, match="DEADLINE_EXCEEDED"):
        HTTPSJsonTransport(clock=clock, connection_factory=lambda *a, **k: no_connection.append(True)).post_json(
            host="space.cn-beijing.maas.aliyuncs.com", path=API_PATH, headers={}, body=b"{}", deadline=clock.value,
        )
    assert no_connection == []


def test_provider_does_not_send_when_deadline_has_already_passed(tmp_path):
    transport = QueueTransport(_ok_response())
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    with pytest.raises(DecisionProviderError, match="DEADLINE_EXCEEDED"):
        provider.assess(synthetic_evidence(), deadline=10.0)
    assert transport.calls == []


def test_cli_off_mode_ignores_environment_key_and_artifacts_are_redacted(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "cli-off"
    environment = dict(os.environ)
    environment["DASHSCOPE_API_KEY"] = FAKE_KEY
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "replay_decision_agent.py"), "--synthetic-scenarios",
         "--provider", "off", "--output", str(output)],
        cwd=root, env=environment, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert FAKE_KEY not in completed.stdout and FAKE_KEY not in completed.stderr
    for artifact in output.iterdir():
        if artifact.is_file():
            assert FAKE_KEY.encode() not in artifact.read_bytes()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["provider"] == "off"
    assert summary["model_called"] is False
    assert summary["network"]["transport_submissions"] == 0


def test_total_budget_survives_restart_and_final_failure_is_queryable(tmp_path):
    class FakeClock:
        wall = 1000.0
        mono = 40.0

        def wall_clock(self):
            return self.wall

        def monotonic_clock(self):
            return self.mono

        def advance(self, seconds):
            self.wall += seconds
            self.mono += seconds

    clock = FakeClock()
    config = _config(tmp_path, timeout=3.0, retries=1)
    transport = QueueTransport(HTTPResponse(503, {"x-request-id": "req-first", "retry-after": "0"}, b""))
    database = tmp_path / "budget.sqlite3"
    service_id = "service-shared-for-budget-test"
    envelope = synthetic_envelope(session_id="budget-session", stream_id="budget-stream", source_time_s=20.0,
                                  suffix="budget-restart")
    loop = DecisionLoop(database, run_id="budget-run", config=config, provider="aliyun", enable_network=True,
                        provider_transport=transport, monotonic_clock=clock.monotonic_clock,
                        wall_clock=clock.wall_clock, service_instance_id=service_id)
    receipt = loop.observe(envelope)
    first = loop.run_one()
    assert first["status"] == "retry_pending"
    task_after_first = loop.store.task(receipt["task_id"])
    fixed_deadline = task_after_first["network_deadline_at"]
    assert fixed_deadline == 1003.0
    assert task_after_first["network_remaining_s"] == 3.0
    loop.close()

    clock.advance(3.1)
    restarted = DecisionLoop(database, run_id="budget-run", config=config, provider="aliyun", enable_network=True,
                             provider_transport=transport, monotonic_clock=clock.monotonic_clock,
                             wall_clock=clock.wall_clock, service_instance_id=service_id)
    second = restarted.run_one()
    assert second["status"] == "unavailable"
    assert second["error_code"] == "DEADLINE_EXCEEDED"
    assert len(transport.calls) == 1
    result = restarted.store.get_result(receipt["decision_id"])["provider_result"]
    assert result["status"] == "unavailable"
    assert result["reason_codes"] == ["DEADLINE_EXCEEDED"]
    assert result["provider_metadata"] is None
    assert restarted.latest("budget-session", "budget-stream")["provider_status"] == "unavailable"
    decision = restarted.list_decisions("budget-session")[0]
    attempts = decision["attempt_history"]
    assert [item["attempt_no"] for item in attempts] == [1]
    assert attempts[0]["request_id"] == "req-first"
    assert attempts[0]["http_status"] == 503
    assert restarted.store.task(receipt["task_id"])["network_deadline_at"] == fixed_deadline
    restarted.close()


def test_restart_uses_persisted_remaining_budget_even_if_wall_clock_moves_back(tmp_path):
    class FakeClock:
        wall = 2000.0
        mono = 80.0

        def wall_clock(self):
            return self.wall

        def monotonic_clock(self):
            return self.mono

    clock = FakeClock()

    class AdvancingTransport(QueueTransport):
        def post_json(self, **kwargs):
            response = super().post_json(**kwargs)
            clock.mono += 1.0
            return response

    transport = AdvancingTransport(
        HTTPResponse(503, {"x-request-id": "req-before-restart", "retry-after": "0"}, b""),
        _ok_response(request_id="req-after-restart"),
    )
    database = tmp_path / "backward-clock.sqlite3"
    service_id = "same-service-for-recovery-check"
    config = _config(tmp_path)
    envelope = synthetic_envelope(session_id="clock-session", stream_id="clock-stream", source_time_s=30.0,
                                  suffix="clock-restart")
    first_loop = DecisionLoop(database, run_id="clock-run", config=config, provider="aliyun",
                              enable_network=True, provider_transport=transport,
                              monotonic_clock=clock.monotonic_clock, wall_clock=clock.wall_clock,
                              service_instance_id=service_id)
    receipt = first_loop.observe(envelope)
    assert first_loop.run_one()["status"] == "retry_pending"
    task = first_loop.store.task(receipt["task_id"])
    assert task["network_remaining_s"] == 2.0
    first_loop.close()

    clock.wall -= 0.5
    waiting = DecisionLoop(database, run_id="clock-run", config=config, provider="aliyun",
                           enable_network=True, provider_transport=transport,
                           monotonic_clock=clock.monotonic_clock, wall_clock=clock.wall_clock,
                           service_instance_id=service_id)
    assert waiting.run_one() is None
    assert len(transport.calls) == 1
    waiting.close()
    clock.wall = 2000.0
    clock.mono += 0.2
    restarted = DecisionLoop(database, run_id="clock-run", config=config, provider="aliyun",
                             enable_network=True, provider_transport=transport,
                             monotonic_clock=clock.monotonic_clock, wall_clock=clock.wall_clock,
                             service_instance_id=service_id)
    second = restarted.run_one()
    assert second["status"] == "accepted"
    assert len(transport.calls) == 2
    assert transport.calls[1]["deadline"] == pytest.approx(83.2)
    assert restarted.store.task(receipt["task_id"])["network_deadline_at"] == 2003.0
    restarted.close()


def test_transient_failure_retries_once_with_one_persisted_budget(tmp_path):
    transport = QueueTransport(
        HTTPResponse(503, {"x-request-id": "req-retry-1", "retry-after": "0"}, b""),
        _ok_response(request_id="req-retry-2"),
    )
    loop = DecisionLoop(tmp_path / "one-retry.sqlite3", run_id="one-retry-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport)
    envelope = synthetic_envelope(session_id="one-retry-session", stream_id="one-retry", source_time_s=10.0,
                                  suffix="one-retry")
    receipt = loop.observe(envelope)
    first = loop.run_one()
    deadline = loop.store.task(receipt["task_id"])["network_deadline_at"]
    assert first["status"] == "retry_pending"
    second = loop.run_one()
    assert second["status"] == "accepted"
    assert len(transport.calls) == 2
    assert loop.store.task(receipt["task_id"])["network_deadline_at"] == deadline
    history = loop.list_decisions("one-retry-session")[0]["attempt_history"]
    assert [item["request_id"] for item in history] == ["req-retry-1", "req-retry-2"]
    assert [item["http_status"] for item in history] == [503, 200]
    loop.close()


def test_retry_after_beyond_remaining_budget_does_not_retry(tmp_path):
    transport = QueueTransport(HTTPResponse(429, {"x-request-id": "req-too-late", "retry-after": "4"}, b""))
    loop = DecisionLoop(
        tmp_path / "retry-after.sqlite3", run_id="retry-run", config=_config(tmp_path),
        provider="aliyun", enable_network=True, provider_transport=transport,
    )
    envelope = synthetic_envelope(session_id="retry-session", stream_id="retry-stream", source_time_s=5.0,
                                  suffix="retry-after-budget")
    receipt = loop.observe(envelope)
    result = loop.run_one()
    assert result["status"] == "unavailable"
    assert len(transport.calls) == 1
    record = loop.store.get_result(receipt["decision_id"])["provider_result"]
    assert record["reason_codes"] == ["HTTP_429"]
    assert record["provider_metadata"]["request_id"] == "req-too-late"
    assert loop.store.task(receipt["task_id"])["status"] == "completed"
    loop.close()


def test_durable_worker_persists_result_duplicate_dedupes_and_quality_stays_local(tmp_path):
    transport = QueueTransport(_ok_response(), _ok_response())
    config = _config(tmp_path)
    database = tmp_path / "worker-flow.sqlite3"
    service_id = "service-worker-flow"
    owner = DecisionLoop(database, run_id="worker-run", config=config, provider="aliyun", enable_network=True,
                         provider_transport=transport, service_instance_id=service_id)
    normal = synthetic_envelope(session_id="worker-session", stream_id="normal", source_time_s=20.0,
                                suffix="worker-normal")
    receipt = owner.observe(normal)
    with DecisionWorker(str(database), config=config, provider="aliyun", enable_network=True,
                        provider_transport=transport, service_instance_id=service_id) as worker:
        outcome = worker.run_one(timeout=5)
        assert outcome["status"] == "accepted"
        before_feedback = owner.store.get_result(receipt["decision_id"])["provider_result"]
        feedback = owner.feedback(receipt["decision_id"], label="uncertain", request_id="human-review-01",
                                  expected_version=0)
        assert feedback["provider_state_changed"] is False
        assert owner.store.get_result(receipt["decision_id"])["provider_result"] == before_feedback
        assert owner.store.list_feedback(receipt["decision_id"])[0]["label"] == "uncertain"
        duplicate = owner.observe(normal)
        assert duplicate["status"] == "duplicate"
        assert worker.run_one(timeout=0.1) is None

        insufficient = synthetic_envelope(session_id="worker-session", stream_id="insufficient", source_time_s=20.0,
                                          suffix="worker-insufficient", sufficient=False)
        local_receipt = owner.observe(insufficient)
        local_outcome = worker.run_one(timeout=5)
        assert local_outcome["status"] == "accepted"
        assert len(transport.calls) == 1
        local_result = owner.store.get_result(local_receipt["decision_id"])["provider_result"]
        assert local_result["status"] == "insufficient"
        assert owner.latest("worker-session", "normal")["current_applicable"] is True
        assert owner.latest("worker-session", "insufficient")["fall_probability"] is None
    saved = owner.store.get_result(receipt["decision_id"])["provider_result"]
    assert saved["provider_metadata"]["request_id"] == "req-ok-001"
    assert owner.store.attempts(receipt["task_id"])[0]["request_id"] == "req-ok-001"
    owner.close()


def test_failed_error_body_and_fake_key_never_enter_database_or_artifacts(tmp_path):
    transport = QueueTransport(HTTPResponse(401, {"x-request-id": "req-auth-fail"}, FAKE_KEY.encode()))
    loop = DecisionLoop(tmp_path / "redaction.sqlite3", run_id="redaction-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport)
    envelope = synthetic_envelope(session_id="redaction-session", stream_id="redaction", source_time_s=20.0,
                                  suffix="redaction")
    receipt = loop.observe(envelope)
    result = loop.run_one()
    assert result["status"] == "unavailable"
    loop.store.db.execute("PRAGMA wal_checkpoint(FULL)")
    loop.close()
    for path in tmp_path.glob("redaction.sqlite3*"):
        assert FAKE_KEY.encode() not in path.read_bytes()
    assert FAKE_KEY not in json.dumps(result, sort_keys=True)
    assert FAKE_KEY not in json.dumps(receipt, sort_keys=True)


@pytest.mark.parametrize(
    ("failure_response", "expected_status", "expected_reason"),
    [
        (HTTPResponse(401, {"x-request-id": "req-r1-auth"}, b"secret-body"), "unavailable", "HTTP_401"),
        (HTTPResponse(200, {}, json.dumps({**_response(), "answers": {
            **_response()["answers"], "fall_occurred": {"type": "noul", "noul": True},
        }}).encode()), "unavailable", "INVALID_RESPONSE_CONTRACT"),
        (_ok_response(sufficiency="insufficient", next_step="observe_more"), "insufficient",
         "MODEL_EVIDENCE_NOT_SUFFICIENT"),
        (_ok_response(sufficiency="conflicting", next_step="request_review"), "insufficient",
         "MODEL_EVIDENCE_NOT_SUFFICIENT"),
    ],
)
def test_newer_nonready_terminal_blocks_older_ready_but_keeps_history(
    tmp_path, failure_response, expected_status, expected_reason,
):
    transport = QueueTransport(_ok_response(fall=0.72), failure_response)
    loop = DecisionLoop(tmp_path / f"r1-{expected_reason}.sqlite3", run_id="r1-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    first = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                                            suffix="r1-ready-before"))
    assert loop.run_one()["status"] == "accepted"
    second = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                                             suffix="r1-terminal-after"))
    outcome = loop.run_one()
    assert outcome["status"] == ("accepted" if expected_status == "insufficient" else "unavailable")

    view = loop.latest("r1-session", "r1-stream")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert view["provider_status"] == expected_status
    assert view["latest_result"]["decision_id"] == second["decision_id"]
    assert expected_reason in view["latest_result"]["reason_codes"]
    assert view["historical_result"]["decision_id"] == first["decision_id"]
    assert len(transport.calls) == 2
    loop.close()


def test_newer_ready_terminal_reopens_projection_after_failure(tmp_path):
    transport = QueueTransport(
        HTTPResponse(401, {"x-request-id": "req-r1-before-ready"}, b""),
        _ok_response(fall=0.64),
    )
    loop = DecisionLoop(tmp_path / "r1-ready-after-failure.sqlite3", run_id="r1-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    failed = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                                             suffix="r1-failure-first"))
    assert loop.run_one()["status"] == "unavailable"
    ready = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                                            suffix="r1-ready-after"))
    assert loop.run_one()["status"] == "accepted"
    view = loop.latest("r1-session", "r1-stream")
    assert view["current_applicable"] is True
    assert view["fall_probability"] == pytest.approx(0.64)
    assert view["current"]["decision_id"] == ready["decision_id"]
    assert view["latest_result"]["status"] == "ready"
    assert loop.store.get_result(failed["decision_id"])["provider_result"]["status"] == "unavailable"
    assert len(transport.calls) == 2
    loop.close()


def test_source_order_wins_when_newer_failure_completes_before_older_ready(tmp_path):
    transport = QueueTransport(
        _ok_response(fall=0.72),
        HTTPResponse(401, {"x-request-id": "req-r1-newer-first"}, b""),
    )
    config = replace(_config(tmp_path), min_interval_s=1.0)
    loop = DecisionLoop(tmp_path / "r1-out-of-order.sqlite3", run_id="r1-run", config=config,
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    older = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                                            suffix="r1-older-ready"))
    assert loop.run_one()["status"] == "accepted"
    newer = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                                            suffix="r1-newer-failure"))
    assert newer["task_id"] != older["task_id"]
    assert loop.run_one()["status"] == "unavailable"
    newer_completed_at = loop.store.get_result(newer["decision_id"])["completed_at"]
    # Simulate a different finalization order; source time remains the authority.
    loop.store.db.execute("UPDATE decision_results SET completed_at=? WHERE decision_id=?",
                          (newer_completed_at - 1.0, newer["decision_id"]))
    view_after_out_of_order_completion = loop.latest("r1-session", "r1-stream")
    assert view_after_out_of_order_completion["current_applicable"] is False
    assert view_after_out_of_order_completion["fall_probability"] is None
    assert view_after_out_of_order_completion["latest_result"]["status"] == "unavailable"
    assert view_after_out_of_order_completion["historical_result"]["decision_id"] == older["decision_id"]
    assert len(transport.calls) == 2
    loop.close()


def test_pending_latest_task_reuses_only_newest_compatible_ready_terminal(tmp_path):
    transport = QueueTransport(_ok_response(fall=0.58), HTTPResponse(401, {}, b""))
    loop = DecisionLoop(tmp_path / "r1-pending-policy.sqlite3", run_id="r1-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    first = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                                            suffix="r1-pending-ready"))
    assert loop.run_one()["status"] == "accepted"
    pending = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                                              suffix="r1-pending-newer"))
    view = loop.latest("r1-session", "r1-stream")
    assert view["current_applicable"] is True
    assert view["fall_probability"] == pytest.approx(0.58)
    assert view["latest_task_status"] == "pending"
    assert view["pending_policy"] == "reuse_latest_compatible_ready_within_freshness"
    assert view["latest_result"]["decision_id"] == first["decision_id"]
    assert loop.run_one()["status"] == "unavailable"
    after_failure = loop.latest("r1-session", "r1-stream")
    assert after_failure["current_applicable"] is False
    assert after_failure["fall_probability"] is None
    assert after_failure["latest_result"]["decision_id"] == pending["decision_id"]
    assert len(transport.calls) == 2
    loop.close()


def test_newer_cancelled_terminal_without_result_blocks_older_ready(tmp_path):
    transport = QueueTransport(_ok_response(fall=0.57))
    loop = DecisionLoop(tmp_path / "r1-cancelled-terminal.sqlite3", run_id="r1-run", config=_config(tmp_path),
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    first = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                                            suffix="r1-ready-before-cancel"))
    assert loop.run_one()["status"] == "accepted"
    cancelled = loop.observe(synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                                                suffix="r1-cancelled-later"))
    loop.store.db.execute("UPDATE decision_tasks SET status='cancelled',reason='TEST_TERMINAL_CANCEL' "
                          "WHERE task_id=?", (cancelled["task_id"],))
    view = loop.latest("r1-session", "r1-stream")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert view["latest_result"] is None
    assert view["provider_status"] == "cancelled"
    assert "LATEST_DECISION_TERMINAL_WITHOUT_RESULT" in view["stale_reason_codes"]
    assert "TEST_TERMINAL_CANCEL" in view["stale_reason_codes"]
    assert view["historical_result"]["decision_id"] == first["decision_id"]
    loop.close()


def test_latest_projection_isolated_by_provider_model_and_task_result_binding(tmp_path):
    config = replace(_config(tmp_path), min_interval_s=1.0, max_source_age_s=5.0)
    database = tmp_path / "configuration-isolation.sqlite3"
    service_id = "one-service-several-query-configs"
    aliyun_transport = QueueTransport(_ok_response(fall=0.72))
    aliyun = DecisionLoop(database, run_id="configuration-run", config=config, provider="aliyun",
                          enable_network=True, provider_transport=aliyun_transport,
                          service_instance_id=service_id)
    ready = aliyun.observe(synthetic_envelope(session_id="configuration-session", stream_id="same-target",
                                              source_time_s=100, suffix="aliyun-ready"))
    assert aliyun.run_one()["status"] == "accepted"
    aliyun.close()

    for provider, model in (
        ("off", config.model),
        ("mock", config.model),
        ("mock", "different-mock-model"),
    ):
        query_config = replace(config, provider=provider, model=model)
        with DecisionLoop(database, run_id="configuration-run", config=query_config, provider=provider,
                          service_instance_id=service_id) as query:
            view = query.latest("configuration-session", "same-target")
            assert view["current_applicable"] is False
            assert view["fall_probability"] is None
            assert view["historical_result"]["decision_id"] == ready["decision_id"]
            assert view["configured_provider"] == provider
            assert view["configured_model"] == model
            if provider == "off":
                assert "PROVIDER_MODE_OFF" in view["stale_reason_codes"]


def test_other_configuration_failure_does_not_mask_compatible_ready_result(tmp_path):
    config = replace(_config(tmp_path), min_interval_s=1.0, max_source_age_s=5.0)
    database = tmp_path / "configuration-failure-isolation.sqlite3"
    service_id = "one-service-config-failure"
    aliyun = DecisionLoop(database, run_id="configuration-run", config=config, provider="aliyun",
                          enable_network=True, provider_transport=QueueTransport(_ok_response(fall=0.67)),
                          service_instance_id=service_id)
    ready = aliyun.observe(synthetic_envelope(session_id="configuration-session", stream_id="same-target",
                                               source_time_s=100, suffix="aliyun-ready-before-mock-failure"))
    assert aliyun.run_one()["status"] == "accepted"
    aliyun.close()

    mock_config = replace(config, provider="mock", model="different-mock-model")

    class MockUnavailable:
        def assess(self, evidence, *, deadline):
            result = MockDecisionProvider(model=mock_config.model, clock=lambda: 1.0).assess(
                evidence, deadline=deadline,
            )
            value = result.model_dump(mode="json")
            value.update(
                status="unavailable", fall_probability=None, fall_probability_raw=None,
                probability_kind=None, evidence_sufficiency="not_assessed", next_step=None,
                reason_codes=["INJECTED_MOCK_FAILURE"],
            )
            return ProviderResult.model_validate(value)

    mock = DecisionLoop(database, run_id="configuration-run", config=mock_config, provider="mock",
                        provider_impl=MockUnavailable(), service_instance_id=service_id)
    failed = mock.observe(synthetic_envelope(session_id="configuration-session", stream_id="same-target",
                                             source_time_s=102, suffix="mock-failure-after-aliyun"))
    mock_outcome = mock.run_one()
    assert mock_outcome["status"] == "accepted", mock_outcome
    assert mock.store.get_result(failed["decision_id"])["provider_result"]["status"] == "unavailable"
    mock.close()

    with DecisionLoop(database, run_id="configuration-run", config=config, provider="aliyun",
                      enable_network=True, provider_transport=QueueTransport(),
                      service_instance_id=service_id) as query:
        view = query.latest("configuration-session", "same-target")
        assert view["current_applicable"] is True
        assert view["fall_probability"] == pytest.approx(0.67)
        assert view["current"]["decision_id"] == ready["decision_id"]
        assert view["latest_result"]["decision_id"] == ready["decision_id"]
        assert "INJECTED_MOCK_FAILURE" not in view["stale_reason_codes"]


def test_latest_projection_rejects_result_whose_provider_model_binding_was_tampered(tmp_path):
    config = _config(tmp_path)
    database = tmp_path / "tampered-result-binding.sqlite3"
    service_id = "binding-service"
    loop = DecisionLoop(database, run_id="binding-run", config=config, provider="aliyun",
                        enable_network=True, provider_transport=QueueTransport(_ok_response()),
                        service_instance_id=service_id)
    receipt = loop.observe(synthetic_envelope(session_id="binding-session", stream_id="binding",
                                              source_time_s=20, suffix="binding-result"))
    assert loop.run_one()["status"] == "accepted"
    result = loop.store.get_result(receipt["decision_id"])["provider_result"]
    result["model"] = "tampered-model"
    loop.store.db.execute("UPDATE decision_results SET result_json=? WHERE decision_id=?",
                          (json.dumps(result, sort_keys=True), receipt["decision_id"]))
    view = loop.latest("binding-session", "binding")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert "RESULT_CONFIGURATION_OR_TASK_BINDING_MISMATCH" in view["stale_reason_codes"]
    loop.close()


def test_mock_ready_result_is_not_current_in_aliyun_configuration(tmp_path):
    database = tmp_path / "mock-to-aliyun.sqlite3"
    service_id = "mock-then-aliyun-query"
    mock_config = replace(_config(tmp_path), provider="mock", model="test-mock")
    with DecisionLoop(database, run_id="switch-run", config=mock_config, provider="mock",
                      service_instance_id=service_id) as mock:
        receipt = mock.observe(synthetic_envelope(session_id="switch-session", stream_id="switch",
                                                  source_time_s=20, suffix="mock-ready"))
        assert mock.run_one()["status"] == "accepted"
    aliyun_config = replace(_config(tmp_path), provider="aliyun")
    with DecisionLoop(database, run_id="switch-run", config=aliyun_config, provider="aliyun",
                      enable_network=True, provider_transport=QueueTransport(),
                      service_instance_id=service_id) as aliyun:
        view = aliyun.latest("switch-session", "switch")
        assert view["current_applicable"] is False
        assert view["fall_probability"] is None
        assert view["historical_result"]["decision_id"] == receipt["decision_id"]


@pytest.mark.parametrize(("phase", "expected_http_requests"), [
    ("connect", 0), ("headers", 1), ("body", 1),
])
def test_isolated_http_deadline_terminates_blocked_connection_and_reaps_process(
    phase, expected_http_requests,
):
    request_counter = multiprocessing.get_context("fork").RawValue("i", 0)

    class BlockingConnection:
        sock = None

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def connect(self):
            if phase == "connect":
                time.sleep(5.0)

        def request(self, *args, **kwargs):
            del args, kwargs
            request_counter.value += 1

        def getresponse(self):
            if phase == "headers":
                time.sleep(5.0)
            return type("Response", (), {
                "status": 200,
                "getheaders": lambda self: [("Transfer-Encoding", "chunked")],
                "read1": lambda self, amount: (time.sleep(5.0), b"")[1],
            })()

        def close(self):
            pass

    transport = HTTPSJsonTransport(connection_factory=BlockingConnection)
    start = time.monotonic()
    with pytest.raises(DecisionProviderError) as captured:
        transport.post_json(host="synthetic-test-space.cn-beijing.maas.aliyuncs.com", path=API_PATH,
                            headers={}, body=b"{}", deadline=start + 0.15)
    elapsed = time.monotonic() - start
    assert captured.value.code == "DEADLINE_EXCEEDED"
    assert elapsed < 0.45
    assert request_counter.value == expected_http_requests
    assert not [child for child in multiprocessing.active_children()
                if child.name == "fallpre-aliyun-http"]


def test_blocked_transport_worker_settles_and_closes_before_shutdown_returns(tmp_path):
    request_counter = multiprocessing.get_context("fork").RawValue("i", 0)
    request_started = threading.Event()

    class BlockingHeadersConnection:
        sock = None

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            del args, kwargs
            request_counter.value += 1

        def getresponse(self):
            time.sleep(5.0)

        def close(self):
            pass

    class CountingBoundedTransport:
        def __init__(self):
            self.inner = HTTPSJsonTransport(connection_factory=BlockingHeadersConnection)
            self.calls = 0

        def post_json(self, **kwargs):
            self.calls += 1
            request_started.set()
            return self.inner.post_json(**kwargs)

    config = _config(tmp_path, timeout=0.15, retries=1)
    database = tmp_path / "blocked-worker.sqlite3"
    service_id = "blocked-worker-service"
    transport = CountingBoundedTransport()
    owner = DecisionLoop(database, run_id="blocked-run", config=config, provider="aliyun", enable_network=True,
                         provider_transport=transport, service_instance_id=service_id)
    receipt = owner.observe(synthetic_envelope(session_id="blocked-session", stream_id="blocked",
                                               source_time_s=20, suffix="slow-headers"))
    worker = DecisionWorker(str(database), config=config, provider="aliyun", enable_network=True,
                            provider_transport=transport, service_instance_id=service_id)
    worker_result = []
    worker_error = []

    def request_work():
        try:
            worker_result.append(worker.run_one(timeout=2.0))
        except BaseException as error:
            worker_error.append(error)

    caller = threading.Thread(target=request_work, daemon=True)
    started = time.monotonic()
    caller.start()
    assert request_started.wait(1.0)
    worker.close(timeout=1.0)
    caller.join(timeout=1.0)
    elapsed = time.monotonic() - started

    assert not caller.is_alive()
    assert not worker.thread.is_alive()
    assert not worker_error
    assert len(worker_result) == 1
    assert worker_result[0]["status"] == "unavailable"
    assert worker_result[0]["error_code"] == "DEADLINE_EXCEEDED"
    assert elapsed < 0.8
    assert transport.calls == 1
    assert request_counter.value == 1
    task = owner.store.task(receipt["task_id"])
    assert task["status"] == "completed"
    assert len(owner.store.attempts(receipt["task_id"])) == 1
    result = owner.store.get_result(receipt["decision_id"])["provider_result"]
    assert result["status"] == "unavailable"
    assert result["fall_probability"] is None
    assert owner.latest("blocked-session", "blocked")["provider_status"] == "unavailable"
    assert not [child for child in multiprocessing.active_children()
                if child.name == "fallpre-aliyun-http"]
    owner.close()


def test_different_run_failure_does_not_hide_pending_targets_own_ready_result(tmp_path):
    transport = QueueTransport(_ok_response(fall=0.61), HTTPResponse(401, {}, b""))
    config = replace(_config(tmp_path), max_source_age_s=10.0)
    loop = DecisionLoop(tmp_path / "r1-run-isolation.sqlite3", run_id="unused", config=config,
                        provider="aliyun", enable_network=True, provider_transport=transport,
                        service_instance_id="r1-service")
    own_ready = loop.observe(
        synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=100,
                           suffix="r1-own-run-ready"),
        run_id="target-run",
    )
    assert loop.run_one()["status"] == "accepted"
    other_run_failure = loop.observe(
        synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=102,
                           suffix="r1-other-run-failure"),
        run_id="other-run",
    )
    assert loop.run_one()["status"] == "unavailable"
    pending_target = loop.observe(
        synthetic_envelope(session_id="r1-session", stream_id="r1-stream", source_time_s=104,
                           suffix="r1-target-pending"),
        run_id="target-run",
    )
    view = loop.latest("r1-session", "r1-stream")
    assert view["current_applicable"] is True
    assert view["current"]["decision_id"] == own_ready["decision_id"]
    assert view["latest_task_status"] == "pending"
    assert view["pending_policy"] == "reuse_latest_compatible_ready_within_freshness"
    assert "HTTP_401" not in view["stale_reason_codes"]
    assert view["latest_result"]["decision_id"] == own_ready["decision_id"]
    assert other_run_failure["decision_id"] != pending_target["decision_id"]
    assert len(transport.calls) == 2
    loop.close()


@pytest.mark.parametrize(
    ("failure", "diagnostic_code", "field_path"),
    [
        ("noul_boolean", "NUMBER_TYPE", "answers.fall_occurred.noul"),
        ("distribution_sum", "DISTRIBUTION_SUM_INVALID", "answers.next_step.probabilities"),
        ("latency_type", "LATENCY_TYPE_INVALID", "latency_ms"),
    ],
)
def test_response_contract_diagnostics_are_allowlisted_and_field_specific(
    tmp_path, failure, diagnostic_code, field_path,
):
    value = _response()
    if failure == "noul_boolean":
        value["answers"]["fall_occurred"]["noul"] = "RAW-SECRET-MARKER"
    elif failure == "distribution_sum":
        value["answers"]["next_step"]["probabilities"]["emit_result"] = 0.5
    else:
        value["latency_ms"] = "RAW-SECRET-MARKER"
    body = json.dumps(value).encode()
    transport = QueueTransport(HTTPResponse(200, {"x-request-id": "req-diag-safe"}, body))
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=lambda: 10.0, transport=transport)
    with pytest.raises(DecisionProviderError) as captured:
        provider.assess(synthetic_evidence(), deadline=13.0)
    error = captured.value
    assert error.code == "INVALID_RESPONSE_CONTRACT"
    assert error.diagnostic_code == diagnostic_code
    assert error.field_path == field_path
    assert error.request_id == "req-ok-001"
    assert "RAW-SECRET-MARKER" not in str(error)
    assert "RAW-SECRET-MARKER" not in repr(error)
    assert error.diagnostic_code in {
        "ANSWER_SET_INVALID", "FIELD_MISSING", "FIELD_TYPE", "ANSWER_TYPE",
        "ENUM_VALUE_INVALID", "NUMBER_TYPE", "NUMBER_RANGE",
        "DISTRIBUTION_KEYS_INVALID", "DISTRIBUTION_SUM_INVALID",
        "USAGE_TOKENS_INVALID", "LATENCY_TYPE_INVALID", "LATENCY_RANGE_INVALID",
        "MODEL_ECHO_MISMATCH", "REQUEST_ID_MISSING_OR_INVALID", "MODEL_VALIDATION_FAILED",
    }


def test_contract_diagnostic_survives_attempt_terminal_query_and_artifact_rows(tmp_path):
    value = _response()
    value["latency_ms"] = "RAW-SECRET-MARKER"
    transport = QueueTransport(HTTPResponse(
        200, {"x-request-id": "req-diagnostic-persisted"}, json.dumps(value).encode(),
    ))
    loop = DecisionLoop(tmp_path / "diagnostic-persisted.sqlite3", run_id="diagnostic-run",
                        config=_config(tmp_path), provider="aliyun", enable_network=True,
                        provider_transport=transport, service_instance_id="diagnostic-service")
    envelope = synthetic_envelope(session_id="diagnostic-session", stream_id="diagnostic", source_time_s=20.0,
                                  suffix="diagnostic-persist")
    receipt = loop.observe(envelope)
    outcome = loop.run_one()
    assert outcome["status"] == "unavailable"
    assert outcome["error_code"] == "INVALID_RESPONSE_CONTRACT"
    assert outcome["diagnostic_code"] == "LATENCY_TYPE_INVALID"
    assert outcome["field_path"] == "latency_ms"

    attempt = loop.store.attempts(receipt["task_id"])[0]
    assert attempt["diagnostic_code"] == "LATENCY_TYPE_INVALID"
    assert attempt["field_path"] == "latency_ms"
    decision = loop.list_decisions("diagnostic-session")[0]
    assert decision["attempt_history"][0]["diagnostic_code"] == "LATENCY_TYPE_INVALID"
    terminal = loop.store.get_result(receipt["decision_id"])["provider_result"]
    assert terminal["status"] == "unavailable"
    assert terminal["provider_metadata"]["diagnostic_code"] == "LATENCY_TYPE_INVALID"
    assert terminal["provider_metadata"]["field_path"] == "latency_ms"
    projection = loop.latest("diagnostic-session", "diagnostic")
    assert projection["latest_result"]["provider_metadata"]["diagnostic_code"] == "LATENCY_TYPE_INVALID"
    loop.store.db.execute("PRAGMA wal_checkpoint(FULL)")
    loop.close()
    for path in tmp_path.glob("diagnostic-persisted.sqlite3*"):
        assert b"RAW-SECRET-MARKER" not in path.read_bytes()


@pytest.mark.parametrize(("returned_at", "expected"), [(12.999, "ready"), (13.0, "deadline"), (13.001, "deadline")])
def test_adapter_rejects_equal_or_late_transport_response(tmp_path, returned_at, expected):
    class Clock:
        value = 10.0

        def __call__(self):
            return self.value

    clock = Clock()

    class ReturningTransport(QueueTransport):
        def post_json(self, **kwargs):
            response = super().post_json(**kwargs)
            clock.value = returned_at
            return response

    transport = ReturningTransport(HTTPResponse(
        200, {"x-request-id": "req-deadline-adapter"}, _ok_response().body,
    ))
    provider = AliyunDecisionProvider(config=_config(tmp_path), clock=clock, transport=transport)
    if expected == "ready":
        result = provider.assess(synthetic_evidence(), deadline=13.0)
        assert result.status == "ready"
    else:
        with pytest.raises(DecisionProviderError) as captured:
            provider.assess(synthetic_evidence(), deadline=13.0)
        assert captured.value.code == "DEADLINE_EXCEEDED"
        assert captured.value.request_id == "req-deadline-adapter"
        assert captured.value.http_status == 200
        assert captured.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.parametrize(("phase", "advance"), [("headers", 2.0), ("eof", 2.0), ("body", 2.0)])
def test_transport_checks_total_deadline_after_headers_and_every_read_including_eof(tmp_path, phase, advance):
    class Clock:
        value = 10.0

        def __call__(self):
            return self.value

    clock = Clock()

    class Socket:
        def settimeout(self, value):
            self.timeout = value

    class Response:
        status = 200

        def getheaders(self):
            if phase == "headers":
                clock.value += advance
            return [("Transfer-Encoding", "chunked"), ("X-Request-Id", "req-header-deadline")]

        def read1(self, amount):
            del amount
            if phase == "eof":
                clock.value += advance
                return b""
            if phase == "body":
                clock.value += advance
                return b"{}"
            return b""

    class Connection:
        sock = Socket()
        requests = 0

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            del args, kwargs
            self.requests += 1

        def getresponse(self):
            return Response()

        def close(self):
            pass

    connection = Connection()
    transport = HTTPSJsonTransport(clock=clock, connection_factory=lambda *a, **k: connection)
    with pytest.raises(DecisionProviderError) as captured:
        transport.post_json(host="synthetic-test-space.cn-beijing.maas.aliyuncs.com", path=API_PATH,
                            headers={}, body=b"{}", deadline=12.0)
    assert captured.value.code == "DEADLINE_EXCEEDED"
    assert captured.value.request_id == "req-header-deadline"
    assert captured.value.http_status == 200
    assert connection.requests == 1


def test_worker_persists_deadline_late_success_as_unavailable_without_retry(tmp_path):
    class Clock:
        wall = 1000.0
        mono = 10.0

        def wall_clock(self):
            return self.wall

        def monotonic_clock(self):
            return self.mono

    clock = Clock()

    class LateTransport(QueueTransport):
        def post_json(self, **kwargs):
            response = super().post_json(**kwargs)
            clock.wall = 1003.0
            clock.mono = 13.0
            return response

    response = HTTPResponse(200, {"x-request-id": "req-worker-deadline"}, _ok_response().body)
    transport = LateTransport(response)
    database = tmp_path / "worker-late-deadline.sqlite3"
    config = _config(tmp_path, timeout=3.0, retries=1)
    loop = DecisionLoop(database, run_id="deadline-run", config=config, provider="aliyun", enable_network=True,
                        provider_transport=transport, monotonic_clock=clock.monotonic_clock,
                        wall_clock=clock.wall_clock, service_instance_id="deadline-service")
    receipt = loop.observe(synthetic_envelope(session_id="deadline-session", stream_id="deadline", source_time_s=20.0,
                                              suffix="late-valid-200"))
    outcome = loop.run_one()
    assert outcome["status"] == "unavailable"
    assert outcome["error_code"] == "DEADLINE_EXCEEDED"
    assert len(transport.calls) == 1
    task = loop.store.task(receipt["task_id"])
    assert task["status"] == "completed"
    result = loop.store.get_result(receipt["decision_id"])["provider_result"]
    assert result["status"] == "unavailable"
    assert result["fall_probability"] is None
    assert result["reason_codes"] == ["DEADLINE_EXCEEDED"]
    assert result["provider_metadata"]["request_id"] == "req-worker-deadline"
    history = loop.list_decisions("deadline-session")[0]["attempt_history"]
    assert len(history) == 1
    assert history[0]["error_code"] == "DEADLINE_EXCEEDED"
    assert history[0]["request_id"] == "req-worker-deadline"
    assert loop.latest("deadline-session", "deadline")["fall_probability"] is None
    loop.close()


def test_worker_rechecks_deadline_after_provider_before_accepting_success(tmp_path):
    class Clock:
        wall = 1000.0
        mono = 10.0

        def wall_clock(self):
            return self.wall

        def monotonic_clock(self):
            return self.mono

    clock = Clock()
    config = _config(tmp_path, timeout=3.0, retries=1)

    class LateProvider:
        def assess(self, evidence, *, deadline):
            del deadline
            clock.wall = 1003.0
            clock.mono = 13.0
            return AliyunDecisionProvider(config=config, clock=lambda: 10.0,
                                          transport=QueueTransport(_ok_response())).assess(
                evidence, deadline=14.0,
            )

    loop = DecisionLoop(tmp_path / "worker-acceptance-deadline.sqlite3", run_id="acceptance-deadline",
                        config=config, provider="aliyun", enable_network=True,
                        provider_impl=LateProvider(), monotonic_clock=clock.monotonic_clock,
                        wall_clock=clock.wall_clock, service_instance_id="acceptance-service")
    receipt = loop.observe(synthetic_envelope(session_id="acceptance-session", stream_id="acceptance",
                                              source_time_s=20.0, suffix="provider-ignored-deadline"))
    outcome = loop.run_one()
    assert outcome["status"] == "unavailable"
    assert outcome["error_code"] == "DEADLINE_EXCEEDED"
    result = loop.store.get_result(receipt["decision_id"])["provider_result"]
    assert result["status"] == "unavailable"
    assert result["fall_probability"] is None
    assert result["provider_metadata"]["request_id"] == "req-ok-001"
    assert loop.store.attempts(receipt["task_id"])[0]["status"] == "unavailable"
    assert loop.latest("acceptance-session", "acceptance")["current_applicable"] is False
    loop.close()
