"""Constrained Aliyun Decision Model transport and response adapter.

Each ``assess`` call sends at most one HTTP request. Retries belong to the
durable DecisionLoop scheduler so its persisted task budget is authoritative.
"""
from __future__ import annotations

import http.client
import json
import math
import multiprocessing
import re
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from pydantic import ValidationError

from ..decision_config import DecisionConfig
from ..decision_contracts import (
    ChoiceAnswer,
    DecisionAnswers,
    DecisionEvidence,
    DecisionStatus,
    EvidenceSufficiency,
    NextStep,
    ProbabilityKind,
    ProviderMetadata,
    ProviderMode,
    ProviderResult,
)
from ..decision_state import build_decision_state, decision_questions
from ..decision_trace import trace_decision

API_PATH = "/compatible-mode/v1/systemone"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
PROBABILITY_SUM_TOLERANCE = 1e-3
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ALLOWED_HOST = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com$"
)
_CHOICE_OPTIONS = {
    "evidence_sufficiency": {"sufficient", "insufficient", "conflicting"},
    "next_step": {"emit_result", "observe_more", "request_review"},
}
_DIAGNOSTIC_CODES = {
    "ANSWER_SET_INVALID", "FIELD_MISSING", "FIELD_TYPE", "ANSWER_TYPE",
    "ENUM_VALUE_INVALID", "NUMBER_TYPE", "NUMBER_RANGE",
    "DISTRIBUTION_KEYS_INVALID", "DISTRIBUTION_SUM_INVALID",
    "USAGE_TOKENS_INVALID", "LATENCY_TYPE_INVALID", "LATENCY_RANGE_INVALID",
    "MODEL_ECHO_MISMATCH", "REQUEST_ID_MISSING_OR_INVALID", "MODEL_VALIDATION_FAILED",
}
_DIAGNOSTIC_PATHS = {
    "model", "request_id", "answers", "answers.fall_occurred",
    "answers.fall_occurred.type", "answers.fall_occurred.noul",
    "answers.evidence_sufficiency", "answers.evidence_sufficiency.type",
    "answers.evidence_sufficiency.choice", "answers.evidence_sufficiency.probabilities",
    "answers.evidence_sufficiency.confidence", "answers.next_step",
    "answers.next_step.type", "answers.next_step.choice", "answers.next_step.probabilities",
    "answers.next_step.confidence", "usage", "usage.input_tokens", "latency_ms",
}


class DecisionProviderError(RuntimeError):
    """Sanitized provider failure; never carries response bodies or secrets."""

    def __init__(self, code: str, *, retryable: bool = False, retry_after_s: float | None = None,
                 request_id: str | None = None, http_status: int | None = None,
                 client_latency_ms: float | None = None, diagnostic_code: str | None = None,
                 field_path: str | None = None, input_tokens: int | None = None,
                 service_latency_ms: float | None = None) -> None:
        self.code = code
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.request_id = request_id
        self.http_status = http_status
        self.client_latency_ms = client_latency_ms
        self.diagnostic_code = diagnostic_code if diagnostic_code in _DIAGNOSTIC_CODES else None
        self.field_path = field_path if field_path in _DIAGNOSTIC_PATHS else None
        self.input_tokens = input_tokens if type(input_tokens) is int and input_tokens >= 0 else None
        self.service_latency_ms = (service_latency_ms if isinstance(service_latency_ms, (int, float))
                                   and not isinstance(service_latency_ms, bool)
                                   and math.isfinite(float(service_latency_ms))
                                   and service_latency_ms >= 0 else None)
        super().__init__(code)

    def __repr__(self) -> str:
        return (f"DecisionProviderError(code={self.code!r}, retryable={self.retryable!r}, "
                f"request_id={self.request_id!r}, http_status={self.http_status!r}, "
                f"diagnostic_code={self.diagnostic_code!r}, field_path={self.field_path!r})")


class _ContractViolation(ValueError):
    """Internal, allowlisted diagnostic; never embeds a received value."""

    def __init__(self, code: str, field_path: str) -> None:
        self.diagnostic_code = code
        self.field_path = field_path
        super().__init__(code)


class JSONTransport(Protocol):
    def post_json(self, *, host: str, path: str, headers: Mapping[str, str], body: bytes,
                  deadline: float) -> "HTTPResponse": ...


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


def _header_value(headers: Mapping[str, str], *names: str) -> str | None:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return next((lowered[name.lower()] for name in names if name.lower() in lowered), None)


def _request_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value) else None


def _retry_after(value: str | None, *, wall_clock: Callable[[], float]) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            delay = parsed.timestamp() - wall_clock()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, 3600.0)


class HTTPSJsonTransport:
    """One verified-TLS POST with a parent-enforced deadline and bounded body.

    The production transport runs the synchronous stdlib HTTP parser in a
    killable child process, so a slow-drip header cannot hold the worker past
    the task deadline. Explicit clock/transport injection remains inline for
    deterministic unit tests; the default real transport is isolated.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 connection_factory: Callable[..., http.client.HTTPSConnection] | None = None,
                 max_response_bytes: int = MAX_RESPONSE_BYTES, isolate: bool | None = None) -> None:
        self.clock = clock
        self._injected_factory = connection_factory
        self.connection_factory = connection_factory or http.client.HTTPSConnection
        self.max_response_bytes = max_response_bytes
        self.isolate = clock is time.monotonic if isolate is None else isolate

    def post_json(self, *, host: str, path: str, headers: Mapping[str, str], body: bytes,
                  deadline: float) -> HTTPResponse:
        if path != API_PATH or not _ALLOWED_HOST.fullmatch(host):
            raise DecisionProviderError("ENDPOINT_NOT_ALLOWED")
        if len(body) > MAX_REQUEST_BYTES:
            raise DecisionProviderError("REQUEST_TOO_LARGE")
        if not self.isolate:
            return self._post_inline(host=host, path=path, headers=headers, body=body, deadline=deadline)
        return self._post_isolated(host=host, path=path, headers=headers, body=body, deadline=deadline)

    def _post_isolated(self, *, host: str, path: str, headers: Mapping[str, str], body: bytes,
                       deadline: float) -> HTTPResponse:
        if deadline <= self.clock():
            raise DecisionProviderError("DEADLINE_EXCEEDED")
        start_method = "fork" if self._injected_factory is not None else "spawn"
        try:
            context = multiprocessing.get_context(start_method)
        except ValueError:
            raise DecisionProviderError("TRANSPORT_ISOLATION_UNAVAILABLE") from None
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_isolated_transport_entry,
            args=(sender, self._injected_factory, self.max_response_bytes, host, path,
                  dict(headers), body, deadline),
            name="fallpre-aliyun-http",
        )
        process.daemon = True

        def stop_child() -> None:
            if not process.is_alive():
                process.join(timeout=0)
                return
            process.terminate()
            process.join(timeout=0.05)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
            process.join(timeout=0.1)

        try:
            process.start()
            sender.close()
            while True:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    stop_child()
                    raise DecisionProviderError("DEADLINE_EXCEEDED")
                if receiver.poll(min(remaining, 0.025)):
                    try:
                        message = receiver.recv()
                    except EOFError:
                        message = None
                    break
                if not process.is_alive():
                    message = None
                    break
            remaining = deadline - self.clock()
            if remaining <= 0:
                stop_child()
                raise DecisionProviderError("DEADLINE_EXCEEDED")
            process.join(timeout=remaining)
            if process.is_alive():
                stop_child()
                raise DecisionProviderError("DEADLINE_EXCEEDED")
            if not isinstance(message, tuple) or not message:
                raise DecisionProviderError("TRANSPORT_WORKER_EXITED")
            if message[0] == "error":
                _, code, retryable, retry_after, request_id, http_status, client_latency, diagnostic, field_path, tokens, service_latency = message
                raise DecisionProviderError(
                    code, retryable=retryable, retry_after_s=retry_after, request_id=request_id,
                    http_status=http_status, client_latency_ms=client_latency,
                    diagnostic_code=diagnostic, field_path=field_path, input_tokens=tokens,
                    service_latency_ms=service_latency,
                )
            if message[0] != "response":
                raise DecisionProviderError("TRANSPORT_WORKER_EXITED")
            _, status, response_headers, response_body = message
            if self.clock() >= deadline:
                raise DecisionProviderError(
                    "DEADLINE_EXCEEDED",
                    request_id=_request_id(_header_value(
                        response_headers, "x-request-id", "request-id",
                        "x-acs-request-id", "x-dashscope-request-id",
                    )),
                    http_status=status,
                )
            return HTTPResponse(status, response_headers, response_body)
        finally:
            sender.close()
            receiver.close()
            if process.pid is not None:
                if process.is_alive():
                    stop_child()
                if not process.is_alive():
                    process.close()

    def _post_inline(self, *, host: str, path: str, headers: Mapping[str, str], body: bytes,
                     deadline: float) -> HTTPResponse:
        if path != API_PATH or not _ALLOWED_HOST.fullmatch(host):
            raise DecisionProviderError("ENDPOINT_NOT_ALLOWED")
        if len(body) > MAX_REQUEST_BYTES:
            raise DecisionProviderError("REQUEST_TOO_LARGE")
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise DecisionProviderError("DEADLINE_EXCEEDED")
        connection: http.client.HTTPSConnection | None = None
        response_status: int | None = None
        response_request_id: str | None = None

        def check_deadline() -> None:
            if self.clock() >= deadline:
                raise DecisionProviderError("DEADLINE_EXCEEDED", request_id=response_request_id,
                                            http_status=response_status)

        try:
            connection = self.connection_factory(
                host,
                timeout=remaining,
                context=ssl.create_default_context(),
            )
            connection.connect()
            check_deadline()
            remaining = deadline - self.clock()
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            connection.request("POST", API_PATH, body=body, headers=dict(headers))
            check_deadline()
            remaining = deadline - self.clock()
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            response = connection.getresponse()
            response_status = response.status
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            response_request_id = _request_id(_header_value(
                response_headers, "x-request-id", "request-id", "x-acs-request-id", "x-dashscope-request-id",
            ))
            check_deadline()
            if response.status != 200:
                # Error bodies may contain sensitive request data. Do not read,
                # log, or persist them; a safe request id may be retained.
                return HTTPResponse(response.status, response_headers, b"")
            content_length = response_headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > self.max_response_bytes:
                        raise DecisionProviderError("RESPONSE_TOO_LARGE", http_status=response.status)
                except ValueError:
                    raise DecisionProviderError("INVALID_CONTENT_LENGTH", http_status=response.status) from None
            payload = bytearray()
            while True:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise DecisionProviderError("DEADLINE_EXCEEDED", retryable=True,
                                                http_status=response.status)
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(16 * 1024, self.max_response_bytes + 1 - len(payload)))
                # Check after read even when it returns EOF; otherwise a slow
                # final read can turn an expired request into an accepted 200.
                check_deadline()
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > self.max_response_bytes:
                    raise DecisionProviderError("RESPONSE_TOO_LARGE", http_status=response.status)
            return HTTPResponse(response.status, response_headers, bytes(payload))
        except DecisionProviderError:
            raise
        except ssl.SSLCertVerificationError:
            raise DecisionProviderError("TLS_VERIFICATION_FAILED") from None
        except (socket.timeout, TimeoutError):
            if self.clock() >= deadline:
                raise DecisionProviderError("DEADLINE_EXCEEDED", request_id=response_request_id,
                                            http_status=response_status) from None
            raise DecisionProviderError("NETWORK_TIMEOUT", retryable=True) from None
        except (ConnectionError, OSError):
            raise DecisionProviderError("NETWORK_CONNECTION_ERROR", retryable=True) from None
        except Exception:
            raise DecisionProviderError("TRANSPORT_ERROR") from None
        finally:
            if connection is not None:
                connection.close()


def _isolated_transport_entry(send_connection: Any, connection_factory: Any, max_response_bytes: int,
                              host: str, path: str, headers: Mapping[str, str], body: bytes,
                              deadline: float) -> None:
    """Run one synchronous HTTP exchange in a process the caller can terminate."""
    try:
        transport = HTTPSJsonTransport(
            clock=time.monotonic,
            connection_factory=connection_factory,
            max_response_bytes=max_response_bytes,
            isolate=False,
        )
        response = transport._post_inline(
            host=host, path=path, headers=headers, body=body, deadline=deadline,
        )
        safe_headers = {}
        for name, value in response.headers.items():
            lower = str(name).lower()
            if lower in {
                "x-request-id", "request-id", "x-acs-request-id", "x-dashscope-request-id", "retry-after",
            } and isinstance(value, str) and len(value) <= 256:
                safe_headers[lower] = value
        send_connection.send(("response", response.status, safe_headers, response.body))
    except DecisionProviderError as error:
        send_connection.send((
            "error", error.code, error.retryable, error.retry_after_s, error.request_id, error.http_status,
            error.client_latency_ms, error.diagnostic_code, error.field_path, error.input_tokens,
            error.service_latency_ms,
        ))
    except BaseException:
        # Do not serialize exception text or a traceback across the boundary.
        try:
            send_connection.send(("error", "TRANSPORT_ERROR", False, None, None, None, None,
                                  None, None, None, None))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send_connection.close()


def _require_field(value: Mapping[str, Any], name: str, field_path: str) -> Any:
    if name not in value:
        raise _ContractViolation("FIELD_MISSING", field_path)
    return value[name]


def _strict_probability(value: Any, *, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ContractViolation("NUMBER_TYPE", field_path)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise _ContractViolation("NUMBER_RANGE", field_path) from None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise _ContractViolation("NUMBER_RANGE", field_path)
    return number


def _choice_answer(value: Any, *, question_id: str) -> ChoiceAnswer:
    prefix = f"answers.{question_id}"
    if not isinstance(value, dict) or value.get("type") != "choice":
        raise _ContractViolation("ANSWER_TYPE", prefix)
    choice = _require_field(value, "choice", f"{prefix}.choice")
    options = _CHOICE_OPTIONS[question_id]
    if not isinstance(choice, str) or choice not in options:
        raise _ContractViolation("ENUM_VALUE_INVALID", f"{prefix}.choice")
    probabilities = _require_field(value, "probabilities", f"{prefix}.probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != options:
        raise _ContractViolation("DISTRIBUTION_KEYS_INVALID", f"{prefix}.probabilities")
    normalized = {
        option: _strict_probability(probabilities[option], field_path=f"{prefix}.probabilities")
        for option in options
    }
    if abs(sum(normalized.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise _ContractViolation("DISTRIBUTION_SUM_INVALID", f"{prefix}.probabilities")
    confidence = _strict_probability(
        _require_field(value, "confidence", f"{prefix}.confidence"), field_path=f"{prefix}.confidence",
    )
    return ChoiceAnswer(type="choice", choice=choice, confidence=confidence, probabilities=normalized)


def parse_response(body: bytes, *, expected_model: str, http_status: int = 200) -> tuple[str, DecisionAnswers, int, float]:
    """Parse only the documented response fields; discard every raw payload."""
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DecisionProviderError("INVALID_JSON", http_status=http_status) from None
    if not isinstance(value, dict):
        raise DecisionProviderError("INVALID_RESPONSE_CONTRACT", http_status=http_status,
                                    diagnostic_code="MODEL_VALIDATION_FAILED", field_path="answers")
    request_id = _request_id(value.get("request_id"))
    if value.get("model") != expected_model:
        raise DecisionProviderError("MODEL_ECHO_MISMATCH", request_id=request_id, http_status=http_status,
                                    diagnostic_code="MODEL_ECHO_MISMATCH", field_path="model")
    if request_id is None:
        raise DecisionProviderError("REQUEST_ID_MISSING_OR_INVALID", http_status=http_status,
                                    diagnostic_code="REQUEST_ID_MISSING_OR_INVALID", field_path="request_id")
    answers = value.get("answers")
    if not isinstance(answers, dict) or set(answers) != {
        "fall_occurred", "evidence_sufficiency", "next_step",
    }:
        raise DecisionProviderError("ANSWER_SET_INVALID", request_id=request_id, http_status=http_status,
                                    diagnostic_code="ANSWER_SET_INVALID", field_path="answers")
    try:
        fall = answers["fall_occurred"]
        if not isinstance(fall, dict):
            raise _ContractViolation("ANSWER_TYPE", "answers.fall_occurred")
        if _require_field(fall, "type", "answers.fall_occurred.type") != "noul":
            raise _ContractViolation("ANSWER_TYPE", "answers.fall_occurred.type")
        fall_probability = _strict_probability(
            _require_field(fall, "noul", "answers.fall_occurred.noul"),
            field_path="answers.fall_occurred.noul",
        )
        sufficiency = _choice_answer(answers["evidence_sufficiency"], question_id="evidence_sufficiency")
        next_step = _choice_answer(answers["next_step"], question_id="next_step")
        usage = value.get("usage")
        if not isinstance(usage, dict):
            raise _ContractViolation("FIELD_TYPE", "usage")
        input_tokens = _require_field(usage, "input_tokens", "usage.input_tokens")
        if type(input_tokens) is not int or input_tokens < 0:
            raise _ContractViolation("USAGE_TOKENS_INVALID", "usage.input_tokens")
        latency = _require_field(value, "latency_ms", "latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise _ContractViolation("LATENCY_TYPE_INVALID", "latency_ms")
        try:
            latency_ms = float(latency)
        except (OverflowError, ValueError):
            raise _ContractViolation("LATENCY_RANGE_INVALID", "latency_ms") from None
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise _ContractViolation("LATENCY_RANGE_INVALID", "latency_ms")
        parsed_answers = DecisionAnswers(
            fall_occurred={"type": "noul", "noul": fall_probability},
            evidence_sufficiency=sufficiency,
            next_step=next_step,
        )
    except _ContractViolation as error:
        raise DecisionProviderError("INVALID_RESPONSE_CONTRACT", request_id=request_id,
                                    http_status=http_status, diagnostic_code=error.diagnostic_code,
                                    field_path=error.field_path) from None
    except (ValueError, TypeError, ValidationError):
        raise DecisionProviderError("INVALID_RESPONSE_CONTRACT", request_id=request_id,
                                    http_status=http_status, diagnostic_code="MODEL_VALIDATION_FAILED",
                                    field_path="answers") from None
    return request_id, parsed_answers, input_tokens, latency_ms


class AliyunDecisionProvider:
    """Structured System One provider; transport is injectable for offline tests."""

    def __init__(self, *, config: DecisionConfig, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time,
                 transport: JSONTransport | None = None) -> None:
        config.require_aliyun_configuration(provider="aliyun")
        self.config = config
        self.clock = clock
        self.wall_clock = wall_clock
        self.transport = transport or HTTPSJsonTransport(clock=clock)

    def _result(self, evidence: DecisionEvidence, **values: Any) -> ProviderResult:
        identity = {
            "run_id": evidence.run_id,
            "session_id": evidence.session_id,
            "stream_id": evidence.stream_id,
            "snapshot_id": evidence.snapshot_id,
            "decision_id": evidence.decision_id,
            "action_epoch": evidence.action_epoch,
            "generation": evidence.generation,
            "source_window": evidence.source_window,
            "event_key": evidence.event_key,
            "event_state": evidence.event_state,
            "recovery_status": evidence.recovery_status,
            "provider": ProviderMode.ALIYUN,
            "model": self.config.model,
        }
        return ProviderResult(**identity, **values)

    def _local_quality_result(self, evidence: DecisionEvidence) -> ProviderResult:
        conflicts = [*evidence.conflict_reason_codes, *evidence.geometry_conflict_reason_codes]
        reasons = list(evidence.eligibility_reasons)
        if conflicts:
            return self._result(
                evidence,
                status=DecisionStatus.INSUFFICIENT,
                evidence_sufficiency=EvidenceSufficiency.CONFLICTING,
                next_step=NextStep.REQUEST_REVIEW,
                reason_codes=list(dict.fromkeys(["EVIDENCE_CONFLICTING", *conflicts, *reasons])),
            )
        stale = "ST_RESULT_STALE_AT_TRIGGER" in reasons
        return self._result(
            evidence,
            status=DecisionStatus.STALE if stale else DecisionStatus.INSUFFICIENT,
            evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
            next_step=NextStep.OBSERVE_MORE,
            reason_codes=list(dict.fromkeys(["EVIDENCE_INSUFFICIENT", *reasons])),
        )

    def assess(self, evidence: DecisionEvidence, *, deadline: float) -> ProviderResult:
        if not evidence.eligible:
            return self._local_quality_result(evidence)
        if self.clock() >= deadline:
            raise DecisionProviderError("DEADLINE_EXCEEDED")
        api_key = self.config.api_key.get_secret_value()
        if not api_key:
            raise DecisionProviderError("MISSING_CONFIGURATION")
        state = build_decision_state(evidence)
        payload = {"model": self.config.model, "state": state, "questions": decision_questions()}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise DecisionProviderError("REQUEST_TOO_LARGE")
        host = f"{self.config.workspace_id}.{self.config.region}.maas.aliyuncs.com"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(encoded)),
            "User-Agent": "FallPreAgent/decision-p2",
        }
        trace_decision("provider_dispatch", run_id=evidence.run_id, decision_id=evidence.decision_id,
                       model=self.config.model, payload_bytes=len(encoded), evidence_digest=evidence.request_digest)
        started = self.clock()
        try:
            response = self.transport.post_json(
                host=host, path=API_PATH, headers=headers, body=encoded, deadline=deadline,
            )
        except DecisionProviderError as error:
            client_latency_ms = max(0.0, (self.clock() - started) * 1000.0)
            if self.clock() >= deadline:
                raise DecisionProviderError(
                    "DEADLINE_EXCEEDED", request_id=error.request_id, http_status=error.http_status,
                    client_latency_ms=client_latency_ms,
                ) from None
            if error.client_latency_ms is None:
                error.client_latency_ms = client_latency_ms
            raise
        except (socket.timeout, TimeoutError):
            elapsed = max(0.0, (self.clock() - started) * 1000.0)
            code = "DEADLINE_EXCEEDED" if self.clock() >= deadline else "NETWORK_TIMEOUT"
            raise DecisionProviderError(code, retryable=code == "NETWORK_TIMEOUT",
                                        client_latency_ms=elapsed) from None
        except (ConnectionError, OSError):
            elapsed = max(0.0, (self.clock() - started) * 1000.0)
            code = "DEADLINE_EXCEEDED" if self.clock() >= deadline else "NETWORK_CONNECTION_ERROR"
            raise DecisionProviderError(code, retryable=code == "NETWORK_CONNECTION_ERROR",
                                        client_latency_ms=elapsed) from None
        except Exception:
            elapsed = max(0.0, (self.clock() - started) * 1000.0)
            code = "DEADLINE_EXCEEDED" if self.clock() >= deadline else "TRANSPORT_ERROR"
            raise DecisionProviderError(code, client_latency_ms=elapsed) from None
        client_latency_ms = max(0.0, (self.clock() - started) * 1000.0)
        header_request_id = _request_id(_header_value(
            response.headers, "x-request-id", "request-id", "x-acs-request-id", "x-dashscope-request-id",
        ))
        trace_decision("provider_http_response", run_id=evidence.run_id, decision_id=evidence.decision_id,
                       http_status=response.status, request_id=header_request_id, client_latency_ms=client_latency_ms)
        if self.clock() >= deadline:
            raise DecisionProviderError("DEADLINE_EXCEEDED", request_id=header_request_id,
                                        http_status=response.status, client_latency_ms=client_latency_ms)
        if response.status != 200:
            retryable = response.status == 429 or 500 <= response.status <= 599
            code = "REDIRECT_REJECTED" if 300 <= response.status <= 399 else f"HTTP_{response.status}"
            retry_after = _retry_after(_header_value(response.headers, "retry-after"), wall_clock=self.wall_clock)
            raise DecisionProviderError(code, retryable=retryable, retry_after_s=retry_after,
                                        request_id=header_request_id, http_status=response.status,
                                        client_latency_ms=client_latency_ms)
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise DecisionProviderError("RESPONSE_TOO_LARGE", request_id=header_request_id,
                                        http_status=response.status)
        try:
            request_id, answers, input_tokens, service_latency_ms = parse_response(
                response.body, expected_model=self.config.model, http_status=response.status,
            )
        except DecisionProviderError as error:
            error.request_id = error.request_id or header_request_id
            error.client_latency_ms = client_latency_ms
            raise
        if self.clock() >= deadline:
            raise DecisionProviderError("DEADLINE_EXCEEDED", request_id=request_id,
                                        http_status=response.status, client_latency_ms=client_latency_ms,
                                        input_tokens=input_tokens, service_latency_ms=service_latency_ms)
        request_id = request_id or header_request_id
        metadata = ProviderMetadata(
            request_id=request_id,
            http_status=response.status,
            input_tokens=input_tokens,
            service_latency_ms=service_latency_ms,
            client_latency_ms=client_latency_ms,
            answers=answers,
        )
        sufficiency = answers.evidence_sufficiency.choice
        suggested_step = answers.next_step.choice
        consistent = (
            (sufficiency == "sufficient" and suggested_step == "emit_result")
            or (sufficiency == "insufficient" and suggested_step == "observe_more")
            or (sufficiency == "conflicting" and suggested_step == "request_review")
        )
        raw_probability = answers.fall_occurred.noul
        if not consistent:
            return self._result(
                evidence,
                status=DecisionStatus.INSUFFICIENT,
                fall_probability=None,
                fall_probability_raw=raw_probability,
                probability_kind=ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE,
                evidence_sufficiency=EvidenceSufficiency.CONFLICTING,
                next_step=NextStep.REQUEST_REVIEW,
                reason_codes=["MODEL_ANSWER_COMBINATION_CONFLICT", "CURRENT_PROBABILITY_WITHHELD"],
                provider_metadata=metadata,
            )
        if sufficiency != "sufficient":
            return self._result(
                evidence,
                status=DecisionStatus.INSUFFICIENT,
                fall_probability=None,
                fall_probability_raw=raw_probability,
                probability_kind=ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE,
                evidence_sufficiency=EvidenceSufficiency(sufficiency),
                next_step=NextStep(suggested_step),
                reason_codes=["MODEL_EVIDENCE_NOT_SUFFICIENT", "CURRENT_PROBABILITY_WITHHELD"],
                provider_metadata=metadata,
            )
        return self._result(
            evidence,
            status=DecisionStatus.READY,
            fall_probability=raw_probability,
            fall_probability_raw=raw_probability,
            fall_probability_calibrated=None,
            probability_kind=ProbabilityKind.UNCALIBRATED_MODEL_ESTIMATE,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            next_step=NextStep.EMIT_RESULT,
            reason_codes=["UNCALIBRATED_MODEL_ESTIMATE"],
            provider_metadata=metadata,
        )


__all__ = [
    "API_PATH",
    "AliyunDecisionProvider",
    "DecisionProviderError",
    "HTTPResponse",
    "HTTPSJsonTransport",
    "parse_response",
]
