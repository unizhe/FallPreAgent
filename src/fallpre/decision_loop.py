"""Local, restartable off/mock decision orchestration for P1.

The loop only appends decisions. It has no authority over upstream event
confirmation, recovery, or simulated notification handling.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from .adapters.aliyun_decision import DecisionProviderError
from .decision_config import DecisionConfig
from .decision_contracts import (
    DecisionEvidence,
    DecisionView,
    ProviderMode,
    ProviderResult,
    decision_view_from_result,
)
from .decision_evidence import DecisionEnvelope, adapt_envelope, canonical_json, sha256_json
from .decision_provider import build_decision_provider
from .decision_state import DECISION_PROMPT_VERSION, DECISION_STATE_SCHEMA_VERSION
from .decision_store import DecisionConflict, DecisionStore, DecisionVersionConflict, InjectedStoreCrash
from .decision_trace import trace_decision

_DEFAULT_SERVICE_INSTANCE_ID = uuid.uuid4().hex


class DecisionLoopError(RuntimeError):
    """An orchestration request violated its local decision contract."""


class InjectedCrash(RuntimeError):
    """Deterministic fault point used by recovery tests and the local demo."""


class DecisionLoop:
    def __init__(
        self,
        store: DecisionStore | str,
        *,
        run_id: str | None = None,
        config: DecisionConfig | None = None,
        provider: str | ProviderMode | None = None,
        provider_impl: Any | None = None,
        provider_transport: Any | None = None,
        enable_network: bool = False,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        process_id: str | None = None,
        service_instance_id: str | None = None,
        max_queue: int = 1000,
    ):
        selected_arg = provider.value if isinstance(provider, ProviderMode) else provider
        selected = selected_arg or (config.provider if config else ProviderMode.OFF.value)
        if selected == ProviderMode.ALIYUN.value and not enable_network:
            raise DecisionLoopError("aliyun requires explicit enable_network=True")
        if selected not in {mode.value for mode in ProviderMode}:
            raise DecisionLoopError("unsupported decision provider")
        if type(max_queue) is not int or not 1 <= max_queue <= 10_000:
            raise ValueError("max_queue must be an integer from 1 to 10000")
        self.config = config or DecisionConfig(provider=selected)
        self.store = store if isinstance(store, DecisionStore) else DecisionStore(store)
        self._owns_store = not isinstance(store, DecisionStore)
        self.run_id = run_id or f"decision-run-{uuid.uuid4().hex}"
        self.provider_mode = ProviderMode(selected)
        self.provider_impl = provider_impl or build_decision_provider(
            self.provider_mode, config=self.config, clock=monotonic_clock,
            wall_clock=wall_clock, enable_network=enable_network, transport=provider_transport,
        )
        self.enable_network = enable_network
        self.monotonic_clock = monotonic_clock
        self.wall_clock = wall_clock
        self.service_instance_id = service_instance_id or _DEFAULT_SERVICE_INSTANCE_ID
        if not isinstance(self.service_instance_id, str) or not self.service_instance_id.strip():
            raise ValueError("service_instance_id must be nonempty")
        # process_id identifies this scheduler/worker instance; service_instance_id
        # identifies the enclosing service boot shared by its query and worker paths.
        self.process_id = process_id or uuid.uuid4().hex
        self.max_queue = max_queue
        self._closed = False

    def close(self) -> None:
        if not self._closed and self._owns_store:
            self.store.close()
        self._closed = True

    def __enter__(self) -> DecisionLoop:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def observe(
        self,
        envelope: DecisionEnvelope | dict[str, Any],
        *,
        run_id: str | None = None,
        end_of_stream: bool = False,
        fail_at: str | None = None,
        allow_task: bool = True,
        force_stale: bool = False,
        task_suppression_reason: str | None = None,
    ) -> dict[str, Any]:
        """Persist an observation, then atomically register an eligible task."""
        item = envelope if isinstance(envelope, DecisionEnvelope) else DecisionEnvelope.model_validate(envelope)
        envelope_json = item.model_dump(mode="json")
        active = self.store.active_workflow(item.session_id, item.stream_id, self.wall_clock())
        parent_id = None
        if active:
            summary = item.evidence_summary or {}
            same = (summary.get("action_epoch") == active["action_epoch"]
                    and summary.get("generation") == active["generation"]
                    and summary.get("fall_target_epoch") == active["fall_target_epoch"])
            if same and item.source_time_s - active["trigger_source_time"] >= 1.0:
                parent_id = active["parent_decision_id"]

        summary = item.evidence_summary if item.evidence_summary is not None else {}
        request_material = {
            "envelope": envelope_json,
            "evidence_summary": summary,
            "provider": self.provider_mode.value,
            "model": self.config.model,
            "fall_head": self.config.fall_head,
            "preprocessing_version": summary.get("preprocessing_version"),
            "yolo_model_version": summary.get("yolo_model_version"),
            "head_version": summary.get("head_version"),
            "head_weights_sha256": summary.get("head_weights_sha256"),
            "prompt_version": (DECISION_PROMPT_VERSION if self.provider_mode == ProviderMode.ALIYUN
                               else summary.get("prompt_version")),
            "state_schema_version": (DECISION_STATE_SCHEMA_VERSION
                                     if self.provider_mode == ProviderMode.ALIYUN else None),
            "policy_version": summary.get("policy_version"),
            "eligibility_policy_version": "decision-eligibility-p0-v1",
            "parent_decision_id": parent_id,
        }
        request_digest = sha256_json(request_material)
        decision_id = f"decision-{request_digest[:32]}"
        snapshot_id = f"snapshot-{sha256_json(envelope_json)[:24]}"
        adaptation = adapt_envelope(
            item,
            run_id=run_id or self.run_id,
            snapshot_id=snapshot_id,
            decision_id=decision_id,
            request_digest=request_digest,
        )
        evidence_json = adaptation.evidence.model_dump(mode="json") if adaptation.evidence else None
        task = {
            "task_id": f"task-{request_digest[:32]}",
            "snapshot_id": snapshot_id,
            "decision_id": decision_id,
            "request_digest": request_digest,
            "provider": self.provider_mode.value,
            "model": self.config.model,
            "max_attempts": self.config.max_retries + 1,
            "max_reobservations": self.config.max_reobservations,
            "parent_decision_id": parent_id,
        } if allow_task else None
        receipt = self.store.record_observation(
            run_id=run_id or self.run_id,
            envelope=envelope_json,
            evidence=evidence_json,
            evidence_digest=adaptation.evidence_digest,
            missing_reasons=list(adaptation.missing_fields),
            task=task,
            min_interval_s=self.config.min_interval_s,
            max_queue=self.max_queue,
            now=self.wall_clock(),
            end_of_stream=end_of_stream,
            fail_at=fail_at,
            force_stale=force_stale,
            task_suppression_reason=task_suppression_reason,
        )
        if receipt["status"] == "duplicate" and receipt.get("decision_id"):
            result = self.store.get_result(receipt["decision_id"])
            if result:
                receipt["decision"] = ProviderResult.model_validate(result["provider_result"]).model_dump(mode="json")
        return receipt

    def accept_result(
        self,
        *,
        task_id: str,
        attempt: int,
        lease_token: str,
        result: ProviderResult | dict[str, Any],
    ) -> dict[str, Any]:
        """Validate provider output against the claimed evidence and persist once."""
        provider_result = result if isinstance(result, ProviderResult) else ProviderResult.model_validate(result)
        task = self.store.task(task_id)
        if task is None:
            raise KeyError(task_id)
        evidence = DecisionEvidence.model_validate(task["evidence"])
        if provider_result.status.value == "ready" and not evidence.eligible:
            raise DecisionLoopError("provider cannot bypass deterministic evidence eligibility")
        expected = {
            "run_id": evidence.run_id,
            "session_id": evidence.session_id,
            "stream_id": evidence.stream_id,
            "snapshot_id": evidence.snapshot_id,
            "decision_id": evidence.decision_id,
            "action_epoch": evidence.action_epoch,
            "generation": evidence.generation,
            "source_window": evidence.source_window.model_dump(mode="json"),
            "event_key": evidence.event_key,
            "event_state": evidence.event_state.value,
            "recovery_status": evidence.recovery_status.value,
            "provider": task["provider"],
            "model": task["model"],
        }
        actual = provider_result.model_dump(mode="json")
        differences = [key for key, value in expected.items() if actual.get(key) != value]
        if differences:
            raise DecisionLoopError("provider result does not match its task evidence: " + ", ".join(differences))
        return self.store.complete_result(
            task_id=task_id,
            attempt=attempt,
            lease_token=lease_token,
            result_json=canonical_json(provider_result.model_dump(mode="json")),
            process_id=self.process_id,
            service_instance_id=self.service_instance_id,
            now=self.wall_clock(),
        )

    def _checkpoint(self, fail_at: str | None, name: str, hook: Callable[[str], None] | None) -> None:
        if hook:
            hook(name)
        if fail_at == name:
            raise InjectedCrash(f"injected crash at {name}")

    def run_one(self, *, fail_at: str | None = None, fault_hook: Callable[[str], None] | None = None) -> dict[str, Any] | None:
        """Claim one durable task and call provider outside every DB transaction."""
        now = self.wall_clock()
        if self.provider_mode == ProviderMode.ALIYUN:
            for expired_task in self.store.expired_aliyun_tasks(now=now):
                evidence = DecisionEvidence.model_validate(expired_task["evidence"])
                if (expired_task.get("network_remaining_s") is not None
                        and expired_task["network_remaining_s"] <= 0) or (
                        expired_task.get("network_deadline_at") is not None
                        and expired_task["network_deadline_at"] <= now):
                    reason = "DEADLINE_EXCEEDED"
                elif expired_task["attempt"] >= expired_task["max_attempts"]:
                    reason = "MAX_ATTEMPTS_EXHAUSTED"
                else:
                    # A lease that elapsed cannot prove whether its HTTP request
                    # reached the service. Stop conservatively instead of replaying it.
                    reason = "WORKER_LEASE_EXPIRED"
                unavailable = self._unavailable_result(evidence, reason, {})
                completed = self.store.complete_unavailable_without_attempt(
                    task_id=expired_task["task_id"],
                    result_json=canonical_json(unavailable.model_dump(mode="json")),
                    error_code=reason,
                    now=now,
                    process_id=self.process_id,
                    service_instance_id=self.service_instance_id,
                )
                if completed:
                    return {"task_id": expired_task["task_id"], "attempt": expired_task["attempt"],
                            "status": "unavailable", "error_code": reason}
        lease_s = max(5.0, self.config.timeout_s * 2.0)
        task = self.store.claim_next(now=now, lease_s=lease_s,
                                     provider=self.provider_mode.value, model=self.config.model,
                                     max_queue=self.max_queue,
                                     network_budget_s=self.config.timeout_s)
        if task is None:
            return None
        attempt = task["attempt"]
        token = task["lease_token"]
        self._checkpoint(fail_at, "after_claim", fault_hook)
        evidence = DecisionEvidence.model_validate(task["evidence"])
        attempt_started_mono = self.monotonic_clock()
        trace_ids = dict(run_id=evidence.run_id, decision_id=evidence.decision_id, task_id=task["task_id"],
                         snapshot_id=evidence.snapshot_id, attempt=attempt)
        trace_decision("task_claimed", **trace_ids, eligible=evidence.eligible,
                       source_window=evidence.source_window.model_dump(), evidence_digest=evidence.request_digest)
        remaining_budget_at_start = task.get("network_remaining_s")
        if self.store.db.in_transaction:
            raise AssertionError("provider called while SQLite transaction is open")
        try:
            remaining_wall = (task.get("network_deadline_at") - self.wall_clock()
                              if self.provider_mode == ProviderMode.ALIYUN and task.get("network_deadline_at") is not None
                              else self.config.timeout_s)
            remaining_budget = (remaining_budget_at_start if self.provider_mode == ProviderMode.ALIYUN
                                and remaining_budget_at_start is not None else self.config.timeout_s)
            attempt_deadline = attempt_started_mono + max(
                0.0, min(self.config.timeout_s, remaining_wall, remaining_budget),
            )
            provider_result = self.provider_impl.assess(
                evidence, deadline=attempt_deadline
            )
            provider_result = ProviderResult.model_validate(provider_result.model_dump(mode="json"))
            if (self.provider_mode == ProviderMode.ALIYUN
                    and (self.monotonic_clock() >= attempt_deadline
                         or (task.get("network_deadline_at") is not None
                             and self.wall_clock() >= task["network_deadline_at"]))):
                metadata = provider_result.provider_metadata
                raise DecisionProviderError(
                    "DEADLINE_EXCEEDED",
                    request_id=metadata.request_id if metadata else None,
                    http_status=metadata.http_status if metadata else None,
                    client_latency_ms=metadata.client_latency_ms if metadata else None,
                    input_tokens=metadata.input_tokens if metadata else None,
                    service_latency_ms=metadata.service_latency_ms if metadata else None,
                )
        except Exception as error:
            if self.provider_mode == ProviderMode.ALIYUN:
                if isinstance(error, DecisionProviderError):
                    error_code = error.code
                    retryable = error.retryable
                    retry_after = error.retry_after_s
                    metadata = {
                        "request_id": error.request_id,
                        "http_status": error.http_status,
                        "client_latency_ms": error.client_latency_ms,
                        "service_latency_ms": error.service_latency_ms,
                        "input_tokens": error.input_tokens,
                        "retry_after_s": retry_after,
                        "diagnostic_code": error.diagnostic_code,
                        "field_path": error.field_path,
                    }
                else:
                    error_code = "PROVIDER_CONTRACT_ERROR"
                    retryable = False
                    retry_after = None
                    metadata = {"request_id": None, "http_status": None, "client_latency_ms": None,
                                "service_latency_ms": None, "input_tokens": None, "retry_after_s": None,
                                "diagnostic_code": None, "field_path": None}
                elapsed_mono = max(0.0, self.monotonic_clock() - attempt_started_mono)
                remaining_after_call = max(0.0, min(
                    remaining_budget_at_start if remaining_budget_at_start is not None else self.config.timeout_s,
                    (task.get("network_deadline_at") or now) - now,
                ) - elapsed_mono)
                backoff = retry_after if retry_after is not None else min(0.05 * (2 ** (attempt - 1)), 0.25)
                will_retry = bool(retryable and attempt < task["max_attempts"] and backoff < remaining_after_call)
                trace_decision("provider_failure", **trace_ids, error_code=error_code, retry_planned=will_retry,
                               **metadata)
                if will_retry:
                    remaining_after_wait = max(0.0, remaining_after_call - backoff)
                    self.store.fail_attempt(
                        task_id=task["task_id"], attempt=attempt, lease_token=token,
                        error_code=error_code, now=self.wall_clock(), retry_delay_s=backoff,
                        network_remaining_s=remaining_after_wait,
                        attempt_metadata=metadata,
                    )
                    return {"task_id": task["task_id"], "attempt": attempt, "status": "retry_pending",
                            "error_code": error_code, "http_status": metadata["http_status"],
                            "request_id": metadata["request_id"], "client_latency_ms": metadata["client_latency_ms"],
                            "diagnostic_code": metadata["diagnostic_code"], "field_path": metadata["field_path"],
                            "retry_delay_s": backoff}
                unavailable = self._unavailable_result(evidence, error_code, metadata)
                persisted = self.store.fail_attempt(
                    task_id=task["task_id"], attempt=attempt, lease_token=token,
                    error_code=error_code, now=self.wall_clock(),
                    terminal_result_json=canonical_json(unavailable.model_dump(mode="json")),
                    process_id=self.process_id, service_instance_id=self.service_instance_id,
                    attempt_metadata=metadata,
                )
                return {"task_id": task["task_id"], "attempt": attempt,
                        "status": "unavailable" if persisted else "late_failure",
                        "error_code": error_code, "http_status": metadata["http_status"],
                        "request_id": metadata["request_id"], "client_latency_ms": metadata["client_latency_ms"],
                        "diagnostic_code": metadata["diagnostic_code"], "field_path": metadata["field_path"]}
            self.store.fail_attempt(task_id=task["task_id"], attempt=attempt, lease_token=token,
                                    error_code=f"PROVIDER_ERROR_{type(error).__name__}",
                                    now=self.wall_clock())
            return {"task_id": task["task_id"], "attempt": attempt,
                    "status": "retry_pending" if attempt < task["max_attempts"] else "failed",
                    "error_code": f"PROVIDER_ERROR_{type(error).__name__}"}
        self._checkpoint(fail_at, "after_provider_before_write", fault_hook)
        if (self.provider_mode == ProviderMode.ALIYUN
                and (self.monotonic_clock() >= attempt_deadline
                     or (task.get("network_deadline_at") is not None
                         and self.wall_clock() >= task["network_deadline_at"]))):
            metadata_obj = provider_result.provider_metadata
            metadata = {
                "request_id": metadata_obj.request_id if metadata_obj else None,
                "http_status": metadata_obj.http_status if metadata_obj else None,
                "client_latency_ms": metadata_obj.client_latency_ms if metadata_obj else None,
                "service_latency_ms": metadata_obj.service_latency_ms if metadata_obj else None,
                "input_tokens": metadata_obj.input_tokens if metadata_obj else None,
                "retry_after_s": None,
                "diagnostic_code": metadata_obj.diagnostic_code if metadata_obj else None,
                "field_path": metadata_obj.field_path if metadata_obj else None,
            }
            unavailable = self._unavailable_result(evidence, "DEADLINE_EXCEEDED", metadata)
            persisted = self.store.fail_attempt(
                task_id=task["task_id"], attempt=attempt, lease_token=token,
                error_code="DEADLINE_EXCEEDED", now=self.wall_clock(),
                terminal_result_json=canonical_json(unavailable.model_dump(mode="json")),
                process_id=self.process_id, service_instance_id=self.service_instance_id,
                attempt_metadata=metadata,
            )
            return {"task_id": task["task_id"], "attempt": attempt,
                    "status": "unavailable" if persisted else "late_failure",
                    "error_code": "DEADLINE_EXCEEDED", "http_status": metadata["http_status"],
                    "request_id": metadata["request_id"], "client_latency_ms": metadata["client_latency_ms"],
                    "diagnostic_code": metadata["diagnostic_code"], "field_path": metadata["field_path"]}
        try:
            accepted = self.accept_result(task_id=task["task_id"], attempt=attempt,
                                          lease_token=token, result=provider_result)
        except DecisionLoopError:
            self.store.fail_attempt(task_id=task["task_id"], attempt=attempt, lease_token=token,
                                    error_code="PROVIDER_RESULT_REJECTED", now=self.wall_clock())
            return {"task_id": task["task_id"], "attempt": attempt,
                    "status": "retry_pending" if attempt < task["max_attempts"] else "failed",
                    "error_code": "PROVIDER_RESULT_REJECTED"}
        self._checkpoint(fail_at, "after_commit_before_receipt", fault_hook)
        meta = provider_result.provider_metadata
        trace_decision("result_commit", **trace_ids, accepted=accepted["accepted"], duplicate=accepted["duplicate"],
                       status=provider_result.status, fall_probability=provider_result.fall_probability,
                       request_id=meta.request_id if meta else None)
        return {"task_id": task["task_id"], "attempt": attempt,
                "status": "accepted" if accepted["accepted"] else "late_result",
                "decision_id": accepted["decision_id"], "duplicate": accepted["duplicate"]}

    def run_pending(self, *, limit: int = 100, fail_at: str | None = None,
                    fault_hook: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000")
        output = []
        for index in range(limit):
            output_item = self.run_one(fail_at=fail_at if index == 0 else None,
                                       fault_hook=fault_hook if index == 0 else None)
            if output_item is None:
                break
            output.append(output_item)
        return output

    def _unavailable_result(self, evidence: DecisionEvidence, reason: str,
                            metadata: dict[str, Any]) -> ProviderResult:
        provider_metadata = None
        if any(value is not None for value in metadata.values()):
            provider_metadata = {
                "request_id": metadata.get("request_id"),
                "http_status": metadata.get("http_status"),
                "client_latency_ms": metadata.get("client_latency_ms"),
                "input_tokens": metadata.get("input_tokens"),
                "service_latency_ms": metadata.get("service_latency_ms"),
                "diagnostic_code": metadata.get("diagnostic_code"),
                "field_path": metadata.get("field_path"),
            }
        values = {
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
            "status": "unavailable",
            "evidence_sufficiency": "not_assessed",
            "next_step": None,
            "reason_codes": [reason],
        }
        if provider_metadata is not None:
            values["provider_metadata"] = provider_metadata
        return ProviderResult.model_validate(values)

    def latest(self, session_id: str, stream_id: str) -> dict[str, Any]:
        now = self.wall_clock()
        observation = self.store.latest_observation(session_id, stream_id)
        workflow = self.store.workflow_status(session_id, stream_id, now)
        stream = self.store.db.execute("SELECT stopped,stop_reason FROM decision_streams "
                                       "WHERE session_id=? AND stream_id=?", (session_id, stream_id)).fetchone()
        reasons: list[str] = []
        if not observation:
            reasons.append("NO_CURRENT_OBSERVATION")
        elif observation["stale"]:
            reasons.append("LATEST_OBSERVATION_STALE")
        elif stream and stream["stopped"]:
            reasons.append("STREAM_STOPPED")
        elif not observation["evidence"]:
            reasons.extend(observation["missing_reasons"] or ["EVIDENCE_INCOMPLETE"])
        elif workflow and workflow["reason"] == "END_OF_STREAM":
            reasons.append("END_OF_STREAM")
        elif workflow and workflow["reason"] == "OBSERVATION_TIMEOUT":
            reasons.append("OBSERVATION_TIMEOUT")
        else:
            current_evidence = DecisionEvidence.model_validate(observation["evidence"])
            reasons.extend(current_evidence.eligibility_reasons)
            reasons.extend(current_evidence.conflict_reason_codes)
            if not current_evidence.eligible or current_evidence.conflict_reason_codes:
                if not reasons:
                    reasons.append("CURRENT_EVIDENCE_NOT_ELIGIBLE")
            elif current_evidence.recovery_status.value == "confirmed":
                reasons.append("UPSTREAM_RECOVERY_CONFIRMED")

        latest_evidence = (DecisionEvidence.model_validate(observation["evidence"])
                           if observation and observation.get("evidence") else None)
        latest_task = (self.store.task(observation["task_id"])
                       if observation and observation.get("task_id") else None)

        def same_target(candidate: DecisionEvidence) -> bool:
            return bool(latest_evidence and (
                candidate.run_id == latest_evidence.run_id
                and candidate.session_id == latest_evidence.session_id
                and candidate.stream_id == latest_evidence.stream_id
                and candidate.action_epoch == latest_evidence.action_epoch
                and candidate.generation == latest_evidence.generation
                and candidate.fall_target_epoch == latest_evidence.fall_target_epoch
                and candidate.event_key == latest_evidence.event_key
                and candidate.event_state == latest_evidence.event_state
                and candidate.recovery_status == latest_evidence.recovery_status
            ))

        # Terminals are ordered by source time and observation id, never by
        # completion time. A newer pending task may reuse the newest compatible
        # ready terminal while it is pending; a newer non-ready terminal closes
        # that fallback boundary.
        decision_rows = self.store.list_decisions(session_id=session_id, stream_id=stream_id, limit=500)
        compatible_terminals: list[tuple[dict[str, Any], ProviderResult | None]] = []
        historical_ready: list[ProviderResult] = []
        identity_mismatch_found = False
        event_transition_found = False
        for row in decision_rows:
            if not row.get("task_id"):
                continue
            task = self.store.task(row["task_id"])
            if not task:
                continue
            candidate_evidence = DecisionEvidence.model_validate(task["evidence"])
            result_payload = row.get("provider_result")
            if result_payload is None:
                result_row = self.store.get_result(task["decision_id"])
                result_payload = result_row["provider_result"] if result_row else None
            result = ProviderResult.model_validate(result_payload) if result_payload else None
            if result and result.status.value == "ready":
                historical_ready.append(result)

            if not latest_evidence:
                continue
            same_run_and_config = (
                candidate_evidence.run_id == latest_evidence.run_id
                and task["run_id"] == latest_evidence.run_id
                and task["provider"] == self.provider_mode.value
                and task["model"] == self.config.model
            )
            result_matches_task = result is None or (
                result.run_id == task["run_id"] == candidate_evidence.run_id
                and result.session_id == task["session_id"] == candidate_evidence.session_id
                and result.stream_id == task["stream_id"] == candidate_evidence.stream_id
                and result.provider.value == task["provider"]
                and result.model == task["model"]
                and result.action_epoch == candidate_evidence.action_epoch
                and result.generation == candidate_evidence.generation
                and result.event_key == candidate_evidence.event_key
                and result.event_state == candidate_evidence.event_state
                and result.recovery_status == candidate_evidence.recovery_status
            )
            same_identity = (
                candidate_evidence.action_epoch == latest_evidence.action_epoch
                and candidate_evidence.generation == latest_evidence.generation
                and candidate_evidence.fall_target_epoch == latest_evidence.fall_target_epoch
            )
            same_event = (
                candidate_evidence.event_key == latest_evidence.event_key
                and candidate_evidence.event_state == latest_evidence.event_state
                and candidate_evidence.recovery_status == latest_evidence.recovery_status
            )
            if (result and result.status.value == "ready" and same_run_and_config and result_matches_task
                    and not same_identity):
                identity_mismatch_found = True
            elif (result and result.status.value == "ready" and same_run_and_config and result_matches_task
                  and same_identity and not same_event):
                event_transition_found = True
            is_terminal_task = row.get("task_status") in {"completed", "cancelled"}
            if not (same_target(candidate_evidence) and same_run_and_config):
                continue
            if result is not None and not result_matches_task:
                reasons.append("RESULT_CONFIGURATION_OR_TASK_BINDING_MISMATCH")
                continue
            if not (result or is_terminal_task):
                continue
            compatible_terminals.append((row, result))

        latest_terminal_row, latest_terminal = compatible_terminals[0] if compatible_terminals else (None, None)
        selected = None
        if not reasons and latest_evidence:
            if self.provider_mode == ProviderMode.OFF:
                reasons.append("PROVIDER_MODE_OFF")
            elif latest_terminal_row is None:
                reasons.append("NO_READY_RESULT")
            elif latest_terminal is None:
                reasons.append("LATEST_DECISION_TERMINAL_WITHOUT_RESULT")
                reasons.append(latest_terminal_row.get("task_reason") or "TERMINAL_TASK_HAS_NO_RESULT")
            elif latest_terminal.status.value != "ready":
                reasons.append("LATEST_DECISION_NOT_PUBLISHABLE")
                reasons.extend(latest_terminal.reason_codes or ["PROVIDER_RESULT_NOT_READY"])
            else:
                if latest_terminal_row.get("service_instance_id") != self.service_instance_id:
                    reasons.append("RESULT_FROM_PRIOR_PROCESS")
                elif latest_terminal.source_window.end_s > observation["source_time"]:
                    reasons.append("RESULT_FROM_FUTURE_SOURCE_WINDOW")
                else:
                    source_age = observation["source_time"] - latest_terminal.source_window.end_s
                    wall_age = now - latest_terminal_row["completed_at"]
                    if source_age > self.config.max_source_age_s:
                        reasons.append("RESULT_SOURCE_AGE_EXCEEDED")
                    elif wall_age < 0 or wall_age > self.config.max_wall_age_s:
                        reasons.append("RESULT_WALL_AGE_EXCEEDED")
                    elif latest_evidence.recovery_status.value == "confirmed":
                        reasons.append("UPSTREAM_RECOVERY_CONFIRMED")
                    else:
                        selected = latest_terminal
        if not selected and latest_evidence and identity_mismatch_found:
            reasons.append("RESULT_IDENTITY_OR_GENERATION_MISMATCH")
        if not selected and latest_evidence and event_transition_found:
            reasons.append("UPSTREAM_EVENT_TRANSITION_AFTER_RESULT")

        latest_result = latest_terminal.model_dump(mode="json") if latest_terminal else None
        if latest_terminal and latest_terminal.status.value != "ready":
            reasons.extend(latest_terminal.reason_codes or ["PROVIDER_RESULT_NOT_READY"])

        current_view: DecisionView | None = decision_view_from_result(selected) if selected else None
        latest_task_status = latest_task["status"] if latest_task else None
        latest_terminal_task_status = (latest_terminal_row.get("task_status") if latest_terminal_row else None)
        pending_policy = None
        if (selected and latest_task_status in {"pending", "running"}
                and observation and observation.get("decision_id") != selected.decision_id):
            pending_policy = "reuse_latest_compatible_ready_within_freshness"
        return {
            "schema_version": "agent-decision-projection.v1",
            "session_id": session_id,
            "stream_id": stream_id,
            "latest_snapshot_id": observation["snapshot_id"] if observation else None,
            "latest_source_time_s": observation["source_time"] if observation else None,
            "current_applicable": current_view is not None,
            "fall_probability": current_view.fall_probability if current_view else None,
            "current": current_view.model_dump(mode="json") if current_view else None,
            "historical_result": (historical_ready[0].model_dump(mode="json") if historical_ready else None),
            "latest_result": latest_result,
            "provider_status": (latest_result.get("status") if latest_result
                                else ("off" if self.provider_mode == ProviderMode.OFF
                                      else latest_terminal_task_status or latest_task_status)),
            "configured_provider": self.provider_mode.value,
            "configured_model": self.config.model,
            "latest_task_status": latest_task_status,
            "latest_terminal_task_status": latest_terminal_task_status,
            "pending_policy": pending_policy,
            "stale_reason_codes": list(dict.fromkeys(reasons)),
            "workflow": workflow,
        }

    def list_decisions(self, session_id: str, *, stream_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_decisions(session_id=session_id, stream_id=stream_id, limit=limit)

    def list_decisions_after(self, run_id: str, session_id: str, stream_id: str, *,
                             after_id: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        """Persisted completion-order cursor; source timestamps are metadata only."""
        return self.store.list_results_after(run_id=run_id, session_id=session_id, stream_id=stream_id,
                                             after_id=after_id, limit=limit)

    def stop(self, session_id: str, stream_id: str, *, reason: str = "STOP_REQUESTED") -> dict[str, Any]:
        return self.store.stop(session_id=session_id, stream_id=stream_id, reason=reason, now=self.wall_clock())

    def feedback(self, decision_id: str, *, label: str, request_id: str, expected_version: int) -> dict[str, Any]:
        return self.store.feedback(decision_id=decision_id, label=label, request_id=request_id,
                                   expected_version=expected_version, now=self.wall_clock())


class DecisionWorker:
    """One local worker with an owner-thread SQLite connection and result queue.

    The caller submits only a wake-up token. The worker creates and owns its
    connection on its thread, claims durable work, then returns the receipt via
    a queue. Closing or losing the process leaves the lease recoverable.
    """

    _STOP = object()

    def __init__(self, database_path: str, *, config: DecisionConfig, provider: str | ProviderMode = "mock",
                 queue_size: int = 8, max_queue: int = 1000,
                 service_instance_id: str | None = None, enable_network: bool = False,
                 provider_transport: Any | None = None):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        if type(max_queue) is not int or not 1 <= max_queue <= 10_000:
            raise ValueError("max_queue must be an integer from 1 to 10000")
        self.commands: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self.results: queue.Queue[Any] = queue.Queue()
        self.database_path = database_path
        self.config = config
        self.provider = provider
        self.enable_network = enable_network
        self.provider_transport = provider_transport
        self.max_queue = max_queue
        self.service_instance_id = service_instance_id or _DEFAULT_SERVICE_INSTANCE_ID
        if not isinstance(self.service_instance_id, str) or not self.service_instance_id.strip():
            raise ValueError("service_instance_id must be nonempty")
        self.thread = threading.Thread(target=self._serve, name="fallpre-decision-worker", daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        loop = DecisionLoop(self.database_path, config=self.config, provider=self.provider,
                            max_queue=self.max_queue, service_instance_id=self.service_instance_id,
                            enable_network=self.enable_network, provider_transport=self.provider_transport)
        try:
            while True:
                command = self.commands.get()
                if command is self._STOP:
                    return
                try:
                    self.results.put(loop.run_one())
                except BaseException as error:
                    self.results.put(error)
        finally:
            loop.close()

    def run_one(self, *, timeout: float | None = None) -> dict[str, Any] | None:
        self.commands.put(True, timeout=timeout)
        result = self.results.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self, *, timeout: float = 5.0) -> None:
        if self.thread.is_alive():
            self.commands.put(self._STOP, timeout=timeout)
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise TimeoutError("decision worker did not stop before timeout")

    def __enter__(self) -> DecisionWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DecisionConflict", "DecisionLoop", "DecisionLoopError", "DecisionVersionConflict", "DecisionWorker",
           "InjectedCrash", "InjectedStoreCrash"]
