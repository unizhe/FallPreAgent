"""Adapt schema-v1 perception observations into validated decision evidence.

The original perception object is retained byte-for-byte at the JSON value
level. Optional evidence summaries are explicit: this adapter never invents a
target epoch, generation, score, source window, or resource hash.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictStr, ValidationError, model_validator

from .decision_contracts import (
    DecisionEvidence,
    EventState,
    EvidenceStatus,
    GeometryTransition,
    RecoveryStatus,
)


class DecisionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "agent-observation-envelope.v1"
    session_id: StrictStr
    stream_id: StrictStr
    source_time_s: StrictFloat
    perception: dict[str, Any]
    evidence_summary: dict[str, Any] | None = None
    evidence_origin: StrictStr = "authorized_structured_input"
    fixture_id: StrictStr | None = None

    @model_validator(mode="after")
    def validate_source(self) -> DecisionEnvelope:
        if self.schema_version != "agent-observation-envelope.v1":
            raise ValueError("unsupported envelope schema_version")
        if not self.session_id.strip() or not self.stream_id.strip():
            raise ValueError("session_id and stream_id must be nonempty")
        if not math.isfinite(self.source_time_s) or self.source_time_s < 0:
            raise ValueError("source_time_s must be finite and nonnegative")
        if self.perception.get("schema_version") != 1:
            raise ValueError("perception must preserve schema v1")
        if self.perception.get("stream_id", self.stream_id) != self.stream_id:
            raise ValueError("envelope and perception stream_id differ")
        if self.perception.get("source_time_s", self.source_time_s) != self.source_time_s:
            raise ValueError("envelope and perception source_time_s differ")
        if self.evidence_origin not in {"authorized_structured_input", "synthetic_fixture"}:
            raise ValueError("evidence_origin must identify the input source")
        return self


class EvidenceAdaptation:
    """Result of adapting one envelope; null evidence is a valid recorded state."""

    def __init__(self, *, evidence: DecisionEvidence | None, missing_fields: list[str], evidence_digest: str):
        self.evidence = evidence
        self.missing_fields = tuple(dict.fromkeys(missing_fields))
        self.evidence_digest = evidence_digest


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _event_values(perception: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    event = perception.get("event")
    if not isinstance(event, dict):
        return {}, ["perception.event"]
    values: dict[str, Any] = {}
    key = event.get("event_key")
    if key is not None:
        values["event_key"] = key
    state = event.get("state")
    if state in {item.value for item in EventState}:
        values["event_state"] = state
    else:
        values["event_state"] = EventState.UNKNOWN.value
    recovery = event.get("recovery_status")
    if recovery in {item.value for item in RecoveryStatus}:
        values["recovery_status"] = recovery
    elif event.get("recovery_confirmed") is True:
        values["recovery_status"] = RecoveryStatus.CONFIRMED.value
    else:
        values["recovery_status"] = RecoveryStatus.UNKNOWN.value
    return values, []


def _summary_from_perception(perception: dict[str, Any]) -> dict[str, Any]:
    """Map only explicitly named schema-v1 fields; absent values remain absent."""
    summary: dict[str, Any] = {}
    action = perception.get("current_action")
    if isinstance(action, dict) and isinstance(action.get("label"), str):
        known_actions = {
            "standing", "sitting", "lying", "walking", "falling", "sit_to_stand",
            "standing_to_sitting", "unknown",
        }
        summary["action_label"] = action["label"] if action["label"] in known_actions else "other"
    identity = perception.get("identity")
    if isinstance(identity, dict):
        for source, target in (("action_epoch", "action_epoch"), ("generation", "generation"),
                               ("fall_target_epoch", "fall_target_epoch"), ("identity_link", "identity_link")):
            if source in identity:
                summary[target] = identity[source]
    observation = perception.get("observation")
    if isinstance(observation, dict):
        for field in ("observable", "event_decidable", "camera_stable", "person_count", "pose_valid_fraction",
                      "continuous"):
            if field in observation:
                summary[field] = observation[field]
        if isinstance(observation.get("missing_reason_codes"), list):
            summary["missing_reason_codes"] = observation["missing_reason_codes"]
    temporal = perception.get("temporal_action")
    if isinstance(temporal, dict):
        if "generation" in temporal and "generation" not in summary:
            summary["generation"] = temporal["generation"]
        window = temporal.get("source_window") or temporal.get("window")
        if isinstance(window, dict):
            summary["source_window"] = window
        if "observed_frames" in temporal:
            summary["observed_frames"] = temporal["observed_frames"]
        if "status" in temporal:
            summary["temporal_status"] = temporal["status"]
        if "model_version" in temporal:
            summary["temporal_model_version"] = temporal["model_version"]
        if "fall_score" in temporal:
            summary["temporal_score"] = temporal["fall_score"]
    process = perception.get("fall_process")
    if isinstance(process, dict):
        process_label = process.get("label")
        if isinstance(process_label, str):
            known_process_labels = {"fall_process", "non_fall_activity", "unknown"}
            summary["fall_process_label"] = process_label if process_label in known_process_labels else "other"
        head = process.get("head") if isinstance(process.get("head"), dict) else process
        for source, target in (("status", "head_status"), ("variant", "head_variant"),
                               ("version", "head_version"), ("raw_score", "head_raw_score"),
                               ("weights_sha256", "head_weights_sha256")):
            if source in head:
                summary[target] = head[source]
        if isinstance(process.get("conflict_reason_codes"), list):
            summary["conflict_reason_codes"] = process["conflict_reason_codes"]
        geometry = process.get("geometry")
        if isinstance(geometry, dict):
            status = geometry.get("status")
            if status in {item.value for item in EvidenceStatus}:
                summary["geometry_status"] = status
            transition = geometry.get("transition")
            if transition in {item.value for item in GeometryTransition}:
                summary["geometry_transition"] = transition
            for source, target in (("downward_motion_observed", "geometry_downward_motion_observed"),
                                   ("score", "geometry_score")):
                if source in geometry:
                    summary[target] = geometry[source]
            if isinstance(geometry.get("conflict_reason_codes"), list):
                summary["geometry_conflict_reason_codes"] = geometry["conflict_reason_codes"]
    return summary


_REQUIRED_SUMMARY = (
    "action_epoch", "generation", "identity_link", "source_window", "observed_frames", "observable",
    "event_decidable", "camera_stable", "person_count", "pose_valid_fraction", "continuous", "temporal_status",
    "head_status", "head_variant", "yolo_model_version", "preprocessing_version", "prompt_version",
    "policy_version", "shared_pose_evidence",
)


def adapt_envelope(
    envelope: DecisionEnvelope | dict[str, Any],
    *,
    run_id: str,
    snapshot_id: str,
    decision_id: str,
    request_digest: str,
) -> EvidenceAdaptation:
    """Build a P0-validated immutable-by-snapshot evidence object when complete."""
    item = envelope if isinstance(envelope, DecisionEnvelope) else DecisionEnvelope.model_validate(envelope)
    perception = item.perception
    perception_summary = _summary_from_perception(perception)
    supplied_summary = item.evidence_summary or {}
    source_conflicts = []
    for name in perception_summary.keys() & supplied_summary.keys():
        raw_value = perception_summary[name]
        summary_value = supplied_summary[name]
        if raw_value is None or summary_value is None:
            continue
        if canonical_json(raw_value) != canonical_json(summary_value):
            source_conflicts.append(f"SOURCE_CONFLICT:{name}")
    summary = dict(perception_summary)
    for name, value in supplied_summary.items():
        # A summary fills absent/null source data; it cannot erase a concrete
        # value copied from the immutable original perception payload.
        if name not in summary or summary[name] is None:
            summary[name] = value
    missing = [name for name in _REQUIRED_SUMMARY if name not in summary or summary[name] is None]
    event_values, event_missing = _event_values(perception)
    missing.extend(event_missing)
    digest = sha256_json({"envelope": item.model_dump(mode="json"), "evidence_summary": summary})
    if source_conflicts:
        return EvidenceAdaptation(evidence=None, missing_fields=source_conflicts,
                                  evidence_digest=digest)
    if missing:
        return EvidenceAdaptation(evidence=None, missing_fields=missing, evidence_digest=digest)

    values = dict(summary)
    values.update(
        run_id=run_id,
        session_id=item.session_id,
        stream_id=item.stream_id,
        snapshot_id=snapshot_id,
        decision_id=decision_id,
        request_digest=request_digest,
        source_time_s=item.source_time_s,
        event_key=event_values.get("event_key"),
        event_state=event_values.get("event_state", EventState.UNKNOWN.value),
        recovery_status=event_values.get("recovery_status", RecoveryStatus.UNKNOWN.value),
        missing_reason_codes=summary.get("missing_reason_codes", []),
        conflict_reason_codes=summary.get("conflict_reason_codes", []),
        temporal_model_version=summary.get("temporal_model_version"),
        temporal_score=summary.get("temporal_score"),
        head_version=summary.get("head_version"),
        head_weights_sha256=summary.get("head_weights_sha256"),
    )
    try:
        evidence = DecisionEvidence.model_validate(values)
    except ValidationError as error:
        fields = []
        for detail in error.errors():
            loc = ".".join(str(part) for part in detail.get("loc", ()))
            fields.append(loc or "evidence_summary")
        return EvidenceAdaptation(evidence=None, missing_fields=fields, evidence_digest=digest)
    return EvidenceAdaptation(evidence=evidence, missing_fields=[], evidence_digest=digest)


def synthetic_envelope(*, session_id: str, stream_id: str, source_time_s: float, suffix: str,
                       sufficient: bool = True, action_epoch: int = 3, generation: int = 2,
                       event_state: str = "normal", event_key: str | None = None,
                       recovery_status: str = "not_established") -> dict[str, Any]:
    """Full schema-v1 envelope; every fixture resource is visibly synthetic."""
    end = float(source_time_s)
    perception = {
        "schema_version": 1,
        "stream_id": stream_id,
        "source_time_s": end,
        "identity": {"action_epoch": action_epoch, "generation": generation,
                     "fall_target_epoch": f"fixture-target-{action_epoch}", "identity_link": "same_target"},
        "observation": {"observable": True, "event_decidable": sufficient, "camera_stable": True,
                        "person_count": 1, "pose_valid_fraction": 0.92, "continuous": True,
                        "missing_reason_codes": [] if sufficient else ["FIXTURE_WINDOW_INSUFFICIENT"]},
        "temporal_action": {"status": "ready" if sufficient else "insufficient",
                            "source_window": {"start_s": end - 2.0, "end_s": end - 0.05},
                            "observed_frames": 8 if sufficient else 3,
                            "model_version": "synthetic-fixture-no-model-v1", "fall_score": 0.71 if sufficient else None},
        "fall_process": {"label": "fall_process" if event_state in {"suspected", "confirmed"} else "non_fall_activity",
                          "head": {"status": "ready" if sufficient else "insufficient",
                                   "variant": "tail_joint", "version": "synthetic-fixture-no-weights-v1",
                                   "raw_score": 0.82 if sufficient else None},
                          "geometry": {"status": "ready" if sufficient else "insufficient",
                                       "transition": "upright_to_low" if event_state == "suspected" else "upright",
                                       "downward_motion_observed": event_state == "suspected",
                                       "score": 0.77 if sufficient else None,
                                       "conflict_reason_codes": []}},
        "event": {"state": event_state, "event_key": event_key,
                  "recovery_status": recovery_status, "recovery_confirmed": recovery_status == "confirmed",
                  "transitions": []},
        "current_action": {"label": "standing"},
    }
    summary = {
        "action_epoch": action_epoch,
        "generation": generation,
        "fall_target_epoch": f"fixture-target-{action_epoch}",
        "identity_link": "same_target",
        "source_window": {"start_s": end - 2.0, "end_s": end - 0.05},
        "observed_frames": 8 if sufficient else 3,
        "observable": True,
        "event_decidable": sufficient,
        "camera_stable": True,
        "person_count": 1,
        "pose_valid_fraction": 0.92,
        "continuous": True,
        "missing_reason_codes": [] if sufficient else ["FIXTURE_WINDOW_INSUFFICIENT"],
        "temporal_status": "ready" if sufficient else "insufficient",
        "temporal_model_version": "synthetic-fixture-no-model-v1" if sufficient else None,
        "temporal_score": 0.71 if sufficient else None,
        "head_status": "ready" if sufficient else "insufficient",
        "head_variant": "tail_joint",
        "head_version": "synthetic-fixture-no-weights-v1" if sufficient else None,
        "head_raw_score": 0.82 if sufficient else None,
        "action_label": perception["current_action"]["label"],
        "fall_process_label": perception["fall_process"]["label"],
        "geometry_status": perception["fall_process"]["geometry"]["status"],
        "geometry_transition": perception["fall_process"]["geometry"]["transition"],
        "geometry_downward_motion_observed": perception["fall_process"]["geometry"]["downward_motion_observed"],
        "geometry_score": perception["fall_process"]["geometry"]["score"],
        "geometry_conflict_reason_codes": [],
        "conflict_reason_codes": [],
        "yolo_model_version": "synthetic-fixture-no-yolo-v1",
        "preprocessing_version": "synthetic-fixture-preprocessing-v1",
        "prompt_version": "synthetic-fixture-prompt-v1",
        "policy_version": "synthetic-fixture-policy-v1",
        "shared_pose_evidence": True,
    }
    return {
        "schema_version": "agent-observation-envelope.v1",
        "session_id": session_id,
        "stream_id": stream_id,
        "source_time_s": float(source_time_s),
        "perception": perception,
        "evidence_summary": summary,
        "evidence_origin": "synthetic_fixture",
        "fixture_id": f"decision-loop-{suffix}",
    }


__all__ = ["DecisionEnvelope", "EvidenceAdaptation", "adapt_envelope", "canonical_json", "sha256_json",
           "synthetic_envelope"]
