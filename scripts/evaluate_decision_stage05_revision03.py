#!/usr/bin/env python3
"""P4 offline revision 03: original observation identity and deterministic label conflicts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_decision_stage05 as base  # noqa: E402

from fallpre.decision_contracts import DecisionEvidence  # noqa: E402
from fallpre.decision_evidence import adapt_envelope, canonical_json, sha256_json  # noqa: E402
from fallpre.decision_state import DECISION_PROMPT_VERSION, DECISION_STATE_SCHEMA_VERSION  # noqa: E402

DEFAULT_DB = base.DEFAULT_DB
DEFAULT_RUN_DIR = ROOT / "runs/decision-stage05-revision-03-20261118"
OLD_RUN_DIR = ROOT / "runs/decision-stage05-revision-02-20261118"
OLD_REPORT = ROOT / "output/agent-review/stage-05/report-revision-02.md"
REVIEW_PATH = ROOT / "output/agent-review/stage-05/reviewer-review-revision-02.md"
OLD_PROBE_DIR = ROOT / "output/agent-review/stage-05/audit-revision-02"
ORIGINAL_READY_PROOF = ROOT / "output/agent-review/stage-04/independent-revision-02/original-ready-input-associations.json"
VERIFIED_LABEL_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {}

# Each run/view is evaluated on its own evidence. The mixed route intentionally
# preserves the prior three-run denominator while raw and composite views are
# also reported separately.
EVENT_VIEW_RUNS = [
    {"run_id": "run-9c09f349998c4b5bb4ccb7f61258b6ec", "video_id": "urfd-adl-04", "input_view": "single_camera_rgb", "cohort": "single_camera_raw"},
    {"run_id": "run-e6d2f672ac064403be865a4b86cd4851", "video_id": "urfd-fall-02", "input_view": "single_camera_rgb", "cohort": "single_camera_raw"},
    {"run_id": "run-0747f0212efb42f1b106a07ecaf4e8ab", "video_id": "urfd-adl-04", "input_view": "three_panel_derived", "cohort": "three_panel_derived"},
    {"run_id": "run-75870cb62dda4919b29865ca2ad77da6", "video_id": "urfd-fall-01", "input_view": "three_panel_derived", "cohort": "three_panel_derived"},
]
MIXED_ROUTE_RUNS = {
    "run-9c09f349998c4b5bb4ccb7f61258b6ec",
    "run-75870cb62dda4919b29865ca2ad77da6",
    "run-e6d2f672ac064403be865a4b86cd4851",
}
EVENT_REFERENCE_RUNS = {
    "urfd-fall-01-fall-process-01": "run-75870cb62dda4919b29865ca2ad77da6",
    "urfd-fall-02-fall-process-01": "run-e6d2f672ac064403be865a4b86cd4851",
}
EVENT_POST_TOLERANCE_S = 2.5
WINDOW_EVENT_OVERLAP_RULE = (
    "A window covers an event only when its typed DecisionEvidence is eligible, "
    "it belongs to the event's designated run and source-video group, its half-open "
    "source window [start_s,end_s) overlaps the official event interval [start_s,end_s) "
    "by strictly more than 0 seconds, and the observation source time is no later than "
    "2.5 seconds after the official event interval ends. A window only before onset does not cover."
)
REVIEWED_STATUSES = {"complete", "completed", "human_reviewed", "human_confirmed", "adjudicated"}
DRAFT_STATUSES = {"draft"}
PENDING_STATUSES = {"needs_review", "pending", "pending_adjudication", "awaiting_adjudication"}
VALID_SCORABLE_LABELS = {"normal_activity", "already_lying", "fall_process_visible"}
VERSION_FIELDS = [
    "mapping_version", "eligibility_policy_version", "preprocessing_version",
    "yolo_model_version", "temporal_model_version", "head_variant", "head_version",
    "head_weights_sha256", "policy_version", "prompt_version", "provider",
    "model_version", "calibration_version",
]
REQUIRED_VERSION_FIELDS = [field for field in VERSION_FIELDS if field != "calibration_version"]
LABEL_FIELDS = [
    "canonical_window_id", "video_id", "source_video_group_id", "artifact_id",
    "source_video_sha256", "partition", "window_start_s", "window_end_s", "source_time_s", "task_id",
    "decision_id", "label", "label_source", "label_source_type", "review_record_id",
    "review_record_sha256", "reviewed_by", "reviewed_at", "annotator", "annotation_status",
    "event_start_s", "event_end_s", "disagreement_notes",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in fields
            })


def sha256_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_evidence(raw: str | None) -> tuple[DecisionEvidence | None, list[str], str]:
    if not raw:
        return None, ["EVIDENCE_JSON_MISSING"], "missing"
    try:
        evidence = DecisionEvidence.model_validate_json(raw)
    except Exception as error:  # export the validation state, never treat it as eligible
        return None, [f"EVIDENCE_INVALID:{type(error).__name__}"], "invalid"
    return evidence, list(evidence.eligibility_reasons) + list(evidence.conflict_reason_codes) + list(evidence.geometry_conflict_reason_codes), "valid"


def provider_settings(run: dict[str, Any]) -> dict[str, Any]:
    receipt = base.decode_json(run.get("receipt_json"), {})
    status = receipt.get("provider_status", {}) if isinstance(receipt, dict) else {}
    mode = status.get("provider")
    return {
        "provider_mode": mode,
        "provider_switch_enabled": mode not in (None, "off"),
        "provider_network_enabled": bool(status.get("network_enabled")),
        "provider_region": status.get("region"),
    }


def verify_original_ready_proof(path: Path = ORIGINAL_READY_PROOF) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "unchanged": False, "by_decision_id": {}}
    proof = json.loads(path.read_text())
    before = proof.get("ready_associations_before", [])
    after = proof.get("ready_associations_after", [])
    unchanged = bool(proof.get("associations_unchanged")) and before == after
    return {
        "available": True,
        "unchanged": unchanged,
        "sha256": base.sha256_file(path),
        "integrity_before": proof.get("integrity_check_before"),
        "integrity_after": proof.get("integrity_check_after"),
        "by_decision_id": {row["decision_id"]: row for row in after},
    }


def build_verified_label_source_registry(frozen_groups: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index review records from the preserved, hash-locked human review file."""
    review_doc = base.read_json(base.LABEL_SOURCE)
    source_sha256 = base.sha256_file(base.LABEL_SOURCE)
    frozen_by_video = {str(row["video_id"]): row for row in frozen_groups["groups"]}
    registry = {}
    for interval in review_doc.get("intervals", []):
        review_id = str(interval.get("review_id") or "")
        frozen = frozen_by_video.get(str(interval.get("clip_id") or ""), {})
        if not (
            review_id and interval.get("review_status") == "human_confirmed"
            and interval.get("reviewed_by") and interval.get("reviewed_at")
            and interval.get("video_sha256") and frozen
        ):
            continue
        registry[review_id] = {
            "review_record_id": review_id,
            "source_type": "human_adjudication",
            "source_artifact_path": str(base.LABEL_SOURCE.resolve().relative_to(ROOT)),
            "source_artifact_sha256": source_sha256,
            "review_record_sha256": sha256_json(interval),
            "label_source": str(review_doc.get("source") or ""),
            "video_id": str(interval["clip_id"]),
            "source_video_group_id": str(frozen["source_video_group_id"]),
            "source_video_sha256": str(interval["video_sha256"]),
            "window_start_s_min": float(interval["start_s"]),
            "window_end_s_max": float(interval["end_s"]),
            "verified_target_label": "normal_activity" if interval.get("label") == "non_fall_activity" else interval.get("label"),
            "annotation_status": str(interval["review_status"]),
            "reviewed_by": str(interval["reviewed_by"]),
            "reviewed_at": str(interval["reviewed_at"]),
        }
    return registry


def evidence_registration_audit(
    raw: dict[str, Any],
    evidence: DecisionEvidence | None,
    evidence_json: str | None,
) -> dict[str, Any]:
    """Re-derive evidence from its registered source envelope and compare content."""
    envelope = base.decode_json(raw.get("envelope_json"), None)
    envelope_identity_matches = bool(
        isinstance(envelope, dict)
        and envelope.get("session_id") == raw.get("session_id")
        and envelope.get("stream_id") == raw.get("stream_id")
        and float_matches(envelope.get("source_time_s"), raw.get("source_time"))
    )
    envelope_digest_matches = bool(
        isinstance(envelope, dict)
        and raw.get("digest")
        and sha256_json(envelope) == raw.get("digest")
        and envelope_identity_matches
    )
    adapted = None
    adaptation_error = None
    try:
        if isinstance(envelope, dict):
            adapted = adapt_envelope(
                envelope,
                run_id=str(raw.get("run_id") or ""),
                snapshot_id=str(raw.get("snapshot_id") or ""),
                decision_id=str(raw.get("decision_id") or ""),
                request_digest=str(raw.get("request_digest") or ""),
            )
    except Exception as error:
        adaptation_error = type(error).__name__
    reconstructed = adapted.evidence if adapted else None
    body_matches_registration = bool(
        evidence is not None and reconstructed is not None
        and canonical_json(evidence.model_dump(mode="json"))
        == canonical_json(reconstructed.model_dump(mode="json"))
    )
    evidence_digest_matches_registration = bool(
        adapted and raw.get("evidence_digest")
        and adapted.evidence_digest == raw.get("evidence_digest")
    )
    identity_checks = {
        "evidence_run_matches_observation": bool(evidence and evidence.run_id == raw.get("run_id")),
        "evidence_session_matches_observation": bool(evidence and evidence.session_id == raw.get("session_id")),
        "evidence_stream_matches_observation": bool(evidence and evidence.stream_id == raw.get("stream_id")),
        "evidence_snapshot_matches_observation": bool(evidence and evidence.snapshot_id == raw.get("snapshot_id")),
        "evidence_decision_matches_observation": bool(evidence and evidence.decision_id == raw.get("decision_id")),
        "evidence_request_matches_observation": bool(evidence and evidence.request_digest == raw.get("request_digest")),
        "evidence_source_time_matches_observation": bool(evidence and float_matches(evidence.source_time_s, raw.get("source_time"))),
        "envelope_registration_matches_observation": envelope_digest_matches,
        "evidence_body_matches_rederived_registration": body_matches_registration,
        "evidence_digest_matches_rederived_registration": evidence_digest_matches_registration,
    }
    reasons = [name.upper() for name, matched in identity_checks.items() if not matched]
    if adaptation_error:
        reasons.append(f"EVIDENCE_RECONSTRUCTION_FAILED:{adaptation_error}")
    return {
        "evidence_identity_checks": identity_checks,
        "evidence_registration_verified": all(identity_checks.values()),
        "evidence_registration_reasons": reasons,
        "reconstructed_evidence_digest": adapted.evidence_digest if adapted else None,
        "reconstructed_evidence_available": reconstructed is not None,
        "evidence_body_matches_registered_content": body_matches_registration,
        "registered_envelope_digest": raw.get("digest"),
        "reconstructed_evidence_digest_matches": evidence_digest_matches_registration,
        "evidence_reconstruction_error": adaptation_error,
    }


def _export_evidence_observations_revision02(
    connection: sqlite3.Connection,
    proof: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations, db_counts = base.export_observation_rows(connection)
    raw_obs = {row["id"]: dict(row) for row in connection.execute("SELECT * FROM decision_observations")}
    runs = {row["run_id"]: dict(row) for row in connection.execute("SELECT * FROM agent_runs")}
    tasks = {row["task_id"]: dict(row) for row in connection.execute("SELECT * FROM decision_tasks")}
    results_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM decision_results ORDER BY completed_at,attempt_no"):
        results_by_decision[row["decision_id"]].append(dict(row))
    frames = {
        (row["run_id"], round(float(row["source_time"]), 6)): dict(row)
        for row in connection.execute("SELECT * FROM agent_frame_receipts")
    }

    output: list[dict[str, Any]] = []
    for row in observations:
        raw = raw_obs[row["observation_row_id"]]
        evidence, evidence_reasons, evidence_status = parse_evidence(raw.get("evidence_json"))
        task = tasks.get(row.get("task_id"), {})
        result_records = results_by_decision.get(row.get("decision_id"), [])
        result_record = next((item for item in reversed(result_records) if item.get("accepted")), result_records[-1] if result_records else {})
        result = base.decode_json(result_record.get("result_json"), {})
        run = runs.get(row["run_id"], {})
        provider = provider_settings(run)
        frame = frames.get((row["run_id"], round(float(row["source_time_s"]), 6)), {})
        response = base.decode_json(frame.get("response_json"), {})
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        fall = payload.get("fall", {}) if isinstance(payload, dict) else {}
        fall_event_id = fall.get("event_id") or fall.get("current_event_id")
        visual_confirmation = bool(fall.get("new_event") and fall_event_id)
        evidence_audit = evidence_registration_audit(raw, evidence, raw.get("evidence_json"))
        if evidence:
            row["window_start_s"] = float(evidence.source_window.start_s)
            row["window_end_s"] = float(evidence.source_window.end_s)
            row["window_id"] = base.canonical_window_id(row["video_id"], row["window_start_s"], row["window_end_s"])
            row["actual_source_samples_in_window"] = evidence.observed_frames
            row["eligibility_policy_version"] = evidence.eligibility_policy_version
            row["preprocessing_version"] = evidence.preprocessing_version
            row["yolo_model_version"] = evidence.yolo_model_version
            row["temporal_model_version"] = evidence.temporal_model_version
            row["head_variant"] = evidence.head_variant
            row["head_version"] = evidence.head_version
            row["head_weights_sha256"] = evidence.head_weights_sha256
            row["policy_version"] = evidence.policy_version
            row["prompt_version"] = evidence.prompt_version
            row["raw_temporal_score_not_probability"] = evidence.temporal_score
            row["raw_head_score_not_probability"] = evidence.head_raw_score
            row["model_derived_fall_process_label_excluded_from_truth"] = evidence.fall_process_label
        proof_row = proof.get("by_decision_id", {}).get(row.get("decision_id"), {})
        internal_association_checks = {
            "evidence_identity_matches_observation": bool(evidence and evidence.run_id == row["run_id"] and evidence.snapshot_id == raw["snapshot_id"] and evidence.decision_id == raw["decision_id"] and evidence.request_digest == raw["request_digest"]),
            "task_identity_matches_observation": bool(task and task.get("run_id") == row["run_id"] and task.get("snapshot_id") == raw["snapshot_id"] and task.get("decision_id") == raw["decision_id"] and task.get("request_digest") == raw["request_digest"] and task.get("evidence_digest") == raw["evidence_digest"]),
            "result_identity_matches_task": bool(result_record and result_record.get("task_id") == row.get("task_id") and result_record.get("decision_id") == row.get("decision_id") and result.get("run_id") == row["run_id"] and result.get("snapshot_id") == raw["snapshot_id"] and result.get("decision_id") == raw["decision_id"] and result.get("source_window", {}).get("start_s") == row.get("window_start_s") and result.get("source_window", {}).get("end_s") == row.get("window_end_s")),
            "source_frame_digest_present": bool(row.get("input_frame_sha256")),
        }
        task_evidence = base.decode_json(task.get("evidence_json"), None)
        observation_evidence = base.decode_json(raw.get("evidence_json"), None)
        task_evidence_matches_observation = bool(
            task_evidence is not None and observation_evidence is not None
            and canonical_json(task_evidence) == canonical_json(observation_evidence)
        )
        task_evidence_version_matches = bool(
            task_evidence is not None and evidence is not None
            and all(task_evidence.get(field) == getattr(evidence, field, None) for field in REQUIRED_VERSION_FIELDS if field in task_evidence)
        )
        task_identity_matches_observation = bool(
            task and task.get("run_id") == row["run_id"]
            and task.get("session_id") == raw.get("session_id")
            and task.get("stream_id") == raw.get("stream_id")
            and task.get("snapshot_id") == raw.get("snapshot_id")
            and task.get("decision_id") == raw.get("decision_id")
            and task.get("request_digest") == raw.get("request_digest")
            and float_matches(task.get("source_time"), raw.get("source_time"))
            and task.get("evidence_digest") == raw.get("evidence_digest")
        )
        task_registered_checks = {
            "task_identity_matches_observation": task_identity_matches_observation,
            "task_evidence_body_matches_observation": task_evidence_matches_observation,
            "task_evidence_digest_matches_registered_body": bool(task and task.get("evidence_digest") == raw.get("evidence_digest") == evidence_audit.get("reconstructed_evidence_digest")),
            "task_evidence_versions_match_observation": task_evidence_version_matches,
        }
        result_identity_matches_task = bool(
            result_record and result_record.get("accepted")
            and result_record.get("task_id") == row.get("task_id")
            and result_record.get("decision_id") == row.get("decision_id")
            and float_matches(result_record.get("source_time"), raw.get("source_time"))
            and result.get("run_id") == row["run_id"]
            and result.get("session_id") == raw.get("session_id")
            and result.get("stream_id") == raw.get("stream_id")
            and result.get("snapshot_id") == raw.get("snapshot_id")
            and result.get("decision_id") == raw.get("decision_id")
            and result.get("provider") == task.get("provider")
            and result.get("model") == task.get("model")
            and float_matches(result.get("source_window", {}).get("start_s"), row.get("window_start_s"))
            and float_matches(result.get("source_window", {}).get("end_s"), row.get("window_end_s"))
            and result.get("fall_probability") == row.get("probability")
        )
        frame_agent = payload.get("agent", {}) if isinstance(payload, dict) else {}
        frame_registration_checks = {
            "frame_registration_status_registered": frame.get("agent_registration_status") == "registered",
            "frame_agent_run_matches_observation": frame_agent.get("run_id") == row.get("run_id"),
            "frame_agent_decision_matches_observation": frame_agent.get("decision_id") == row.get("decision_id"),
            "frame_agent_task_matches_observation": frame_agent.get("task_id") == row.get("task_id"),
            "frame_agent_snapshot_matches_observation": frame_agent.get("snapshot_id") == raw.get("snapshot_id"),
        }
        proof_matches = bool(
            proof.get("unchanged") and proof_row
            and proof_row.get("input_registration_status") == "registered"
            and proof_row.get("decision_id") == row.get("decision_id")
            and proof_row.get("task_id") == row.get("task_id")
            and proof_row.get("snapshot_id") == raw.get("snapshot_id")
            and proof_row.get("evidence_digest") == raw.get("evidence_digest")
            and proof_row.get("request_digest") == raw.get("request_digest")
            and proof_row.get("input_image_sha256") == row.get("input_frame_sha256")
            and proof_row.get("input_envelope_sha256") == sha256_text(frame.get("observation_envelope_json"))
            and proof_row.get("input_response_sha256") == sha256_text(frame.get("response_json"))
            and proof_row.get("provider_request_id") == row.get("provider_request_id")
            and proof_row.get("model") == row.get("model_version")
            and proof_row.get("provider") == row.get("provider")
            and proof_row.get("source_time") == row.get("source_time_s")
            and proof_row.get("source_window", {}).get("start_s") == row.get("window_start_s")
            and proof_row.get("source_window", {}).get("end_s") == row.get("window_end_s")
            and proof_row.get("fall_probability") == row.get("probability")
            and proof_row.get("probability_kind") == row.get("probability_kind")
            and proof_row.get("status") == row.get("result_status")
            and all(frame_registration_checks.values())
        )
        internal_association_checks = {
            **evidence_audit["evidence_identity_checks"],
            **task_registered_checks,
            "result_identity_matches_task": result_identity_matches_task,
            **frame_registration_checks,
            "source_frame_digest_present": bool(row.get("input_frame_sha256")),
            "original_ready_input_proof_matches": proof_matches if row.get("result_status") == "ready" else True,
        }
        ready_input_verified = bool(
            evidence_audit["evidence_registration_verified"]
            and all(task_registered_checks.values())
            and result_identity_matches_task
            and proof_matches
        )
        result_status = result.get("status")
        metadata = result.get("provider_metadata") or {}
        http_status = metadata.get("http_status")
        provider_response_valid = bool(
            result_record and result_record.get("accepted")
            and result.get("provider") == provider.get("provider_mode")
            and result.get("provider") not in (None, "off")
            and result_status == "ready"
            and isinstance(http_status, int) and 200 <= http_status < 300
            and metadata.get("request_id")
        )
        row.update({
            **provider,
            **evidence_audit,
            "evidence_validation_status": evidence_status if evidence_audit["evidence_registration_verified"] else "registered_body_mismatch",
            "evidence_contract_eligible": bool(evidence and evidence.eligible),
            "evidence_gate_eligible": bool(evidence and evidence.eligible and evidence_audit["evidence_registration_verified"]),
            "evidence_eligibility_reasons": evidence_reasons + evidence_audit["evidence_registration_reasons"],
            "evidence_source_time_s": evidence.source_time_s if evidence else None,
            "evidence_event_state": evidence.event_state.value if evidence else None,
            "evidence_observable": evidence.observable if evidence else None,
            "evidence_event_decidable": evidence.event_decidable if evidence else None,
            "visual_fall_state": fall.get("state"),
            "visual_confirmation": visual_confirmation,
            "visual_event_id": str(fall_event_id) if fall_event_id else None,
            "provider_response_present": bool(result_record),
            "provider_response_accepted": bool(result_record.get("accepted")) if result_record else False,
            "provider_response_valid": provider_response_valid,
            "provider_response_status": result_status,
            "provider_response_http_status": http_status,
            "input_association_checks": internal_association_checks,
            "original_ready_input_proof_status": (
                "verified" if ready_input_verified
                else "mismatch" if proof_row
                else "not_available"
            ),
            "task_evidence_body_matches_observation": task_evidence_matches_observation,
            "task_evidence_versions_match_observation": task_evidence_version_matches,
            "original_input_association_verified": ready_input_verified if row.get("result_status") == "ready" else evidence_audit["evidence_registration_verified"],
            "version_record_complete": all(row.get(field) for field in REQUIRED_VERSION_FIELDS) and task_evidence_version_matches if row.get("result_status") == "ready" else all(row.get(field) for field in REQUIRED_VERSION_FIELDS),
            "label": None,
            "label_source": None,
            "label_annotator": None,
            "label_review_status": "needs_review" if row.get("window_id") else "no_source_window",
            "label_inclusion_status": "not_evaluated",
            "label_exclusion_reasons": [],
        })
        output.append(row)
    return output, db_counts


def reconstruct_original_observation_identity(raw: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Rebuild immutable observe() identity from the original envelope and run config."""
    envelope = base.decode_json(raw.get("envelope_json"), None)
    if not isinstance(envelope, dict):
        return {"available": False, "envelope": None}
    summary = envelope.get("evidence_summary") or {}
    provider = provider_settings(run).get("provider_mode") or raw.get("provider") or "off"
    model = raw.get("model") or base.decode_json(run.get("receipt_json"), {}).get("model")
    if not model or not isinstance(summary, dict):
        return {"available": False, "envelope": envelope}
    request_material = {
        "envelope": envelope,
        "evidence_summary": summary,
        "provider": provider,
        "model": model,
        "fall_head": summary.get("head_variant"),
        "preprocessing_version": summary.get("preprocessing_version"),
        "yolo_model_version": summary.get("yolo_model_version"),
        "head_version": summary.get("head_version"),
        "head_weights_sha256": summary.get("head_weights_sha256"),
        "prompt_version": DECISION_PROMPT_VERSION if provider == "aliyun" else summary.get("prompt_version"),
        "state_schema_version": DECISION_STATE_SCHEMA_VERSION if provider == "aliyun" else None,
        "policy_version": summary.get("policy_version"),
        "eligibility_policy_version": "decision-eligibility-p0-v1",
        "parent_decision_id": raw.get("parent_decision_id"),
    }
    request_digest = sha256_json(request_material)
    return {
        "available": True,
        "envelope": envelope,
        "registered_snapshot_id": f"snapshot-{sha256_json(envelope)[:24]}",
        "registered_decision_id": f"decision-{request_digest[:32]}",
        "registered_request_digest": request_digest,
        "registered_task_id_if_new": f"task-{request_digest[:32]}",
        "registered_provider": provider,
        "registered_model": model,
        "request_material_sha256": request_digest,
    }


def evidence_registration_audit_revision03(
    raw: dict[str, Any],
    evidence: DecisionEvidence | None,
    evidence_json: str | None,
    identity: dict[str, Any],
    frame: dict[str, Any],
) -> dict[str, Any]:
    """Separate missing evidence from unverifiable origin and confirmed conflicts."""
    envelope = identity.get("envelope")
    frame_response = base.decode_json(frame.get("response_json"), {}) if frame else {}
    payload = frame_response.get("payload", {}) if isinstance(frame_response, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    frame_agent = payload.get("agent", {})
    if not isinstance(frame_agent, dict):
        frame_agent = {}
    frame_envelope = base.decode_json(frame.get("observation_envelope_json"), None) if frame else None
    registered_envelope = envelope if isinstance(envelope, dict) else {}
    origin_available = bool(
        identity.get("available") and isinstance(envelope, dict) and frame
        and isinstance(frame_response, dict) and isinstance(frame_agent, dict)
    )
    reconstructed = None
    adapted = None
    adaptation_error = None
    if origin_available:
        try:
            adapted = adapt_envelope(
                envelope,
                run_id=str(raw.get("run_id") or ""),
                snapshot_id=str(identity["registered_snapshot_id"]),
                decision_id=str(identity["registered_decision_id"]),
                request_digest=str(identity["registered_request_digest"]),
            )
            reconstructed = adapted.evidence
        except Exception as error:
            adaptation_error = type(error).__name__
    body_matches_registration = bool(
        evidence is not None and reconstructed is not None
        and canonical_json(evidence.model_dump(mode="json"))
        == canonical_json(reconstructed.model_dump(mode="json"))
    )
    evidence_digest_matches_registration = bool(
        adapted and raw.get("evidence_digest")
        and adapted.evidence_digest == raw.get("evidence_digest")
    )
    frame_envelope_matches = bool(
        isinstance(frame_envelope, dict) and isinstance(envelope, dict)
        and all(envelope.get(key) == value for key, value in frame_envelope.items())
    )
    checks = {
        "evidence_run_matches_original_registration": bool(evidence and evidence.run_id == raw.get("run_id")),
        "evidence_session_matches_original_registration": bool(evidence and evidence.session_id == raw.get("session_id") == registered_envelope.get("session_id")),
        "evidence_stream_matches_original_registration": bool(evidence and evidence.stream_id == raw.get("stream_id") == registered_envelope.get("stream_id")),
        "evidence_snapshot_matches_original_registration": bool(evidence and evidence.snapshot_id == identity.get("registered_snapshot_id") == raw.get("snapshot_id")),
        "evidence_decision_matches_original_registration": bool(evidence and evidence.decision_id == identity.get("registered_decision_id") and frame_agent.get("decision_id") == identity.get("registered_decision_id")),
        "evidence_request_matches_original_registration": bool(evidence and evidence.request_digest == identity.get("registered_request_digest")),
        "evidence_source_time_matches_original_registration": bool(evidence and float_matches(evidence.source_time_s, raw.get("source_time")) and float_matches(registered_envelope.get("source_time_s"), raw.get("source_time"))),
        "raw_envelope_digest_matches_registration": bool(registered_envelope and raw.get("digest") and sha256_json(registered_envelope) == raw.get("digest")),
        "registered_frame_receipt_matches_run_time": bool(frame and frame.get("run_id") == raw.get("run_id") and float_matches(frame.get("source_time"), raw.get("source_time"))),
        "registered_frame_status_is_registered": bool(frame and frame.get("agent_registration_status") == "registered"),
        "registered_frame_agent_run_matches": bool(frame_agent.get("run_id") == raw.get("run_id")),
        "registered_frame_agent_snapshot_matches": bool(frame_agent.get("snapshot_id") == identity.get("registered_snapshot_id")),
        "registered_frame_agent_task_matches_original_observation": bool(frame_agent.get("task_id") == raw.get("task_id")),
        "registered_frame_input_envelope_matches": frame_envelope_matches,
        "evidence_body_matches_rederived_original_registration": body_matches_registration,
        "evidence_digest_matches_rederived_original_registration": evidence_digest_matches_registration,
    }
    if not evidence_json:
        status = "missing"
        reasons = ["EVIDENCE_BODY_MISSING"]
    elif not origin_available:
        status = "unverifiable"
        reasons = ["ORIGINAL_REGISTRATION_INPUTS_UNAVAILABLE"]
    elif not all(checks.values()) or adaptation_error:
        status = "conflict"
        reasons = [name.upper() for name, matched in checks.items() if not matched]
        if adaptation_error:
            reasons.append(f"ORIGINAL_EVIDENCE_RECONSTRUCTION_FAILED:{adaptation_error}")
    else:
        status = "verified"
        reasons = []
    return {
        "evidence_identity_checks": checks,
        "evidence_registration_status": status,
        "evidence_registration_verified": status == "verified",
        "evidence_registration_reasons": reasons,
        "evidence_registration_origin_available": origin_available,
        "evidence_body_matches_registered_content": body_matches_registration,
        "reconstructed_evidence_available": reconstructed is not None,
        "reconstructed_evidence_digest": adapted.evidence_digest if adapted else None,
        "registered_envelope_digest": raw.get("digest"),
        "reconstructed_evidence_digest_matches": evidence_digest_matches_registration,
        "evidence_reconstruction_error": adaptation_error,
    }


def export_evidence_observations(
    connection: sqlite3.Connection,
    proof: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations, db_counts = base.export_observation_rows(connection)
    raw_obs = {row["id"]: dict(row) for row in connection.execute("SELECT * FROM decision_observations")}
    runs = {row["run_id"]: dict(row) for row in connection.execute("SELECT * FROM agent_runs")}
    tasks = {row["task_id"]: dict(row) for row in connection.execute("SELECT * FROM decision_tasks")}
    frames = {
        (row["run_id"], round(float(row["source_time"]), 6)): dict(row)
        for row in connection.execute("SELECT * FROM agent_frame_receipts")
    }
    task_observation_disposition = {
        (row["task_id"], int(row["observation_id"])): row["disposition"]
        for row in connection.execute("SELECT task_id,observation_id,disposition FROM decision_task_observations")
    }
    results_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result_row in connection.execute("SELECT * FROM decision_results ORDER BY completed_at,attempt_no"):
        results_by_task[result_row["task_id"]].append(dict(result_row))
    task_evidence_by_id = {task_id: base.decode_json(task.get("evidence_json"), None) for task_id, task in tasks.items()}

    identities: dict[int, dict[str, Any]] = {}
    parsed_evidence: dict[int, tuple[DecisionEvidence | None, list[str], str]] = {}
    audits: dict[int, dict[str, Any]] = {}
    for row in observations:
        row_id = int(row["observation_row_id"])
        raw = raw_obs[row_id]
        run = runs.get(row["run_id"], {})
        frame = frames.get((row["run_id"], round(float(row["source_time_s"]), 6)), {})
        identity = reconstruct_original_observation_identity(raw, run)
        evidence, evidence_reasons, evidence_status = parse_evidence(raw.get("evidence_json"))
        identities[row_id] = identity
        parsed_evidence[row_id] = (evidence, evidence_reasons, evidence_status)
        audits[row_id] = evidence_registration_audit_revision03(raw, evidence, raw.get("evidence_json"), identity, frame)

    # A mutable task can point at many observations after coalescing. Recover
    # the one actual task input from its current identity and evidence body.
    task_input_candidates: dict[str, list[int]] = defaultdict(list)
    task_input_check_templates: dict[str, dict[str, bool]] = {}
    for (task_id, observation_id), disposition in task_observation_disposition.items():
        task = tasks.get(task_id, {})
        raw = raw_obs.get(observation_id, {})
        identity = identities.get(observation_id, {})
        frame = frames.get((raw.get("run_id"), round(float(raw.get("source_time") or 0), 6)), {})
        frame_response = base.decode_json(frame.get("response_json"), {})
        frame_agent = frame_response.get("payload", {}).get("agent", {}) if isinstance(frame_response, dict) else {}
        task_evidence = task_evidence_by_id.get(task_id)
        observation_evidence = base.decode_json(raw.get("evidence_json"), None)
        body_match = bool(
            task_evidence is not None and observation_evidence is not None
            and canonical_json(task_evidence) == canonical_json(observation_evidence)
        )
        checks = {
            "task_observation_link_exists": bool(task and raw),
            "task_and_observation_run_match": bool(task and task.get("run_id") == raw.get("run_id")),
            "task_source_time_matches_original_observation": bool(task and float_matches(task.get("source_time"), raw.get("source_time"))),
            "task_snapshot_matches_original_observation": bool(task and task.get("snapshot_id") == identity.get("registered_snapshot_id")),
            "task_decision_matches_original_observation": bool(task and task.get("decision_id") == identity.get("registered_decision_id")),
            "task_request_matches_original_observation": bool(task and task.get("request_digest") == identity.get("registered_request_digest")),
            "task_evidence_digest_matches_original_observation": bool(task and task.get("evidence_digest") == raw.get("evidence_digest")),
            "task_evidence_body_matches_original_observation": body_match,
            "task_frame_receipt_points_to_task": bool(frame_agent and frame_agent.get("task_id") == task_id),
            "original_observation_registration_verified": bool(audits.get(observation_id, {}).get("evidence_registration_verified")),
        }
        task_input_check_templates[f"{task_id}:{observation_id}"] = checks
        if all(checks.values()):
            task_input_candidates[task_id].append(observation_id)
    task_input_by_id = {
        task_id: members[0]
        for task_id, members in task_input_candidates.items()
        if len(members) == 1
    }
    task_input_status = {
        task_id: "verified" if len(members) == 1 else "ambiguous" if len(members) > 1 else "unverified"
        for task_id, members in task_input_candidates.items()
    }
    for task_id in tasks:
        task_input_status.setdefault(task_id, "unverified")

    output: list[dict[str, Any]] = []
    for row in observations:
        row_id = int(row["observation_row_id"])
        raw = raw_obs[row_id]
        run = runs.get(row["run_id"], {})
        frame = frames.get((row["run_id"], round(float(row["source_time_s"]), 6)), {})
        frame_response = base.decode_json(frame.get("response_json"), {})
        payload = frame_response.get("payload", {}) if isinstance(frame_response, dict) else {}
        fall = payload.get("fall", {}) if isinstance(payload, dict) else {}
        fall_event_id = fall.get("event_id") or fall.get("current_event_id")
        visual_confirmation = bool(fall.get("new_event") and fall_event_id)
        evidence, evidence_reasons, evidence_status = parsed_evidence[row_id]
        identity = identities[row_id]
        evidence_audit = audits[row_id]
        task_id = str(row.get("task_id") or "")
        task = tasks.get(task_id, {})
        task_input_id = task_input_by_id.get(task_id)
        task_input_verified = task_input_status.get(task_id) == "verified"
        is_actual_task_input = bool(task_input_verified and task_input_id == row_id)
        link_disposition = task_observation_disposition.get((task_id, row_id)) if task_id else None
        task_link_checks = task_input_check_templates.get(f"{task_id}:{row_id}", {}) if task_id else {}
        task_evidence = task_evidence_by_id.get(task_id)
        task_evidence_version_matches = bool(
            task_evidence is not None and evidence is not None
            and all(task_evidence.get(field) == getattr(evidence, field, None) for field in REQUIRED_VERSION_FIELDS)
        )
        task_evidence_matches_this_observation = bool(
            task_evidence is not None and raw.get("evidence_json")
            and canonical_json(task_evidence) == canonical_json(base.decode_json(raw.get("evidence_json"), {}))
        )
        result_rows = results_by_task.get(task_id, []) if task_id else []
        result_record = next((item for item in reversed(result_rows) if item.get("accepted")), result_rows[-1] if result_rows else {})
        task_result = base.decode_json(result_record.get("result_json"), {}) if result_record else {}
        task_evidence_body = task_evidence if isinstance(task_evidence, dict) else {}
        task_window = task_evidence_body.get("source_window", {}) if isinstance(task_evidence_body, dict) else {}
        task_result_identity_matches = bool(
            result_record and result_record.get("accepted") and task and is_actual_task_input
            and result_record.get("task_id") == task_id
            and result_record.get("decision_id") == task.get("decision_id")
            and float_matches(result_record.get("source_time"), task.get("source_time"))
            and task_result.get("run_id") == task.get("run_id")
            and task_result.get("session_id") == task.get("session_id")
            and task_result.get("stream_id") == task.get("stream_id")
            and task_result.get("snapshot_id") == task.get("snapshot_id")
            and task_result.get("decision_id") == task.get("decision_id")
            and task_result.get("provider") == task.get("provider")
            and task_result.get("model") == task.get("model")
            and float_matches(task_result.get("source_window", {}).get("start_s"), task_window.get("start_s"))
            and float_matches(task_result.get("source_window", {}).get("end_s"), task_window.get("end_s"))
        )
        result_status = task_result.get("status") if is_actual_task_input and result_record else None
        probability = task_result.get("fall_probability_raw")
        if probability is None:
            probability = task_result.get("fall_probability")
        probability_kind = task_result.get("probability_kind") if is_actual_task_input and result_record else None
        result_accepted = bool(result_record.get("accepted")) if is_actual_task_input and result_record else None
        result_provider = task_result.get("provider") if is_actual_task_input and result_record else None
        result_model = task_result.get("model") if is_actual_task_input and result_record else None
        result_matches_task_input = task_result_identity_matches if result_record and is_actual_task_input else False

        proof_row = proof.get("by_decision_id", {}).get(identity.get("registered_decision_id"), {})
        frame_agent = payload.get("agent", {}) if isinstance(payload, dict) else {}
        frame_registered_for_observation = bool(
            frame and frame.get("agent_registration_status") == "registered"
            and frame.get("run_id") == raw.get("run_id")
            and float_matches(frame.get("source_time"), raw.get("source_time"))
            and frame_agent.get("run_id") == raw.get("run_id")
            and frame_agent.get("snapshot_id") == identity.get("registered_snapshot_id")
            and frame_agent.get("decision_id") == identity.get("registered_decision_id")
            and frame_agent.get("task_id") == raw.get("task_id")
        )
        proof_matches = bool(
            result_status == "ready" and proof.get("unchanged") and proof_row
            and proof_row.get("input_registration_status") == "registered"
            and proof_row.get("decision_id") == identity.get("registered_decision_id")
            and proof_row.get("task_id") == task_id
            and proof_row.get("snapshot_id") == identity.get("registered_snapshot_id")
            and proof_row.get("evidence_digest") == raw.get("evidence_digest")
            and proof_row.get("request_digest") == identity.get("registered_request_digest")
            and proof_row.get("input_image_sha256") == row.get("input_frame_sha256")
            and proof_row.get("input_envelope_sha256") == sha256_text(frame.get("observation_envelope_json"))
            and proof_row.get("input_response_sha256") == sha256_text(frame.get("response_json"))
            and proof_row.get("provider_request_id") == row.get("provider_request_id")
            and proof_row.get("model") == result_model
            and proof_row.get("provider") == result_provider
            and float_matches(proof_row.get("source_time"), row.get("source_time_s"))
            and float_matches(proof_row.get("source_window", {}).get("start_s"), row.get("window_start_s"))
            and float_matches(proof_row.get("source_window", {}).get("end_s"), row.get("window_end_s"))
            and proof_row.get("fall_probability") == probability
            and proof_row.get("probability_kind") == probability_kind
            and proof_row.get("status") == result_status
        )
        ready_input_verified = bool(
            evidence_audit["evidence_registration_verified"]
            and frame_registered_for_observation
            and is_actual_task_input
            and task_evidence_matches_this_observation
            and task_evidence_version_matches
            and result_matches_task_input
            and proof_matches
        ) if result_status == "ready" else evidence_audit["evidence_registration_verified"]
        task_result_probability = task_result.get("fall_probability_raw")
        if task_result_probability is None:
            task_result_probability = task_result.get("fall_probability")
        if result_record and not is_actual_task_input:
            result_attribution_status = "shared_merged_task_result_not_this_input" if task_input_verified else "unattributed_task_result"
        elif result_record and is_actual_task_input:
            result_attribution_status = "assigned_to_verified_task_input"
        else:
            result_attribution_status = "no_task_result"

        if evidence:
            row["window_start_s"] = float(evidence.source_window.start_s)
            row["window_end_s"] = float(evidence.source_window.end_s)
            row["window_id"] = base.canonical_window_id(row["video_id"], row["window_start_s"], row["window_end_s"])
            row["actual_source_samples_in_window"] = evidence.observed_frames
            row["eligibility_policy_version"] = evidence.eligibility_policy_version
            row["preprocessing_version"] = evidence.preprocessing_version
            row["yolo_model_version"] = evidence.yolo_model_version
            row["temporal_model_version"] = evidence.temporal_model_version
            row["head_variant"] = evidence.head_variant
            row["head_version"] = evidence.head_version
            row["head_weights_sha256"] = evidence.head_weights_sha256
            row["policy_version"] = evidence.policy_version
            row["prompt_version"] = evidence.prompt_version
            row["raw_temporal_score_not_probability"] = evidence.temporal_score
            row["raw_head_score_not_probability"] = evidence.head_raw_score
            row["model_derived_fall_process_label_excluded_from_truth"] = evidence.fall_process_label

        # Preserve both immutable source identity and the mutable coalesced pointer.
        row.update({
            "current_task_pointer_decision_id": raw.get("decision_id"),
            "current_task_pointer_request_digest": raw.get("request_digest"),
            "current_task_pointer_snapshot_id": raw.get("snapshot_id"),
            "registered_snapshot_id": identity.get("registered_snapshot_id"),
            "registered_decision_id": identity.get("registered_decision_id"),
            "registered_request_digest": identity.get("registered_request_digest"),
            "original_identity_reconstruction_available": identity.get("available", False),
            "current_task_pointer_drifted_from_original": bool(
                raw.get("decision_id") != identity.get("registered_decision_id")
                or raw.get("request_digest") != identity.get("registered_request_digest")
            ),
            "decision_id": identity.get("registered_decision_id") or raw.get("decision_id"),
            "task_association_disposition": link_disposition,
            "task_association_status": (
                "no_task" if not task_id else
                "verified_task_input" if is_actual_task_input else
                "coalesced_prior_observation" if task_input_verified else
                "task_input_unverified"
            ),
            "task_current_source_time_s": task.get("source_time"),
            "task_current_snapshot_id": task.get("snapshot_id"),
            "task_current_decision_id": task.get("decision_id"),
            "task_current_request_digest": task.get("request_digest"),
            "task_current_evidence_digest": task.get("evidence_digest"),
            "task_actual_input_observation_id": task_input_id,
            "task_actual_input_verified": task_input_verified,
            "task_is_actual_input_for_this_observation": is_actual_task_input,
            "task_actual_input_checks": task_link_checks,
            "task_evidence_body_matches_this_observation": task_evidence_matches_this_observation,
            "task_evidence_versions_match_this_observation": task_evidence_version_matches,
            "merged_task_result_status": task_result.get("status") if result_record else None,
            "merged_task_result_decision_id": task_result.get("decision_id") if result_record else None,
            "merged_task_result_source_time_s": result_record.get("source_time") if result_record else None,
            "merged_task_result_probability": task_result_probability if result_record else None,
            "merged_task_result_accepted": bool(result_record.get("accepted")) if result_record else False,
            "task_result_attribution_status": result_attribution_status,
            "task_result_identity_matches_actual_input": result_matches_task_input,
        })
        # Do not let base exporter aliases make a merged task result look like
        # a prediction for every earlier observation sharing the task pointer.
        row.update({
            "result_status": result_status,
            "probability": probability if is_actual_task_input else None,
            "probability_kind": probability_kind,
            "provider": result_provider or task.get("provider") or raw.get("provider"),
            "model_version": result_model or task.get("model") or raw.get("model"),
            "result_accepted": result_accepted if is_actual_task_input else None,
            "provider_response_present": bool(result_record) if is_actual_task_input else False,
            "provider_response_accepted": bool(result_record.get("accepted")) if is_actual_task_input and result_record else False,
        })
        metadata = task_result.get("provider_metadata") or {}
        http_status = metadata.get("http_status")
        provider = provider_settings(run)
        provider_response_valid = bool(
            is_actual_task_input and result_record and result_record.get("accepted")
            and task_result.get("provider") == provider.get("provider_mode")
            and task_result.get("provider") not in (None, "off")
            and task_result.get("status") == "ready"
            and isinstance(http_status, int) and 200 <= http_status < 300
            and metadata.get("request_id")
        )
        check_map = {
            **evidence_audit["evidence_identity_checks"],
            "task_actual_input_verified": task_input_verified,
            "task_evidence_body_matches_this_observation": task_evidence_matches_this_observation,
            "task_evidence_versions_match_this_observation": task_evidence_version_matches,
            "task_result_identity_matches_actual_input": result_matches_task_input if is_actual_task_input else not bool(result_record),
            "original_ready_input_proof_matches": proof_matches if result_status == "ready" and is_actual_task_input else not bool(result_record and task_result.get("status") == "ready"),
        }
        row.update({
            **provider,
            **evidence_audit,
            "evidence_validation_status": evidence_status,
            "evidence_contract_eligible": bool(evidence and evidence.eligible),
            "evidence_gate_eligible": bool(evidence and evidence.eligible and evidence_audit["evidence_registration_verified"]),
            "evidence_eligibility_reasons": evidence_reasons + evidence_audit["evidence_registration_reasons"],
            "evidence_source_time_s": evidence.source_time_s if evidence else None,
            "evidence_event_state": evidence.event_state.value if evidence else None,
            "evidence_observable": evidence.observable if evidence else None,
            "evidence_event_decidable": evidence.event_decidable if evidence else None,
            "visual_fall_state": fall.get("state"),
            "visual_confirmation": visual_confirmation,
            "visual_event_id": str(fall_event_id) if fall_event_id else None,
            "provider_response_valid": provider_response_valid,
            "provider_response_status": task_result.get("status") if is_actual_task_input and result_record else None,
            "provider_response_http_status": http_status if is_actual_task_input and result_record else None,
            "input_association_checks": check_map,
            "original_ready_input_proof_status": (
                "verified" if ready_input_verified
                else "mismatch" if proof_row
                else "not_available"
            ),
            "original_input_association_verified": ready_input_verified,
            "version_record_complete": bool(
                is_actual_task_input and all(row.get(field) for field in REQUIRED_VERSION_FIELDS)
                and task_evidence_version_matches
            ) if result_status == "ready" else all(row.get(field) for field in REQUIRED_VERSION_FIELDS),
            "label": None,
            "label_source": None,
            "label_annotator": None,
            "label_review_status": "needs_review" if row.get("window_id") else "no_source_window",
            "label_inclusion_status": "not_evaluated",
            "label_exclusion_reasons": [],
        })
        output.append(row)
    db_counts["task_observation_link_count"] = len(task_observation_disposition)
    db_counts["task_input_verified_count"] = len(task_input_by_id)
    return output, db_counts


def real_probability_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("result_status") == "ready"
        and row.get("probability") is not None
        and row.get("probability_kind") == "uncalibrated_model_estimate"
        and row.get("provider") == "aliyun"
        and row.get("result_accepted")
    ]


def prepare_window_labels(path: Path, candidates: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    old_path = OLD_RUN_DIR / "window-labels.csv"
    old_rows: list[dict[str, str]] = []
    if old_path.is_file():
        with old_path.open(newline="") as stream:
            old_rows = list(csv.DictReader(stream))
    ids = {row.get("canonical_window_id") for row in old_rows}
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row.get("window_id"):
            by_id[row["window_id"]].append(row)
    output = []
    for old in old_rows:
        row = {key: old.get(key, "") for key in LABEL_FIELDS}
        matches = by_id.get(row.get("canonical_window_id", ""), [])
        matching_views = {candidate.get("artifact_id") for candidate in matches}
        matching_video_hashes = {candidate.get("source_video_sha256") for candidate in matches if candidate.get("source_video_sha256")}
        if not row.get("artifact_id") and len(matching_views) == 1:
            row["artifact_id"] = next(iter(matching_views))
        if not row.get("source_video_sha256") and len(matching_video_hashes) == 1:
            row["source_video_sha256"] = next(iter(matching_video_hashes))
        output.append(row)
    for window_id, group in by_id.items():
        if window_id in ids:
            continue
        first = group[0]
        output.append({
            "canonical_window_id": window_id,
            "video_id": first["video_id"],
            "source_video_group_id": first["source_video_group_id"],
            "artifact_id": first["artifact_id"],
            "source_video_sha256": first.get("source_video_sha256", ""),
            "partition": first["partition"],
            "window_start_s": first["window_start_s"],
            "window_end_s": first["window_end_s"],
            "source_time_s": first["source_time_s"],
            "task_id": first["task_id"],
            "decision_id": first["decision_id"],
            "label": "",
            "label_source": "",
            "annotator": "",
            "annotation_status": "needs_review",
            "event_start_s": "",
            "event_end_s": "",
            "disagreement_notes": "",
        })
    write_csv(path, output, LABEL_FIELDS)


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, start=2):
        row["_label_row_number"] = index
    return rows


def label_lookup_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("canonical_window_id") or ""), str(row.get("artifact_id") or ""))


def label_signature(row: dict[str, Any]) -> tuple[str, ...]:
    # A duplicate is collapsible only when every serialized review/provenance
    # field is identical. This makes qualification independent of row order.
    fields = LABEL_FIELDS
    return tuple(str(row.get(field) or "").strip() for field in fields)


def label_duplicate_groups(label_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[int, str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dispositions: dict[int, str] = {}
    for row in label_rows:
        key = label_lookup_key(row)
        if key[0]:
            groups[key].append(row)
    for members in groups.values():
        if len(members) < 2:
            continue
        signatures = {label_signature(row) for row in members}
        if len(signatures) > 1:
            for row in members:
                dispositions[row["_label_row_number"]] = "conflicting_duplicate_key"
        else:
            for row in members[1:]:
                dispositions[row["_label_row_number"]] = "identical_duplicate_collapsed"
    return groups, dispositions


def float_matches(value: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def version_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in VERSION_FIELDS}


def metric_group_key(row: dict[str, Any]) -> str:
    payload = {
        "partition": row.get("partition"),
        "versions": version_signature(row),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def prediction_dedup_dispositions(candidates: list[dict[str, Any]]) -> dict[int, str]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            row.get("window_id"), row.get("artifact_id"), row.get("partition"),
            json.dumps(version_signature(row), sort_keys=True, ensure_ascii=False),
        )
        groups[key].append(row)
    disposition: dict[int, str] = {}
    for rows in groups.values():
        if len(rows) < 2:
            continue
        signature = {
            (row.get("probability"), row.get("evidence_digest"), row.get("input_frame_sha256"))
            for row in rows
        }
        status = "identical_duplicate_prediction_collapsed" if len(signature) == 1 else "conflicting_duplicate_prediction"
        for row in rows[1:] if len(signature) == 1 else rows:
            disposition[int(row["observation_row_id"])] = status
    return disposition


def score_label_compatibility(prediction: dict[str, Any], label: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    label_window_id = str(label.get("canonical_window_id") or "")
    try:
        expected_label_window_id = base.canonical_window_id(
            str(label.get("video_id") or ""),
            float(label.get("window_start_s")),
            float(label.get("window_end_s")),
        )
    except (TypeError, ValueError):
        expected_label_window_id = ""
    if label_window_id != expected_label_window_id:
        reasons.append("label_canonical_window_identity_mismatch")
    if label_window_id != str(prediction.get("window_id") or ""):
        reasons.append("prediction_label_window_key_mismatch")
    if str(label.get("video_id") or "") != str(prediction.get("video_id") or ""):
        reasons.append("label_video_mismatch")
    if str(label.get("source_video_group_id") or "") != str(prediction.get("source_video_group_id") or ""):
        reasons.append("label_source_video_group_mismatch")
    if str(label.get("source_video_sha256") or "") != str(prediction.get("source_video_sha256") or ""):
        reasons.append("label_source_video_hash_mismatch")
    if str(label.get("artifact_id") or "") != str(prediction.get("artifact_id") or ""):
        reasons.append("label_view_mismatch")
    if not float_matches(label.get("window_start_s"), prediction.get("window_start_s")):
        reasons.append("label_window_start_mismatch")
    if not float_matches(label.get("window_end_s"), prediction.get("window_end_s")):
        reasons.append("label_window_end_mismatch")
    if str(label.get("partition") or "") != str(prediction.get("partition") or ""):
        reasons.append("label_partition_mismatch")

    status = str(label.get("annotation_status") or "").strip().lower()
    target = str(label.get("label") or "").strip()
    label_source = str(label.get("label_source") or "").strip()
    annotator = str(label.get("annotator") or "").strip()
    if status in DRAFT_STATUSES:
        reasons.append("label_status_draft")
    elif status in PENDING_STATUSES:
        reasons.append("label_not_reviewed")
    elif status == "undetermined":
        reasons.append("label_status_uncertain")
    elif status not in REVIEWED_STATUSES:
        reasons.append("label_status_unrecognized_or_missing")
    if target == "undetermined":
        reasons.append("label_uncertain")
    elif target not in VALID_SCORABLE_LABELS:
        reasons.append("label_value_missing_or_invalid")
    source_type = str(label.get("label_source_type") or "").strip().lower()
    machine_markers = ("model", "prediction", "inference", "ai pre", "st-gcn", "自动输出", "自动预测", "机器输出", "模型")
    machine_text = any(
        marker in text.lower()
        for text in (label_source, annotator, str(label.get("label_source_type") or ""))
        for marker in machine_markers
    )
    if source_type == "machine_output" or machine_text:
        reasons.append("machine_generated_label")
    elif source_type != "human_adjudication":
        reasons.append("label_source_type_unknown")
    else:
        record_id = str(label.get("review_record_id") or "")
        record = VERIFIED_LABEL_SOURCE_REGISTRY.get(record_id)
        if not record:
            reasons.append("human_review_record_unverified")
        else:
            try:
                label_start = float(label.get("window_start_s"))
                label_end = float(label.get("window_end_s"))
                window_in_reviewed_interval = (
                    label_start >= float(record.get("window_start_s_min")) - 1e-6
                    and label_end <= float(record.get("window_end_s_max")) + 1e-6
                )
            except (TypeError, ValueError):
                window_in_reviewed_interval = False
            record_matches = all([
                record.get("source_type") == "human_adjudication",
                record.get("review_record_sha256") == label.get("review_record_sha256"),
                record.get("label_source") == label_source,
                record.get("video_id") == label.get("video_id") == prediction.get("video_id"),
                record.get("source_video_group_id") == label.get("source_video_group_id") == prediction.get("source_video_group_id"),
                record.get("source_video_sha256") == label.get("source_video_sha256") == prediction.get("source_video_sha256"),
                record.get("reviewed_by") == label.get("reviewed_by") == annotator,
                record.get("reviewed_at") == label.get("reviewed_at"),
                record.get("annotation_status") == status,
                record.get("verified_target_label") == target,
                window_in_reviewed_interval,
            ])
            if not record_matches:
                reasons.append("human_review_record_identity_or_content_mismatch")
    if not label_source:
        reasons.append("label_source_missing")
    if not annotator:
        reasons.append("annotator_missing")
    return reasons


def probability_metrics(rows: list[dict[str, Any]], bins: int = 10, bootstrap_iterations: int = 2000, seed: int = 20261118) -> dict[str, Any]:
    n = len(rows)
    n_positive = sum(int(row["label"] == "fall_process_visible") for row in rows)
    n_negative = n - n_positive
    brier = statistics.mean((float(row["probability"]) - int(row["label"] == "fall_process_visible")) ** 2 for row in rows) if rows else None
    reliability_bins = []
    ece = None
    if rows:
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            members = [row for row in rows if lower <= float(row["probability"]) < upper or (index == bins - 1 and float(row["probability"]) == 1.0)]
            reliability_bins.append({
                "bin_index": index,
                "lower_inclusive": lower,
                "upper_exclusive": upper if index < bins - 1 else 1.0,
                "n": len(members),
                "mean_prediction": statistics.mean(float(row["probability"]) for row in members) if members else None,
                "observed_fall_fraction": statistics.mean(int(row["label"] == "fall_process_visible") for row in members) if members else None,
            })
        ece = sum(item["n"] * abs(item["mean_prediction"] - item["observed_fall_fraction"]) for item in reliability_bins if item["n"]) / n
    independent_groups = len({row["source_video_group_id"] for row in rows})
    brier_ci = ece_ci = None
    if n and independent_groups >= 2 and bootstrap_iterations > 0:
        import random

        rng = random.Random(seed)
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[row["source_video_group_id"]].append(row)
        ids = list(by_group)
        brier_samples, ece_samples = [], []
        for _ in range(bootstrap_iterations):
            selected = [rng.choice(ids) for _ in ids]
            sample = [row for group_id in selected for row in by_group[group_id]]
            targets = [int(row["label"] == "fall_process_visible") for row in sample]
            brier_samples.append(statistics.mean((float(row["probability"]) - target) ** 2 for row, target in zip(sample, targets)))
            sample_ece = 0.0
            for index in range(bins):
                lower, upper = index / bins, (index + 1) / bins
                members = [(row, target) for row, target in zip(sample, targets) if lower <= float(row["probability"]) < upper or (index == bins - 1 and float(row["probability"]) == 1.0)]
                if members:
                    sample_ece += len(members) / len(sample) * abs(statistics.mean(float(row["probability"]) for row, _ in members) - statistics.mean(target for _, target in members))
            ece_samples.append(sample_ece)
        brier_samples.sort()
        ece_samples.sort()
        brier_ci = [brier_samples[int(.025 * (len(brier_samples) - 1))], brier_samples[int(.975 * (len(brier_samples) - 1))]]
        ece_ci = [ece_samples[int(.025 * (len(ece_samples) - 1))], ece_samples[int(.975 * (len(ece_samples) - 1))]]
    return {
        "eligible_labeled_windows": n,
        "positive_windows": n_positive,
        "negative_windows": n_negative,
        "labeled_independent_source_video_groups": independent_groups,
        "brier_score": brier,
        "ece": ece,
        "calibration_bins": bins,
        "reliability_bins": reliability_bins,
        "brier_95_percentile": brier_ci,
        "ece_95_percentile": ece_ci,
        "uncertainty_method": "source-video cluster bootstrap; unavailable with fewer than 2 labeled groups",
    }


def render_reliability_panels(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Draw one auditable reliability panel per version/partition bucket."""
    buckets = metrics.get("by_version_partition", {})
    ordered = [(key, buckets[key]) for key in sorted(buckets)]
    width, panel_height = 860, 430
    if not ordered:
        ordered = [("no-buckets", {"partition": "none", "versions": {}, "eligible_labeled_windows": 0, "candidate_real_ready_rows": 0, "positive_windows": 0, "negative_windows": 0, "reliability_bins": []})]
    height = panel_height * len(ordered)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.diag{stroke:#777;stroke-dasharray:5 5}.panel{fill:#fff;stroke:#aaa}</style>',
    ]
    manifest_panels = []
    for index, (bucket_id, bucket) in enumerate(ordered):
        offset = index * panel_height
        left, top, plot_w, plot_h = 86, offset + 96, 500, 270
        version = bucket.get("versions", {})
        heading = (
            f"{bucket.get('partition', 'unknown')} | head={version.get('head_version', 'unknown')} | "
            f"model={version.get('model_version', 'unknown')} | bucket={bucket_id}"
        )
        n = int(bucket.get("eligible_labeled_windows", 0))
        bin_rows = bucket.get("reliability_bins", [])
        bin_total = sum(int(row.get("n", 0)) for row in bin_rows)
        if bin_total != n:
            raise ValueError(f"Reliability bins for bucket {bucket_id} sum to {bin_total}, expected {n}")
        parts.extend([
            f'<rect class="panel" x="10" y="{offset + 8}" width="840" height="{panel_height - 16}" rx="8"/>',
            f'<text x="28" y="{offset + 35}" font-size="15" font-weight="bold">{html.escape(heading)}</text>',
            f'<text x="28" y="{offset + 61}" font-size="13">n={n} reviewed labels | candidates={int(bucket.get("candidate_real_ready_rows", 0))} | positive={int(bucket.get("positive_windows", 0))} | negative={int(bucket.get("negative_windows", 0))}</text>',
        ])
        for tick in range(11):
            x = left + tick * plot_w / 10
            y = top + plot_h - tick * plot_h / 10
            parts.extend([
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>',
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>',
                f'<text x="{x - 9:.1f}" y="{top + plot_h + 19}" font-size="10">{tick / 10:.1f}</text>',
                f'<text x="{left - 33}" y="{y + 4:.1f}" font-size="10">{tick / 10:.1f}</text>',
            ])
        parts.extend([
            f'<line class="diag" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text x="{left + plot_w / 2 - 67}" y="{top + plot_h + 43}" font-size="12">Mean predicted probability</text>',
            f'<text transform="translate(29 {top + plot_h / 2 + 59}) rotate(-90)" font-size="12">Observed fall-process fraction</text>',
        ])
        if n == 0:
            parts.append(f'<text x="{left + plot_w / 2}" y="{top + plot_h / 2}" text-anchor="middle" font-size="16">No reviewed labels in this version/partition bucket (n=0)</text>')
        else:
            for row in bin_rows:
                count = int(row["n"])
                if not count:
                    continue
                x = left + float(row["mean_prediction"]) * plot_w
                y = top + plot_h - float(row["observed_fall_fraction"]) * plot_h
                radius = 4 + 3 * math.sqrt(count)
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2673aa" fill-opacity="0.8"/>')
                parts.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" font-size="11">n={count}</text>')
        manifest_panels.append({
            "bucket_id": bucket_id,
            "partition": bucket.get("partition"),
            "versions": version,
            "reviewed_label_sample_count": n,
            "candidate_real_ready_rows": int(bucket.get("candidate_real_ready_rows", 0)),
            "reliability_bin_counts": [int(row.get("n", 0)) for row in bin_rows],
            "reliability_bin_count_sum": bin_total,
            "matches_window_metrics_json": bin_total == n,
        })
    parts.append('</svg>')
    return "\n".join(parts) + "\n", {
        "schema_version": "stage05-reliability-panels.v1",
        "panel_count": len(manifest_panels),
        "panels": manifest_panels,
        "sample_count_source": "window-metrics.json by_version_partition bucket eligible_labeled_windows",
    }


def evaluate_probability_labels(candidates: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels_by_key, duplicate_dispositions = label_duplicate_groups(label_rows)
    labels_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in label_rows:
        if label.get("canonical_window_id"):
            labels_by_window[str(label["canonical_window_id"])].append(label)
    row_to_audit: dict[int, dict[str, Any]] = {}
    for row in label_rows:
        row_to_audit[row["_label_row_number"]] = {
            "label_row_number": row["_label_row_number"],
            "canonical_window_id": row.get("canonical_window_id"),
            "video_id": row.get("video_id"),
            "source_video_group_id": row.get("source_video_group_id"),
            "artifact_id": row.get("artifact_id"),
            "source_video_sha256": row.get("source_video_sha256"),
            "partition": row.get("partition"),
            "window_start_s": row.get("window_start_s"),
            "window_end_s": row.get("window_end_s"),
            "label": row.get("label"),
            "annotation_status": row.get("annotation_status"),
            "label_source": row.get("label_source"),
            "label_source_type": row.get("label_source_type"),
            "review_record_id": row.get("review_record_id"),
            "review_record_sha256": row.get("review_record_sha256"),
            "reviewed_by": row.get("reviewed_by"),
            "reviewed_at": row.get("reviewed_at"),
            "annotator": row.get("annotator"),
            "inclusion_status": "pending_prediction_match",
            "matched_prediction_observation_ids": [],
            "matched_prediction_included_count": 0,
            "matched_prediction_excluded_count": 0,
            "exclusion_reasons": [],
        }
        if row["_label_row_number"] in duplicate_dispositions:
            row_to_audit[row["_label_row_number"]]["inclusion_status"] = "excluded"
            row_to_audit[row["_label_row_number"]]["exclusion_reasons"] = [duplicate_dispositions[row["_label_row_number"]]]

    partitions_by_group: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        partitions_by_group[str(row.get("source_video_group_id"))].add(str(row.get("partition")))
    cross_partition_groups = {group for group, partitions in partitions_by_group.items() if len(partitions) > 1}
    duplicate_predictions = prediction_dedup_dispositions(candidates)
    prediction_audit: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for prediction in candidates:
        reasons: list[str] = []
        if not prediction.get("evidence_gate_eligible"):
            reasons.append("prediction_evidence_ineligible")
        if not prediction.get("original_input_association_verified"):
            reasons.append("prediction_input_association_unverified")
        if not prediction.get("version_record_complete"):
            reasons.append("prediction_version_incomplete")
        if str(prediction.get("source_video_group_id")) in cross_partition_groups:
            reasons.append("source_video_group_cross_partition")
        if not prediction.get("frozen_partition_match"):
            reasons.append("prediction_partition_not_frozen")
        duplicate_status = duplicate_predictions.get(int(prediction["observation_row_id"]))
        if duplicate_status:
            reasons.append(duplicate_status)

        key = (str(prediction.get("window_id") or ""), str(prediction.get("artifact_id") or ""))
        label_group = labels_by_key.get(key, [])
        if not label_group:
            # Keep an explicit view mismatch in the row audit instead of
            # misreporting a label from the same source window as merely absent.
            label_group = labels_by_window.get(key[0], [])
        if not label_group:
            reasons.append("label_not_found")
            selected_label = None
        elif any(row["_label_row_number"] in duplicate_dispositions and duplicate_dispositions[row["_label_row_number"]] == "conflicting_duplicate_key" for row in label_group):
            reasons.append("conflicting_duplicate_key")
            selected_label = None
        else:
            selected_label = next((row for row in label_group if row["_label_row_number"] not in duplicate_dispositions), label_group[0])
            reasons.extend(score_label_compatibility(prediction, selected_label))

        row_audit = {
            **{key: prediction.get(key) for key in [
                "observation_row_id", "video_id", "source_video_group_id", "artifact_id", "partition",
                "source_video_sha256",
                "run_id", "window_id", "window_start_s", "window_end_s", "source_time_s", "probability",
                "provider", "model_version", "mapping_version", "eligibility_policy_version", "head_version",
                "head_weights_sha256", "temporal_model_version", "yolo_model_version", "policy_version",
                "frozen_partition", "frozen_partition_status",
            ]},
            "input_association_status": prediction.get("original_ready_input_proof_status"),
            "version_record_complete": prediction.get("version_record_complete"),
            "label_row_number": selected_label.get("_label_row_number") if selected_label else None,
            "label": selected_label.get("label") if selected_label else None,
            "annotation_status": selected_label.get("annotation_status") if selected_label else None,
            "label_source": selected_label.get("label_source") if selected_label else None,
            "label_source_type": selected_label.get("label_source_type") if selected_label else None,
            "review_record_id": selected_label.get("review_record_id") if selected_label else None,
            "review_record_sha256": selected_label.get("review_record_sha256") if selected_label else None,
            "reviewed_by": selected_label.get("reviewed_by") if selected_label else None,
            "reviewed_at": selected_label.get("reviewed_at") if selected_label else None,
            "annotator": selected_label.get("annotator") if selected_label else None,
            "included_in_metrics": not reasons,
            "exclusion_reasons": reasons,
        }
        prediction_audit.append(row_audit)
        prediction["label"] = selected_label.get("label") if selected_label else None
        prediction["label_source"] = selected_label.get("label_source") if selected_label else None
        prediction["label_annotator"] = selected_label.get("annotator") if selected_label else None
        prediction["label_review_status"] = selected_label.get("annotation_status") if selected_label else None
        prediction["label_inclusion_status"] = "included" if not reasons else "excluded"
        prediction["label_exclusion_reasons"] = reasons
        if not reasons and selected_label:
            accepted_rows.append({**prediction, "label": str(selected_label["label"]).strip()})

    audits_by_label_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in prediction_audit:
        if prediction.get("label_row_number") is not None:
            audits_by_label_row[int(prediction["label_row_number"])].append(prediction)
    for label in label_rows:
        audit = row_to_audit[label["_label_row_number"]]
        if audit["inclusion_status"] == "excluded":
            continue
        matched = audits_by_label_row.get(label["_label_row_number"], [])
        audit["matched_prediction_observation_ids"] = [row["observation_row_id"] for row in matched]
        included_count = sum(bool(row["included_in_metrics"]) for row in matched)
        audit["matched_prediction_included_count"] = included_count
        audit["matched_prediction_excluded_count"] = len(matched) - included_count
        reasons = sorted({reason for row in matched for reason in row["exclusion_reasons"]})
        audit["exclusion_reasons"] = reasons
        if not matched:
            audit["inclusion_status"] = "excluded"
            audit["exclusion_reasons"] = ["no_scored_prediction_for_label"]
        elif included_count == len(matched):
            audit["inclusion_status"] = "included_in_metrics"
        elif included_count:
            audit["inclusion_status"] = "partially_included_across_versions_or_partitions"
        else:
            audit["inclusion_status"] = "excluded_from_metrics"

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_bucket_counts: Counter[str] = Counter()
    candidate_bucket_meta: dict[str, dict[str, Any]] = {}
    for prediction in candidates:
        key = metric_group_key(prediction)
        candidate_bucket_counts[key] += 1
        candidate_bucket_meta[key] = {"partition": prediction.get("partition"), "versions": version_signature(prediction)}
    for row in accepted_rows:
        buckets[metric_group_key(row)].append(row)
    by_version_partition = {}
    all_keys = sorted(set(candidate_bucket_counts) | set(buckets))
    for key in all_keys:
        scored = buckets.get(key, [])
        partition = candidate_bucket_meta.get(key, {}).get("partition")
        by_version_partition[key] = {
            **candidate_bucket_meta.get(key, {}),
            "evaluation_claim": "descriptive_only_development_exposed" if partition == "development_exposed" else "partition_specific",
            "candidate_real_ready_rows": candidate_bucket_counts.get(key, 0),
            **probability_metrics(scored),
        }
    unique_metric_buckets = len(all_keys)
    aggregate = probability_metrics(accepted_rows) if unique_metric_buckets == 1 else None
    summary = {
        "schema_version": "stage05-r1-window-metrics.v1",
        "candidate_real_ready_rows": len(candidates),
        "eligible_real_scored_windows": sum(bool(row.get("evidence_gate_eligible") and row.get("original_input_association_verified") and row.get("version_record_complete")) for row in candidates),
        "labeled_rows_included": len(accepted_rows),
        "labeled_rows_excluded": len(prediction_audit) - len(accepted_rows),
        "conflicting_duplicate_label_keys": sum(
            any(duplicate_dispositions.get(row["_label_row_number"]) == "conflicting_duplicate_key" for row in members)
            for members in labels_by_key.values()
        ),
        "identical_duplicate_label_rows_collapsed": sum(status == "identical_duplicate_collapsed" for status in duplicate_dispositions.values()),
        "verified_human_source_records_available": len(VERIFIED_LABEL_SOURCE_REGISTRY),
        "label_source_type_counts": dict(Counter(str(row.get("label_source_type") or "unknown") for row in label_rows)),
        "source_video_groups_cross_partition": sorted(cross_partition_groups),
        "metrics_pooled_across_versions_or_partitions": False,
        "aggregate_metrics": aggregate,
        "by_version_partition": by_version_partition,
        "calibrator_fit": False,
        "candidate_calibrated_strategy": "not evaluated; no fit or threshold selection",
    }
    # Keep the required nullable fields explicit at top level. They remain null
    # whenever there are zero labels or more than one version/partition bucket.
    summary["brier_score"] = aggregate["brier_score"] if aggregate else None
    summary["ece"] = aggregate["ece"] if aggregate else None
    summary["labeled_version_matched_windows"] = len(accepted_rows)
    summary["positive_windows"] = sum(row["label"] == "fall_process_visible" for row in accepted_rows)
    summary["negative_windows"] = len(accepted_rows) - summary["positive_windows"]
    label_audit = list(row_to_audit.values())
    return summary, prediction_audit, label_audit


def source_run_for_event(event: dict[str, Any]) -> str:
    return str(event.get("reference_run_id") or EVENT_REFERENCE_RUNS[event["event_id"]])


def evidence_window_overlap(row: dict[str, Any], event: dict[str, Any]) -> float:
    start = row.get("window_start_s")
    end = row.get("window_end_s")
    if start is None or end is None:
        return 0.0
    return max(0.0, min(float(end), float(event["end_s_exclusive"])) - max(float(start), float(event["start_s"])))


def event_prediction_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    run_ids = {row["run_id"] for row in EVENT_VIEW_RUNS}
    predictions = []
    for run_id in sorted(run_ids):
        view = next(row for row in EVENT_VIEW_RUNS if row["run_id"] == run_id)
        for frame in connection.execute(
            "SELECT source_time,response_json,created_at,completed_at,image_sha256 FROM agent_frame_receipts WHERE run_id=? ORDER BY source_time",
            (run_id,),
        ):
            response = base.decode_json(frame["response_json"], {})
            payload = response.get("payload", response) if isinstance(response, dict) else {}
            fall = payload.get("fall", {}) if isinstance(payload, dict) else {}
            event_id = fall.get("event_id") or fall.get("current_event_id")
            if not (fall.get("new_event") and event_id):
                continue
            predictions.append({
                "prediction_id": f"{run_id}:{event_id}:{frame['source_time']}",
                "video_id": view["video_id"],
                "source_video_group_id": base.VIDEO_GROUPS[view["video_id"]]["source_video_group_id"],
                "run_id": run_id,
                "input_view": view["input_view"],
                "event_id": str(event_id),
                "source_time_s": float(frame["source_time"]),
                "visual_fall_state": fall.get("state"),
                "frame_completed_at_unix_s": frame["completed_at"],
                "frame_created_at_unix_s": frame["created_at"],
                "input_frame_sha256": frame["image_sha256"],
            })
    return predictions


def build_event_window_links(
    events: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    run_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    links: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    used_visual_prediction_ids: set[str] = set()
    for event in sorted(events, key=lambda item: (float(item["start_s"]), str(item["event_id"]))):
        run_id = source_run_for_event(event)
        view = next((row for row in EVENT_VIEW_RUNS if row["run_id"] == run_id), {
            "run_id": run_id,
            "video_id": event["video_id"],
            "input_view": event.get("input_view", "synthetic_view"),
            "cohort": event.get("cohort", "synthetic"),
        })
        event_obs = [
            row for row in observations
            if row.get("run_id") == run_id
            and row.get("video_id") == event["video_id"]
            and row.get("source_video_group_id") == event["source_video_group_id"]
        ]
        event_links: list[dict[str, Any]] = []
        for row in event_obs:
            overlap_s = evidence_window_overlap(row, event)
            timely = row.get("source_time_s") is not None and float(row["source_time_s"]) <= float(event["end_s_exclusive"]) + EVENT_POST_TOLERANCE_S
            if not row.get("evidence_gate_eligible"):
                reason = "evidence_missing_or_ineligible"
            elif row.get("window_start_s") is None or row.get("window_end_s") is None:
                reason = "eligible_evidence_without_window"
            elif float(row["window_end_s"]) <= float(event["start_s"]):
                reason = "window_only_before_event"
            elif overlap_s <= 0:
                reason = "window_does_not_overlap_event"
            elif not timely:
                reason = "overlap_window_observed_too_late"
            else:
                reason = "eligible_window_overlaps_event"
            qualifies = reason == "eligible_window_overlaps_event"
            link = {
                "event_id": event["event_id"],
                "video_id": event["video_id"],
                "source_video_group_id": event["source_video_group_id"],
                "reference_run_id": run_id,
                "input_view": view["input_view"],
                "observation_row_id": row.get("observation_row_id"),
                "source_time_s": row.get("source_time_s"),
                "window_id": row.get("window_id"),
                "window_start_s": row.get("window_start_s"),
                "window_end_s": row.get("window_end_s"),
                "event_start_s": event["start_s"],
                "event_end_s_exclusive": event["end_s_exclusive"],
                "overlap_s": overlap_s,
                "observation_within_post_event_tolerance": timely,
                "evidence_validation_status": row.get("evidence_validation_status"),
                "evidence_gate_eligible": row.get("evidence_gate_eligible"),
                "eligibility_reasons": row.get("evidence_eligibility_reasons"),
                "provider_mode": row.get("provider_mode"),
                "provider_switch_enabled": row.get("provider_switch_enabled"),
                "provider_network_enabled": row.get("provider_network_enabled"),
                "provider_response_valid": row.get("provider_response_valid"),
                "provider_response_status": row.get("provider_response_status"),
                "visual_confirmation": row.get("visual_confirmation"),
                "link_status": reason,
                "qualifies_for_event_gate_coverage": qualifies,
            }
            event_links.append(link)
            links.append(link)
        qualifying = [row for row in event_links if row["qualifies_for_event_gate_coverage"]]
        run_settings = run_meta.get(run_id, {})
        run_predictions = run_settings.get("event_predictions", [])
        prediction_candidates = [
            row for row in run_predictions
            if row["prediction_id"] not in used_visual_prediction_ids
            and row.get("run_id") == run_id
            and row.get("video_id") == event["video_id"]
            and row.get("source_video_group_id") == event["source_video_group_id"]
            and float(event["start_s"]) <= float(row["source_time_s"]) <= float(event["end_s_exclusive"]) + EVENT_POST_TOLERANCE_S
        ]
        matching_predictions = sorted(
            prediction_candidates,
            key=lambda row: (float(row["source_time_s"]) - float(event["start_s"]), float(row["source_time_s"]), str(row["prediction_id"])),
        )[:1]
        used_visual_prediction_ids.update(row["prediction_id"] for row in matching_predictions)
        visually_confirmed = bool(matching_predictions)
        confirmation_times = {float(row["source_time_s"]) for row in matching_predictions}
        valid_responses = [row for row in qualifying if row.get("provider_response_valid")]
        qualified_without_valid_response = bool(qualifying) and not valid_responses
        provider_enabled_without_response = any(
            bool(row.get("provider_switch_enabled")) and not bool(row.get("provider_response_valid"))
            for row in qualifying
        )
        source_time_latency_s = (
            float(matching_predictions[0]["source_time_s"]) - float(event["start_s"])
            if matching_predictions else None
        )
        event_rows.append({
            "event_id": event["event_id"],
            "video_id": event["video_id"],
            "source_video_group_id": event["source_video_group_id"],
            "reference_run_id": run_id,
            "input_view": view["input_view"],
            "cohort": view["cohort"],
            "event_start_s": event["start_s"],
            "event_end_s_exclusive": event["end_s_exclusive"],
            "label_source": event["label_source"],
            "reference_annotator": event["annotator"],
            "qualified_window_observations_overlapping_event": len(qualifying),
            "unique_qualified_window_ids_overlapping_event": sorted({row.get("window_id") for row in qualifying if row.get("window_id")}),
            "maximum_qualified_window_overlap_s": max((row["overlap_s"] for row in qualifying), default=0.0),
            "window_gate_coverage": bool(qualifying),
            "visual_confirmation_in_event_match_interval": visually_confirmed,
            "visual_confirmation_observations": len(confirmation_times),
            "provider_mode": run_settings.get("provider_mode"),
            "provider_switch_enabled": run_settings.get("provider_switch_enabled", False),
            "provider_network_enabled": run_settings.get("provider_network_enabled", False),
            "provider_enabled_qualified_window_observations": sum(bool(row.get("provider_switch_enabled")) for row in qualifying),
            "valid_provider_responses_on_qualified_windows": len(valid_responses),
            "qualified_but_no_valid_provider_response": qualified_without_valid_response,
            "provider_enabled_qualified_window_without_valid_response": provider_enabled_without_response,
            "provider_response_statuses_on_qualified_windows": sorted({str(row.get("provider_response_status")) for row in qualifying if row.get("provider_response_status")}),
            "visual_event_prediction_ids": [row["prediction_id"] for row in run_predictions if row["prediction_id"] in {match["prediction_id"] for match in matching_predictions}],
            "visually_detected_event": visually_confirmed,
            "source_time_latency_s": source_time_latency_s,
            "end_to_end_latency_ms": None,
            "event_abstained_by_quality_gate": not bool(qualifying),
            "missed_by_quality_gate": not bool(qualifying),
            "visual_event_missed": not visually_confirmed,
            "outcome_classification": (
                "no_qualified_window"
                if not qualifying
                else "qualified_but_no_visual_confirmation"
                if not visually_confirmed
                else "visually_confirmed"
            ),
        })
    return links, event_rows


def event_view_run_rows(
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    run_meta: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    asset_by_video = {}
    for asset in inventory["p3_video_artifacts"]:
        if asset.get("artifact_type") == "original_single_camera_rgb":
            asset_by_video[asset["video_id"]] = asset
    rows = []
    for view in EVENT_VIEW_RUNS:
        run_id = view["run_id"]
        video_id = view["video_id"]
        obs = [row for row in observations if row.get("run_id") == run_id]
        run_events = [row for row in event_rows if row["reference_run_id"] == run_id]
        run_predictions = [row for row in predictions if row["run_id"] == run_id]
        matched_ids = {prediction_id for event in run_events for prediction_id in event["visual_event_prediction_ids"]}
        run_settings = run_meta[run_id]
        duration = asset_by_video.get(video_id, {}).get("duration_s")
        event_counts = summarize_event_rows(run_events)
        abstention_reasons: Counter[str] = Counter()
        for row in obs:
            reasons = row.get("evidence_eligibility_reasons") or []
            abstention_reasons.update(set(str(reason) for reason in reasons))
        abstained = sum(not bool(row.get("evidence_gate_eligible")) for row in obs)
        rows.append({
            "run_id": run_id,
            "video_id": video_id,
            "source_video_group_id": base.VIDEO_GROUPS[video_id]["source_video_group_id"],
            "artifact_id": base.RUN_SOURCE_MAP[run_id]["artifact_id"],
            "input_view": view["input_view"],
            "cohort": view["cohort"],
            "partition": "development_exposed",
            "source_duration_s_view_run": duration,
            "sampled_observations": len(obs),
            "typed_evidence_rows": sum(row.get("evidence_validation_status") == "valid" for row in obs),
            "eligible_window_observations": sum(bool(row.get("evidence_gate_eligible")) for row in obs),
            "abstained_observations": abstained,
            "eligibility_coverage_rate_by_observation": sum(bool(row.get("evidence_gate_eligible")) for row in obs) / len(obs) if obs else None,
            "abstention_rate_by_observation": abstained / len(obs) if obs else None,
            "abstention_reason_counts_multilabel": dict(sorted(abstention_reasons.items())),
            "decision_result_status_counts": dict(sorted(Counter(str(row.get("result_status") or "missing") for row in obs).items())),
            "unique_eligible_source_windows": len({row.get("window_id") for row in obs if row.get("evidence_gate_eligible") and row.get("window_id")}),
            "visual_confirmation_observations": sum(bool(row.get("visual_confirmation")) for row in obs),
            "visual_fall_state_counts": dict(Counter(str(row.get("visual_fall_state")) for row in obs)),
            "provider_mode": run_settings["provider_mode"],
            "provider_switch_enabled": run_settings["provider_switch_enabled"],
            "provider_network_enabled": run_settings["provider_network_enabled"],
            "provider_switch_disabled_observations": sum(not bool(row.get("provider_switch_enabled")) for row in obs),
            "valid_provider_response_observations": sum(bool(row.get("provider_response_valid")) for row in obs),
            "reference_event_ids": [row["event_id"] for row in run_events],
            "reference_events": event_counts["reference_events"],
            "gate_covered_events": event_counts["gate_covered_events"],
            "visual_confirmed_events": event_counts["visual_confirmed_events"],
            "events_qualified_but_not_visually_confirmed": event_counts["qualified_but_not_visually_confirmed"],
            "events_without_qualified_window": event_counts["events_without_qualified_window"],
            "event_predictions": len(run_predictions),
            "false_positive_event_predictions": sum(row["prediction_id"] not in matched_ids for row in run_predictions),
            "source_time_latency_p50_s": base.percentile([float(row["source_time_latency_s"]) for row in run_events if row.get("source_time_latency_s") is not None], .5),
            "source_time_latency_p95_s": base.percentile([float(row["source_time_latency_s"]) for row in run_events if row.get("source_time_latency_s") is not None], .95),
            "end_to_end_latency_p50_ms": None,
            "end_to_end_latency_p95_ms": None,
        })
    return rows


def summarize_event_rows(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(event_rows)
    covered = sum(bool(row["window_gate_coverage"]) for row in event_rows)
    visual = sum(bool(row["visually_detected_event"]) for row in event_rows)
    return {
        "reference_events": total,
        "gate_covered_events": covered,
        "window_gate_event_coverage": covered / total if total else None,
        "visual_confirmed_events": visual,
        "visual_event_recall": visual / total if total else None,
        "missed_by_quality_gate": sum(bool(row["missed_by_quality_gate"]) for row in event_rows),
        "events_abstained_by_quality_gate": sum(bool(row["event_abstained_by_quality_gate"]) for row in event_rows),
        "qualified_but_not_visually_confirmed": sum(row["outcome_classification"] == "qualified_but_no_visual_confirmation" for row in event_rows),
        "events_without_qualified_window": sum(row["outcome_classification"] == "no_qualified_window" for row in event_rows),
        "visually_missed_events": total - visual,
    }


def summarize_event_cohort(
    cohort: str,
    selected_runs: set[str],
    run_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    chosen_views = [row for row in EVENT_VIEW_RUNS if row["run_id"] in selected_runs]
    video_ids = {row["video_id"] for row in chosen_views}
    unique_durations = {video_id: ({"urfd-adl-04": 5.0, "urfd-fall-01": 160 / 30, "urfd-fall-02": 110 / 30}[video_id]) for video_id in video_ids}
    selected_events = [row for row in event_rows if row["reference_run_id"] in selected_runs]
    selected_predictions = [row for row in predictions if row["run_id"] in selected_runs]
    matched_prediction_ids = {prediction_id for event in selected_events for prediction_id in event["visual_event_prediction_ids"]}
    false_positives = [row for row in selected_predictions if row["prediction_id"] not in matched_prediction_ids]
    # Iterate in a fixed key order and use a correctly rounded summation so
    # the mixed-view duration is independent of set/hash iteration order.
    duration_s = math.fsum(unique_durations[video_id] for video_id in sorted(unique_durations))
    event_counts = summarize_event_rows(selected_events)
    n_events = event_counts["reference_events"]
    gate_count = event_counts["gate_covered_events"]
    visual_count = event_counts["visual_confirmed_events"]
    provider_enabled_events = sum(bool(row["provider_switch_enabled"]) and bool(row["window_gate_coverage"]) for row in selected_events)
    response_events = sum(bool(row["valid_provider_responses_on_qualified_windows"]) for row in selected_events)
    source_delays = [float(row["source_time_latency_s"]) for row in selected_events if row.get("source_time_latency_s") is not None]
    wall_delays = [float(row["end_to_end_latency_ms"]) for row in selected_events if row.get("end_to_end_latency_ms") is not None]
    selected_video_rows = [row for row in run_rows if row["run_id"] in selected_runs]
    sampled_observations = sum(int(row["sampled_observations"]) for row in selected_video_rows)
    abstained_observations = sum(int(row["abstained_observations"]) for row in selected_video_rows)
    eligible_window_observations = sum(int(row["eligible_window_observations"]) for row in selected_video_rows)
    reason_counts: Counter[str] = Counter()
    result_status_counts: Counter[str] = Counter()
    for row in selected_video_rows:
        reason_counts.update(row["abstention_reason_counts_multilabel"])
        result_status_counts.update(row["decision_result_status_counts"])
    return {
        "cohort": cohort,
        "run_ids": sorted(selected_runs),
        "input_views": sorted({row["input_view"] for row in chosen_views}),
        "distinct_source_video_groups": len(video_ids),
        "source_video_ids": sorted(video_ids),
        "unique_source_video_duration_s": duration_s,
        "observation_denominator": "view/run observations; raw and three-panel derivatives are separate exposures",
        "sampled_observations": sampled_observations,
        "eligible_window_observations": eligible_window_observations,
        "abstained_observations": abstained_observations,
        "abstention_rate_by_observation": abstained_observations / sampled_observations if sampled_observations else None,
        "abstention_reason_counts_multilabel": dict(sorted(reason_counts.items())),
        "decision_result_status_counts": dict(sorted(result_status_counts.items())),
        "provider_switch_disabled_observations": sum(int(row["provider_switch_disabled_observations"]) for row in selected_video_rows),
        "valid_provider_response_observations": sum(int(row["valid_provider_response_observations"]) for row in selected_video_rows),
        "official_reference_events": n_events,
        "events_with_qualified_overlapping_window": gate_count,
        "window_gate_event_coverage": gate_count / n_events if n_events else None,
        "events_with_visual_confirmation": visual_count,
        "visual_event_recall": visual_count / n_events if n_events else None,
        "provider_enabled_and_window_covered_events": provider_enabled_events,
        "events_with_valid_provider_response_on_qualified_window": response_events,
        "missed_by_quality_gate": event_counts["missed_by_quality_gate"],
        "events_abstained_by_quality_gate": event_counts["events_abstained_by_quality_gate"],
        "qualified_but_not_visually_confirmed": event_counts["qualified_but_not_visually_confirmed"],
        "events_without_qualified_window": event_counts["events_without_qualified_window"],
        "visually_missed_events": event_counts["visually_missed_events"],
        "events_on_provider_disabled_runs": sum(not bool(row["provider_switch_enabled"]) for row in selected_events),
        "false_positive_event_predictions": len(false_positives),
        "false_alarms_per_hour": len(false_positives) / (duration_s / 3600) if duration_s else None,
        "events_with_qualified_window_but_no_valid_provider_response": sum(bool(row["qualified_but_no_valid_provider_response"]) for row in selected_events),
        "events_with_provider_enabled_qualified_window_but_no_valid_response": sum(bool(row["provider_enabled_qualified_window_without_valid_response"]) for row in selected_events),
        "source_time_latency_p50_s": base.percentile(source_delays, .5),
        "source_time_latency_p95_s": base.percentile(source_delays, .95),
        "end_to_end_latency_p50_ms": base.percentile(wall_delays, .5),
        "end_to_end_latency_p95_ms": base.percentile(wall_delays, .95),
        "latency_status": (
            "no matched event predictions" if visual_count == 0
            else "source-time latency available; event-onset wall-clock anchor unavailable" if source_delays and not wall_delays
            else "matched predictions lack latency anchors"
        ),
    }


def build_event_evaluation(
    official: dict[str, Any],
    observations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    run_meta: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = []
    for source_event in official["events"]:
        event = dict(source_event)
        event["reference_run_id"] = source_run_for_event(event)
        events.append(event)
    for run_id, settings in run_meta.items():
        settings["event_predictions"] = [row for row in predictions if row["run_id"] == run_id]
    links, event_rows = build_event_window_links(events, observations, run_meta)
    view_rows = event_view_run_rows(observations, predictions, event_rows, run_meta, inventory)
    cohort_rows = [
        summarize_event_cohort("single_camera_raw", {row["run_id"] for row in EVENT_VIEW_RUNS if row["cohort"] == "single_camera_raw"}, view_rows, event_rows, predictions),
        summarize_event_cohort("three_panel_derived", {row["run_id"] for row in EVENT_VIEW_RUNS if row["cohort"] == "three_panel_derived"}, view_rows, event_rows, predictions),
        summarize_event_cohort("designated_mixed_unique_video_route", MIXED_ROUTE_RUNS, view_rows, event_rows, predictions),
    ]
    result = {
        "schema_version": "stage05-r1-event-metrics.v1",
        "window_event_overlap_rule": WINDOW_EVENT_OVERLAP_RULE,
        "event_confirmation_rule": "Same designated run and source group; new_event=true with non-empty event_id from event onset through 2.5 seconds after the official event interval ends; one-to-one matching.",
        "events": event_rows,
        "cohorts": cohort_rows,
        "single_source_of_truth": "All event and video summaries are derived from the event rows and run-view rows exported alongside this file; no provider-off aggregate is an input to event gate coverage.",
    }
    return result, links, view_rows, event_rows


def external_summary_crosscheck(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the prior aggregate with row-level results; never feed the metric."""
    previous = base.read_json(base.STAGE04_OFF_SUMMARY)
    prior_by_run = {row.get("run_id"): row for row in previous.get("runs", [])}
    comparisons = []
    for view in EVENT_VIEW_RUNS:
        old = prior_by_run.get(view["run_id"])
        if not old:
            continue
        rows = [row for row in observations if row.get("run_id") == view["run_id"]]
        computed = sum(bool(row.get("evidence_gate_eligible")) for row in rows)
        comparisons.append({
            "run_id": view["run_id"],
            "video_id": view["video_id"],
            "input_view": view["input_view"],
            "prior_summary_eligible_windows": old.get("eligible_windows"),
            "row_level_eligible_observations": computed,
            "eligible_counts_match": old.get("eligible_windows") == computed,
        })
    return {
        "purpose": "audit-only comparison; values are never read by event or video metric functions",
        "prior_summary_sha256": base.sha256_file(base.STAGE04_OFF_SUMMARY),
        "run_level_comparisons": comparisons,
        "all_available_comparisons_match": all(row["eligible_counts_match"] for row in comparisons),
    }


def old_artifact_hashes() -> dict[str, Any]:
    path = OLD_RUN_DIR / "artifact-hashes.json"
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": base.sha256_file(path),
        "content": json.loads(path.read_text()) if path.is_file() else {},
    }


def write_reused_frozen_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import shutil

    copied = {}
    for name in ["frozen-groups.json", "video-inventory.json", "human-label-inventory.json", "official-event-labels.json"]:
        source = OLD_RUN_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Frozen Stage 05 source artifact missing: {source}")
        destination = run_dir / name
        shutil.copyfile(source, destination)
        if base.sha256_file(source) != base.sha256_file(destination):
            raise RuntimeError(f"Frozen artifact copy changed: {name}")
        copied[name] = base.sha256_file(destination)
    groups = json.loads((run_dir / "frozen-groups.json").read_text())
    inventory = json.loads((run_dir / "video-inventory.json").read_text())
    official = json.loads((run_dir / "official-event-labels.json").read_text())
    base.validate_group_assignments(groups["groups"])
    return groups, inventory, official


def make_run_view_metadata(connection: sqlite3.Connection, predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    runs = {row["run_id"]: dict(row) for row in connection.execute("SELECT * FROM agent_runs")}
    metadata = {}
    for view in EVENT_VIEW_RUNS:
        run_id = view["run_id"]
        raw = runs[run_id]
        metadata[run_id] = {
            **provider_settings(raw),
            "run_status": raw.get("status"),
            "event_predictions": [row for row in predictions if row["run_id"] == run_id],
        }
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.overwrite:
        parser.error(f"{args.run_dir} already contains data; pass --overwrite to regenerate")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for required in [args.db, base.LABEL_SOURCE, base.EVENT_SOURCE, base.URFD_LABELS, ORIGINAL_READY_PROOF]:
        if not required.is_file():
            parser.error(f"Required local input missing: {required}")

    frozen_groups, inventory, official = write_reused_frozen_inputs(args.run_dir)
    frozen_partition_by_group = {
        str(row["source_video_group_id"]): str(row["partition"])
        for row in frozen_groups["groups"]
    }
    frozen_group_by_id = {str(row["source_video_group_id"]): row for row in frozen_groups["groups"]}
    VERIFIED_LABEL_SOURCE_REGISTRY.clear()
    VERIFIED_LABEL_SOURCE_REGISTRY.update(build_verified_label_source_registry(frozen_groups))
    write_json(args.run_dir / "label-source-registry.json", {
        "schema_version": "stage05-verified-label-source-registry.v1",
        "source_artifact": str(base.LABEL_SOURCE.resolve().relative_to(ROOT)),
        "source_artifact_sha256": base.sha256_file(base.LABEL_SOURCE),
        "verified_record_count": len(VERIFIED_LABEL_SOURCE_REGISTRY),
        "records": VERIFIED_LABEL_SOURCE_REGISTRY,
    })
    connection = base.read_only_connection(args.db)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Source SQLite integrity check failed: {integrity}")
    proof = verify_original_ready_proof()
    observations, db_counts = export_evidence_observations(connection, proof)
    for row in observations:
        source_group = str(row.get("source_video_group_id"))
        frozen_partition = frozen_partition_by_group.get(source_group)
        frozen_group = frozen_group_by_id.get(source_group, {})
        row["frozen_partition"] = frozen_partition
        row["source_video_sha256"] = frozen_group.get("source_file_sha256")
        row["frozen_partition_match"] = frozen_partition is not None and frozen_partition == str(row.get("partition"))
        row["frozen_partition_status"] = (
            "matched" if row["frozen_partition_match"]
            else "source_group_not_in_frozen_manifest" if frozen_partition is None
            else "observation_partition_mismatch"
        )
    candidates = real_probability_candidates(observations)
    label_path = args.run_dir / "window-labels.csv"
    prepare_window_labels(label_path, candidates)
    labels = load_label_rows(label_path)
    window_metrics, prediction_audit, label_audit = evaluate_probability_labels(candidates, labels)
    audit_by_observation = {row["observation_row_id"]: row for row in prediction_audit}
    for row in observations:
        audit = audit_by_observation.get(row["observation_row_id"])
        if audit:
            row["label"] = audit.get("label")
            row["label_source"] = audit.get("label_source")
            row["label_annotator"] = audit.get("annotator")
            row["label_review_status"] = audit.get("annotation_status")
            row["label_inclusion_status"] = "included" if audit["included_in_metrics"] else "excluded"
            row["label_exclusion_reasons"] = audit["exclusion_reasons"]

    predictions = event_prediction_rows(connection)
    run_meta = make_run_view_metadata(connection, predictions)
    event_metrics, event_window_links, video_rows, event_rows = build_event_evaluation(official, observations, predictions, run_meta, inventory)
    crosscheck = external_summary_crosscheck(observations)
    attempt_rows = base.build_task_attempt_export(connection)
    usage = base.calculate_cache_and_usage(connection, attempt_rows)
    usage["approved_new_p4_http_budget"] = 0
    usage["new_p4_cloud_requests_sent"] = 0
    usage["proposed_plan_execution_status"] = "not executed; 120 expected / 240 maximum remains pending separate approval"

    write_csv(args.run_dir / "decision-observations.csv", observations, list(observations[0].keys()))
    with (args.run_dir / "decision-observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(args.run_dir / "decision-attempts.csv", attempt_rows, list(attempt_rows[0].keys()) if attempt_rows else ["task_id"])
    write_csv(args.run_dir / "event-predictions.csv", predictions, list(predictions[0].keys()) if predictions else ["prediction_id"])
    write_csv(args.run_dir / "event-window-links.csv", event_window_links, list(event_window_links[0].keys()) if event_window_links else ["event_id"])
    write_csv(args.run_dir / "event-coverage.csv", event_rows, list(event_rows[0].keys()) if event_rows else ["event_id"])
    write_csv(args.run_dir / "video-coverage.csv", video_rows, list(video_rows[0].keys()) if video_rows else ["run_id"])
    write_csv(args.run_dir / "probability-row-audit.csv", prediction_audit, list(prediction_audit[0].keys()) if prediction_audit else ["observation_row_id"])
    write_csv(args.run_dir / "window-label-audit.csv", label_audit, list(label_audit[0].keys()) if label_audit else ["label_row_number"])
    attribution_fields = [
        "observation_row_id", "run_id", "source_time_s", "window_id", "evidence_registration_status",
        "evidence_gate_eligible", "current_task_pointer_drifted_from_original", "registered_decision_id",
        "current_task_pointer_decision_id", "task_id", "task_association_disposition", "task_association_status",
        "task_actual_input_observation_id", "task_actual_input_verified", "task_is_actual_input_for_this_observation",
        "task_current_source_time_s", "task_current_snapshot_id", "task_current_decision_id",
        "task_current_request_digest", "task_current_evidence_digest", "task_result_attribution_status",
        "merged_task_result_status", "merged_task_result_decision_id", "merged_task_result_source_time_s",
        "merged_task_result_accepted", "task_result_identity_matches_actual_input",
    ]
    write_csv(args.run_dir / "merged-task-attribution.csv", observations, attribution_fields)
    run_registration_summary = {}
    for run_id in sorted({str(row.get("run_id")) for row in observations}):
        rows = [row for row in observations if str(row.get("run_id")) == run_id]
        run_registration_summary[run_id] = {
            "observation_count": len(rows),
            "evidence_registration_status_counts": dict(Counter(str(row.get("evidence_registration_status")) for row in rows)),
            "eligible_original_windows": sum(bool(row.get("evidence_gate_eligible")) for row in rows),
            "coalesced_prior_evidence_recovered": sum(
                row.get("task_association_status") == "coalesced_prior_observation" and row.get("evidence_gate_eligible")
                for row in rows
            ),
        }
    write_json(args.run_dir / "evidence-registration-summary.json", {
        "schema_version": "stage05-evidence-registration-summary.v1",
        "status_counts": dict(Counter(str(row.get("evidence_registration_status")) for row in observations)),
        "per_run": run_registration_summary,
        "definitions": {
            "missing": "no serialized evidence body exists on the observation row",
            "unverifiable": "body exists but original envelope/frame registration inputs are insufficient to authenticate it",
            "conflict": "origin is available and one or more identity, digest, frame, or body checks disagree",
            "verified": "original identity, envelope, registered frame, rederived body, and evidence digest all agree",
        },
    })
    write_json(args.run_dir / "window-metrics.json", window_metrics)
    write_json(args.run_dir / "event-metrics.json", event_metrics)
    write_json(args.run_dir / "p3-aggregate-crosscheck.json", crosscheck)
    reliability_svg, reliability_manifest = render_reliability_panels(window_metrics)
    (args.run_dir / "reliability.svg").write_text(reliability_svg)
    write_json(args.run_dir / "reliability-panels.json", reliability_manifest)
    write_json(args.run_dir / "cloud-usage.json", usage)

    view_counts = Counter(row["input_view"] for row in video_rows)
    summary = {
        "schema_version": "stage05-revision03-summary.v1",
        "source_database_counts": db_counts,
        "source_sqlite_integrity_check": integrity,
        "read_only_source_database": True,
        "external_network_calls": 0,
        "approved_new_p4_http_budget": 0,
        "observation_rows": len(observations),
        "real_probability_candidate_rows": len(candidates),
        "evidence_registration_status_counts": dict(Counter(str(row.get("evidence_registration_status")) for row in observations)),
        "evidence_missing_observations": sum(row.get("evidence_registration_status") == "missing" for row in observations),
        "evidence_unverifiable_observations": sum(row.get("evidence_registration_status") == "unverifiable" for row in observations),
        "evidence_conflict_observations": sum(row.get("evidence_registration_status") == "conflict" for row in observations),
        "evidence_verified_observations": sum(row.get("evidence_registration_status") == "verified" for row in observations),
        "coalesced_prior_observations_with_verified_original_evidence": sum(
            row.get("task_association_status") == "coalesced_prior_observation" and row.get("evidence_gate_eligible")
            for row in observations
        ),
        "merged_task_result_alias_observations": sum(
            row.get("task_result_attribution_status") == "shared_merged_task_result_not_this_input"
            for row in observations
        ),
        "label_source_registry_record_count": len(VERIFIED_LABEL_SOURCE_REGISTRY),
        "reliability_panel_count": reliability_manifest["panel_count"],
        "frozen_inventory_sha256_reused": {
            name: base.sha256_file(args.run_dir / name)
            for name in ["frozen-groups.json", "video-inventory.json", "human-label-inventory.json", "official-event-labels.json"]
        },
        "frozen_group_count": len(frozen_groups["groups"]),
        "frozen_partition_mismatches": sum(not bool(row.get("frozen_partition_match")) for row in observations),
        "partition_counts": dict(Counter(row["partition"] for row in frozen_groups["groups"])),
        "event_view_run_rows_by_input_view": dict(view_counts),
        "event_metrics": event_metrics,
        "window_metrics": window_metrics,
        "p3_summary_used_for_event_metrics": False,
        "p3_summary_audit_only_crosscheck": crosscheck,
        "source_input_modes": {
            "single_camera_rgb_runs": [row for row in video_rows if row["input_view"] == "single_camera_rgb"],
            "three_panel_derived_runs": [row for row in video_rows if row["input_view"] == "three_panel_derived"],
        },
        "old_report_and_probe_preserved": True,
        "conclusion": {
            "r1_event_coverage_is_row_level_and_event_time_matched": True,
            "r2_probability_labels_are_review_and_identity_audited": True,
            "calibration_accepted": False,
            "deployment_policy_changed": False,
        },
    }
    write_json(args.run_dir / "summary.json", summary)
    source_paths = [
        args.db, Path(str(args.db) + "-wal"), base.STAGE04_SOURCE_MANIFEST,
        base.STAGE04_OFF_SUMMARY, base.LABEL_SOURCE, base.EVENT_SOURCE,
        base.HEAD_PROTOCOL, base.HEAD_PROVENANCE, base.URFD_LABELS, base.URFD_ADL_LABELS,
        base.STAGE04_MAIN_REPORT, base.STAGE04_R1_REPORT, base.STAGE04_R2_APPROVAL,
        OLD_REPORT, REVIEW_PATH, OLD_RUN_DIR / "artifact-hashes.json",
        OLD_PROBE_DIR / "reviewer_probes.py", OLD_PROBE_DIR / "reviewer_probes.json",
        ORIGINAL_READY_PROOF, Path(__file__), ROOT / "scripts/evaluate_decision_stage05.py",
        ROOT / "output/agent-review/stage-05/revision-03/r1_probe.py",
        ROOT / "output/agent-review/stage-05/revision-03/r2_probe.py",
        ROOT / "output/agent-review/stage-05/independent-revision-03/reviewer_probes.py",
        ROOT / "output/agent-review/stage-05/revision-03/revision02_probe.py",
        ROOT / "output/agent-review/stage-05/revision-03/revision03_probe.py",
        ROOT / "output/agent-review/stage-05/revision-03/reviewer_checks.py",
    ]
    source_hashes = {
        str(path.resolve().relative_to(ROOT)): base.sha256_file(path)
        for path in source_paths if path.is_file()
    }
    lock = {
        "schema_version": "stage05-revision03-source-lock.v1",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "git_worktree_dirty_at_run": True,
        "revision_evaluator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "revision_evaluator_sha256": base.sha256_file(Path(__file__)),
        "independent_probe_sha256": {
            "r1": base.sha256_file(ROOT / "output/agent-review/stage-05/revision-03/r1_probe.py"),
            "r2": base.sha256_file(ROOT / "output/agent-review/stage-05/revision-03/r2_probe.py"),
            "revision02": base.sha256_file(ROOT / "output/agent-review/stage-05/revision-03/revision02_probe.py"),
            "revision03": base.sha256_file(ROOT / "output/agent-review/stage-05/revision-03/revision03_probe.py"),
            "reviewer": base.sha256_file(ROOT / "output/agent-review/stage-05/independent-revision-03/reviewer_probes.py"),
            "revision03_reviewer_checks": base.sha256_file(ROOT / "output/agent-review/stage-05/revision-03/reviewer_checks.py"),
        },
        "previous_evaluator_sha256": base.sha256_file(ROOT / "scripts/evaluate_decision_stage05.py"),
        "reused_frozen_groups_sha256": base.sha256_file(args.run_dir / "frozen-groups.json"),
        "event_view_runs": EVENT_VIEW_RUNS,
        "event_reference_runs": EVENT_REFERENCE_RUNS,
        "window_event_overlap_rule": WINDOW_EVENT_OVERLAP_RULE,
        "original_ready_input_association_proof_sha256": proof.get("sha256"),
        "source_sha256": source_hashes,
        "read_only_source_database": True,
        "external_network_calls": 0,
        "approved_new_p4_http_budget": 0,
    }
    write_json(args.run_dir / "code-and-source-lock.json", lock)
    (args.run_dir / "run-command.txt").write_text(
        "PYTHONHASHSEED=1 .venv/bin/python scripts/evaluate_decision_stage05_revision03.py --db runs/decision-stage04-live/decision-agent.sqlite3 --run-dir runs/decision-stage05-revision-03-20261118 --overwrite\n"
        "cp runs/decision-stage05-revision-03-20261118/artifact-hashes.json /tmp/stage05-revision03-seed1-artifact-hashes.json\n"
        "PYTHONHASHSEED=77 .venv/bin/python scripts/evaluate_decision_stage05_revision03.py --db runs/decision-stage04-live/decision-agent.sqlite3 --run-dir runs/decision-stage05-revision-03-20261118 --overwrite\n"
        "cmp /tmp/stage05-revision03-seed1-artifact-hashes.json runs/decision-stage05-revision-03-20261118/artifact-hashes.json\n"
        "cp runs/decision-stage05-revision-03-20261118/window-labels.csv output/agent-review/stage-05/independent-revision-03/generated/window-labels.csv\n"
        ".venv/bin/python output/agent-review/stage-05/independent-revision-03/reviewer_probes.py\n"
        ".venv/bin/python output/agent-review/stage-05/revision-03/r1_probe.py\n"
        ".venv/bin/python output/agent-review/stage-05/revision-03/r2_probe.py\n"
        ".venv/bin/python output/agent-review/stage-05/revision-03/revision02_probe.py\n"
        ".venv/bin/python output/agent-review/stage-05/revision-03/revision03_probe.py\n"
        ".venv/bin/python output/agent-review/stage-05/revision-03/reviewer_checks.py\n"
        ".venv/bin/python scripts/check_portable.py\n"
        ".venv/bin/python scripts/prepare_github.py --report output/agent-review/stage-05/revision-03/publish-boundary.json\n"
    )
    connection.close()
    write_json(args.run_dir / "artifact-hashes.json", base.hash_manifest(args.run_dir))
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "observations": len(observations),
        "eligible_probability_candidates": window_metrics["eligible_real_scored_windows"],
        "labeled_probability_rows": window_metrics["labeled_version_matched_windows"],
        "event_gate_coverage": {row["cohort"]: row["window_gate_event_coverage"] for row in event_metrics["cohorts"]},
        "visual_event_recall": {row["cohort"]: row["visual_event_recall"] for row in event_metrics["cohorts"]},
        "sqlite_integrity": integrity,
        "network_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
