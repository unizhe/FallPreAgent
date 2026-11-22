"""Deterministic, clearly synthetic evidence used only by P0 demos/tests."""
from __future__ import annotations

import hashlib

from .decision_contracts import DecisionEvidence, EvidenceStatus, IdentityLink


def synthetic_evidence(*, sufficient: bool = True, suffix: str = "complete") -> DecisionEvidence:
    """Return a fixed fixture; no video, model weights, credentials or network."""
    digest = hashlib.sha256(f"decision-stage01:{suffix}".encode()).hexdigest()
    return DecisionEvidence(
        run_id=f"synthetic-run-{suffix}",
        session_id=f"synthetic-session-{suffix}",
        stream_id="synthetic-stream-01",
        snapshot_id=f"synthetic-snapshot-{suffix}",
        decision_id=f"synthetic-decision-{suffix}",
        request_digest=digest,
        action_epoch=3,
        generation=2,
        fall_target_epoch="synthetic-fall-target-3",
        identity_link=IdentityLink.SAME_TARGET if sufficient else IdentityLink.AMBIGUOUS,
        source_time_s=12.5,
        source_window={"start_s": 10.0, "end_s": 12.5},
        observed_frames=24 if sufficient else 2,
        observable=True,
        event_decidable=sufficient,
        camera_stable=True,
        person_count=1 if sufficient else 2,
        pose_valid_fraction=0.92 if sufficient else 0.25,
        continuous=sufficient,
        missing_reason_codes=[] if sufficient else ["SYNTHETIC_WINDOW_TOO_SHORT"],
        temporal_status=EvidenceStatus.READY if sufficient else EvidenceStatus.INSUFFICIENT,
        temporal_model_version="synthetic-stgcn-fixture-v1" if sufficient else None,
        temporal_score=0.71 if sufficient else None,
        head_status=EvidenceStatus.READY if sufficient else EvidenceStatus.INSUFFICIENT,
        head_variant="tail_joint",
        head_version="synthetic-tail-joint-fixture-v1" if sufficient else None,
        head_raw_score=0.82 if sufficient else None,
        event_state="suspected" if sufficient else "unknown",
        recovery_status="not_established" if sufficient else "unknown",
        yolo_model_version="synthetic-yolo-fixture-v1",
        preprocessing_version="synthetic-preprocess-v1",
        prompt_version="decision-prompt-p0-v1",
        policy_version="decision-policy-p0-v1",
        shared_pose_evidence=True,
    )
