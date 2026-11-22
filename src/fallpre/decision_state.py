"""Whitelisted state and fixed questions sent to the decision model."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .decision_contracts import DecisionEvidence

DECISION_STATE_SCHEMA_VERSION = "fall-prevention-decision-state.v1"
DECISION_PROMPT_VERSION = "fall-process-review-prompt.v1"

_QUESTIONS: dict[str, Any] = {
    "fall_occurred": {
        "type": "noul",
        "instructions": (
            "判断该目标在给定观察窗口内是否已经发生跌倒过程。仅依据窗口中的已观测过程证据；"
            "高分、风险或观测不足不能单独证明发生了跌倒。"
        ),
        "criteria": {
            "true": "观察窗口内存在明确的跌倒过程证据。",
            "false": "观察窗口内没有明确的跌倒过程证据。",
        },
    },
    "evidence_sufficiency": {
        "type": "choice",
        "instructions": "判断当前结构化证据是否足以支持窗口内判断，并识别证据不足或相互冲突。",
        "criteria": {
            "sufficient": "目标、窗口和质量信息明确，证据可用于作出窗口内判断。",
            "insufficient": "证据缺少关键内容、覆盖不足或不够新鲜。",
            "conflicting": "身份、动作、过程、几何或事件信息之间存在明显矛盾。",
        },
    },
    "next_step": {
        "type": "choice",
        "instructions": "只从下列本地工作流建议中选择；不要执行通知、工具或其他动作。",
        "criteria": {
            "emit_result": "证据充分且问题答案一致，可记录本窗口判断。",
            "observe_more": "证据不足且仍可能通过新的连续观测补足。",
            "request_review": "证据冲突或无法安全自动判断，应保留给人工复核。",
        },
    },
}
_SAFE_CODE = re.compile(r"^[A-Z0-9_:-]{1,64}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9._+-]{1,128}$")


def _safe_codes(values: list[str]) -> list[str]:
    return [value for value in values if _SAFE_CODE.fullmatch(value)]


def _safe_version(value: str | None) -> str | None:
    return value if value is not None and _SAFE_VERSION.fullmatch(value) else None


def decision_questions() -> dict[str, Any]:
    """Return a fresh copy so callers cannot mutate the versioned prompt."""
    return deepcopy(_QUESTIONS)


def build_decision_state(evidence: DecisionEvidence) -> dict[str, Any]:
    """Build a small, strict, non-identifying JSON object from validated evidence.

    This intentionally copies only named evidence fields. It never accepts an
    envelope, video, image, path, person name, feedback, or arbitrary text.
    """
    return {
        "state_schema_version": DECISION_STATE_SCHEMA_VERSION,
        "prompt_version": DECISION_PROMPT_VERSION,
        "target_continuity": {
            "identity_link": evidence.identity_link.value,
            "action_epoch": evidence.action_epoch,
            "generation": evidence.generation,
            "fall_target_epoch_present": evidence.fall_target_epoch is not None,
        },
        "window": {
            "start_s": evidence.source_window.start_s,
            "end_s": evidence.source_window.end_s,
            "source_time_s": evidence.source_time_s,
            "real_observed_frames": evidence.observed_frames,
        },
        "quality": {
            "observable": evidence.observable,
            "event_decidable": evidence.event_decidable,
            "camera_stable": evidence.camera_stable,
            "person_count": evidence.person_count,
            "pose_valid_fraction": evidence.pose_valid_fraction,
            "continuous": evidence.continuous,
            "missing_reason_codes": _safe_codes(evidence.missing_reason_codes),
        },
        "action": {"label": evidence.action_label},
        "process": {
            "fall_process_label": evidence.fall_process_label,
            "temporal": {
                "status": evidence.temporal_status.value,
                "fall_score": evidence.temporal_score,
                "model_version": _safe_version(evidence.temporal_model_version),
            },
            "fall_head": {
                "status": evidence.head_status.value,
                "variant": evidence.head_variant,
                "raw_score": evidence.head_raw_score,
                "version": _safe_version(evidence.head_version),
                "weights_sha256": evidence.head_weights_sha256,
            },
        },
        "geometry": {
            "status": evidence.geometry_status.value if evidence.geometry_status else None,
            "transition": evidence.geometry_transition.value if evidence.geometry_transition else None,
            "downward_motion_observed": evidence.geometry_downward_motion_observed,
            "score": evidence.geometry_score,
            "conflict_reason_codes": _safe_codes(evidence.geometry_conflict_reason_codes),
        },
        "event": {
            "state": evidence.event_state.value,
            "recovery_status": evidence.recovery_status.value,
            "alarm_authority": evidence.alarm_authority,
        },
        "evidence_relationships": {
            "temporal_and_head_share_pose": evidence.shared_pose_evidence,
            "independent_sensor_confirmation": False if evidence.shared_pose_evidence else None,
            "conflict_reason_codes": _safe_codes(evidence.conflict_reason_codes),
        },
        "versions": {
            "eligibility_policy": evidence.eligibility_policy_version,
            "yolo": _safe_version(evidence.yolo_model_version),
            "temporal": _safe_version(evidence.temporal_model_version),
            "preprocessing": _safe_version(evidence.preprocessing_version),
            "fall_head": _safe_version(evidence.head_version),
            "prompt": DECISION_PROMPT_VERSION,
            "policy": _safe_version(evidence.policy_version),
        },
    }


__all__ = [
    "DECISION_PROMPT_VERSION",
    "DECISION_STATE_SCHEMA_VERSION",
    "build_decision_state",
    "decision_questions",
]
