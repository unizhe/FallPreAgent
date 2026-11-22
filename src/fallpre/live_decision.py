"""Build schema-v1 Agent evidence from the actual live perception result.

This adapter is deliberately server-side. It never accepts evidence values
from the browser and never changes the primary event state.
"""
from __future__ import annotations

import math
from typing import Any

from .decision_contracts import DECISION_ELIGIBILITY_POLICY_VERSION
from .decision_state import DECISION_PROMPT_VERSION

DECISION_POLICY_VERSION = "decision-policy-p0-v1"
HEAD_VERSION = "fall-process-cd-20261107"
LIVE_EVIDENCE_MAPPING_VERSION = "live-evidence-mapping-v2"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = tuple(_num(item) for item in value)
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1e-12, area_left + area_right - intersection)


def _pose_fraction(person: dict[str, Any] | None, width: int, height: int) -> float:
    points = person.get("keypoints") if isinstance(person, dict) else None
    if not isinstance(points, list) or len(points) != 17:
        return 0.0
    visible = 0
    for point in points:
        if not isinstance(point, list) or len(point) < 3:
            continue
        x, y, score = (_num(point[index]) for index in range(3))
        if x is not None and y is not None and score is not None and score >= 0.25 \
                and 0 < x < width and 0 < y < height:
            visible += 1
    return visible / 17.0


def _event_state(fall: dict[str, Any]) -> str:
    value = fall.get("state")
    return value if value in {"normal", "unobservable", "suspected", "confirmed"} else "unknown"


def _recovery_status(fall: dict[str, Any]) -> str:
    lifecycle = fall.get("lifecycle")
    transitions = fall.get("transitions") or []
    rearmed = next((row for row in reversed(transitions)
                    if isinstance(row, dict) and row.get("kind") == "rearmed"), None)
    if rearmed:
        return "confirmed" if rearmed.get("recovery_scope") == "standing_or_seated" else "low_support_only"
    if lifecycle == "recovering":
        return "recovering"
    return "not_established"


def live_observation_envelope(
    result: dict[str, Any], *, run_id: str, session_id: str, stream_id: str,
    source_time_s: float, head_variant: str = "cd",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a complete immutable observation wrapper and a local trace.

    The summary is a field-for-field projection of the ``perception`` object;
    ``adapt_envelope`` checks for disagreement before it creates DecisionEvidence.
    Missing model output stays missing and makes the observation ineligible.
    """
    if head_variant not in {"cd", "tail_joint"}:
        raise ValueError("unsupported fall head variant")
    temporal = result.get("temporal") if isinstance(result.get("temporal"), dict) else {}
    temporal_window = temporal.get("window") if isinstance(temporal.get("window"), dict) else {}
    temporal_model = temporal.get("model") if isinstance(temporal.get("model"), dict) else {}
    action = result.get("action") if isinstance(result.get("action"), dict) else {}
    action_selection = action.get("selection") if isinstance(action.get("selection"), dict) else {}
    fall_observation = result.get("fall_observation") if isinstance(result.get("fall_observation"), dict) else {}
    fall_selection = fall_observation.get("selection") if isinstance(fall_observation.get("selection"), dict) else {}
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    fall = result.get("fall") if isinstance(result.get("fall"), dict) else {}
    persons = result.get("persons") if isinstance(result.get("persons"), list) else []
    action_index = action.get("target_index")
    action_person = (persons[action_index] if isinstance(action_index, int) and not isinstance(action_index, bool)
                     and 0 <= action_index < len(persons) else None)
    action_epoch = action_selection.get("target_epoch")
    generation = temporal.get("generation")
    target_epoch = fall_selection.get("target_epoch")
    fall_person = fall_observation.get("person") if isinstance(fall_observation.get("person"), dict) else None

    action_box = _box(action_person.get("box")) if action_person else None
    fall_box = _box(fall_person.get("box")) if fall_person else None
    identity_link = "ambiguous"
    if action_person is not None and fall_person is not None and len(persons) == 1:
        same_current_pose = (fall_observation.get("source") == "raw"
                             and fall_selection.get("raw_index") == action_index)
        same_rgb_box = (fall_observation.get("source") == "rgb_fall" and action_box and fall_box
                        and _iou(action_box, fall_box) >= 0.5)
        if same_current_pose or same_rgb_box:
            identity_link = "same_target"
        elif fall_box is not None and action_box is not None:
            identity_link = "different_target"
    elif action_person is None and fall_person is None:
        identity_link = "not_applicable"

    real_observations = temporal_window.get("observations")
    if type(real_observations) is not int:
        real_observations = temporal.get("observed_frames")
    window_start = _num(temporal_window.get("start_s"))
    window_end = _num(temporal_window.get("end_s"))
    if window_start is None or window_end is None:
        span = _num(temporal.get("span_s"))
        temporal_time = _num(temporal.get("source_time"))
        if span is not None and temporal_time is not None and span > 0:
            window_start, window_end = temporal_time - span, temporal_time

    temporal_is_ready = (temporal.get("status") == "ready" and window_start is not None
                         and window_end is not None and type(real_observations) is int
                         and type(generation) is int and type(action_epoch) is int
                         and temporal.get("generation") == generation
                         and temporal.get("target_epoch") == action_epoch)
    if temporal_is_ready:
        result_age = source_time_s - window_end
        if result_age < -1e-9:
            temporal_status = "stale"
        elif result_age > 0.6 + 1e-9:
            temporal_status = "stale"
        else:
            temporal_status = "ready"
    elif temporal.get("status") in {"warming_up", "pending", "unobservable", "insufficient"}:
        temporal_status = "insufficient"
    else:
        temporal_status = "unavailable"

    heads = temporal.get("fall_heads") if isinstance(temporal.get("fall_heads"), dict) else {}
    selected_head_name = "st_geometry" if head_variant == "cd" else "tail_joint"
    selected_head = heads.get(selected_head_name) if isinstance(heads.get(selected_head_name), dict) else {}
    head_hash = selected_head.get("weights_sha256")
    head_score = _num(selected_head.get("fall_process_score"))
    head_matches_window = (selected_head.get("status") == "ready" and temporal_is_ready
                           and selected_head.get("window_end_s") == window_end
                           and selected_head.get("generation") == generation
                           and selected_head.get("target_epoch") == action_epoch
                           and isinstance(head_hash, str) and len(head_hash) == 64)
    head_status = "ready" if temporal_status == "ready" and head_matches_window and head_score is not None else (
        "insufficient" if temporal_status == "insufficient" else "stale" if temporal_status == "stale"
        else "unavailable")
    head_label = "fall_process" if head_score is not None and head_score >= 0.5 else (
        "non_fall_activity" if head_score is not None else "unknown")

    fall_quality = result.get("fall_temporal_evidence")
    fall_quality = fall_quality if isinstance(fall_quality, dict) else {}
    geometry_raw = fall_quality.get("geometry") if isinstance(fall_quality.get("geometry"), dict) else {}
    geometry_metrics = geometry_raw.get("metrics") if isinstance(geometry_raw.get("metrics"), dict) else {}
    geometry_status = ("ready" if geometry_raw.get("quality") == "observable" else
                       "insufficient" if geometry_raw.get("quality") == "insufficient" else "unavailable")
    fall_metric = fall.get("metrics") if isinstance(fall.get("metrics"), dict) else {}
    descending = geometry_raw.get("descending")
    descending_source = "fall_temporal_evidence.geometry.descending"
    if type(descending) is not bool:
        descending = fall_metric.get("descending")
        descending_source = "fall.metrics.descending"
    if type(descending) is not bool:
        descending = None
        descending_source = "missing"
    low_pose = geometry_raw.get("low_pose")
    low_pose_source = "fall_temporal_evidence.geometry.low_pose"
    if type(low_pose) is not bool:
        low_pose = fall_metric.get("low_pose")
        low_pose_source = "fall.metrics.low_pose"
    if type(low_pose) is not bool:
        low_pose = None
        low_pose_source = "missing"
    upright_witness_source = "missing"
    upright_witness = False
    for source, value in (
        ("fall_temporal_evidence.geometry.upright", geometry_raw.get("upright")),
        ("fall_temporal_evidence.geometry.geometry_upright", geometry_raw.get("geometry_upright")),
        ("fall.metrics.pose_upright_witness", fall_metric.get("pose_upright_witness")),
    ):
        if type(value) is bool:
            upright_witness_source = source
            upright_witness = value
            if value:
                break
    if low_pose is True:
        geometry_transition = ("upright_to_low" if descending is True
                               and fall_metric.get("had_upright") is True else "low")
    elif upright_witness:
        geometry_transition = ("recovering" if _recovery_status(fall) == "recovering" else "upright")
    else:
        # A normal event or an explicit low_pose=False does not prove upright.
        geometry_transition = "unknown"

    width, height = result.get("width"), result.get("height")
    pose_fraction = (_pose_fraction(action_person, width, height)
                     if type(width) is int and type(height) is int else 0.0)
    camera_stable = quality.get("camera_stable") is True
    observable = bool(action_person is not None and action_selection.get("state") == "locked" and camera_stable)
    event_decidable = bool(fall.get("state") in {"normal", "suspected", "confirmed"} and camera_stable)
    window_continuous = (temporal_window.get("causal") is True and temporal_window.get("temporal_wrap") is False
                         and (_num(temporal_window.get("max_observation_gap_s")) or math.inf) <= 0.45 + 1e-9)
    identity = {}
    if type(action_epoch) is int and action_epoch >= 0:
        identity["action_epoch"] = action_epoch
    if type(generation) is int and generation >= 0:
        identity["generation"] = generation
    if target_epoch is not None:
        identity["fall_target_epoch"] = str(target_epoch)
    identity["identity_link"] = identity_link

    temporal_action: dict[str, Any] = {
        "status": temporal_status,
        "result_current": temporal_is_ready and temporal_status == "ready",
        "observed_frames": real_observations if type(real_observations) is int else None,
        "model_version": temporal_model.get("id") or temporal.get("model_id"),
        "source_window": ({"start_s": window_start, "end_s": window_end}
                          if window_start is not None and window_end is not None else None),
        "all_scores": temporal.get("all_scores") if temporal_status == "ready" else None,
        "top3": temporal.get("top3", []) if temporal_status == "ready" else [],
        "window": temporal_window or None,
        "semantics": "past_window_60_classes_not_current_pose",
    }
    all_scores = temporal.get("all_scores")
    temporal_score = _num(temporal.get("relevant_scores", {}).get("falling")) \
        if isinstance(temporal.get("relevant_scores"), dict) else None
    if temporal_score is None and isinstance(all_scores, list) and len(all_scores) > 42:
        temporal_score = _num(all_scores[42])
    if temporal_status == "ready":
        temporal_action["fall_score"] = temporal_score

    c_head = heads.get("st_only") if isinstance(heads.get("st_only"), dict) else {}
    d_head = heads.get("st_geometry") if isinstance(heads.get("st_geometry"), dict) else {}
    head_payload = {
        "status": head_status,
        "variant": head_variant,
        "version": HEAD_VERSION if head_status == "ready" else None,
        "raw_score": head_score if head_status == "ready" else None,
        "weights_sha256": head_hash if head_status == "ready" else None,
        "deployment_status": "experimental_not_alarm_authority",
        "selected_head_name": selected_head_name,
        "heads": {"C": c_head, "D": d_head},
    }
    fall_process = {
        "status": "experimental" if head_status == "ready" else "abstain",
        "label": head_label if head_status == "ready" else "unknown",
        "head": head_payload,
        "geometry": {
            "status": geometry_status,
            "mapping_version": LIVE_EVIDENCE_MAPPING_VERSION,
            "transition": geometry_transition,
            "downward_motion_observed": descending,
            "score": None,
            "metrics": geometry_metrics,
            "conflict_reason_codes": [],
        },
        "conflict_reason_codes": [item.get("code") for item in fall_quality.get("conflicts", [])
                                   if isinstance(item, dict) and isinstance(item.get("code"), str)],
        "shared_pose_evidence": True,
        "changes_alarm": False,
        "score_semantics": "uncalibrated observed-process class score, not future risk",
    }
    primary_state = _event_state(fall)
    event = {
        "state": primary_state,
        "event_key": (f"{stream_id}/{fall.get('current_event_id') or fall.get('event_id')}"
                      if fall.get("current_event_id") or fall.get("event_id") else None),
        "recovery_status": _recovery_status(fall),
        "recovery_confirmed": _recovery_status(fall) == "confirmed",
        "alarm_authority": "upstream_only",
        "transitions": fall.get("transitions") if isinstance(fall.get("transitions"), list) else [],
    }
    action_label = action.get("label") if action.get("label") in {
        "standing", "sitting", "lying", "walking", "falling", "sit_to_stand",
        "standing_to_sitting", "unknown",
    } else "other"
    person_count = len(persons)
    missing_reasons = []
    if action_person is None:
        missing_reasons.append("ACTION_TARGET_UNAVAILABLE")
    if not camera_stable:
        missing_reasons.append("CAMERA_UNSTABLE")
    if temporal_status != "ready":
        missing_reasons.append("TEMPORAL_" + temporal_status.upper())
    if head_status != "ready":
        missing_reasons.append("FALL_HEAD_" + head_status.upper())
    if identity_link != "same_target":
        missing_reasons.append("IDENTITY_LINK_" + identity_link.upper())

    observation = {
        "observable": observable,
        "event_decidable": event_decidable,
        "camera_stable": camera_stable,
        "person_count": person_count,
        "pose_valid_fraction": pose_fraction,
        "continuous": window_continuous,
        "missing_reason_codes": list(dict.fromkeys(missing_reasons)),
    }
    process = {
        "schema_version": 1,
        "stream_id": stream_id,
        "source_time_s": float(source_time_s),
        "identity": identity,
        "observation": observation,
        "current_action": {"label": action_label, "source": "live-yolo-v7"},
        "temporal_action": temporal_action,
        "fall_process": fall_process,
        "event": event,
        "notification": {"emit": False, "external_action_performed": False},
        "shared_pose_evidence": True,
    }
    summary: dict[str, Any] = {
        "action_epoch": identity.get("action_epoch"),
        "generation": identity.get("generation"),
        "fall_target_epoch": identity.get("fall_target_epoch"),
        "identity_link": identity_link,
        "observable": observable,
        "event_decidable": event_decidable,
        "camera_stable": camera_stable,
        "person_count": person_count,
        "pose_valid_fraction": pose_fraction,
        "continuous": window_continuous,
        "missing_reason_codes": list(dict.fromkeys(missing_reasons)),
        "temporal_status": temporal_status,
        "temporal_model_version": temporal_action.get("model_version"),
        "temporal_score": temporal_score if temporal_status == "ready" else None,
        "head_status": head_status,
        "head_variant": head_variant,
        "head_version": HEAD_VERSION if head_status == "ready" else None,
        "head_raw_score": head_score if head_status == "ready" else None,
        "head_weights_sha256": head_hash if head_status == "ready" else None,
        "conflict_reason_codes": fall_process["conflict_reason_codes"],
        "action_label": action_label,
        "fall_process_label": fall_process["label"],
        "geometry_status": "ready" if geometry_status == "ready" else geometry_status,
        "geometry_transition": geometry_transition,
        "geometry_downward_motion_observed": fall_process["geometry"]["downward_motion_observed"],
        "geometry_conflict_reason_codes": [],
        "yolo_model_version": (result.get("model") or {}).get("id", "unknown"),
        "preprocessing_version": (temporal_model.get("preprocessing") or "unknown") + "/" +
                                str(temporal_model.get("resampling") or temporal_window.get("resampling") or "unknown"),
        "prompt_version": DECISION_PROMPT_VERSION,
        "policy_version": DECISION_POLICY_VERSION,
        "shared_pose_evidence": True,
    }
    if window_start is not None and window_end is not None and window_start >= 0 and window_end > window_start:
        summary["source_window"] = {"start_s": window_start, "end_s": window_end}
    if type(real_observations) is int and real_observations >= 1:
        summary["observed_frames"] = real_observations
    envelope = {
        "schema_version": "agent-observation-envelope.v1",
        "session_id": session_id,
        "stream_id": stream_id,
        "source_time_s": float(source_time_s),
        "perception": process,
        "evidence_summary": summary,
        "evidence_origin": "authorized_structured_input",
    }
    trace = {
        "run_id": run_id,
        "source_time_s": float(source_time_s),
        "observation": observation,
        "temporal": {"status": temporal_status, "window": summary.get("source_window"),
                     "real_observations": summary.get("observed_frames"),
                     "model_id": temporal_model.get("id"), "model_sha256": temporal_model.get("sha256"),
                     "falling_class_score": temporal_score},
        "head": {"status": head_status, "variant": head_variant, "name": selected_head_name,
                 "weights_sha256": head_hash if head_status == "ready" else None,
                 "raw_score": head_score if head_status == "ready" else None},
        "geometry_mapping_version": LIVE_EVIDENCE_MAPPING_VERSION,
        "geometry_field_sources": {
            "descending": descending_source,
            "low_pose": low_pose_source,
            "upright_witness": upright_witness_source,
        },
        "event_state": primary_state,
        "missing_reason_codes": observation["missing_reason_codes"],
        "eligibility_policy_version": DECISION_ELIGIBILITY_POLICY_VERSION,
    }
    return envelope, trace
