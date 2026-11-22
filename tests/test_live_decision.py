from __future__ import annotations

from copy import deepcopy

from fallpre.decision_evidence import adapt_envelope
from fallpre.decision_state import build_decision_state
from fallpre.live_decision import live_observation_envelope


def live_result(*, with_head=True, source_time=2.0):
    points = [[100.0 + index, 100.0 + index, 0.9] for index in range(17)]
    scores = [0.01] * 60
    scores[42] = 0.1
    head = {
        "status": "ready",
        "fall_process_score": 0.61,
        "weights_sha256": "a" * 64,
        "generation": 2,
        "target_epoch": 3,
        "window_end_s": 2.0,
    }
    heads = {"st_only": {**head, "fall_process_score": 0.2}}
    if with_head:
        heads["st_geometry"] = head
    return {
        "width": 640,
        "height": 480,
        "persons": [{"box": [50, 40, 250, 450], "confidence": 0.9, "keypoints": points}],
        "model": {"id": "yolo26n-pose", "sha256": "b" * 64},
        "quality": {"camera_stable": True},
        "action": {"label": "standing", "target_index": 0,
                   "selection": {"target_epoch": 3, "state": "locked"}},
        "fall_observation": {"source": "raw", "person": {"box": [50, 40, 250, 450]},
                             "selection": {"raw_index": 0, "target_epoch": 9, "state": "locked"}},
        "fall": {"state": "normal", "lifecycle": "observing", "metrics": {}, "transitions": []},
        "fall_temporal_evidence": {"geometry": {"quality": "observable", "metrics": {}}, "conflicts": []},
        "temporal": {
            "status": "ready", "generation": 2, "target_epoch": 3, "observed_frames": 12,
            "source_time": source_time, "window_start_s": 0.0, "window_end_s": 2.0,
            "window": {"start_s": 0.0, "end_s": 2.0, "observations": 12,
                       "max_observation_gap_s": 0.2, "causal": True, "temporal_wrap": False,
                       "preprocessing": "image-normalized", "resampling": "linear_supported"},
            "model_id": "stgcnpp-ntu60-2d-coco17-joint",
            "model": {"id": "stgcnpp-ntu60-2d-coco17-joint", "sha256": "c" * 64,
                      "preprocessing": "coco17-image-normalized", "resampling": "linear_supported"},
            "all_scores": scores, "fall_heads": heads, "relevant_scores": {"falling": 0.1}, "top3": [],
        },
        "inference_ms": 80.0,
        "local_visual_ms": 145.0,
    }


def adapt_result(result, run_id="run-" + "1" * 32):
    envelope, trace = live_observation_envelope(
        result, run_id=run_id, session_id="session-live", stream_id="stream-live-1234567890",
        source_time_s=2.0, head_variant="cd",
    )
    adaptation = adapt_envelope(envelope, run_id=run_id, snapshot_id="snapshot-live",
                                decision_id="decision-live", request_digest="d" * 64)
    return envelope, trace, adaptation


def test_real_live_mapping_builds_consistent_cd_evidence_without_changing_primary_event():
    result = live_result()
    original_event = deepcopy(result["fall"])
    envelope, trace, adaptation = adapt_result(result)

    assert adaptation.missing_fields == ()
    assert adaptation.evidence is not None
    assert adaptation.evidence.eligible
    assert adaptation.evidence.head_variant == "cd"
    assert adaptation.evidence.head_raw_score == 0.61
    assert adaptation.evidence.temporal_score == 0.1
    assert adaptation.evidence.observed_frames == 12
    assert adaptation.evidence.source_window.start_s == 0.0
    assert adaptation.evidence.source_window.end_s == 2.0
    assert adaptation.evidence.identity_link.value == "same_target"
    assert trace["head"]["name"] == "st_geometry"
    assert trace["head"]["weights_sha256"] == "a" * 64
    assert result["fall"] == original_event
    assert envelope["perception"]["event"]["state"] == "normal"


def test_missing_selected_head_stays_missing_and_never_fabricates_probability():
    envelope, trace, adaptation = adapt_result(live_result(with_head=False))

    assert trace["head"]["status"] == "unavailable"
    assert envelope["perception"]["fall_process"]["head"]["raw_score"] is None
    assert adaptation.evidence is not None
    assert not adaptation.evidence.eligible
    assert "FALL_HEAD_EVIDENCE_NOT_READY" in adaptation.evidence.eligibility_reasons


def test_stale_temporal_output_is_rejected_without_head_relabeling():
    result = live_result()
    result["temporal"]["window_end_s"] = 1.0
    result["temporal"]["window"]["end_s"] = 1.0
    result["temporal"]["fall_heads"]["st_geometry"]["window_end_s"] = 1.0
    envelope, trace, adaptation = adapt_result(result)

    assert trace["temporal"]["status"] == "stale"
    assert trace["head"]["status"] == "stale"
    assert envelope["perception"]["temporal_action"]["status"] == "stale"
    assert adaptation.evidence is not None
    assert not adaptation.evidence.eligible
    assert "ST_RESULT_STALE_AT_TRIGGER" in adaptation.evidence.eligibility_reasons


def test_inconsistent_source_timestamp_remains_unavailable():
    result = live_result()
    result["temporal"]["window_end_s"] = 2.1
    result["temporal"]["window"]["end_s"] = 2.1
    result["temporal"]["fall_heads"]["st_geometry"]["window_end_s"] = 2.1
    envelope, trace, adaptation = adapt_result(result)

    assert envelope["perception"]["temporal_action"]["source_window"]["end_s"] == 2.1
    assert adaptation.evidence is None
    assert "TEMPORAL_STALE" in trace["missing_reason_codes"]
    assert "evidence_summary" in adaptation.missing_fields


def test_normal_sitting_without_geometry_does_not_claim_upright_or_block_st_eligibility():
    result = live_result()
    result["action"]["label"] = "sitting"
    result["fall"]["metrics"] = {}
    result["fall_temporal_evidence"]["geometry"] = {}

    envelope, trace, adaptation = adapt_result(result)
    geometry = envelope["perception"]["fall_process"]["geometry"]

    assert geometry["status"] == "unavailable"
    assert geometry["transition"] == "unknown"
    assert geometry["downward_motion_observed"] is None
    assert geometry["mapping_version"] == "live-evidence-mapping-v2"
    assert trace["geometry_mapping_version"] == geometry["mapping_version"]
    assert trace["geometry_field_sources"] == {
        "descending": "missing", "low_pose": "missing",
        "upright_witness": "missing",
    }
    assert adaptation.evidence is not None and adaptation.evidence.eligible
    state = build_decision_state(adaptation.evidence)
    assert state["geometry"]["transition"] == "unknown"
    assert state["geometry"]["downward_motion_observed"] is None


def test_observed_descent_and_low_pose_map_to_upright_to_low():
    result = live_result()
    result["fall"]["metrics"] = {"had_upright": True}
    result["fall_temporal_evidence"]["geometry"].update(
        quality="observable", descending=True, low_pose=True, metrics={"center_drop_reference_torso": 0.4})

    envelope, trace, adaptation = adapt_result(result)

    assert envelope["perception"]["fall_process"]["geometry"]["transition"] == "upright_to_low"
    assert envelope["perception"]["fall_process"]["geometry"]["downward_motion_observed"] is True
    assert trace["geometry_field_sources"]["descending"] == "fall_temporal_evidence.geometry.descending"
    assert trace["geometry_field_sources"]["low_pose"] == "fall_temporal_evidence.geometry.low_pose"
    assert adaptation.evidence is not None and adaptation.evidence.eligible
    state = build_decision_state(adaptation.evidence)
    assert state["geometry"]["transition"] == "upright_to_low"
    assert state["geometry"]["downward_motion_observed"] is True


def test_explicit_negative_descent_stays_false_and_does_not_infer_upright_from_sitting():
    result = live_result()
    result["action"]["label"] = "sitting"
    result["fall"]["metrics"] = {"had_upright": True, "low_pose": False, "pose_upright_witness": False}
    result["fall_temporal_evidence"]["geometry"].update(
        quality="observable", descending=False, low_pose=False, metrics={})

    envelope, trace, adaptation = adapt_result(result)

    geometry = envelope["perception"]["fall_process"]["geometry"]
    assert geometry["status"] == "ready"
    assert geometry["transition"] == "unknown"
    assert geometry["downward_motion_observed"] is False
    assert trace["geometry_field_sources"]["descending"] == "fall_temporal_evidence.geometry.descending"
    assert trace["geometry_field_sources"]["low_pose"] == "fall_temporal_evidence.geometry.low_pose"
    assert adaptation.evidence is not None and adaptation.evidence.eligible
    state = build_decision_state(adaptation.evidence)
    assert state["geometry"]["downward_motion_observed"] is False


def test_explicit_false_fall_metrics_are_used_when_geometry_layer_omits_fields():
    result = live_result()
    result["fall"]["metrics"] = {"descending": False, "low_pose": False,
                                  "had_upright": True, "pose_upright_witness": False}
    result["fall_temporal_evidence"]["geometry"] = {"quality": "observable", "metrics": {}}

    envelope, trace, adaptation = adapt_result(result)

    geometry = envelope["perception"]["fall_process"]["geometry"]
    assert geometry["transition"] == "unknown"
    assert geometry["downward_motion_observed"] is False
    assert trace["geometry_field_sources"]["descending"] == "fall.metrics.descending"
    assert trace["geometry_field_sources"]["low_pose"] == "fall.metrics.low_pose"
    assert adaptation.evidence is not None and adaptation.evidence.eligible
    state = build_decision_state(adaptation.evidence)
    assert state["geometry"]["downward_motion_observed"] is False


def test_explicit_low_pose_without_observed_descent_is_not_called_transition():
    result = live_result()
    result["fall"]["metrics"] = {"had_upright": True}
    result["fall_temporal_evidence"]["geometry"].update(
        quality="observable", descending=False, low_pose=True, metrics={})

    envelope, trace, adaptation = adapt_result(result)

    geometry = envelope["perception"]["fall_process"]["geometry"]
    assert geometry["transition"] == "low"
    assert geometry["downward_motion_observed"] is False
    assert trace["geometry_field_sources"]["descending"] == "fall_temporal_evidence.geometry.descending"
    assert adaptation.evidence is not None and adaptation.evidence.eligible
