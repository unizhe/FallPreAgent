"""Checks for audit clock handling and descriptive summaries; no model downloads."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/compare_live_models.py"
spec = importlib.util.spec_from_file_location("compare_live_models", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_sampling_preserves_sparse_source_frames_without_repeating_them():
    times = [0., .033, .067, .15, .30, .333, .90]
    indices = audit.sample_indices(times, 10)
    assert indices == [0, 2, 3, 4, 5, 6]
    assert len(indices) == len(set(indices))
    assert [times[i] for i in indices] == [0., .067, .15, .30, .333, .90]


def test_manifest_source_times_override_nominal_container_fps():
    entry = dict(clip_id="sparse", frame_count=3, fps=30, source_times_s=[0, .3, .7])
    assert audit.source_times(entry).tolist() == [0, .3, .7]
    with pytest.raises(ValueError, match="clock"):
        audit.source_times(dict(entry, source_times_s=[0, .3, .3]))


def test_jpeg_profile_keeps_aspect_and_caps_dimensions_without_upscaling():
    for shape, expected in [((1080, 1920, 3), (540, 960)), ((1200, 800, 3), (720, 480)),
                            ((240, 320, 3), (240, 320))]:
        image = np.zeros(shape, np.uint8)
        body = audit.browser_jpeg(image)
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == expected


def result(state, target, epoch=1):
    return dict(persons=[dict(box=[1, 2, 3, 4])],
                action=dict(label="unknown", raw_label="unavailable", phase="unknown",
                            target_index=target, selection=dict(target_epoch=epoch), reset_reasons=[]),
                fall=dict(state=state, label=state, reason="test"),
                quality=dict(camera_stable=None, code="insufficient_background"),
                inference_ms=12.0, audit_pipeline_ms=16.0)


def test_summary_does_not_confuse_any_box_with_available_target():
    rows = [dict(source_time_s=t, source_frame_id=i+1,
                 models={audit.MODELS[0]:result(s, target, epoch)})
            for i, (t, s, target, epoch) in enumerate([
                (0., "unobservable", None, 0), (.3, "suspected", 0, 1),
                (.6, "unobservable", None, 1), (.9, "normal", 0, 2)])]
    summary = audit.summarize(rows, audit.MODELS[0])
    assert summary["any_pose_frames"] == 4
    assert summary["target_available_frames"] == 2
    assert summary["target_epoch_changes"] == 2
    assert summary["first_candidate_s"] == .3
    assert summary["first_confirmation_s"] is None
    assert summary["camera_stable"] == {"None":4}


def test_observed_confirmation_time_is_taken_from_output_not_label_annotations():
    rows = [dict(source_time_s=1.4, source_frame_id=9,
                 models={audit.MODELS[0]:result("confirmed", 0)})]
    summary = audit.summarize(rows, audit.MODELS[0])
    assert summary["first_confirmation_s"] == 1.4
    assert summary["first_candidate_s"] is None


def test_action_disagreements_are_not_scored_as_errors():
    left, right = result("normal", 0), result("normal", 0)
    left["action"]["label"] = "standing"
    right["action"]["label"] = "unknown"
    row = dict(source_time_s=0., source_frame_id=1, models=dict(zip(audit.MODELS, [left, right])))
    stats = audit.comparison_stats([row])
    assert stats["action_pairs"] == {"standing → unknown":1}
    assert stats["differing_frames"] == 1
    assert "accuracy" not in stats


def test_independent_fall_person_is_not_raw_action_target():
    value = result("suspected", None)
    recovered = dict(box=[10, 20, 30, 40])
    value["fall_observation"] = dict(person=recovered, source="recovered",
        selection=dict(raw_index=None, target_epoch=2, state="tracked"),
        recovery=dict(executed=True, accepted=True, inference_passes=3))
    row = dict(source_time_s=1., source_frame_id=1, models={audit.MODELS[0]:value})
    stats = audit.summarize([row], audit.MODELS[0])
    assert stats["action_target_available_frames"] == 0
    assert stats["fall_target_available_frames"] == 1
    assert stats["recovery_accepted_frames"] == 1
    assert stats["recovery_inference_passes"] == 3
    assert audit.fall_target_info(value)["person"] == recovered


def test_baseline_requires_exact_inputs_raw_poses_and_action_but_allows_fall_changes():
    import copy
    prior = dict(source_time_s=0., source_frame_id=1, input_jpeg_sha256="abc",
                 models={model:result("normal", 0) for model in audit.MODELS})
    current = copy.deepcopy(prior)
    current["models"][audit.MODELS[0]]["fall"]["state"] = "suspected"
    assert audit.baseline_comparison([current], [prior])["exact_input_raw_action_match"]
    current["models"][audit.MODELS[0]]["action"]["label"] = "standing"
    comparison = audit.baseline_comparison([current], [prior])
    assert not comparison["exact_input_raw_action_match"]
    assert comparison["mismatches"] == [dict(frame=1, model=audit.MODELS[0], field="action")]
    current["source_time_s"] = .1
    with pytest.raises(ValueError, match="frame/time"):
        audit.baseline_comparison([current], [prior])


def test_rgb_box_without_keypoints_renders_and_keeps_pose_fall_separate():
    value = result("confirmed", None)
    value.update(fall_mode="rgb", rgb_inference_ms=40.0,
                 rgb_detections=[dict(box=[10, 20, 70, 80], label="fallen")],
                 pose_fall=dict(state="unobservable", label="UNKNOWN", reason="missing torso"),
                 pose_fall_observation=dict(person=None, source="none", recovery={}),
                 fall_observation=dict(person=dict(box=[10, 20, 70, 80], keypoints=[]),
                                       source="rgb", selection=dict(target_epoch=3)))
    # The current raw detections and the independent fall box may both lack joints.
    value["persons"][0]["keypoints"] = []
    panel = audit.render_panel(np.zeros((100, 100, 3), np.uint8), value, "model", 1.4)
    assert panel.shape == (650, 640, 3)
    row = dict(source_time_s=1.4, source_frame_id=9, models={audit.MODELS[0]:value})
    stats = audit.summarize([row], audit.MODELS[0])
    assert stats["first_confirmation_s"] == 1.4
    assert stats["fall_target_sources"] == {"rgb":1}
    assert stats["pose_fall"]["states"] == {"unobservable":1}
    assert stats["pose_fall"]["first_confirmation_s"] is None
    assert stats["timing"]["rgb_inference_ms"]["median_ms"] == 40.0
    assert stats["rgb"]["detection_labels"] == {"fallen":1}


def test_model_watch_includes_nested_weights_and_ignores_remote_urls(tmp_path):
    weights = tmp_path/"models"
    weights.mkdir()
    first, second = weights/"fallen.pt", weights/"person.onnx"
    first.write_bytes(b"test")
    second.write_bytes(b"test")
    config = dict(path="models/fallen.pt", person=dict(path="models/person.onnx"),
                  url="https://example.org/fallen.pt")
    assert audit.referenced_weights(config, tmp_path) == [first, second]
    with pytest.raises(FileNotFoundError, match="missing"):
        audit.referenced_weights(dict(path="models/absent.pt"), tmp_path)
    with pytest.raises(ValueError, match="within"):
        audit.referenced_weights(dict(path="../outside.pt"), tmp_path)
