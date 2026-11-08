"""Causal support, abstention accounting, and leakage checks for five-action baseline.

Synthetic skeletons here test software contracts, never model performance.
"""
import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from fallpre.weekly.five_actions import (
    ALL_LABELS,
    FEATURE_NAMES,
    LABELS,
    FeatureConfig,
    FiveActionClassifier,
    OnlineFiveAction,
    classification_metrics,
    evaluate,
    extract_features,
    fit_model,
    label_at,
    load_manifest,
)
from fallpre.weekly.human import POSE_SHA256


def observation(index, *, dy=0., dx=0., confidence=.95):
    points = [[100., 30., confidence] for _ in range(17)]
    for indices, coords in (((5, 6), ((80, 50), (120, 50))),
                            ((11, 12), ((80, 100), (120, 100))),
                            ((13, 14), ((80, 150), (120, 150))),
                            ((15, 16), ((80, 200), (120, 200)))):
        for idx, (x, y) in zip(indices, coords):
            points[idx] = [x + dx, y + dy, confidence]
    return {"t": index / 10, "source_frame_id": index + 1,
            "instances": [{"box": [50.+dx, 20.+dy, 150.+dx, 220.+dy], "det": .9, "kpts": points}]}


def constant_model(config=None):
    n = len(FEATURE_NAMES)
    return FiveActionClassifier({"schema": "fallpre.five_actions.model.v1", "labels": LABELS,
                                "feature_names": FEATURE_NAMES, "config": asdict(config or FeatureConfig()),
                                "mean": [0.] * n, "scale": [1.] * n,
                                "weights": np.zeros((n, len(LABELS))).tolist(),
                                "bias": [0., 0., 0., 0., 0., 8.]})


def test_direction_survives_normalization_and_translation_does_not():
    forward = [observation(i, dy=i*2) for i in range(11)]
    backward = [observation(i, dy=-i*2) for i in range(11)]
    shifted = [observation(i, dy=500+i*2, dx=300) for i in range(11)]
    a, b, c = [extract_features(rows) for rows in (forward, backward, shifted)]
    assert a["quality"] == "available"
    index = FEATURE_NAMES.index("hip_y_slope")
    assert a["features"][index] > 0 > b["features"][index]
    np.testing.assert_allclose(a["features"], c["features"], atol=1e-12)
    assert a["reliable_body_side_fraction"] == 1


def test_warmup_missing_pose_and_identity_reset_are_unknown():
    online = OnlineFiveAction(constant_model())
    for i in range(8):
        assert online.update(observation(i))["label"] == "unknown"
    assert online.update(observation(8))["label"] == "background"
    missing = observation(9)
    missing["instances"] = []
    result = online.update(missing)
    assert result["label"] == "unknown"
    assert result["reason"] == "missing_person"
    assert not online.history
    assert online.update(observation(10))["support_frames"] == 1
    online.reset("identity_change")
    assert online.update(observation(11))["support_frames"] == 1


@pytest.mark.parametrize("change", ["gap", "backwards", "jump", "multiple"])
def test_discontinuous_source_cannot_reuse_past_person(change):
    online = OnlineFiveAction(constant_model())
    for i in range(11):
        online.update(observation(i))
    row = observation(11)
    if change == "gap":
        row["source_frame_id"] += 1
    elif change == "backwards":
        row["t"] = .1
    elif change == "jump":
        row = observation(11, dx=500)
    else:
        row["instances"] *= 2
    assert online.update(row)["label"] == "unknown"
    assert len(online.history) <= 1


def test_low_confidence_does_not_become_background():
    rows = [observation(i, confidence=.1) for i in range(11)]
    result = constant_model().predict(rows)
    assert result["label"] == "unknown"
    assert result["reason"] == "insufficient_joint_confidence"
    assert not result["model_inference_executed"]


def test_low_softmax_confidence_abstains_and_feature_input_is_not_modified():
    model = constant_model()
    model.bias[:] = 0
    rows = [observation(i) for i in range(11)]
    original = copy.deepcopy(rows)
    result = model.predict(rows)
    assert result["label"] == "unknown"
    assert result["reason"] == "low_classifier_confidence"
    assert rows == original


def test_metric_unknown_counts_as_false_negative_and_unlabeled_is_excluded():
    records = [{"target": "falling", "label": "falling", "clip_id": "heldout-1"},
               {"target": "falling", "label": "unknown", "clip_id": "heldout-1"},
               {"target": "background", "label": "falling", "clip_id": "heldout-2"},
               {"target": "unknown", "label": "falling", "clip_id": "heldout-3"}]
    metrics = classification_metrics(records)
    falling = metrics["per_class"]["falling"]
    assert falling["precision"] == .5
    assert falling["recall"] == .5
    assert falling["f1"] == .5
    assert falling["unknown_predictions"] == 1
    assert falling["support_clips"] == ["heldout-1"]
    assert metrics["accuracy_including_unknown_as_error"] == pytest.approx(1/3)
    assert metrics["classified_coverage"] == pytest.approx(2/3)
    assert metrics["selective_accuracy"] == .5
    assert metrics["five_action_macro_f1"] is None
    assert np.asarray(metrics["confusion_matrix_rows_true_columns_predicted"]).shape == (len(ALL_LABELS),) * 2


def test_annotations_are_half_open_and_gaps_are_unknown():
    entry = {"segments": [{"start_s": 1., "end_s": 2., "label": "sitting_down"},
                           {"start_s": 2., "end_s": 3., "label": "background"}]}
    assert label_at(entry, .9) == "unknown"
    assert label_at(entry, 1.) == "sitting_down"
    assert label_at(entry, 2.) == "background"
    assert label_at(entry, 3.) == "unknown"


def test_optimizer_really_learns_and_model_roundtrips(tmp_path):
    rng = np.random.default_rng(71)
    x, y = [], []
    for i, label in enumerate(LABELS):
        center = np.zeros(len(FEATURE_NAMES))
        center[i] = 4
        for _ in range(5):
            x.append(center + rng.normal(0, .05, len(FEATURE_NAMES)))
            y.append(label)
    x = np.asarray(x)
    fitted = fit_model(x, y)
    assert fitted.artifact["training"]["converged"]
    assert fitted.artifact["training"]["iterations"] > 0
    np.testing.assert_allclose(fitted.mean, x.mean(axis=0))
    logits = ((x - fitted.mean) / fitted.scale) @ fitted.weights + fitted.bias
    assert [LABELS[i] for i in logits.argmax(axis=1)] == y
    path = tmp_path / "model.json"
    path.write_text(json.dumps(fitted.artifact))
    loaded = FiveActionClassifier.load(path)
    np.testing.assert_allclose(loaded.weights, fitted.weights)
    with pytest.raises(ValueError, match="all six classes"):
        fit_model(x[:-5], y[:-5])


def test_duplicate_video_under_renamed_clip_cannot_cross_splits(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"same public video fixture")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": [
        {"clip_id": "a", "split": "train", "video_path": str(video), "segments": []},
        {"clip_id": "b", "split": "test", "video_path": str(video), "segments": []}]}))
    with pytest.raises(ValueError, match="Duplicate source video"):
        load_manifest(path)


def test_reencoded_same_source_group_is_not_an_independent_test_clip(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for i, path in enumerate(videos):
        path.write_bytes(f"different encode {i}".encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"clip_id": "crop-a", "source_group_id": "same-source", "split": "train",
         "video_path": str(videos[0]), "segments": []},
        {"clip_id": "crop-b", "source_group_id": "same-source", "split": "test",
         "video_path": str(videos[1]), "segments": []}]}))
    with pytest.raises(ValueError, match="underlying source group"):
        load_manifest(manifest)


def test_declared_sampling_stride_preserves_true_source_ids():
    stream = OnlineFiveAction(constant_model(), expected_frame_stride=3)
    for i in range(11):
        row = observation(i)
        row["source_frame_id"] = i * 3 + 1
        result = stream.update(row)
    assert result["label"] == "background"
    assert result["source_frame_id"] == 31
    assert result["support_frames"] == 11


def test_evaluation_refuses_changed_frozen_manifest_before_reading_test_data(tmp_path):
    model = constant_model()
    model.artifact["training"] = {"provenance": {"manifest_sha256": "frozen", "train_clips": []}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.artifact))
    with pytest.raises(ValueError, match="Manifest changed after fitting"):
        evaluate({"manifest_sha256": "changed", "entries": []}, tmp_path, path, tmp_path / "evaluation")


def test_evaluation_preserves_online_missing_and_ambiguous_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr("fallpre.weekly.five_actions._plot_evaluation", lambda *args: None)
    model = constant_model()
    model.artifact["training"] = {"provenance": {"manifest_sha256": "frozen", "train_clips": []}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.artifact))
    rows = [dict(observation(i), clip_id="test-clip") for i in range(24)]
    rows[11]["instances"] = []
    rows[12]["instances"] *= 2
    entry = {"clip_id": "test-clip", "split": "test", "video_sha256": "video-source",
             "segments": [{"start_s": 0., "end_s": 2.4, "label": "background"}]}
    cached = tmp_path / "poses" / "test-clip" / "observations.jsonl"
    cached.parent.mkdir(parents=True)
    header = {"kind": "meta", "clip_id": "test-clip", "source_sha256": "video-source",
              "weights_sha256": POSE_SHA256, "frame_count": len(rows), "fps": 10.}
    cached.write_text("\n".join(json.dumps(row) for row in [header, *rows]) + "\n")
    result = evaluate({"manifest_sha256": "frozen", "entries": [entry]}, tmp_path / "poses",
                      path, tmp_path / "evaluation")
    assert result["unknown_reasons"] == {"warming_up": 16, "missing_person": 1, "ambiguous_target_person": 1}
    predictions = [json.loads(line) for line in (tmp_path / "evaluation" / "predictions.jsonl").read_text().splitlines()]
    assert predictions[11]["label"] == predictions[12]["label"] == "unknown"
    assert predictions[11]["reason"] == "missing_person"
    assert predictions[12]["reason"] == "ambiguous_target_person"
    assert result["metrics"]["per_class"]["background"]["unknown_predictions"] == 18


def test_future_row_does_not_change_previous_prediction():
    online = OnlineFiveAction(constant_model())
    for i in range(11):
        old = online.update(observation(i))
    old_copy = copy.deepcopy(old)
    online.update(observation(11, dy=70))
    assert old == old_copy
    assert old["evidence_window_s"] == [0., 1.]
