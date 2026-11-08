import copy
import json

import numpy as np
import pytest
from test_action_v2 import person, row

from fallpre.weekly.action_v3 import MotionForest, OnlineMotion, motion_features, observations


def history():
    return [row(i, [person(x=100 + i, y=100 - i)]) for i in range(16)]


def test_features_are_translation_and_scale_invariant():
    h = history()
    changed = copy.deepcopy(h)
    for r in changed:
        p = r["instances"][0]
        p["box"] = (np.array(p["box"]) * 2 + np.array([100, 50, 100, 50])).tolist()
        a = np.array(p["kpts"])
        a[:, :2] = a[:, :2] * 2 + [100, 50]
        p["kpts"] = a.tolist()
    a, info = motion_features(h)
    b, other = motion_features(changed)
    assert np.allclose(a, b, atol=1e-5)
    assert info["upward_torso_units"] == pytest.approx(other["upward_torso_units"])


def test_static_pose_has_no_rising_evidence():
    _, info = motion_features([row(i, [person()]) for i in range(16)])
    assert info["upward_torso_units"] == 0


def test_short_transition_available_without_point_eight_second_warmup():
    features, info = motion_features(history()[:7])
    assert features is not None
    assert info["upward_torso_units"] > 0


def test_missing_current_observation_never_replays_an_action():
    h = history() + [row(16, [])]
    features, info = motion_features(h)
    assert features is None and info["reason"] == "missing_observation"


def test_action_labels_and_clip_identity_are_not_features():
    h = history()
    other = copy.deepcopy(h)
    for r in other:
        r.update(label="standing_up", target="falling", clip_id="hidden_test_subject")
    assert np.array_equal(motion_features(h)[0], motion_features(other)[0])


def test_frozen_fall_branch_has_priority():
    class Frozen:
        def predict(self, h):
            return dict(label="falling", scores={"falling": 0.9})

    model = object.__new__(MotionForest)
    model.frozen = Frozen()
    assert model.predict(history())["branch"] == "frozen_v2_falling"


def test_identity_switch_resets_temporal_history():
    class Model:
        def predict(self, h):
            return dict(label="unknown", support=len(h))

    online = OnlineMotion(Model())
    for r in history():
        online.update(r)
    switched = row(16, [person()])
    switched["target_epoch"] = 2
    assert online.update(switched)["support"] == 1


def test_non_monotone_source_clock_is_rejected():
    class Model:
        def predict(self, h):
            return dict(label="unknown")

    online = OnlineMotion(Model())
    online.update(row(1, [person()]))
    with pytest.raises(ValueError):
        online.update(row(1, [person()]))


def test_pose_cache_checks_original_source_clock(tmp_path):
    from fallpre.weekly.human import POSE_SHA256

    e = dict(clip_id="test", video_sha256="abc", frame_count=1, fps=15.0, source_times_s=[0.2])
    p = tmp_path / "test"
    p.mkdir()
    meta = dict(source_sha256="abc", weights_sha256=POSE_SHA256)
    (p / "observations.jsonl").write_text(json.dumps(meta) + "\n" + json.dumps(row(0, [person()])) + "\n")
    with pytest.raises(ValueError, match="source-clock"):
        observations(e, tmp_path)


def test_subject_cannot_cross_development_and_independent_test(tmp_path):
    from fallpre.weekly.action_v3 import load_manifest

    entries = [
        dict(
            clip_id=f"clip-{i}",
            source_group_id=f"group-{i}",
            video_path=f"missing-{i}.mp4",
            split=split,
            subject_id="same-person",
            segments=[],
        )
        for i, split in enumerate(["train", "test"])
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": entries}))
    with pytest.raises(ValueError, match="subject_id crosses"):
        load_manifest(path)


def test_historical_test_cannot_be_rebranded_as_independent(tmp_path):
    from fallpre.weekly.action_v3 import load_manifest

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    dict(
                        clip_id="old-test",
                        video_path="missing.mp4",
                        split="test",
                        legacy_v2=True,
                        segments=[],
                    )
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Historical source"):
        load_manifest(path)
