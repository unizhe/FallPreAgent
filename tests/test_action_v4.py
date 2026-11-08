import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_action_v2 import person, row

from fallpre.weekly.action_v4 import OnlinePhase, phase_features


def stream(label="background"):
    class Base:
        def predict(self, h):
            return dict(label=label, scores={label: 1.0})

    model = SimpleNamespace(
        base=Base(),
        config=dict(
            seat_probability=0.45,
            min_up=0.1,
            min_opening=0.04,
            confirm_s=0.1,
            stable_speed=0.15,
            stable_s=0.2,
        ),
        scores=lambda x: {"seated": 1.0, "rising": 0.0, "standing": 0.0, "forward_lean": 0.0},
    )
    return OnlinePhase(model)


def test_static_seated_never_outputs_rising():
    s = stream()
    for i in range(100):
        r = row(i, [person()])
        assert s.update(r)["label"] != "standing_up"


def test_phase_history_resets_on_identity_change():
    s = stream()
    for i in range(20):
        s.update(row(i, [person()]))
    s.anchor = {"hip_y": 300, "t": 0.5}
    s.phase = "rising"
    r = row(21, [person()])
    r["target_epoch"] = 2
    assert s.update(r)["phase"] == "unknown"
    assert s.anchor is None


def test_missing_observation_is_not_phase_evidence():
    s = stream()
    for i in range(10):
        s.update(row(i, [person()]))
    assert s.update(row(11, []))["phase"] == "unknown"


@pytest.mark.parametrize("label", ["walking", "bending", "falling"])
def test_protected_baseline_decisions_are_preserved(label):
    s = stream(label)
    for i in range(20):
        assert s.update(row(i, [person()]))["label"] == label


def test_non_monotone_clock_rejected():
    s = stream()
    s.update(row(1, [person()]))
    with pytest.raises(ValueError):
        s.update(row(1, [person()]))


def test_features_ignore_reference_labels_and_identity():
    h = [row(i, [person(x=100 + i, y=100 - i)]) for i in range(10)]
    other = copy.deepcopy(h)
    for r in other:
        r.update(target="standing_up", reference_phase="rising", clip_id="secret", subject_id="person10")
    assert np.array_equal(phase_features(h)[0], phase_features(other)[0])


def test_phase_features_translation_scale_invariant():
    h = [row(i, [person(x=100 + i, y=100 - i)]) for i in range(10)]
    other = copy.deepcopy(h)
    for r in other:
        p = r["instances"][0]
        p["box"] = (np.array(p["box"]) * 2 + [100, 50, 100, 50]).tolist()
        a = np.array(p["kpts"])
        a[:, :2] = a[:, :2] * 2 + [100, 50]
        p["kpts"] = a.tolist()
    assert np.allclose(phase_features(h)[0], phase_features(other)[0], atol=1e-5)


def test_slow_cumulative_rise_requires_seat_and_finishes(monkeypatch):
    import fallpre.weekly.action_v4 as v4

    def feature(h):
        r = h[-1]
        info = dict(
            t=r["t"],
            hip_y=r.get("hip", 100),
            shoulder_y=50,
            scale=100,
            knee=r.get("knee", 0.6),
            hip_angle=r.get("hip_angle", 0.7),
            tilt=0,
            reliable=True,
            hip_speed=r.get("speed", 0),
        )
        return np.zeros(1), info

    monkeypatch.setattr(v4, "phase_features", feature)
    s = stream()
    for i in range(10):
        s.update(row(i, [person()]))
    assert s.phase == "seated"
    for i in range(10, 18):
        r = row(i, [person()])
        r.update(hip_angle=0.6)
        assert s.update(r)["label"] != "standing_up"
    for i in range(18, 70):
        r = row(i, [person()])
        delta = (i - 18) * 0.4
        r.update(hip=100 - delta, knee=0.6 + delta / 70, speed=0.06)
        result = s.update(r)
    assert result["label"] == "standing_up"
    s.model.scores = lambda x: {"seated": 0.0, "rising": 0.0, "standing": 1.0, "forward_lean": 0.0}
    for i in range(70, 90):
        r = row(i, [person()])
        r.update(hip=65, knee=0.97, speed=0)
        result = s.update(r)
    assert result["phase"] == "standing"
    assert result["label"] != "standing_up"


def test_long_gap_clears_armed_seat(monkeypatch):
    s = stream()
    s.update(row(0, [person()]))
    s.anchor = dict(hip_y=100, t=0)
    s.phase = "seated"
    assert s.update(row(100, [person()]))["phase"] == "unknown"
    assert s.anchor is None


def test_classifier_rising_without_prior_seat_cannot_start_event():
    s = stream()
    s.model.scores = lambda x: {"rising": 1.0, "seated": 0.0, "standing": 0.0}
    for i in range(30):
        result = s.update(row(i, [person()]))
        assert result["label"] != "standing_up"
        assert result["phase"] != "rising"


def test_manifest_rejects_old_v3_as_new_test(tmp_path):
    import json

    from fallpre.weekly.action_v4 import load_manifest

    p = tmp_path / "m.json"
    p.write_text(
        json.dumps(
            {
                "entries": [
                    dict(clip_id="old", split="test", legacy_v3=True, video_path="missing.mp4", segments=[])
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Historical V3"):
        load_manifest(p)
