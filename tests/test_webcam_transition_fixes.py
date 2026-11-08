"""Transitions require measured geometry; conflicting fall votes stay protected."""

import copy

import numpy as np
import pytest
from test_webcam_behavior import person
from test_webcam_daily_activity import straight_pose
from test_webcam_static_recovery import sample

from fallpre.weekly.webcam_behavior import BehaviorAxes, apply_auxiliary_bend
from fallpre.weekly.webcam_daily_activity import DailyActivity
from fallpre.weekly.webcam_reconcile import StaticRecovery


def test_auxiliary_bend_handoff_transfers_only_current_posture():
    observer = BehaviorAxes()
    behavior = observer.update(None, 1.0, None, {"label": "bending"}, 960, 540)
    action = dict(
        label="bending",
        raw_label="background",
        target_index=None,
        auxiliary_posture=dict(label="bending", target_index=0, target_epoch=4, source_time=1.0),
    )
    original = copy.deepcopy(action)
    out = apply_auxiliary_bend(behavior, action, [person()], 1.0, 960, 540)
    assert out["posture"]["label"] == "bending"
    assert out["target_index"] == 0 and out["target_epoch"] == "bend-4"
    assert all(out[k]["label"] == "unknown" for k in ["locomotion", "turning", "path_change"])
    assert not observer.history and action == original
    # Main track comes back without inheriting any auxiliary movement history.
    resumed = observer.update(person(), 1.13, 7, {"label": "standing"}, 960, 540)
    assert resumed["locomotion"]["label"] != "walking"


@pytest.mark.parametrize(
    "condition", ["stale", "epoch", "wrong_target", "out_of_range", "falling", "camera", "missing"]
)
def test_auxiliary_posture_cannot_cross_invalid_evidence(condition):
    b = BehaviorAxes().update(None, 1.0, None, {"label": "unknown"}, 960, 540)
    a = dict(
        label="bending",
        raw_label="background",
        target_index=None,
        auxiliary_posture=dict(label="bending", target_index=0, target_epoch=1, source_time=1.0),
    )
    pp = [person()]
    if condition == "stale":
        a["auxiliary_posture"]["source_time"] = 0.87
    if condition == "epoch":
        a["auxiliary_posture"]["target_epoch"] = None
    if condition == "wrong_target":
        a["target_index"] = 1
    if condition == "out_of_range":
        a["auxiliary_posture"]["target_index"] = 2
    if condition == "falling":
        a["raw_label"] = "falling"
    if condition == "missing":
        pp = []
    before = copy.deepcopy(b)
    assert apply_auxiliary_bend(b, a, pp, 1.0, 960, 540, condition != "camera") == before


def confirmed_bend():
    d = DailyActivity()
    p = straight_pose(100, 0)
    for t in [0.0, 0.13, 0.26]:
        out = d.update(t, 1, p, dict(label="unknown", raw_label="background"))
    assert out["label"] == "bending"
    return d


@pytest.mark.parametrize(
    "condition", ["valid", "knee_flexion", "foot_motion", "falling", "gap", "identity", "missing"]
)
def test_bend_recovery_veto_needs_current_straightening_and_same_track(condition):
    d = confirmed_bend()
    p = straight_pose()
    if condition == "knee_flexion":
        for j in [13, 14]:
            p["kpts"][j][0] += 60
    if condition == "foot_motion":
        for j in [15, 16]:
            p["kpts"][j][0] += 20
    if condition == "missing":
        p = None
    out = d.update(
        0.8 if condition == "gap" else 0.39,
        2 if condition == "identity" else 1,
        p,
        dict(label="sitting_down", raw_label="falling" if condition == "falling" else "sitting_down"),
    )
    if condition == "valid":
        assert out["label"] == "unknown" and out["decision_code"] == "bend_recovery"
    else:
        assert out.get("decision_code") != "bend_recovery"


def descent_reference():
    r = StaticRecovery()
    _, f, pp = sample()
    origin = np.asarray(pp["kpts"])
    points = origin.copy()
    points[[5, 6, 11, 12], 1] += 25
    values = dict(f, knee=0.65, hip_angle=0.65)
    r.descent = dict(t=0.0, points=points.copy(), scale=100.0, origin=dict(points=origin, scale=100.0))
    return r, values, points


@pytest.mark.parametrize("condition", ["valid", "feet_move", "floor", "stale", "low_confidence"])
def test_short_descent_conflict_keeps_evidence_only_with_planted_seat_height(condition):
    r, f, points = descent_reference()
    if condition == "feet_move":
        points[[15, 16], 0] += 15
    if condition == "floor":
        points[[11, 12], 1] = points[[15, 16], 1] - 25
    if condition == "low_confidence":
        points[:, 2] = 0.2
    assert r.update(0.5 if condition == "stale" else 0.13, {"label": "falling"}, f, points) is None
    assert not r.details["stable"] and r.posture is None
    assert (r.descent is not None) == (condition == "valid")
    if condition == "valid":
        # A sustained conflict expires. It never becomes a seated output.
        assert r.update(0.52, {"label": "falling"}, f, points) is None
        assert r.descent is None


def test_adjacent_descent_observations_cannot_accumulate_unbounded_foot_drift():
    r, f, points = descent_reference()
    for t, shift in [(0.1, 4), (0.2, 8)]:
        current = points.copy()
        current[[15, 16], 0] += shift
        assert r.update(t, {"label": "falling"}, f, current) is None
        assert r.descent is not None
        assert r.descent["t"] == 0.0
    current = points.copy()
    current[[15, 16], 0] += 12
    assert r.update(0.3, {"label": "falling"}, f, current) is None
    assert r.descent is None
