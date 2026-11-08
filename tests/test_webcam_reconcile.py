"""Behavior checks for action/phase reconciliation, independent of model weights."""
import copy
import json

import pytest

from fallpre.weekly.action_v4 import phase_features
from fallpre.weekly.webcam_reconcile import ReconcileActions


def observation(step=0, phase="descending", label="bending", mode="articulated"):
    points = [[100., 80., .99] for _ in range(17)]
    for shoulder, hip, knee, ankle, x in [(5, 11, 13, 15, 90.), (6, 12, 14, 16, 110.)]:
        points[shoulder] = [x, 100., .99]
        points[hip] = [x, 200., .99]
        points[knee] = [x + 15, 270., .99]
        points[ankle] = [x, 350., .99]
    direction = -1 if phase == "rising" else 1
    dy = direction * step * 2
    for i in (5, 6, 11, 12):
        points[i][1] += dy
    if mode == "translation":
        for i in range(17):
            if i not in (5, 6, 11, 12):
                points[i][1] += dy
    elif mode == "bend_only":
        for i in (11, 12):
            points[i][1] -= dy
    elif mode == "horizontal":
        for point in points:
            point[0] += step * 4
    elif mode == "arm_wave":
        points[9][0] += step * 5
    static = phase in {"standing", "seated"}
    knee_value = (.98 if phase == "standing" else .65) if static else .8 - direction * step * .015
    hip_value = .98 if phase == "standing" else .65
    features = dict(reliable=True, knee=knee_value, hip_angle=hip_value, tilt=.05, scale=100.,
                    hip_speed=0. if static else -direction * .2,
                    hip_y=points[11][1], shoulder_y=points[5][1])
    return dict(label=label, phase=phase, phase_scores={phase: .8}), features, dict(kpts=points)


def run(phase="descending", label="bending", mode="articulated", steps=None, stream=None):
    stream = stream or ReconcileActions()
    steps = list(range(6)) if steps is None else steps
    output = None
    for i, step in enumerate(steps):
        p, f, person = observation(step, phase, label, mode)
        output = stream.update(i * .05, p, f, person)
    return output


@pytest.mark.parametrize("label", ["bending", "background", "unknown"])
def test_descent_requires_observed_motion_and_flexion(label):
    out = run(label=label)
    assert out["changed"] and out["prediction"]["label"] == "sitting_down"
    assert out["code"] == "descent_confirmed"
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("mode,steps", [
    ("articulated", [0] * 7),              # phase alone, no actual motion
    ("bend_only", list(range(7))),        # shoulders move, hips do not
    ("translation", list(range(7))),      # whole image moves, limbs do not
    ("articulated", [0, 0, 6, 0, 0, 0]), # single observation outlier
    ("articulated", [0, 6, 6, 6, 6, 6]), # one jump followed by a still pose
])
def test_false_motion_does_not_create_sitting_down(mode, steps):
    assert not run(mode=mode, steps=steps)["changed"]


def test_body_descent_without_knee_flexion_is_rejected():
    stream = ReconcileActions()
    for i in range(7):
        p, f, person = observation(i)
        f["knee"] = .8
        out = stream.update(i * .05, p, f, person)
    assert not out["changed"] and out["code"] == "descent_without_flexion"


@pytest.mark.parametrize("label", ["walking", "falling"])
@pytest.mark.parametrize("phase", ["descending", "rising", "seated", "standing"])
def test_walking_and_falling_preserved_without_enough_static_evidence(label, phase):
    out = run(phase=phase, label=label, steps=[0] * 7)
    assert not out["changed"] and out["prediction"]["label"] == label
    assert out["code"] == "protected_action"


@pytest.mark.parametrize("phase", ["standing", "seated"])
@pytest.mark.parametrize("label", ["standing_up", "sitting_down", "bending", "unknown"])
def test_stable_geometry_and_phase_can_release_stale_action(phase, label):
    out = run(phase=phase, label=label, steps=[0] * 8)
    assert out["changed"] and out["prediction"]["label"] == "background"
    assert out["code"] == "static_confirmed"


@pytest.mark.parametrize("mode", ["horizontal", "arm_wave"])
def test_static_phase_cannot_hide_body_or_limb_motion(mode):
    stream = ReconcileActions()
    for i in range(8):
        p, f, person = observation(0, "standing", "standing_up")
        if mode == "horizontal":
            for point in person["kpts"]:
                point[0] += i * 3
        else:
            person["kpts"][9][0] += i * 4
        out = stream.update(i * .05, p, f, person)
    if mode == 'arm_wave':
        assert out['prediction']['label'] == 'background'
        assert out['prediction']['phase'] == 'standing'
    else:
        assert out["changed"] and out["prediction"]["label"] == "unknown"
        assert out["code"] == "static_motion_conflict"


@pytest.mark.parametrize("bad", ["score", "geometry", "instantaneous_speed"])
def test_static_confirmation_keeps_existing_gates(bad):
    stream = ReconcileActions()
    for i in range(8):
        p, f, person = observation(0, "standing", "sitting_down")
        if bad == "score":
            p["phase_scores"]["standing"] = .49
        elif bad == "geometry":
            f["knee"] = .89
        else:
            f["hip_speed"] = .11
            for j in (5,6,11,12):person['kpts'][j][1] += i*5
        out = stream.update(i * .05, p, f, person)
    assert out["changed"] and out["prediction"]["label"] == ('background' if bad=='score' else 'unknown')


def test_rising_phase_requires_extension_and_motion():
    out = run(phase="rising", label="background")
    assert out["changed"] and out["prediction"]["label"] == "standing_up"
    assert not run(phase="rising", label="background", steps=[0] * 7)["changed"]
    assert not run(phase="rising", label="background", mode="translation")["changed"]


@pytest.mark.parametrize("phase", ["descending", "rising"])
def test_current_reversal_overrides_previous_direction(phase):
    out = run(phase=phase, label="background", steps=[0, 1, 2, 3, 4, 5, 3])
    assert not out["changed"] and out["code"] == "transition_motion_conflict"


def test_phase_change_restarts_confirmation_window():
    stream = ReconcileActions()
    assert run(stream=stream)["changed"]
    p, f, person = observation(0, "seated", "sitting_down")
    out = stream.update(.30, p, f, person)
    assert out["changed"] and out["prediction"]["label"] == "unknown"
    assert out["code"] == "warming_up"


def test_no_rise_is_created_from_unconfirmed_phase():
    stream = ReconcileActions()
    for i in range(8):
        p, f, person = observation(i, "rising", "background")
        p["phase"] = "unknown"
        out = stream.update(i * .05, p, f, person)
    assert not out["changed"] and out["code"] == "unsupported_phase"


def test_missing_joints_cannot_be_replaced_by_another_visible_side():
    stream = ReconcileActions()
    for i in range(8):
        p, f, person = observation(i)
        hidden = (5, 11, 13, 15) if i % 2 else (6, 12, 14, 16)
        for joint in hidden:
            person["kpts"][joint][2] = 0
        out = stream.update(i * .05, p, f, person)
    assert not out["changed"] and out["code"] == "insufficient_common_joints"


def test_input_is_not_mutated_and_state_does_not_cross_missing_or_time_gap():
    stream = ReconcileActions()
    assert run(stream=stream)["changed"]
    p, f, person = observation(7)
    original = copy.deepcopy((p, f, person))
    out = stream.update(.35, p, f, person)
    assert (p, f, person) == original
    assert out["prediction"] is not p
    stream.update(.40, p, None, None)
    assert not stream.update(.45, p, f, person)["changed"]
    assert not stream.update(2., p, f, person)["changed"]
    assert not ReconcileActions().update(.35, p, f, person)["changed"]


def test_nonfinite_input_cannot_leak_nan_into_evidence():
    stream = ReconcileActions()
    p, f, person = observation()
    f["scale"] = float("nan")
    out = stream.update(0., p, f, person)
    assert not out["changed"]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("label,phase", [("sitting_down", "descending"), ("standing_up", "rising")])
def test_existing_same_phase_transition_has_no_new_warmup_gate(label, phase):
    stream = ReconcileActions()
    for i in range(6):
        p, f, person = observation(i, phase, label)
        original = copy.deepcopy(p)
        out = stream.update(i * .05, p, f, person)
        assert p == original
        if i < 3:
            assert not out["changed"] and out["prediction"]["label"] == label
            assert out["code"] == "warming_up"
            assert "transition_blocked" not in out["evidence"]
    assert out["prediction"]["label"] == label and not out["changed"]


@pytest.mark.parametrize("label,phase", [("sitting_down", "rising"), ("standing_up", "descending")])
def test_existing_transition_cannot_survive_opposite_confirmed_motion(label, phase):
    out = run(label=label, phase=phase)
    assert out["changed"] and out["prediction"]["label"] == "unknown"
    assert out["code"] == "opposed_transition"


def straight_pose(dy=0., translate_feet=True):
    """Coordinates, not supplied knee/speed values, drive the regression below."""
    points = [[100., 80. + dy, .99] for _ in range(17)]
    for shoulder, hip, knee, ankle, x in [(5, 11, 13, 15, 90.), (6, 12, 14, 16, 110.)]:
        points[shoulder] = [x, 100. + dy, .99]
        points[hip] = [x, 200. + dy, .99]
        points[knee] = [x, 270. + dy, .99]
        points[ankle] = [x, 350. + (dy if translate_feet else 0), .99]
    return dict(kpts=points, box=[70., 60. + dy, 150., 370. + dy], det=.99)


@pytest.mark.parametrize("label,phase,direction,translate_feet,expected_code", [
    ("sitting_down", "standing", 1, True, "static_motion_conflict"),
    ("standing_up", "standing", -1, True, "static_motion_conflict"),
    ("sitting_down", "descending", 1, True, "rigid_translation_confirmed"),
    ("standing_up", "rising", -1, True, "rigid_translation_confirmed"),
    ("sitting_down", "descending", 1, False, "descent_without_flexion"),
    ("standing_up", "rising", -1, False, "rise_without_extension"),
    ("sitting_down", "descending", -1, False, "opposite_motion_confirmed"),
    ("standing_up", "rising", 1, False, "opposite_motion_confirmed"),
])
def test_original_transition_cannot_bypass_coordinate_derived_conflicts(
        label, phase, direction, translate_feet, expected_code):
    stream = ReconcileActions()
    history = []
    predictions = []
    for i in range(11):
        t = i * .1
        person = straight_pose(direction * i * 10 / 3, translate_feet)
        history.append(dict(t=t, instances=[person]))
        _, features = phase_features(history)
        prediction = dict(label=label, phase=phase, phase_scores={phase: .9})
        out = stream.update(t, prediction, features, person)
        predictions.append(out["prediction"]["label"])
        assert prediction["label"] == label
    assert features["knee"] == pytest.approx(1.)
    assert features["hip_speed"] == pytest.approx(-direction / 3)
    if phase == "standing":
        assert predictions == ["unknown"] * len(predictions)
    elif expected_code in {"descent_without_flexion", "rise_without_extension"}:
        # A 2D knee with no visible angle change cannot establish a *new* action,
        # but it also cannot disprove the frozen action when its phase agrees.
        assert predictions == [label] * len(predictions)
    else:
        # Missing features and same-phase short windows preserve the old model
        # context. Once motion can be checked, its contradictory label is blocked.
        assert predictions[:4] == [label] * 4
        assert predictions[4:] == ["unknown"] * (len(predictions) - 4)
    assert out["changed"] == (predictions[-1] != label)
    assert out["code"] == expected_code


@pytest.mark.parametrize("label", ["standing_up", "sitting_down"])
def test_unavailable_phase_and_missing_observation_preserve_raw_context(label):
    stream = ReconcileActions()
    p, f, person = observation(0, "rising", label)
    p["phase"] = "unknown"
    out = stream.update(0., p, f, person)
    assert not out["changed"] and out["prediction"]["label"] == label
    assert out["code"] == "unsupported_phase"
    out = stream.update(.1, p, None, None)
    assert not out["changed"] and out["prediction"]["label"] == label
    assert out["code"] == "insufficient_pose"


@pytest.mark.parametrize("label,phase", [("sitting_down", "descending"), ("standing_up", "rising")])
@pytest.mark.parametrize("steps", [[0] * 7, [0, .1, .2, .3, .4, .5], [0, 1, 2, 3, 4, 3.8]])
def test_same_action_and_phase_not_rejected_for_weak_new_motion_evidence(label, phase, steps):
    out = run(label=label, phase=phase, steps=steps)
    assert not out["changed"] and out["prediction"]["label"] == label
    assert "transition_blocked" not in out["evidence"]


@pytest.mark.parametrize("label,phase", [("sitting_down", "descending"), ("standing_up", "rising")])
def test_same_action_and_phase_survive_small_projected_knee_progress(label, phase):
    stream = ReconcileActions()
    for i in range(8):
        prediction, features, person = observation(i, phase, label)
        features["knee"] = .8  # foreshortening can hide 2D knee-angle progress
        out = stream.update(i * .05, prediction, features, person)
    assert not out["changed"] and out["prediction"]["label"] == label
    assert "transition_blocked" not in out["evidence"]
