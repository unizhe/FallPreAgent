import copy
import json

import numpy as np
import pytest

from fallpre.weekly.webcam_behavior import BehaviorAxes


def person(x=300.0, y=0.0, step=0.0, facing=1.0, head=0.0):
    p = np.zeros((17, 3))
    for index, xx, yy in [
        (0, x + head, 80),
        (1, x - 6 + head, 76),
        (2, x + 6 + head, 76),
        (3, x - 12 + head, 80),
        (4, x + 12 + head, 80),
    ]:
        p[index] = [xx, yy + y, 0.99]
    for s, h, k, a, sign in [(5, 11, 13, 15, -1), (6, 12, 14, 16, 1)]:
        p[s] = [x + sign * 32 * facing, 120 + y, 0.99]
        p[h] = [x + sign * 20 * facing, 220 + y, 0.99]
        p[k] = [x + sign * 20 + sign * step * 0.4, 290 + y, 0.99]
        p[a] = [x + sign * 20 + sign * step, 365 + y, 0.99]
    return dict(keypoints=p.tolist(), box=[x - 65, 50 + y, x + 65, 390 + y], confidence=0.99)


def run(observer, p, t, label="unknown", **kwargs):
    return observer.update(
        p,
        t,
        kwargs.pop("epoch", 1),
        {"label": label, "raw_label": kwargs.pop("raw", "background")},
        960,
        540,
        **kwargs,
    )


def moving_sequence(observer, start=0.0):
    return [
        run(observer, person(x=300 + i * 8, step=15 * np.sin(i * 0.8)), start + i * 0.13) for i in range(10)
    ]


def test_walking_is_independent_from_raw_action_vote_and_preserves_input():
    b = BehaviorAxes()
    original = person()
    action = {"label": "unknown", "raw_label": "background"}
    before = copy.deepcopy((original, action))
    b.update(original, 0.0, 1, action, 960, 540)
    assert (original, action) == before
    out = moving_sequence(BehaviorAxes())
    assert any(r["locomotion"]["label"] == "walking" for r in out)
    assert all(r["posture"]["label"] == "upright" for r in out)
    assert not any(r["turning"]["label"] == "turning" for r in out)


def test_raw_walking_without_actual_motion_is_stationary():
    b = BehaviorAxes()
    out = [run(b, person(), i * 0.13, raw="walking") for i in range(8)]
    assert out[-1]["locomotion"]["label"] == "stationary"
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)


def test_head_turn_does_not_become_body_turn():
    b = BehaviorAxes()
    out = [run(b, person(head=i * 7), i * 0.13) for i in range(7)]
    assert out[-1]["turning"]["label"] == "not_turning"
    assert out[-1]["locomotion"]["label"] == "stationary"


def test_in_place_body_turn_is_detected_without_translation():
    b = BehaviorAxes()
    out = [run(b, person(facing=1 - i * 0.20), i * 0.13) for i in range(9)]
    assert any(r["turning"]["label"] == "turning" for r in out)
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)
    assert all(r["path_change"]["label"] == "unknown" for r in out)


def test_walking_and_body_turn_can_coexist():
    b = BehaviorAxes()
    out = [
        run(b, person(x=300 + i * 8, step=15 * np.sin(i * 0.8), facing=1 - i * 0.14), i * 0.13)
        for i in range(10)
    ]
    assert any(
        r["turning"]["label"] == "turning" and r["locomotion"]["label"] in {"starting", "walking"}
        for r in out
    )


def test_path_change_does_not_imply_body_turn():
    b = BehaviorAxes()
    out = []
    for i in range(12):
        x = 300 + min(i, 5) * 12
        y = max(i - 5, 0) * 12
        out.append(run(b, person(x=x, y=y, step=15 * np.sin(i)), i * 0.13))
    assert any(r["path_change"]["label"] == "changing" for r in out)
    assert not any(r["turning"]["label"] == "turning" for r in out)


def test_pure_rigid_translation_is_moving_not_walking():
    out = [run(b, person(x=300 + i * 12), i * 0.13) for b in [BehaviorAxes()] for i in range(8)]
    assert out[-1]["locomotion"]["label"] == "moving"
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)


def test_missing_feet_do_not_prevent_turn_observation_or_fabricate_steps():
    b = BehaviorAxes()
    out = []
    for i in range(8):
        p = person(x=300 + i * 8, facing=1 - i * 0.2)
        p["keypoints"][15] = p["keypoints"][16] = [0, 0, 0]
        out.append(run(b, p, i * 0.13, label="standing"))
    assert any(r["turning"]["label"] == "turning" for r in out)
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)
    assert not out[-1]["quality"]["feet"]


def test_missing_opposite_hip_keeps_turn_unknown():
    b = BehaviorAxes()
    out = []
    for i in range(8):
        p = person(facing=1 - i * 0.2)
        p["keypoints"][12] = [0, 0, 0]
        out.append(run(b, p, i * 0.13))
    assert all(r["turning"]["label"] == "unknown" for r in out)


def test_isolated_shoulder_noise_and_one_frame_identity_flip_cannot_confirm_turn():
    for mode in ["shoulder", "swap"]:
        b = BehaviorAxes()
        out = []
        for i in range(8):
            p = person()
            if mode == "shoulder":
                p["keypoints"][5][0] += i * 10
                p["keypoints"][6][0] -= i * 10
            elif i >= 4:
                for a, c in [(5, 6), (11, 12)]:
                    p["keypoints"][a], p["keypoints"][c] = p["keypoints"][c], p["keypoints"][a]
            out.append(run(b, p, i * 0.13))
        assert not any(r["turning"]["label"] == "turning" for r in out)


@pytest.mark.parametrize("label", ["seated", "bending", "squatting", "standing_up", "sitting_down"])
def test_existing_postures_and_transitions_cannot_be_relabelled_walking(label):
    b = BehaviorAxes()
    out = [
        run(b, person(x=300 + i * 8, step=15 * np.sin(i * 0.8), facing=1 - i * 0.14), i * 0.13, label=label)
        for i in range(10)
    ]
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)
    if label in {"standing_up", "sitting_down"}:
        assert out[-1]["transition"]["label"] == label
    else:
        assert out[-1]["posture"]["label"] == label


def test_stationary_seated_person_keeps_seat_and_stationary_separate():
    b = BehaviorAxes()
    out = [run(b, person(), i * 0.13, label="seated") for i in range(5)]
    assert out[-1]["posture"]["label"] == "seated"
    assert out[-1]["locomotion"]["label"] == "stationary"


@pytest.mark.parametrize("mode", ["gap", "epoch", "missing", "camera", "falling", "duplicate", "invalid"])
def test_interruption_resets_evidence_and_never_carries_old_walk(mode):
    b = BehaviorAxes()
    moving_sequence(b)
    t = 1.3
    kwargs = {}
    p = person(x=390, step=12)
    if mode == "gap":
        t = 2.0
    if mode == "epoch":
        kwargs["epoch"] = 2
    if mode == "missing":
        p = None
    if mode == "camera":
        kwargs["camera_stable"] = False
    if mode == "falling":
        kwargs["raw"] = "falling"
    if mode == "duplicate":
        t = 1.17
    if mode == "invalid":
        t = float("nan")
    out = run(b, p, t, **kwargs)
    assert out["locomotion"]["label"] == "unknown"
    assert out["turning"]["label"] == "unknown"
    assert out["quality"]["reset_reason"]


def test_stop_does_not_keep_walking_from_old_window():
    b = BehaviorAxes()
    moving_sequence(b)
    p = person(x=372, step=15 * np.sin(9 * 0.8))
    out = [run(b, p, 1.30 + i * 0.13) for i in range(7)]
    assert all(r["locomotion"]["label"] not in {"starting", "walking"} for r in out[1:])
    assert out[-1]["locomotion"]["label"] == "stationary"


def test_rise_history_cannot_be_borrowed_by_first_upright_frame():
    b = BehaviorAxes()
    for i in range(6):
        run(b, person(x=300 + i * 8, step=15 * np.sin(i * 0.8)), i * 0.13, label="standing_up")
    out = run(b, person(x=350, step=-12), 0.78, label="standing")
    assert out["locomotion"]["label"] == "unknown"


def test_projection_shrink_without_actual_width_change_is_not_body_turn():
    b = BehaviorAxes()
    out = []
    for i in range(8):
        p = person()
        for j in [5, 6]:
            p["keypoints"][j][1] += i * 6
        out.append(run(b, p, i * 0.13, label="standing"))
    assert not any(r["turning"]["label"] == "turning" for r in out)


def test_output_is_json_serializable_without_numpy_or_nan():
    for r in moving_sequence(BehaviorAxes()):
        json.dumps(r, allow_nan=False)


def test_upright_with_hidden_feet_cannot_claim_stationary_or_stopped():
    b = BehaviorAxes()
    moving_sequence(b)
    p = person(x=372, step=15 * np.sin(9 * 0.8))
    p["keypoints"][15] = p["keypoints"][16] = [0, 0, 0]
    out = [run(b, p, 1.30 + i * 0.13, label="standing") for i in range(8)]
    assert out[-1]["posture"]["label"] == "upright"
    assert out[-1]["locomotion"]["label"] == "unknown"
    assert "脚步不可完整观察" in out[-1]["locomotion"]["reason"]
    assert all(r["locomotion"]["label"] not in {"stationary", "stopping"} for r in out)


def test_established_gait_bridges_small_steps_but_weak_evidence_cannot_renew_itself():
    b = BehaviorAxes()
    moving_sequence(b)
    out = []
    for i in range(1, 16):
        out.append(run(b, person(x=372 + i * 1.6, step=15 * np.sin((9 + i) * 0.8)), (9 + i) * 0.13))
    assert any(r["metrics"].get("bounded_walk_continuation") for r in out)
    assert all(r["locomotion"]["label"] not in {"starting", "walking"} for r in out[-3:])


def test_small_steps_cannot_start_walking_using_continuation_threshold():
    b = BehaviorAxes()
    out = [run(b, person(x=300 + i * 1.6, step=15 * np.sin(i * 0.8)), i * 0.13) for i in range(16)]
    assert not any(r["locomotion"]["label"] in {"starting", "walking"} for r in out)


def test_first_stopped_observation_does_not_borrow_old_footsteps():
    b = BehaviorAxes()
    moving_sequence(b)
    out = run(b, person(x=372, step=15 * np.sin(9 * 0.8)), 1.30)
    assert out["metrics"]["latest_foot_speed"] == 0
    assert out["locomotion"]["label"] not in {"starting", "walking"}


def test_path_cannot_borrow_rising_trajectory_before_walking():
    b = BehaviorAxes()
    for i in range(6):
        run(b, person(y=70 - i * 14), i * 0.13, label="standing_up")
    out = [run(b, person(x=300 + i * 12, step=15 * np.sin(i)), 0.78 + i * 0.13) for i in range(5)]
    assert all(r["path_change"]["label"] == "unknown" for r in out)


def test_stationary_or_stopping_cannot_keep_a_past_path_change():
    b = BehaviorAxes()
    for i in range(12):
        p = person(x=300 + min(i, 5) * 12, y=max(i - 5, 0) * 12, step=15 * np.sin(i))
        run(b, p, i * 0.13)
    out = [run(b, p, 1.56 + i * 0.13) for i in range(8)]
    assert any(r["locomotion"]["label"] == "stationary" for r in out)
    assert all(
        r["path_change"]["label"] == "unknown"
        for r in out
        if r["locomotion"]["label"] in {"stationary", "stopping"}
    )


def test_brief_torso_projection_mismatch_abstains_then_resumes_existing_gait():
    b = BehaviorAxes()
    moving_sequence(b)
    onset = b.walk_since
    p = person(x=380, step=15 * np.sin(8))
    for j in [5, 6]:
        p["keypoints"][j][0] += 28
    uncertain = run(b, p, 1.3, label="standing")
    assert uncertain["locomotion"]["label"] == "moving"
    assert b.walk_since == onset
    recovered = run(b, person(x=388, step=15 * np.sin(8.8)), 1.43, label="standing")
    assert recovered["locomotion"]["label"] == "walking"
    assert b.walk_since == onset


@pytest.mark.parametrize("interruption", ["missing", "new_target"])
def test_projection_continuity_cannot_bridge_missing_or_new_person(interruption):
    b = BehaviorAxes()
    moving_sequence(b)
    p = person(x=380, step=15 * np.sin(8))
    for j in [5, 6]:
        p["keypoints"][j][0] += 28
    run(b, p, 1.3, label="standing")
    out = run(
        b,
        None if interruption == "missing" else person(x=388),
        1.43,
        epoch=2 if interruption == "new_target" else 1,
    )
    assert out["locomotion"]["label"] == "unknown"
    assert b.walk_since is None


def test_settling_feet_do_not_keep_walking_after_pelvis_progression_stops():
    b = BehaviorAxes()
    moving_sequence(b)
    run(b, person(x=372, step=10), 1.3, label="standing")
    out = run(b, person(x=372, step=-10), 1.43, label="standing")
    assert out["metrics"]["latest_foot_speed"] > 0.1
    assert out["locomotion"]["label"] == "stopping"


def test_scale_change_during_approach_is_not_mistaken_for_a_stop():
    b = BehaviorAxes()
    moving_sequence(b)
    out = []
    for i in range(4):
        p = person(x=372, step=15 * np.sin((10 + i) * 0.8))
        points = np.array(p["keypoints"])
        points[:, :2] = (points[:, :2] - [372, 220]) * (1.05 + i * 0.04) + [372, 220]
        p["keypoints"] = points.tolist()
        out.append(run(b, p, 1.3 + i * 0.13, label="standing"))
    assert all(not r["metrics"]["settling"] for r in out)
    assert all(r["locomotion"]["label"] != "stopping" for r in out)
