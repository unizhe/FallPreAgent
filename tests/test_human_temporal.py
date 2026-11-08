"""Synthetic observations verify time/identity semantics, not clinical accuracy."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from fallpre.config import RuleConfig
from fallpre.perception.human import HumanPerceptor
from fallpre.schemas import Behavior, HumanObservation, MotionState, Validity
from fallpre.temporal.behavior import BehaviorEstimator
from fallpre.temporal.fall_evidence import FallDetector


def person(t, *, epoch="epoch-a", pose="standing", tilt=None, hip_y=None, height=None):
    points = [(30.0, 10.0, 0.95) for _ in range(17)]
    if pose == "lying":
        bbox = (0.0, 100.0, 200.0, 160.0)
        points[5], points[6] = (40, 120, 0.95), (40, 140, 0.95)
        points[11], points[12] = (130, 120, 0.95), (130, 140, 0.95)
    else:
        bbox = (0.0, 0.0, 60.0, height or 200.0)
        hy = 100.0 if hip_y is None else hip_y
        angle = math.radians(tilt or 0)
        sx, sy = 30 - 50 * math.sin(angle), hy - 50 * math.cos(angle)
        points[5], points[6] = (sx - 10, sy, .95), (sx + 10, sy, .95)
        points[11], points[12] = (20, hy, .95), (40, hy, .95)
    points[13], points[14] = (20, 150, .95), (40, 150, .95)
    points[15], points[16] = (20, 195, .95), (40, 195, .95)
    return HumanObservation(
        t_s=t, frame_id=int(t * 100), track_id=7, track_epoch=epoch,
        bbox_px=bbox, keypoints=points, score=.9,
    )


def motion(t, speed, epoch="epoch-a", validity=Validity.VALID):
    return MotionState(
        t_s=t, xy_m=(0, 0), velocity_mps=(speed, 0), speed_mps=speed,
        track_epoch=epoch, validity=validity,
    )


def establish_reference(detector):
    for t in (0., .25, .5):
        state = detector.update(person(t), t)
    assert state.had_upright
    return state


def test_behavior_requires_dwell_and_has_speed_hysteresis():
    estimator = BehaviorEstimator(RuleConfig())
    assert estimator.update(person(0), motion(0, .2), 0) == Behavior.UNKNOWN
    assert estimator.update(person(.25), motion(.25, .2), .25) == Behavior.UNKNOWN
    assert estimator.update(person(.5), motion(.5, .2), .5) == Behavior.WALKING
    assert estimator.update(person(.75), motion(.75, .08), .75) == Behavior.WALKING
    for t in (1., 1.25):
        assert estimator.update(person(t), motion(t, .01), t) == Behavior.WALKING
    assert estimator.update(person(1.5), motion(1.5, .01), 1.5) == Behavior.STATIONARY


def test_behavior_no_pixel_speed_fallback_and_no_cross_epoch_motion():
    estimator = BehaviorEstimator(RuleConfig())
    for t in (0., .25, .5):
        estimator.update(person(t), motion(t, .2), t)
    assert estimator.update(person(.75), motion(.75, .2, validity=Validity.UNKNOWN), .75) == Behavior.UNKNOWN
    assert estimator.update(person(1, epoch="new"), motion(1, .2), 1) == Behavior.UNKNOWN


def test_behavior_cached_observation_cannot_complete_dwell():
    estimator = BehaviorEstimator(RuleConfig(behavior_dwell_s=.4))
    h, m = person(0), motion(0, .2)
    assert estimator.update(h, m, 0) == Behavior.UNKNOWN
    assert estimator.update(h, m, .4) == Behavior.UNKNOWN
    assert estimator.update(h, m, .6) == Behavior.UNKNOWN


def test_behavior_moving_lying_person_is_not_walking():
    estimator = BehaviorEstimator(RuleConfig())
    for t in (0., .25, .5, .75):
        assert estimator.update(person(t, pose="lying"), motion(t, .2), t) == Behavior.UNKNOWN


def test_rapid_fall_evidence_uses_frozen_reference_and_no_metric_inputs():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    state = detector.update(person(.75, pose="lying"), .75)
    assert state.had_upright and state.phase == "LOW_POSE"
    assert state.vertical_drop_norm == pytest.approx(30 / 200)
    assert "rapid_pose_change" in state.reasons
    first_candidate = state.candidate_since_s
    for t in (1., 1.25, 1.5, 1.75):
        state = detector.update(person(t, pose="lying"), t)
    assert state.low_pose_duration_s == pytest.approx(1)
    assert state.candidate_since_s == first_candidate
    assert state.vertical_drop_norm == pytest.approx(30 / 200)
    assert state.phase not in ("ALARM", "CLEAR", "WAIT")


def test_frontal_descent_and_shortening_can_be_low_without_sideways_tilt():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    h = person(.75, hip_y=180, height=100)
    h.bbox_px = (0, 120, 60, 220)
    state = detector.update(h, .75)
    assert state.tilt_deg == pytest.approx(0)
    assert state.low_pose is True
    assert state.vertical_drop_norm == pytest.approx(.4)


def test_slow_fall_has_an_entry_without_rapid_descent():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    # Smooth rotations never cross the rapid-change threshold in the short window.
    for n, angle in enumerate((15, 30, 45, 60, 75, 90), 1):
        t = .5 + n * .25
        h = person(t, tilt=angle)
        if angle >= 60:
            h.bbox_px = (0, 80, 200, 180)
        state = detector.update(h, t)
        assert "rapid_pose_change" not in state.reasons
    for n in range(1, 7):
        t = 2 + n * .25
        h = person(t, tilt=90)
        h.bbox_px = (0, 80, 200, 180)
        state = detector.update(h, t)
    assert state.phase == "LOW_POSE"
    assert "slow_low_pose" in state.reasons
    assert state.vertical_drop_norm == pytest.approx(0)


def test_initial_low_pose_is_review_only_without_observed_upright():
    detector = FallDetector(RuleConfig())
    for t in (0., .25, .5, .75, 1., 1.25):
        state = detector.update(person(t, pose="lying"), t)
    assert state.phase == "LOW_POSE"
    assert state.had_upright is False
    assert state.vertical_drop_norm is None
    assert "initial_low_pose" in state.reasons


@pytest.mark.parametrize("camera_stable", [False, None])
def test_camera_motion_or_unknown_stability_discards_old_reference(camera_stable):
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    state = detector.update(person(.75, pose="lying"), .75, camera_stable)
    assert state.phase == "UNOBSERVABLE" and state.had_upright is False
    state = detector.update(person(1, pose="lying"), 1, True)
    assert "initial_low_pose" in state.reasons


def test_missing_person_does_not_accumulate_low_duration_or_fake_recovery():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    detector.update(person(.75, pose="lying"), .75)
    detector.update(person(1, pose="lying"), 1)
    missing = detector.update(None, 1.25)
    assert missing.phase == "UNOBSERVABLE" and not missing.recovered
    state = detector.update(person(1.5, pose="lying"), 1.5)
    assert state.low_pose_duration_s == 0
    assert not state.had_upright


@pytest.mark.parametrize("t,epoch", [(1.5, "epoch-a"), (.25, "epoch-a"), (.75, "epoch-b")])
def test_gap_rewind_and_epoch_change_break_evidence(t, epoch):
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    state = detector.update(person(t, epoch=epoch, pose="lying"), t)
    assert state.phase == "UNOBSERVABLE"
    assert state.vertical_drop_norm is None and not state.had_upright


def test_recovery_requires_continuous_observations_and_keeps_candidate_origin():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    candidate = detector.update(person(.75, pose="lying"), .75)
    for t in (1., 1.25, 1.5, 1.75):
        state = detector.update(person(t), t)
        assert state.phase == "RECOVERING" and not state.recovered
    state = detector.update(person(2), 2)
    assert state.recovered
    assert state.candidate_since_s == candidate.candidate_since_s
    assert detector.update(person(2.25), 2.25).phase == "NONE"


def test_missing_wrist_does_not_discard_torso_evidence_but_missing_hip_does():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    h = person(.75, pose="lying")
    h.keypoints[9] = (0, 0, 0)
    assert detector.update(h, .75).phase == "LOW_POSE"
    h = person(1, pose="lying")
    h.keypoints[11] = (0, 0, 0)
    assert detector.update(h, 1).phase == "UNOBSERVABLE"


def test_repeated_cached_pose_does_not_accumulate_low_duration():
    detector = FallDetector(RuleConfig())
    establish_reference(detector)
    h = person(.75, pose="lying")
    detector.update(h, .75)
    state = detector.update(h, 1)
    assert state.low_pose_duration_s == 0
    assert state.t_s == .75


class FakeTracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeModel:
    def __init__(self):
        self.predictor = SimpleNamespace(trackers=[FakeTracker()])
        self.calls = []
        self.visible = True

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        h = person(0)
        boxes = SimpleNamespace(id=np.array([7]), xyxy=np.array([h.bbox_px]), conf=np.array([.9]))
        keypoints = SimpleNamespace(data=np.array([h.keypoints]))
        return [SimpleNamespace(boxes=boxes if self.visible else None, keypoints=keypoints)]


def perceptor():
    instance = HumanPerceptor("yolo11n-pose.pt", rules=RuleConfig())
    instance._model = FakeModel()
    return instance


def test_perceptor_explicit_tracker_and_stable_epoch():
    p = perceptor()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    first = p.infer(frame, 0, 0)[0]
    second = p.infer(frame, .25, 1)[0]
    assert first.track_epoch == second.track_epoch
    assert p._model.calls[0]["tracker"] == "bytetrack.yaml"
    assert p._model.calls[0]["persist"] is True


@pytest.mark.parametrize("t,frame_id,shape", [(1, 1, (240, 320, 3)), (0, 1, (240, 320, 3)), (.25, 0, (240, 320, 3)), (.25, 1, (480, 640, 3))])
def test_perceptor_reset_even_when_tracker_id_is_reused(t, frame_id, shape):
    p = perceptor()
    first = p.infer(np.zeros((240, 320, 3)), 0, 0)[0]
    current = p.infer(np.zeros(shape), t, frame_id)[0]
    assert current.track_id == first.track_id
    assert current.track_epoch != first.track_epoch
    assert p._model.predictor.trackers[0].resets == 1


def test_per_track_absence_and_explicit_reset_create_new_epochs():
    p = perceptor()
    frame = np.zeros((240, 320, 3))
    first = p.infer(frame, 0, 0)[0]
    p._model.visible = False
    assert p.infer(frame, .25, 1) == []
    assert p.infer(frame, .5, 2) == []
    p._model.visible = True
    current = p.infer(frame, .75, 3)[0]
    assert current.track_epoch != first.track_epoch
    p.reset()
    assert p.infer(frame, 1, 4)[0].track_epoch != current.track_epoch
