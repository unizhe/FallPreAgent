"""Prototype displacement states. STATIONARY does not mean upright or safe."""
from __future__ import annotations

import math

from ..config import RuleConfig
from ..schemas import Behavior, HumanObservation, MotionState, Validity


class BehaviorEstimator:
    def __init__(self, rules: RuleConfig):
        self.rules = rules
        self.reset()

    def reset(self) -> None:
        self._state = Behavior.UNKNOWN
        self._pending: Behavior | None = None
        self._pending_since: float | None = None
        self._epoch: str | None = None
        self._last_observation_t: float | None = None
        self._last_update_t: float | None = None

    def _upright(self, human: HumanObservation) -> bool:
        if human.pose_format != "coco17" or len(human.keypoints) != 17:
            return False
        if any(human.keypoints[i][2] < self.rules.keypoint_conf for i in (5, 6, 11, 12)):
            return False
        shoulder_x = (human.keypoints[5][0] + human.keypoints[6][0]) / 2
        shoulder_y = (human.keypoints[5][1] + human.keypoints[6][1]) / 2
        hip_x = (human.keypoints[11][0] + human.keypoints[12][0]) / 2
        hip_y = (human.keypoints[11][1] + human.keypoints[12][1]) / 2
        if hip_y <= shoulder_y:
            return False
        tilt = math.degrees(math.atan2(abs(hip_x - shoulder_x), hip_y - shoulder_y))
        return tilt <= self.rules.upright_tilt_deg

    def update(self, human: HumanObservation | None, motion: MotionState, t_s: float) -> Behavior:
        if not math.isfinite(t_s):
            raise ValueError("source time must be finite")
        if self._last_update_t is not None and t_s < self._last_update_t:
            self.reset()
        self._last_update_t = t_s
        usable = (
            human is not None
            and human.validity in (Validity.VALID, Validity.DEGRADED)
            and motion.validity == Validity.VALID
            and motion.track_epoch == human.track_epoch
            and motion.speed_mps is not None and math.isfinite(motion.speed_mps)
            and motion.speed_mps >= 0
            and 0 <= t_s - human.t_s <= self.rules.human_max_age_s
            and 0 <= t_s - motion.t_s <= self.rules.human_max_age_s
            and abs(motion.t_s - human.t_s) <= 1e-6
        )
        if not usable:
            self.reset()
            return Behavior.UNKNOWN
        assert human is not None
        if self._epoch != human.track_epoch or (
            self._last_observation_t is not None and (
                human.t_s < self._last_observation_t
                or human.t_s - self._last_observation_t > self.rules.max_gap_s
            )
        ):
            self.reset()
        self._epoch, self._last_update_t = human.track_epoch, t_s
        if human.t_s == self._last_observation_t:
            return self._state  # Polling cached data never extends dwell time.
        self._last_observation_t = human.t_s
        speed = motion.speed_mps
        if speed <= self.rules.walking_exit_mps:
            desired = Behavior.STATIONARY
        elif self._upright(human) and (
            speed >= self.rules.walking_enter_mps or self._state == Behavior.WALKING
        ):
            desired = Behavior.WALKING
        elif self._state == Behavior.STATIONARY and speed < self.rules.walking_enter_mps:
            desired = Behavior.STATIONARY
        else:
            self._state, self._pending, self._pending_since = Behavior.UNKNOWN, None, None
            return self._state
        if desired == self._state:
            self._pending, self._pending_since = None, None
        else:
            if desired != self._pending:
                self._pending, self._pending_since = desired, human.t_s
            if human.t_s - self._pending_since >= self.rules.behavior_dwell_s:
                self._state, self._pending, self._pending_since = desired, None, None
        return self._state
