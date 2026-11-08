"""Image-space fall evidence; experimental rules, never an alarm decision.

No metric position, floor calibration, gait label or personal-risk background is
required here. A low pose is evidence to review, not a diagnosis of a fall.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median

from ..config import RuleConfig
from ..schemas import FallState, HumanObservation, Validity


class FallDetector:
    def __init__(self, rules: RuleConfig):
        self.rules = rules
        self.reset()

    def reset(self) -> None:
        self._epoch: str | None = None
        self._last_t: float | None = None
        self._last_update_t: float | None = None
        self._history: deque[tuple[float, float, float]] = deque()
        self._standing: deque[tuple[float, float, float]] = deque()
        self._h_ref: float | None = None
        self._hip_ref: float | None = None
        self._candidate_since: float | None = None
        self._candidate_reason: str | None = None
        self._previous_low = False
        self._low_duration = 0.0
        self._recovery_since: float | None = None
        self._recovered = False
        self._last_state = FallState()

    def _unknown(self, t_s: float, reason: str) -> FallState:
        self.reset()
        self._last_state = FallState(t_s=t_s, reasons=[reason])
        return self._last_state.model_copy(deep=True)

    def _features(self, human: HumanObservation) -> tuple[float, float, float, float, bool] | None:
        if human.pose_format != "coco17" or len(human.keypoints) != 17:
            return None
        needed = (5, 6, 11, 12)
        if any(human.keypoints[i][2] < self.rules.keypoint_conf for i in needed):
            return None
        x1, y1, x2, y2 = human.bbox_px
        height, width = y2 - y1, x2 - x1
        if height <= 0 or width <= 0:
            return None
        shoulders = tuple((human.keypoints[5][i] + human.keypoints[6][i]) / 2 for i in (0, 1))
        hips = tuple((human.keypoints[11][i] + human.keypoints[12][i]) / 2 for i in (0, 1))
        dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
        if math.hypot(dx, dy) < 1e-6:
            return None
        tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
        aspect = width / height
        legs_visible = all(human.keypoints[i][2] >= self.rules.keypoint_conf for i in (13, 14, 15, 16))
        legs_below_hips = all(
            human.keypoints[hip][1] < human.keypoints[knee][1] < human.keypoints[ankle][1]
            for hip, knee, ankle in ((11, 13, 15), (12, 14, 16))
        )
        upright = (
            dy > 0 and tilt <= self.rules.upright_tilt_deg and aspect < self.rules.fall_aspect_ratio
            and legs_visible and legs_below_hips
        )
        return height, hips[1], tilt, aspect, upright

    def update(
        self, human: HumanObservation | None, t_s: float, camera_stable: bool | None = True
    ) -> FallState:
        if not math.isfinite(t_s):
            raise ValueError("source time must be finite")
        if camera_stable is not True:
            return self._unknown(t_s, "camera_motion" if camera_stable is False else "camera_stability_unknown")
        if human is None:
            return self._unknown(t_s, "missing_person")
        if human.validity not in (Validity.VALID, Validity.DEGRADED):
            return self._unknown(t_s, "invalid_person")
        if not 0 <= t_s - human.t_s <= self.rules.human_max_age_s:
            return self._unknown(t_s, "stale_or_future_observation")
        features = self._features(human)
        if features is None:
            return self._unknown(t_s, "insufficient_torso_keypoints")
        if self._last_update_t is not None and t_s < self._last_update_t:
            return self._unknown(t_s, "source_time_reversed")
        if self._epoch is not None and human.track_epoch != self._epoch:
            return self._unknown(t_s, "track_epoch_changed")
        if self._last_t is not None and (
            human.t_s < self._last_t or human.t_s - self._last_t > self.rules.max_gap_s
        ):
            return self._unknown(t_s, "observation_discontinuity")
        self._last_update_t = t_s
        if human.t_s == self._last_t:
            return self._last_state.model_copy(deep=True)
        self._epoch = human.track_epoch
        previous_t = self._last_t
        self._last_t = now = human.t_s
        height, hip_y, tilt, aspect, upright = features

        if self._recovered:
            self._candidate_since = self._candidate_reason = None
            self._recovered = False
            self._recovery_since = None
            self._standing.clear()
            self._history.clear()

        while self._history and now - self._history[0][0] > self.rules.max_gap_s:
            self._history.popleft()
        drop = None if self._h_ref is None else (hip_y - self._hip_ref) / self._h_ref
        collapsed = self._h_ref is not None and height <= self._h_ref * self.rules.fall_height_ratio
        # Sideways evidence, or frontal descent plus shortening, gives two low-pose routes.
        low = (tilt >= self.rules.fall_tilt_deg and aspect >= self.rules.fall_aspect_ratio) or (
            collapsed and drop is not None and drop >= self.rules.fall_drop_norm
        )
        rapid = False
        if self._h_ref is not None and self._history:
            _, old_hip, old_tilt = self._history[0]
            rapid = (hip_y - old_hip) / self._h_ref >= self.rules.fall_drop_norm or (
                tilt - old_tilt >= self.rules.fall_tilt_deg
            )
        self._history.append((now, hip_y, tilt))
        if low:
            if self._previous_low and previous_t is not None:
                self._low_duration += now - previous_t
        else:
            self._low_duration = 0.0
        self._previous_low = low

        if self._candidate_since is None:
            reason = None
            if low and self._h_ref is None:
                reason = "initial_low_pose"
            elif rapid:
                reason = "rapid_pose_change"
            elif low and self._low_duration >= self.rules.fall_slow_duration_s:
                reason = "slow_low_pose"
            if reason:
                self._candidate_since, self._candidate_reason = now, reason

        phase, recovered = "NONE", False
        if self._candidate_since is not None:
            phase = "LOW_POSE" if low else "SUSPECTED"
            if upright and not low:
                if self._recovery_since is None:
                    self._recovery_since = now
                phase = "RECOVERING"
                recovered = now - self._recovery_since >= self.rules.recovery_duration_s
                self._recovered = recovered
            else:
                self._recovery_since = None
        elif upright and not low and self._h_ref is None:
            self._standing.append((now, height, hip_y))
            # Retain enough samples to span the requested window at irregular frame rates.
            while len(self._standing) > 1 and now - self._standing[1][0] >= self.rules.upright_reference_s:
                self._standing.popleft()
            if now - self._standing[0][0] >= self.rules.upright_reference_s:
                self._h_ref = median(item[1] for item in self._standing)
                self._hip_ref = median(item[2] for item in self._standing)
                drop = (hip_y - self._hip_ref) / self._h_ref
        elif not upright or low:
            self._standing.clear()

        reasons = [self._candidate_reason] if self._candidate_reason else []
        if low and self._candidate_since is None:
            reasons.append("low_pose_under_observation")
        if self._h_ref is None:
            reasons.append("upright_reference_unavailable")
        if recovered:
            reasons.append("sustained_upright_recovery")
        self._last_state = FallState(
            t_s=now, phase=phase, candidate_since_s=self._candidate_since,
            vertical_drop_norm=drop, tilt_deg=tilt, aspect_ratio=aspect, low_pose=low,
            low_pose_duration_s=self._low_duration, had_upright=self._h_ref is not None,
            recovered=recovered, validity=Validity.VALID, reasons=reasons,
        )
        return self._last_state.model_copy(deep=True)
