"""Small variable-dt constant-velocity Kalman filter; no future observations."""
from __future__ import annotations

import numpy as np

from ..config import RuleConfig
from ..schemas import GroundObservation, MotionState, Validity


class MotionEstimator:
    def __init__(self, rules: RuleConfig):
        self.rules = rules
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._p = None
        self._t = None
        self._epoch = None
        self._calibration = None

    def _initialize(self, ground: GroundObservation, track_epoch: str, reason: str) -> MotionState:
        self._x = np.array([*ground.xy_m, 0.0, 0.0], dtype=float)
        variance = max(ground.uncertainty_m, 0.01) ** 2
        self._p = np.diag([variance, variance, 4.0, 4.0])
        self._t = ground.t_s
        self._epoch = track_epoch
        self._calibration = ground.calibration_id
        return MotionState(t_s=ground.t_s, xy_m=ground.xy_m, uncertainty_m=ground.uncertainty_m,
                           calibration_id=ground.calibration_id, track_epoch=track_epoch,
                           validity=Validity.DEGRADED, reasons=[reason, "VELOCITY_WARMUP"])

    def update(self, ground: GroundObservation, track_epoch: str) -> MotionState:
        if ground.validity != Validity.VALID or ground.xy_m is None or not np.isfinite(ground.xy_m).all():
            self.reset()
            return MotionState(t_s=ground.t_s, calibration_id=ground.calibration_id,
                               track_epoch=track_epoch, reasons=["GROUND_OBSERVATION_INVALID"])
        if self._x is None:
            return self._initialize(ground, track_epoch, "NEW_TRACK")
        dt = ground.t_s - self._t
        if track_epoch != self._epoch or ground.calibration_id != self._calibration:
            return self._initialize(ground, track_epoch, "TRACK_OR_CALIBRATION_CHANGED")
        if dt <= 0 or dt > self.rules.max_gap_s:
            return self._initialize(ground, track_epoch, "TIME_DISCONTINUITY")
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        acceleration = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]], dtype=float)
        sigma = float(getattr(self.rules, "motion_acceleration_noise_mps2", 1.0))
        process_noise = acceleration @ acceleration.T * sigma ** 2
        predicted = transition @ self._x
        predicted_p = transition @ self._p @ transition.T + process_noise
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        noise = np.eye(2) * max(ground.uncertainty_m, 0.01) ** 2
        residual = np.asarray(ground.xy_m) - observation @ predicted
        innovation = observation @ predicted_p @ observation.T + noise
        gain = np.linalg.solve(innovation, observation @ predicted_p).T
        self._x = predicted + gain @ residual
        correction = np.eye(4) - gain @ observation
        self._p = correction @ predicted_p @ correction.T + gain @ noise @ gain.T
        self._t = ground.t_s
        position_sigma = float(np.sqrt(max(0.0, np.linalg.eigvalsh(self._p[:2, :2]).max())))
        velocity = self._x[2:]
        return MotionState(t_s=ground.t_s, xy_m=tuple(float(v) for v in self._x[:2]),
                           velocity_mps=tuple(float(v) for v in velocity), speed_mps=float(np.linalg.norm(velocity)),
                           uncertainty_m=max(ground.uncertainty_m, 2 * position_sigma),
                           calibration_id=ground.calibration_id, track_epoch=track_epoch,
                           validity=Validity.VALID)
