"""Independent qualified walking clips; descriptive reference, not a medical risk score."""
from __future__ import annotations

import math
from statistics import median

import numpy as np

from .config import RuleConfig
from .schemas import Behavior, Snapshot, Validity, new_id


def mad(values: list[float]) -> float:
    mid = median(values)
    return median([abs(value - mid) for value in values])


class PersonalBaseline:
    def __init__(self, rules: RuleConfig, *, min_clip_s: float = 3.0,
                 measured_noise_floor: float | None = None):
        self.rules = rules
        self.min_clip_s = min_clip_s
        self.measured_noise_floor = measured_noise_floor
        self.clips: list[dict] = []
        self.archived: list[dict] = []
        self._samples: list[tuple[float, float]] = []
        self._direction: np.ndarray | None = None
        self._version: tuple | None = None
        self._baseline_id = new_id("baseline")
        self._until_s = -math.inf
        self._last_epoch: str | None = None
        self._reference: dict | None = None
        self._revision = 0
        self._updated_at_s: float | None = None

    def _finish(self):
        if len(self._samples) >= 2:
            duration = self._samples[-1][0] - self._samples[0][0]
            if duration >= self.min_clip_s:
                values = [speed for _, speed in self._samples]
                self.clips.append({"start_s": self._samples[0][0], "end_s": self._samples[-1][0],
                                   "duration_s": duration, "speed_median_mps": median(values),
                                   "speed_mad_mps": mad(values)})
                self._revision += 1
                self._updated_at_s = self._samples[-1][0]
                if self._reference is None and len(self.clips) >= self.rules.baseline_min_clips and sum(
                    clip["duration_s"] for clip in self.clips
                ) >= self.rules.baseline_min_duration_s:
                    speeds = [clip["speed_median_mps"] for clip in self.clips]
                    self._reference = {"clip_count": len(self.clips), "speed_median_mps": median(speeds),
                                       "speed_mad_mps": mad(speeds)}
        self._samples = []
        self._direction = None

    def update(self, snapshot: Snapshot, *, excluded: bool = False, intervention: bool = False) -> dict:
        version = (snapshot.person_id, snapshot.calibration_id, snapshot.pipeline_version, self.rules.version)
        if self._updated_at_s is None:
            self._updated_at_s = snapshot.t_s
        if self._version is not None and version != self._version:
            # Never join a partially observed clip across measurement pipelines.
            self._samples = []
            self.archived.append(self.report(snapshot.t_s))
            self.clips = []
            self._reference = None
            self._direction = None
            self._baseline_id = new_id("baseline")
            self._revision = 0
            self._updated_at_s = snapshot.t_s
        self._version = version
        if intervention:
            self._until_s = snapshot.t_s + 5.0
        m = snapshot.motion
        valid = (not excluded and snapshot.t_s > self._until_s and snapshot.binding_status == "BOUND"
                 and snapshot.behavior == Behavior.WALKING and m.validity == Validity.VALID
                 and snapshot.quality.validity == Validity.VALID
                 and snapshot.quality.camera_stable is True and snapshot.quality.valid_until_s > snapshot.t_s
                 and m.speed_mps is not None and m.velocity_mps is not None
                 and m.speed_mps >= self.rules.walking_enter_mps
                 and snapshot.fall.phase == "NONE"
                 and snapshot.spatial.validity == Validity.VALID
                 and snapshot.spatial.ttc_status in ("NO_INTERSECTION_WITHIN_HORIZON", "STATIONARY_OUTSIDE"))
        if excluded or intervention:
            self._samples = []  # Do not retain the seconds approaching an event as normal walking.
            self._direction = None
        if not valid:
            self._finish()
            return self.report(snapshot.t_s)
        direction = np.asarray(m.velocity_mps) / m.speed_mps
        discontinuity = (self._samples and
                         (snapshot.t_s - self._samples[-1][0] > self.rules.max_gap_s
                          or snapshot.t_s <= self._samples[-1][0]
                          or snapshot.track_epoch != self._last_epoch))
        turning = self._direction is not None and float(direction @ self._direction) < math.cos(math.radians(20))
        if discontinuity or turning:
            self._finish()
            # Turning sample is excluded; a new straight clip can start on the next sample.
            self._last_epoch = snapshot.track_epoch
            return self.report(snapshot.t_s)
        if self._direction is None:
            self._direction = direction
        self._samples.append((snapshot.t_s, float(m.speed_mps)))
        self._last_epoch = snapshot.track_epoch
        return self.report(snapshot.t_s)

    def finish(self, now_s: float) -> dict:
        self._finish()
        return self.report(now_s)

    def report(self, now_s: float) -> dict:
        reference = self._reference
        deviations = []
        # Keep the initial reference frozen; later clips cannot normalize their own deviation away.
        if reference and len(self.clips) > reference["clip_count"]:
            current = self.clips[-1]["speed_median_mps"]
            center = reference["speed_median_mps"]
            scale = 1.4826 * reference["speed_mad_mps"]
            if self.measured_noise_floor is not None:
                scale = max(scale, self.measured_noise_floor)
            deviations.append({"metric": "walking_speed", "absolute_change_mps": current - center,
                               "speed_drop_ratio": (center - current) / center if center > 0.03 else None,
                               "robust_deviation": (current - center) / scale if scale > 0 else None})
        updated_at_s = self._updated_at_s if self._updated_at_s is not None else now_s
        return {"report_id": f"{self._baseline_id}_r{self._revision}", "baseline_id": self._baseline_id,
                "baseline_status": "temporary" if reference else "collecting", "risk_level": None,
                "updated_at_s": updated_at_s, "valid_until_s": updated_at_s + 86400,
                "measurement_version": list(self._version) if self._version else None,
                "reference": reference, "main_deviations": deviations,
                "data_coverage": {"qualified_clip_count": len(self.clips),
                                  "qualified_walking_duration_s": sum(c["duration_s"] for c in self.clips),
                                  "scope": "visible qualified walking only; not daily activity"},
                "clips": list(self.clips),
                "missing_domains": ["clinical_history", "sleep", "validated_risk_thresholds"],
                "limitations": ["prototype reference; no clinical risk classification"]}
