"""A causal, per-session standing reference for foreshortened frontal bends.

No learned weights or stored person calibration. Full current leg chains must
remain visible and planted; a compressed torso alone never establishes bending.
"""

from collections import deque

import numpy as np

from .action_v2 import angle

SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))
JOINTS = [5, 6, 11, 12, 13, 14, 15, 16]


class FrontalBend:
    def __init__(self):
        self.last_t = None
        self.epoch = None
        self.clear()

    def clear(self):
        self.standing = deque()
        self.reference = None
        self.candidates = deque()

    def update(self, t, epoch, person, result, camera_stable=True):
        if (
            epoch != self.epoch
            or self.last_t is not None
            and not 0 < t - self.last_t <= 0.35
            or camera_stable is not True
        ):
            self.clear()
        self.last_t, self.epoch = t, epoch
        info = dict(ready=False, candidate=False, confirmed=False, code="standing_reference_needed")
        try:
            p = np.asarray(person["kpts"], dtype=float)
        except (TypeError, KeyError, ValueError):
            p = np.empty((0, 3))
        if (
            camera_stable is not True
            or epoch is None
            or p.shape != (17, 3)
            or not np.isfinite(p).all()
            or not np.all(p[JOINTS, 2] >= 0.60)
            or result.get("raw_label") == "falling"
        ):
            self.clear()
            info["code"] = "current_pose_or_camera_unavailable"
            return info
        if self.reference is not None and t - self.reference[0] > 8.0:
            self.reference = None
        shoulder, hip, feet = (p[ids, :2].mean(0) for ids in ([5, 6], [11, 12], [15, 16]))
        torso = np.linalg.norm(hip - shoulder)
        legs = np.linalg.norm(feet - hip)
        knees = [angle(p[h, :2], p[k, :2], p[a, :2]) for _, h, k, a in SIDES]
        upright = (
            result.get("label") == "standing"
            and torso >= 24
            and legs >= 40
            and min(knees) > 0.90
            and (hip[1] - shoulder[1]) / torso > 0.85
            and (feet[1] - hip[1]) / legs > 0.85
        )
        if self.reference is not None:
            q = self.reference[1]
            old_torso = np.linalg.norm(q[[11, 12], :2].mean(0) - q[[5, 6], :2].mean(0))
            upright = upright and torso >= 0.85 * old_torso
        if upright:
            self.standing.append((t, p.copy()))
            while self.standing and t - self.standing[0][0] > 0.65:
                self.standing.popleft()
            if len(self.standing) >= 4 and t - self.standing[0][0] >= 0.4:
                points = np.stack([row[1][JOINTS, :2] for row in self.standing])
                if np.max(np.linalg.norm(points - points[0], axis=2)) / torso < 0.08:
                    self.reference = (t, np.median(np.stack([row[1] for row in self.standing]), axis=0))
        else:
            self.standing.clear()
        if self.reference is None:
            self.candidates.clear()
            return info
        rt, q = self.reference
        qs, qh, qf = (q[ids, :2].mean(0) for ids in ([5, 6], [11, 12], [15, 16]))
        scale = float(np.linalg.norm(qh - qs))
        vertical = (qf - qh) / np.linalg.norm(qf - qh)
        shoulder_drop = float((shoulder - qs) @ vertical / scale)
        hip_travel = float(np.linalg.norm(hip - qh) / scale)
        foot_travel = float(np.max(np.linalg.norm(p[[15, 16], :2] - q[[15, 16], :2], axis=1)) / scale)
        compression = float((hip - shoulder) @ vertical / ((qh - qs) @ vertical))
        # Both legs stay extended in their own observed reference orientation.
        leg_support = all(
            0.86 <= np.linalg.norm(p[h, :2] - p[a, :2]) / np.linalg.norm(q[h, :2] - q[a, :2]) <= 1.14
            and (p[a, :2] - p[h, :2]) @ vertical > 0.85 * np.linalg.norm(q[h, :2] - q[a, :2])
            for _, h, _, a in SIDES
        )
        candidate = bool(
            result.get("label") not in {"seated", "squatting", "standing_up"}
            and min(knees) > 0.85
            and leg_support
            and shoulder_drop >= 0.30
            and hip_travel <= 0.15
            and foot_travel <= 0.10
            and 0.10 < compression < 0.62
        )
        if candidate:
            self.candidates.append(t)
            while self.candidates and t - self.candidates[0] > 0.6:
                self.candidates.popleft()
        else:
            self.candidates.clear()
        confirmed = len(self.candidates) >= 3 and t - self.candidates[0] >= 0.25
        info.update(
            ready=True,
            candidate=candidate,
            confirmed=confirmed,
            code="confirmed" if confirmed else "confirming" if candidate else "reference_ready",
            reference_age_s=round(t - rt, 4),
            shoulder_drop=round(shoulder_drop, 4),
            hip_travel=round(hip_travel, 4),
            foot_travel=round(foot_travel, 4),
            compression=round(compression, 4),
            minimum_knee=round(min(knees), 4),
        )
        return info
