"""Visible setup diagnostics and a separate current lying-posture observer.

The observer cannot create/red-confirm a fall or reset its event lifecycle.
It reports what is visible when a fall's preceding transition was not observed.
"""

from collections import deque

import numpy as np

from .webcam_fall_geometry import pose_geometry
from .webcam_rgb_fall import _angle, _box, _overlap


class LiveSetup:
    def __init__(self):
        self.last_t = None
        self.gaps = deque(maxlen=20)
        self.lying_times = deque()
        self.box = None

    def update(self, result, t, rgb=None):
        dt = None if self.last_t is None else t - self.last_t
        self.last_t = t
        if dt is not None and dt > 0:
            self.gaps.append(dt)
        quality = result.get("quality") or {}
        stable = quality.get("camera_stable") is True
        persons = result.get("persons") or []
        person = persons[0] if len(persons) == 1 else None
        width, height = result["width"], result["height"]
        try:
            points = np.asarray(person["keypoints"], dtype=float)
            valid = (
                (points[:, 2] >= 0.5)
                & (points[:, 0] > 2)
                & (points[:, 0] < width - 2)
                & (points[:, 1] > 2)
                & (points[:, 1] < height - 2)
            )
            if points.shape != (17, 3) or not np.isfinite(points).all():
                raise ValueError("invalid pose")
        except (TypeError, KeyError, IndexError, ValueError):
            points, valid = np.zeros((17, 3)), np.zeros(17, bool)
        chains = [s for s in ((5, 11, 13, 15), (6, 12, 14, 16)) if all(valid[list(s)])]
        raw = (rgb or {}).get("raw_candidates") or {}
        falls = [f for f in raw.get("fallen", []) if _box(f.get("box")) is not None]
        people = [
            p
            for p in raw.get("persons", [])
            if _box(p.get("box")) is not None and p.get("confidence", 0) >= 0.25
        ]
        current_falls = [
            f
            for f in falls
            if person and f.get("confidence", 0) >= 0.75 and _overlap(f["box"], person["box"]) >= 0.60
        ]
        geometry = pose_geometry(person, person["box"], width, height) if person else None
        horizontal_legs = bool(
            chains
            and all(
                np.linalg.norm(points[a, :2] - points[h, :2]) >= 30
                and abs(points[a, 1] - points[h, 1]) < 0.50 * np.linalg.norm(points[a, :2] - points[h, :2])
                for _, h, _, a in chains
            )
        )
        # A side-lying person can keep both knees bent, so the hip-to-ankle
        # lines need not be horizontal. This separate route requires the
        # whole visible torso to be low beside the feet, not just a bent knee
        # or a horizontal torso (which also occurs during an ordinary bend).
        folded_side_lying = False
        if geometry and len(chains) == 2 and geometry["torso_confidence"] >= 0.75:
            length = geometry["length"]
            axis = (np.asarray(geometry["hip"]) - geometry["shoulder"]) / length
            folded_side_lying = bool(
                geometry["tilt"] >= 70
                and (person["box"][2] - person["box"][0]) >= 1.6 * (person["box"][3] - person["box"][1])
                and all(
                    abs(points[a, 1] - points[h, 1]) <= 0.90 * length
                    and abs(points[a, 1] - points[s, 1]) <= 0.90 * length
                    and np.dot(points[a, :2] - points[h, :2], axis) >= 0.35 * length
                    and np.linalg.norm(points[k, :2] - points[h, :2]) >= 0.25 * length
                    and np.linalg.norm(points[a, :2] - points[k, :2]) >= 0.25 * length
                    and 35 <= _angle(points[h], points[k], points[a]) <= 145
                    for s, h, k, a in chains
                )
            )
        current_person = bool(
            len(people) == 1
            and person
            and people[0].get("confidence", 0) >= 0.55
            and _overlap(people[0]["box"], person["box"]) >= 0.50
        )
        unclipped = bool(
            person
            and person["box"][0] > 2
            and person["box"][1] > 2
            and person["box"][2] < width - 2
            and person["box"][3] < height - 2
        )
        lying = bool(
            stable
            and current_person
            and len(current_falls) == 1
            and geometry
            and geometry["tilt"] >= 60
            and (horizontal_legs or folded_side_lying)
            and unclipped
            and (person["box"][2] - person["box"][0]) > 1.15 * (person["box"][3] - person["box"][1])
        )
        continuous = bool(
            self.box is not None
            and person
            and _overlap(self.box, person["box"]) >= 0.35
            and dt is not None
            and 0 < dt <= 0.35
        )
        if not lying or not continuous:
            self.lying_times.clear()
        self.box = list(person["box"]) if lying else None
        if lying:
            self.lying_times.append(t)
            while self.lying_times and t - self.lying_times[0] > 1.0:
                self.lying_times.popleft()
        observed = bool(len(self.lying_times) >= 5 and t - self.lying_times[0] >= 0.60)
        posture = dict(
            label="lying" if observed else "unknown",
            observed=observed,
            text="观察到倒卧（跌倒过程未确认）" if observed else "尚无独立倒卧姿态证据",
            target_index=0 if observed else None,
            source_time=t,
            samples=len(self.lying_times),
            new_event=False,
            event_id=None,
            evidence_type=("horizontal_legs" if horizontal_legs else "bent_knee_side_lying") if lying else None,
            reason=("当前横向躯干、低位屈膝腿部与独立倒地框持续支持；不计作新的跌倒事件"
                    if not horizontal_legs else "当前躯干与腿部横向、独立倒地框持续支持；不计作新的跌倒事件")
            if observed
            else "需同一当前人体、横向或屈膝侧卧姿态及连续图像检测共同支持",
        )
        median_gap = float(np.median(self.gaps)) if self.gaps else None
        check = dict(
            camera_stable=quality.get("camera_stable"),
            camera_code=quality.get("code"),
            camera_reason=quality.get("reason"),
            person_count=len(persons),
            complete_sides=len(chains),
            feet_visible=int(valid[[15, 16]].sum()),
            source_gap_s=dt,
            median_gap_s=median_gap,
            sampling_slow=bool(len(self.gaps) >= 5 and median_gap > 0.30),
            fall_reason=(result.get("fall") or {}).get("reason"),
            fall_code=(result.get("fall") or {}).get("reason_code"),
            fallen_candidates=len(falls),
            strongest_fallen_score=max((f.get("confidence", 0) for f in falls), default=0.0),
            weights_finetuned_on_personal_videos=False,
        )
        return check, posture
