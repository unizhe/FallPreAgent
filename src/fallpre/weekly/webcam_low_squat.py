"""Conservative low-squat evidence when a frontal knee projects almost straight.

Image angles are not anatomical joint angles. This route needs an independently
visible bent leg, both hips close to the feet, an upright upper body and a real
stationary window. It deliberately excludes the seat-height overlap region.
"""

from collections import deque

import numpy as np

from .action_v2 import angle


SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))
JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)


class ProjectedLowSquat:
    def __init__(self):
        self.samples = deque()
        self.epoch = None
        self.last_t = None

    def update(self, t, epoch, person, result, camera_stable=True):
        info = dict(candidate=False, confirmed=False, code='geometry_insufficient')
        if (epoch != self.epoch or self.last_t is not None
                and not 0 < t - self.last_t <= .35):
            self.samples.clear()
        self.epoch, self.last_t = epoch, t
        try:
            p = np.asarray(person['kpts'], dtype=float)
        except (TypeError, KeyError, ValueError):
            p = np.empty((0, 3))
        if (camera_stable is not True or epoch is None or p.shape != (17, 3)
                or not np.isfinite(p).all() or not np.all(p[list(JOINTS), 2] >= .80)
                or result.get('raw_label') == 'falling'
                or result.get('label') in {'seated', 'standing_up', 'bending'}):
            self.samples.clear()
            info['code'] = 'current_evidence_unavailable_or_conflicting'
            return info

        torsos = [np.linalg.norm(p[s, :2] - p[h, :2]) for s, h, _, _ in SIDES]
        scale = float(np.median(torsos))
        knees = [angle(p[h, :2], p[k, :2], p[a, :2]) for _, h, k, a in SIDES]
        heights = [(p[a, 1] - p[h, 1]) / max(scale, 1) for _, h, _, a in SIDES]
        # A strongly flexed visible leg plus a foreshortened opposite thigh is
        # useful only well below the existing seated-support height range.
        candidate = bool(scale >= 24 and min(knees) < .40
            and min(heights) > .15 and max(heights) < .60
            and all((p[h, 1] - p[s, 1]) / max(torso, 1) > .85
                    and np.linalg.norm(p[k, :2] - p[h, :2]) / scale < .50
                    and -.45 < (p[k, 1] - p[h, 1]) / scale < .25
                    for (s, h, k, _), torso in zip(SIDES, torsos)))
        info.update(candidate=candidate, minimum_knee=round(min(knees), 4),
                    maximum_hip_ankle_height=round(max(heights), 4))
        if not candidate:
            self.samples.clear()
            return info
        self.samples.append((t, p.copy(), scale))
        while self.samples and t - self.samples[0][0] > .55:
            self.samples.popleft()
        _, reference, reference_scale = self.samples[0]
        feet_motion = max(float(np.linalg.norm(row[1][j, :2] - reference[j, :2]) / reference_scale)
                          for row in self.samples for j in (15, 16))
        body_motion = max(float(np.linalg.norm(row[1][j, :2] - reference[j, :2]) / reference_scale)
                          for row in self.samples for j in (5, 6, 11, 12))
        stationary = feet_motion <= .12 and body_motion <= .08
        if not stationary:
            self.samples.clear()
            self.samples.append((t, p.copy(), scale))
        duration = t - self.samples[0][0]
        confirmed = stationary and len(self.samples) >= 3 and duration >= .25
        info.update(confirmed=confirmed, samples=len(self.samples), duration_s=round(duration, 4),
                    feet_motion=round(feet_motion, 4), body_motion=round(body_motion, 4),
                    code='confirmed' if confirmed else 'confirming' if stationary else 'still_moving')
        return info
