"""Causal display evidence for a selected person; independent of action labels.

These fixed prototype thresholds are not a trained fall classifier. An alarm-like
display requires observed upright history, descent and sustained low geometry.
Missing observations are never interpolated or counted as positive evidence.
"""
from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class FallDisplayConfig:
    keypoint_conf: float = .30
    max_gap_s: float = .50
    upright_reference_s: float = .50
    upright_tilt_deg: float = 25.0
    upright_aspect_max: float = .80
    standing_joint_angle_deg: float = 150.0
    low_tilt_deg: float = 55.0
    low_aspect_ratio: float = 1.10
    drop_norm: float = .35
    collapsed_height_ratio: float = .60
    collapsed_aspect_ratio: float = .90
    collapsed_tilt_deg: float = 35.0
    candidate_drop_norm: float = .20
    candidate_tilt_deg: float = 40.0
    candidate_aspect_ratio: float = .85
    candidate_height_ratio: float = .85
    confirm_duration_s: float = 1.0
    recovery_duration_s: float = 1.0


def _angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    length = math.hypot(*u) * math.hypot(*v)
    if length < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / length))))


class LiveFallDetector:
    """One selected identity at a time, with separately retained session events."""

    def __init__(self, config: FallDisplayConfig | None = None):
        self.config = config or FallDisplayConfig()
        self._last_t = None
        self._epoch = None
        self._shape = None
        self._last_display_key = None
        self.events = []
        self.confirmed_events = []
        self.statistics = dict(updates=0, observable_updates=0, unobservable_updates=0,
                               candidates=0, confirmed=0, recoveries=0, resets=0)
        self._clear_evidence()

    def _clear_evidence(self):
        self._last_valid_t = None
        self._previous_low = False
        self._previous_upright = False
        self._upright_duration = 0.0
        self._standing = deque()
        self._reference = None
        self._candidate_since = None
        self._low_duration = 0.0
        self._recovery_duration = 0.0

    def _reset(self, reason, t):
        self._clear_evidence()
        self.statistics['resets'] += 1
        self.events.append(dict(kind='reset', t_s=t, target_epoch=self._epoch, reason=reason))

    def _features(self, person, width, height):
        points, box = person.get('keypoints', []), person.get('box', [])
        if len(points) != 17 or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = map(float, box)
            if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
                return None
            valid = [len(p) == 3 and all(math.isfinite(float(v)) for v in p)
                     and 0 <= float(p[0]) < width and 0 <= float(p[1]) < height
                     and float(p[2]) >= self.config.keypoint_conf for p in points]
            if not all(valid[i] for i in (5, 6, 11, 12)):
                return None
            shoulder = tuple((float(points[5][i]) + float(points[6][i])) / 2 for i in (0, 1))
            hip = tuple((float(points[11][i]) + float(points[12][i])) / 2 for i in (0, 1))
            dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
            if math.hypot(dx, dy) < 1e-6:
                return None
            tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
            aspect = (x2 - x1) / (y2 - y1)
            # Ankles are only needed to establish reliable standing history.
            # Low-pose evidence itself needs shoulders and hips, not feet.
            extended_leg = any(
                all(valid[i] for i in (s, h, k, a)) and points[h][1] < points[k][1] < points[a][1]
                and _angle(points[s], points[h], points[k]) >= self.config.standing_joint_angle_deg
                and _angle(points[h], points[k], points[a]) >= self.config.standing_joint_angle_deg
                for s, h, k, a in ((5, 11, 13, 15), (6, 12, 14, 16)))
        except (TypeError, ValueError, IndexError):
            return None
        return dict(height=y2 - y1, hip_y=hip[1], torso_tilt_deg=tilt, aspect_ratio=aspect,
                    upright=dy > 0 and tilt <= self.config.upright_tilt_deg
                    and aspect < self.config.upright_aspect_max and extended_leg)

    def _result(self, state, reason, t, target_epoch, observed, metrics, start):
        names = dict(normal=('OK', '未见跌倒证据'), suspected=('LYING?', '疑似跌倒'),
                     confirmed=('FALL', '已确认跌倒事件'), unobservable=('UNKNOWN', '暂不可判断'))
        label, text = names[state]
        key = state, target_epoch
        if key != self._last_display_key:
            self.events.append(dict(kind='state', t_s=t, state=state, label=label,
                                    target_epoch=target_epoch, reason=reason))
            self._last_display_key = key
        latest = self.confirmed_events[-1] if self.confirmed_events else None
        current = next((e for e in reversed(self.confirmed_events)
                        if e['target_epoch'] == target_epoch), None) if target_epoch is not None else None
        return dict(state=state, label=label, text=text, reason=reason, observed=observed,
                    target_epoch=target_epoch, candidate_duration_s=round(self._low_duration, 4),
                    candidate_since_s=self._candidate_since,
                    candidate_age_s=round(t - self._candidate_since, 4) if self._candidate_since is not None else 0,
                    confirmed_at_s=latest['t_s'] if latest else None,
                    event_id=latest['event_id'] if latest else None,
                    event_target_epoch=latest['target_epoch'] if latest else None,
                    current_target_confirmed=current is not None,
                    metrics=metrics, transitions=deepcopy(self.events[start:]))

    def update(self, person, t, width, height, target_epoch, raw_label='unknown',
               camera_stable=True, quality_reason=None):
        if not math.isfinite(t) or t < 0 or self._last_t is not None and t <= self._last_t:
            raise ValueError('跌倒分支的源时间必须有限、非负并严格递增')
        if width <= 0 or height <= 0:
            raise ValueError('图像尺寸必须为正')
        start = len(self.events)
        self._last_t = t
        self.statistics['updates'] += 1
        if target_epoch is not None and target_epoch != self._epoch:
            self._reset('target_changed', t)
            self._epoch = target_epoch
        if self._shape is not None and self._shape != (width, height):
            self._reset('image_size_changed', t)
        self._shape = width, height
        if self._last_valid_t is not None and t - self._last_valid_t > self.config.max_gap_s + 1e-9:
            self._reset('observation_gap', t)
        if camera_stable is not True:
            self._reset('camera_motion' if camera_stable is False else 'camera_stability_unknown', t)
            person = None
            reason = quality_reason or '相机移动或稳定性未知，已清除姿态参考'
        elif person is None or target_epoch is None:
            reason = '当前目标不可见，缺失时间不计入确认'
        else:
            reason = '肩髋关键点不足或人体框无效，缺失时间不计入确认'
        features = self._features(person, width, height) if person is not None and target_epoch is not None else None
        if features is None:
            self._previous_low = self._previous_upright = False
            self._upright_duration = self._recovery_duration = 0.0
            self._standing.clear()
            self.statistics['unobservable_updates'] += 1
            return self._result('unobservable', reason, t, target_epoch, False, {}, start)

        self.statistics['observable_updates'] += 1
        dt = 0 if self._last_valid_t is None else t - self._last_valid_t
        self._last_valid_t = t
        c = self.config
        drop = ((features['hip_y'] - self._reference['hip_y']) / self._reference['height']
                if self._reference else None)
        height_ratio = features['height'] / self._reference['height'] if self._reference else None
        sideways = features['torso_tilt_deg'] >= c.low_tilt_deg and features['aspect_ratio'] >= c.low_aspect_ratio
        # A near-vertical seated torso cannot satisfy either route. Frontal low
        # posture still requires orientation evidence, not height change alone.
        collapsed = (drop is not None and drop >= c.drop_norm and height_ratio <= c.collapsed_height_ratio
                     and features['aspect_ratio'] >= c.collapsed_aspect_ratio
                     and features['torso_tilt_deg'] >= c.collapsed_tilt_deg)
        low = sideways or collapsed
        changing = (drop is not None and drop >= c.candidate_drop_norm
                    and features['torso_tilt_deg'] >= c.candidate_tilt_deg
                    and features['aspect_ratio'] >= c.candidate_aspect_ratio
                    and height_ratio <= c.candidate_height_ratio)
        if low:
            if self._previous_low:
                self._low_duration += dt
        else:
            self._low_duration = 0.0
        upright = features['upright'] and not low
        self._upright_duration = self._upright_duration + dt if upright and self._previous_upright else 0.0
        self._recovery_duration = (self._recovery_duration + dt
                                   if upright and self._previous_upright else 0.0)
        if self._candidate_since is None and (low or changing):
            self._candidate_since = t
            self.statistics['candidates'] += 1
            self.events.append(dict(kind='candidate', t_s=t, target_epoch=self._epoch,
                                    reason='low_pose' if low else 'descending_pose'))
        if self._candidate_since is not None and upright and self._recovery_duration >= c.recovery_duration_s - 1e-9:
            self._candidate_since = None
            self._low_duration = 0.0
            self.statistics['recoveries'] += 1
            self.events.append(dict(kind='recovery', t_s=t, target_epoch=self._epoch))
        if self._candidate_since is None and upright:
            self._standing.append((t, features['height'], features['hip_y']))
            while len(self._standing) > 1 and t - self._standing[1][0] >= c.upright_reference_s:
                self._standing.popleft()
            if self._upright_duration >= c.upright_reference_s - 1e-9:
                self._reference = dict(height=median(x[1] for x in self._standing),
                                       hip_y=median(x[2] for x in self._standing))
        elif not upright:
            self._standing.clear()
        self._previous_low, self._previous_upright = low, upright
        current_event = next((e for e in reversed(self.confirmed_events) if e['target_epoch'] == self._epoch), None)
        if (current_event is None and low and self._reference is not None and drop is not None
                and drop >= c.drop_norm and self._low_duration >= c.confirm_duration_s - 1e-9):
            current_event = dict(kind='confirmed', event_id=f'fall-{len(self.confirmed_events) + 1:04d}',
                                 t_s=t, target_epoch=self._epoch, candidate_since_s=self._candidate_since,
                                 observed_low_duration_s=round(self._low_duration, 4))
            self.confirmed_events.append(current_event)
            self.events.append(deepcopy(current_event))
            self.statistics['confirmed'] += 1
        metrics = dict(features, vertical_drop_norm=drop, height_ratio=height_ratio,
                       low_pose=low, had_upright=self._reference is not None,
                       action_falling_support=raw_label == 'falling', recovery_duration_s=self._recovery_duration)
        metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        if current_event:
            state, reason = 'confirmed', '已观察到直立后下降和持续低姿态；事件在本会话保留'
        elif self._candidate_since is not None:
            state = 'suspected'
            reason = ('当前为低姿态，但缺少此前可靠直立证据，不能确认跌倒' if self._reference is None else
                      '低姿态或下降仍在观察，等待持续证据或可靠直立恢复')
        else:
            state, reason = 'normal', '当前没有足够的跌倒候选证据'
        return self._result(state, reason, t, target_epoch, True, metrics, start)

    def export(self):
        return deepcopy(dict(config=asdict(self.config), events=self.events,
                             confirmed_events=self.confirmed_events, statistics=self.statistics))
