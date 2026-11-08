"""Causal RGB-classification evidence, independent of the pose/action branches.

The caller supplies current-image standing/sitting/fallen detections. Pose only
witnesses initial upright posture; reference and subsequent boxes always come
from this same RGB detector. Boxes never become joints or metric body heights.
This cannot distinguish a persistently misclassified sit/bend from an identical
fallen detection sequence, nor establish identity from boxes in a crowded scene.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class RgbFallConfig:
    proposal_confidence: float = .25
    class_confidence: float = .60
    max_gap_s: float = .50
    admission_observations: int = 4
    admission_duration_s: float = .50
    pose_box_overlap: float = .50
    pose_joint_confidence: float = .30
    pose_upright_tilt_deg: float = 25.0
    pose_upright_aspect_max: float = .80
    pose_standing_angle_deg: float = 150.0
    minimum_box_overlap: float = .05
    maximum_center_distance: float = .60
    minimum_area_ratio: float = .40
    maximum_area_ratio: float = 2.50
    border_margin_px: float = 2.0
    center_drop_norm: float = .35
    maximum_height_ratio: float = .80
    confirm_duration_s: float = 1.0


def _box(value):
    try:
        if len(value) != 4:
            return None
        box = tuple(float(v) for v in value)
        return box if all(map(math.isfinite, box)) and box[2] > box[0] and box[3] > box[1] else None
    except (TypeError, ValueError):
        return None


def _overlap(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def _angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*u) * math.hypot(*v)
    if norm < 1e-6:
        return 0.
    return math.degrees(math.acos(max(-1., min(1., (u[0] * v[0] + u[1] * v[1]) / norm))))


class LiveRgbFall:
    """One conservative RGB identity, with separately retained session events.

    Initial fallen observations are provisional yellow observations with no
    identity epoch. Only matched, reliable standing observations establish an
    epoch/reference. Later pose loss or pose-epoch changes do not affect RGB
    identity. All timings use strictly increasing source timestamps.
    """

    def __init__(self, config: RgbFallConfig | None = None):
        self.config = config or RgbFallConfig()
        self._last_t = None
        self._shape = None
        self._next_epoch = 0
        self._last_display_key = None
        self.events = []
        self.confirmed_events = []
        self.statistics = dict(updates=0, observable_updates=0, unobservable_updates=0,
                               candidates=0, confirmed=0, recoveries=0, resets=0, acquisitions=0)
        self._clear()

    def _clear(self):
        self._epoch = None
        self._anchor = None
        self._last_seen = None
        self._pending = []
        self._pending_pose_epoch = None
        self._reference = None
        self._reference_pose_epoch = None
        self._candidate_since = None
        self._low_duration = 0.
        self._previous_qualifying = False

    def _reset(self, reason, t):
        if self._anchor is not None or self._pending or self._candidate_since is not None:
            self.events.append(dict(kind='reset', t_s=t, target_epoch=self._epoch, reason=reason))
            self.statistics['resets'] += 1
        self._clear()

    def _continuous(self, old, new):
        old_width, old_height = old[2] - old[0], old[3] - old[1]
        ratio = ((new[2] - new[0]) * (new[3] - new[1])) / (old_width * old_height)
        distance = math.hypot((new[0] + new[2] - old[0] - old[2]) / 2,
                              (new[1] + new[3] - old[1] - old[3]) / 2)
        c = self.config
        return (c.minimum_area_ratio <= ratio <= c.maximum_area_ratio
                and distance / math.hypot(old_width, old_height) <= c.maximum_center_distance
                and _overlap(old, new) >= c.minimum_box_overlap)

    def _near(self, old, new):
        # A weak proposal may veto identity, but may never refresh it. Avoid
        # rejecting a competitor merely because its partial box has a different
        # area from the established target's box.
        distance = math.hypot((new[0] + new[2] - old[0] - old[2]) / 2,
                              (new[1] + new[3] - old[1] - old[3]) / 2)
        diagonal = math.hypot(old[2] - old[0], old[3] - old[1])
        return (_overlap(old, new) >= self.config.minimum_box_overlap
                or distance / diagonal <= self.config.maximum_center_distance)

    def _pose_upright(self, person, rgb_box, width, height):
        if not isinstance(person, dict):
            return False
        box, points, c = _box(person.get('box')), person.get('keypoints', []), self.config
        if box is None or _overlap(box, rgb_box) < c.pose_box_overlap:
            return False
        try:
            if len(points) != 17:
                return False
            points = [tuple(float(v) for v in p) for p in points]
            valid = [len(p) == 3 and all(map(math.isfinite, p)) and p[2] >= c.pose_joint_confidence
                     and 0 <= p[0] < width and 0 <= p[1] < height for p in points]
            if not all(valid[i] for i in (5, 6, 11, 12)):
                return False
            shoulder = tuple((points[5][i] + points[6][i]) / 2 for i in (0, 1))
            hip = tuple((points[11][i] + points[12][i]) / 2 for i in (0, 1))
            dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
            upright = (dy > 0 and math.degrees(math.atan2(abs(dx), abs(dy))) <= c.pose_upright_tilt_deg
                       and (box[2] - box[0]) / (box[3] - box[1]) < c.pose_upright_aspect_max)
            extended = any(all(valid[i] for i in (s, h, k, a))
                           and points[h][1] < points[k][1] < points[a][1]
                           and _angle(points[s], points[h], points[k]) >= c.pose_standing_angle_deg
                           and _angle(points[h], points[k], points[a]) >= c.pose_standing_angle_deg
                           for s, h, k, a in ((5, 11, 13, 15), (6, 12, 14, 16)))
            return upright and extended
        except (TypeError, ValueError, IndexError):
            return False

    def _output(self, state, reason, reason_code, t, start, detection=None, index=None, metrics=None):
        names = dict(normal=('OK', '未见跌倒证据'), suspected=('LYING?', '疑似跌倒'),
                     confirmed=('FALL', '已确认跌倒事件'), unobservable=('UNKNOWN', '暂不可判断'))
        label, text = names[state]
        observed = detection is not None
        self.statistics['observable_updates' if observed else 'unobservable_updates'] += 1
        key = state, self._epoch
        if key != self._last_display_key:
            self.events.append(dict(kind='state', t_s=t, state=state, label=label,
                                    target_epoch=self._epoch, reason=reason, reason_code=reason_code))
            self._last_display_key = key
        latest = self.confirmed_events[-1] if self.confirmed_events else None
        current = next((e for e in reversed(self.confirmed_events)
                        if e['target_epoch'] == self._epoch), None) if self._epoch is not None else None
        fall = dict(state=state, label=label, text=text, reason=reason, reason_code=reason_code,
                    observed=observed, target_epoch=self._epoch,
                    candidate_duration_s=round(self._low_duration, 4), candidate_since_s=self._candidate_since,
                    candidate_age_s=round(t - self._candidate_since, 4) if self._candidate_since is not None else 0.,
                    confirmed_at_s=latest['t_s'] if latest else None, event_id=latest['event_id'] if latest else None,
                    event_target_epoch=latest['target_epoch'] if latest else None,
                    current_target_confirmed=current is not None, metrics=metrics or {},
                    transitions=deepcopy(self.events[start:]))
        selection_state = ('locked' if self._epoch is not None else 'provisional') if observed else 'unobservable'
        observation = dict(person=deepcopy(detection), source='rgb_fall' if observed else 'none',
                           selection=dict(raw_index=None, rgb_index=index, target_epoch=self._epoch, state=selection_state,
                                          reason=reason, reason_code=reason_code), recovery={})
        diagnostics = dict(reference_ready=self._reference is not None,
                           reference_pose_epoch=self._reference_pose_epoch,
                           admission_observations=len(self._pending),
                           admission_duration_s=round(self._pending[-1][0] - self._pending[0][0], 4)
                           if self._pending else 0., geometry_space='rgb_detector_image_box')
        return deepcopy(dict(fall=fall, observation=observation, diagnostics=diagnostics))

    def _missing(self, reason, code, t, start):
        # Keep at most a short reference/candidate, but the next observation
        # cannot integrate across the unobserved interval.
        self._previous_qualifying = False
        self._pending = []
        self._pending_pose_epoch = None
        return self._output('unobservable', reason, code, t, start)

    def update(self, detections, t, width, height, pose_person=None, pose_epoch=None,
               camera_stable=True, quality_reason=None):
        if not math.isfinite(t) or t < 0 or self._last_t is not None and t <= self._last_t:
            raise ValueError('RGB 跌倒分支的源时间必须有限、非负并严格递增')
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError('图像尺寸必须有限且为正')
        start = len(self.events)
        self.statistics['updates'] += 1
        if self._shape is not None and self._shape != (width, height):
            self._reset('image_size_changed', t)
        if self._last_seen is not None and t - self._last_seen > self.config.max_gap_s + 1e-9:
            self._reset('observation_gap', t)
        self._last_t, self._shape = t, (width, height)
        if camera_stable is not True:
            self._reset('camera_motion' if camera_stable is False else 'camera_stability_unknown', t)
            return self._missing(quality_reason or '相机稳定性未知或已移动，已清除 RGB 身份与直立参考',
                                 'camera_quality_unavailable', t, start)

        candidates = []
        for i, item in enumerate(detections or []):
            if not isinstance(item, dict) or item.get('label') not in ('standing', 'sitting', 'fallen'):
                continue
            box = _box(item.get('box'))
            try:
                confidence = float(item.get('confidence', 0))
            except (TypeError, ValueError):
                continue
            if box is None or not math.isfinite(confidence) or not self.config.proposal_confidence <= confidence <= 1:
                continue
            candidates.append((i, dict(box=list(box), confidence=confidence, label=item['label'], keypoints=[])))
        strong = [(i, item) for i, item in candidates if item['confidence'] >= self.config.class_confidence]
        weak = [(i, item) for i, item in candidates if item['confidence'] < self.config.class_confidence]
        if len(strong) > 1:
            # With boxes only there is no robust way to prove which person fell.
            # Even a distant additional person prevents transferring a reference.
            self._reset('multiple_rgb_candidates', t)
            return self._missing('存在多个 RGB 人体候选；已清除身份与参考，不按分数强选',
                                 'multiple_rgb_candidates', t, start)
        anchors = ([strong[0][1]['box']] if strong else []) + ([self._anchor] if self._anchor is not None else [])
        nearby_weak = [(i, item) for i, item in weak if any(self._near(anchor, item['box']) for anchor in anchors)]
        if (strong and nearby_weak) or len(nearby_weak) > 1:
            self._reset('weak_rgb_identity_competitor', t)
            return self._missing('附近存在低置信人体竞争框；它们只能否决身份，不能替代当前观测',
                                 'weak_rgb_identity_competitor', t, start)
        if not strong:
            return self._missing('当前无可信 RGB 人体观测；缺失时间不计入确认',
                                 'no_confident_rgb_detection', t, start)
        index, detection = strong[0]
        box, c = detection['box'], self.config
        if (box[0] <= c.border_margin_px or box[1] <= c.border_margin_px
                or box[2] >= width - c.border_margin_px or box[3] > height):
            return self._missing('人体框顶部或侧边出画，当前下降或身份依据不足', 'unreliable_clipped_box', t, start)
        bottom_clipped = box[3] >= height - c.border_margin_px
        if self._anchor is not None and not self._continuous(self._anchor, box):
            self._reset('box_identity_discontinuity', t)
        dt = t - self._last_seen if self._last_seen is not None else 0.
        self._anchor, self._last_seen = tuple(box), t

        upright = (not bottom_clipped and detection['label'] == 'standing'
                   and self._pose_upright(pose_person, box, width, height))
        if upright:
            if self._pending and pose_epoch != self._pending_pose_epoch:
                self._pending = []
            self._pending_pose_epoch = pose_epoch
            self._pending.append((t, box[3] - box[1], (box[1] + box[3]) / 2, box[1]))
            while len(self._pending) > c.admission_observations and t - self._pending[1][0] >= c.admission_duration_s:
                self._pending.pop(0)
            if (len(self._pending) >= c.admission_observations
                    and t - self._pending[0][0] >= c.admission_duration_s - 1e-9):
                self._reference = dict(height=median(p[1] for p in self._pending),
                                       center_y=median(p[2] for p in self._pending),
                                       top_y=median(p[3] for p in self._pending))
                self._reference_pose_epoch = pose_epoch
                if self._epoch is None:
                    self._next_epoch += 1
                    self._epoch = f'rgb-{self._next_epoch}'
                    self.statistics['acquisitions'] += 1
                    self.events.append(dict(kind='acquired', t_s=t, target_epoch=self._epoch))
        else:
            self._pending = []
            self._pending_pose_epoch = None

        drop = (((box[1] + box[3]) / 2 - self._reference['center_y']) / self._reference['height']
                if self._reference is not None else None)
        ratio = (box[3] - box[1]) / self._reference['height'] if self._reference is not None else None
        top_drop = ((box[1] - self._reference['top_y']) / self._reference['height']
                    if self._reference is not None else None)
        fallen = detection['label'] == 'fallen'
        # When the bottom is clipped, shrinking height is not evidence. A
        # visible top moving down and a clipped center moving down still are
        # current-image measurements; the center descent is a lower bound.
        descended = (drop is not None and drop >= c.center_drop_norm
                     and (top_drop >= c.center_drop_norm if bottom_clipped else ratio <= c.maximum_height_ratio))
        qualifying = fallen and descended
        if fallen:
            if self._candidate_since is None:
                self._candidate_since = t
                self.statistics['candidates'] += 1
                self.events.append(dict(kind='candidate', t_s=t, target_epoch=self._epoch, reason='rgb_fallen'))
            if qualifying and self._previous_qualifying:
                self._low_duration += dt
            elif not qualifying:
                self._low_duration = 0.
        else:
            if self._candidate_since is not None:
                self.events.append(dict(kind='recovery', t_s=t, target_epoch=self._epoch,
                                        reason='rgb_no_longer_fallen'))
                self.statistics['recoveries'] += 1
            self._candidate_since = None
            self._low_duration = 0.
        self._previous_qualifying = qualifying
        current = next((e for e in reversed(self.confirmed_events)
                        if e['target_epoch'] == self._epoch), None) if self._epoch is not None else None
        if (current is None and self._epoch is not None and qualifying
                and self._low_duration >= c.confirm_duration_s - 1e-9):
            current = dict(kind='confirmed', event_id=f'rgb-fall-{len(self.confirmed_events) + 1:04d}',
                           t_s=t, target_epoch=self._epoch, candidate_since_s=self._candidate_since,
                           observed_low_duration_s=round(self._low_duration, 4),
                           center_drop_norm=round(drop, 4), height_ratio=round(ratio, 4),
                           geometry_evidence=('bottom_clipped_fallen_with_visible_top_descent' if bottom_clipped
                                              else 'full_box_descent_and_height_reduction'))
            self.confirmed_events.append(current)
            self.events.append(deepcopy(current))
            self.statistics['confirmed'] += 1
        metrics = dict(rgb_label=detection['label'], rgb_confidence=detection['confidence'],
                       had_upright=self._reference is not None, pose_upright_witness=upright,
                       center_drop_norm=round(drop, 4) if drop is not None else None,
                       top_drop_norm=round(top_drop, 4) if top_drop is not None else None,
                       height_ratio=round(ratio, 4) if ratio is not None else None,
                       bottom_clipped=bottom_clipped, height_reduction_is_evidence=not bottom_clipped,
                       low_pose=fallen, qualifying_fallen=qualifying)
        if current:
            state = 'confirmed'
            code = ('bottom_clipped_fallen_with_visible_top_descent' if bottom_clipped and qualifying
                    else 'rgb_fall_confirmed')
            reason = ('框底部出画，仅用可见顶部与中心下降及持续倒地分类确认；框高收缩不作为证据'
                      if code == 'bottom_clipped_fallen_with_visible_top_descent' else
                      '已观察到直立后下降与持续倒地分类；事件在本会话保留')
        elif fallen:
            state, code = 'suspected', 'rgb_fallen_candidate'
            reason = ('当前为倒地分类，但缺少同模型的可靠直立参考，不能确认跌倒' if self._reference is None else
                      '倒地分类仍在观察，需当前框下降、收缩与至少一秒有效持续证据')
        elif self._epoch is None:
            state, code, reason = 'normal', 'reference_pending', '尚未建立直立参考；需要唯一站立框与可靠直立姿态连续对应'
        else:
            state, code, reason = 'normal', 'rgb_not_fallen', '当前 RGB 分类未见跌倒证据'
        return self._output(state, reason, code, t, start, detection, index, metrics)

    def export(self):
        return deepcopy(dict(config=asdict(self.config), events=self.events,
                             confirmed_events=self.confirmed_events, statistics=self.statistics))
