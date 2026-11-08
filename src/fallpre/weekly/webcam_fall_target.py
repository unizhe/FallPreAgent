"""Current-image association for the fall branch, separate from action tracking.

Admission retains the existing strong-pose criteria. Once admitted, torso
continuity can follow a rotating or shortened body without requiring knees or
ankles. A historical box may request current-image recovery, never supply pose.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from .webcam_target import candidate, match_cost


@dataclass(frozen=True)
class FallTargetConfig:
    admission_observations: int = 4
    admission_duration_s: float = .30
    max_gap_s: float = .50
    joint_confidence: float = .30
    minimum_torso_length_px: float = 12.0
    minimum_area_ratio: float = .15
    maximum_area_ratio: float = 6.0
    maximum_box_center_distance: float = .75
    minimum_box_overlap: float = .02
    close_box_center_distance: float = .40
    minimum_torso_scale_ratio: float = .25
    maximum_torso_scale_ratio: float = 4.0
    maximum_torso_center_distance: float = .65
    maximum_joint_distance: float = 1.75
    duplicate_box_overlap: float = .65
    duplicate_torso_distance: float = .20


def _overlap(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection, 1))


class LiveFallSubject:
    """``prepare`` selects raw evidence; optional ``finish`` uses this same frame.

    Recovery candidates must have ``recovery_verified=True`` from a current-image
    verifier. This class independently applies identity and ambiguity checks.
    Neither recovery nor a box can initialize an identity or extend an expired one.
    """

    def __init__(self, config: FallTargetConfig | None = None):
        self.config = config or FallTargetConfig()
        self._last_t = None
        self._last_seen = None
        self._shape = None
        self._anchor = None
        self._pending_admission = None
        self._pending_recovery = None
        self._epoch = 0

    def configuration(self):
        return asdict(self.config)

    def _forget(self):
        self._last_seen = None
        self._anchor = None
        self._pending_admission = None
        self._pending_recovery = None

    def _box(self, person):
        try:
            box = np.asarray(person.get('box', []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                return None
            return box
        except (ValueError, TypeError):
            return None

    def _torso(self, person, width, height):
        box = self._box(person)
        try:
            points = np.asarray(person.get('keypoints', []), dtype=float)
            if box is None or points.shape != (17, 3):
                return None
            torso = points[[5, 6, 11, 12]]
            valid = (np.isfinite(torso).all(axis=1) & (torso[:, 2] >= self.config.joint_confidence)
                     & (torso[:, 0] >= 0) & (torso[:, 0] < width)
                     & (torso[:, 1] >= 0) & (torso[:, 1] < height))
            margin = .10 * (box[2:] - box[:2])
            valid &= ((torso[:, :2] >= box[:2] - margin) & (torso[:, :2] <= box[2:] + margin)).all(axis=1)
            if not valid.all():
                return None
            shoulder, hip = torso[:2, :2].mean(axis=0), torso[2:, :2].mean(axis=0)
            scale = float(np.linalg.norm(hip - shoulder))
            if scale < self.config.minimum_torso_length_px:
                return None
            return dict(box=box, torso=torso[:, :2].copy(), center=(hip + shoulder) / 2, scale=scale)
        except (ValueError, TypeError):
            return None

    def _near(self, box):
        old = self._anchor['box']
        ratio = float(np.prod(box[2:] - box[:2]) / np.prod(old[2:] - old[:2]))
        distance = float(np.linalg.norm((box[:2] + box[2:] - old[:2] - old[2:]) / 2)
                         / max(np.linalg.norm(old[2:] - old[:2]), 1))
        cfg = self.config
        return (cfg.minimum_area_ratio <= ratio <= cfg.maximum_area_ratio
                and distance <= cfg.maximum_box_center_distance
                and (_overlap(old, box) >= cfg.minimum_box_overlap
                     or distance <= cfg.close_box_center_distance))

    def _continuous(self, torso):
        if torso is None or not self._near(torso['box']):
            return False
        old, cfg = self._anchor, self.config
        ratio = torso['scale'] / old['scale']
        joint_distance = np.median(np.linalg.norm(torso['torso'] - old['torso'], axis=1))
        joint_distance /= max(torso['scale'], old['scale'])
        diagonal = max(np.linalg.norm(old['box'][2:] - old['box'][:2]),
                       np.linalg.norm(torso['box'][2:] - torso['box'][:2]), 1)
        center_distance = np.linalg.norm(torso['center'] - old['center']) / diagonal
        return bool(cfg.minimum_torso_scale_ratio <= ratio <= cfg.maximum_torso_scale_ratio
                    and joint_distance <= cfg.maximum_joint_distance
                    and center_distance <= cfg.maximum_torso_center_distance)

    def _duplicates(self, a, b):
        if a is None or b is None or _overlap(a['box'], b['box']) < self.config.duplicate_box_overlap:
            return False
        delta = np.median(np.linalg.norm(a['torso'] - b['torso'], axis=1))
        return bool(delta / max(a['scale'], b['scale']) <= self.config.duplicate_torso_distance)

    def _distinct(self, candidates):
        distinct = []
        for item in candidates:
            if not any(self._duplicates(item['torso'], previous['torso']) for previous in distinct):
                distinct.append(item)
        return distinct

    def _result(self, state, code, reason, person=None, raw_index=None, source='none', recovery_allowed=False):
        return dict(person=deepcopy(person), source=source, raw_index=raw_index,
                    target_epoch=self._epoch if self._anchor is not None else None,
                    state=state, reason_code=code, reason=reason, recovery_allowed=recovery_allowed,
                    anchor_box=self._anchor['box'].tolist() if self._anchor is not None else None)

    def _accept(self, item, t, source):
        self._anchor = item['torso']
        self._last_seen = t
        self._pending_recovery = None
        return self._result('locked', 'current_torso_continuity',
                            '当前双肩双髋与已确认目标连续，允许身体旋转且不要求膝踝',
                            item['person'], item['index'] if source == 'raw' else None, source)

    def prepare(self, persons, t, width, height, camera_stable=True):
        if not math.isfinite(t) or t < 0 or self._last_t is not None and t <= self._last_t:
            raise ValueError('跌倒目标源时间必须有限、非负并严格递增')
        if width <= 0 or height <= 0:
            raise ValueError('图像尺寸必须为正')
        if self._last_t is not None and t - self._last_t > self.config.max_gap_s + 1e-9:
            self._forget()
        self._last_t = t
        self._pending_recovery = None
        if self._shape is not None and self._shape != (width, height):
            self._forget()
        self._shape = width, height
        if camera_stable is not True:
            self._forget()
            return self._result('unobservable', 'camera_quality_unavailable',
                                '相机稳定性未知或已移动；不延续身份，也不执行补检')
        if self._last_seen is not None and t - self._last_seen > self.config.max_gap_s + 1e-9:
            self._forget()

        entries = [dict(index=i, person=p, box=self._box(p), torso=self._torso(p, width, height))
                   for i, p in enumerate(persons)]
        if self._anchor is None:
            strong = []
            for item in entries:
                if item['torso'] is None:
                    continue
                try:
                    body = candidate(item['person'], width, height)
                except (ValueError, TypeError, KeyError):
                    continue
                if body is not None and body['strong']:
                    strong.append(dict(item, acquisition=body))
            strong = self._distinct(strong)
            if len(strong) != 1:
                self._pending_admission = None
                return self._result('ambiguous' if len(strong) > 1 else 'searching',
                                    'multiple_strong_candidates' if len(strong) > 1 else 'no_strong_candidate',
                                    '需要唯一可靠人体连续确认；恢复候选不能创建新身份')
            chosen = strong[0]
            if (self._pending_admission is None
                    or match_cost(self._pending_admission['body'], chosen['acquisition']) is None):
                self._pending_admission = dict(body=chosen['acquisition'], since=t, count=1)
            else:
                self._pending_admission.update(body=chosen['acquisition'], count=self._pending_admission['count'] + 1)
            pending = self._pending_admission
            if (pending['count'] < self.config.admission_observations
                    or t - pending['since'] < self.config.admission_duration_s - 1e-9):
                return self._result('confirming', 'initial_confirmation', '唯一强候选正在确认：至少4次、0.3秒')
            self._epoch += 1
            self._pending_admission = None
            return self._accept(chosen, t, 'raw')

        nearby = self._distinct([item for item in entries if item['box'] is not None and self._near(item['box'])])
        if len(nearby) > 1:
            return self._result('ambiguous', 'nearby_candidates_compete',
                                '附近存在不同人体候选；不借用其他人，也不以补检覆盖身份歧义')
        if nearby and self._continuous(nearby[0]['torso']):
            return self._accept(nearby[0], t, 'raw')
        self._pending_recovery = dict(t=t, width=width, height=height)
        return self._result('missing', 'current_torso_unavailable',
                            '近期目标缺少可关联躯干，仅允许核对本帧图像；历史框不是观测', recovery_allowed=True)

    def finish(self, recovered, t, width, height):
        pending = self._pending_recovery
        if pending is None or pending != dict(t=t, width=width, height=height):
            raise ValueError('补检必须属于刚prepare且允许恢复的同一源帧')
        self._pending_recovery = None
        # prepare already checked the source-time TTL. Inference wall time may
        # not turn an old-frame result into evidence at a later source time.
        candidates = []
        for p in recovered:
            competitor = p.get('recovery_identity_competitor') is True
            verified = p.get('recovery_verified') is True and not competitor
            if not verified and not competitor:
                continue
            box = self._box(p)
            if box is None or not self._near(box):
                continue
            torso = self._torso(p, width, height)
            candidates.append(dict(person=p, index=None, torso=torso, verified=verified))
        # Unverified current-image competitors can veto an identity guess but
        # can never be selected. Prefer verified observations when a strict
        # duplicate is supplied in both roles, regardless of caller ordering.
        distinct = self._distinct(sorted(candidates, key=lambda item: not item['verified']))
        if len(distinct) > 1:
            return self._result('ambiguous', 'recovered_candidates_compete',
                                '补检发现多个可关联人体，身份仍有歧义')
        if not distinct or not distinct[0]['verified'] or not self._continuous(distinct[0]['torso']):
            return self._result('missing', 'recovery_not_verified',
                                '本帧补检未提供唯一、通过多视角验证且身份连续的双肩双髋')
        return self._accept(distinct[0], t, 'recovered')
