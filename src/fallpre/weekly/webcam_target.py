"""Causal webcam subject association. No area ranking or missing-pose synthesis."""
from collections import deque

import numpy as np

BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SUPPORT = [13, 14, 15, 16]


def _extended_support(points):
    """Current high-confidence straight legs, measured without torso scale."""
    shins = []
    for h, k, a in [(11, 13, 15), (12, 14, 16)]:
        thigh = points[h, :2] - points[k, :2]
        shin = points[a, :2] - points[k, :2]
        thigh_length, shin_length = np.linalg.norm(thigh), np.linalg.norm(shin)
        if (thigh_length < 12 or shin_length < 18
                or np.dot(thigh, shin) / (thigh_length * shin_length) > -.95
                or shin[1] < .85 * shin_length):
            return None
        shins.append(shin_length)
    return float(np.median(shins))


def _reference_support(body):
    """Only full, uncompressed observations may refresh the fixed reference."""
    if body is None or not body['strong'] or body.get('folded_continuation') or body.get('support_bridge'):
        return None
    p = body['points']
    joints = [5, 6, 11, 12, 13, 14, 15, 16]
    if not all(body['valid'][j] and p[j, 2] >= .85 for j in joints):
        return None
    scale = _extended_support(p)
    if scale is None:
        return None
    for s, h in [(5, 11), (6, 12)]:
        torso = p[h, :2] - p[s, :2]
        if np.linalg.norm(torso) < .8 * scale or torso[1] < .6 * np.linalg.norm(torso):
            return None
    return scale


def _support_bridge(person, width, height, reference):
    """Associate an existing identity through a short torso projection failure.

    Hip projection may jump in a frontal bend. Both actual knees and ankles
    must still match a fixed, independently observed support reference. This
    only supplies identity; it neither confirms bending nor supplies joints.
    """
    p = np.asarray(person['keypoints'], dtype=float)
    box = np.asarray(person['box'], dtype=float)
    if (p.shape != (17, 3) or box.shape != (4,) or not np.isfinite(p).all()
            or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
            or person['confidence'] < .70):
        return None
    margin = .1 * (box[2:] - box[:2])
    valid = ((p[:, 2] >= .30) & (p[:, 0] > 0) & (p[:, 0] < width)
             & (p[:, 1] > 0) & (p[:, 1] < height)
             & ((p[:, :2] >= box[:2]-margin) & (p[:, :2] <= box[2:]+margin)).all(1))
    joints = [5, 6, 11, 12, 13, 14, 15, 16]
    if not all(valid[j] and p[j, 2] >= .85 for j in joints) or _extended_support(p) is None:
        return None
    q, scale = reference['points'], reference['shin_scale']
    delta = np.linalg.norm(p[SUPPORT, :2] - q[SUPPORT, :2], axis=1) / scale
    if max(delta[:2]) > .35 or max(delta[2:]) > .40:
        return None
    for k, a in [(13, 15), (14, 16)]:
        if not .80 <= np.linalg.norm(p[a, :2]-p[k, :2]) / np.linalg.norm(q[a, :2]-q[k, :2]) <= 1.30:
            return None
    return dict(points=p, valid=valid, box=box, scale=reference['body_scale'],
                strong=False, support_bridge=True, support_distance=float(np.median(delta)))


def candidate(person, width, height, reference=None):
    p = np.asarray(person['keypoints'], dtype=float)
    box = np.asarray(person['box'], dtype=float)
    if p.shape != (17, 3) or box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None
    valid = (np.isfinite(p).all(1) & (p[:, 2] >= .30) & (p[:, 0] > 0) & (p[:, 0] < width)
             & (p[:, 1] > 0) & (p[:, 1] < height))
    margin = .1 * (box[2:] - box[:2])
    valid &= ((p[:, :2] >= box[:2]-margin) & (p[:, :2] <= box[2:]+margin)).all(1)
    sides = [(s, h, k) for s, h, k in [(5, 11, 13), (6, 12, 14)]
             if valid[s] and valid[h] and valid[k]]
    if not sides:
        return None
    # Current shoulder/hip/knee chain, any orientation; lying bodies are not
    # rejected just because their shoulders are no longer above their hips.
    plausible = [(s, h, k) for s, h, k in sides
                 if np.linalg.norm(p[s, :2]-p[h, :2]) >= 12
                 and .15 <= np.linalg.norm(p[k, :2]-p[h, :2]) / np.linalg.norm(p[s, :2]-p[h, :2]) <= 2.5]
    if not plausible:
        # A folded torso can be foreshortened in 2D. Only continue an already
        # locked identity when both real legs remain confident and stationary.
        legs = [11, 12, 13, 14, 15, 16]
        if reference is None or not all(valid[j] and p[j, 2] >= .60
                                       and reference['valid'][j] for j in legs):
            return None
        old = reference['points']
        lengths = [np.linalg.norm(old[h, :2]-old[a, :2]) for h,a in [(11,15),(12,16)]]
        leg_scale = float(np.median(lengths))
        if leg_scale < 24 or any(not .75 <= np.linalg.norm(p[h,:2]-p[a,:2])/length <= 1.3
                                for (h,a),length in zip([(11,15),(12,16)], lengths)):
            return None
        delta = np.linalg.norm(p[legs,:2]-old[legs,:2],axis=1)/leg_scale
        if max(delta) > .25 or not any(np.linalg.norm(p[s,:2]-p[h,:2]) >= 8 for s,h,k in sides):
            return None
        return dict(points=p, valid=valid, box=box, scale=reference['scale'],
                    strong=False, folded_continuation=True, leg_distance=float(np.median(delta)))
    ids = [j for j in BODY if valid[j]]
    scale = float(np.median([np.linalg.norm(p[s, :2]-p[h, :2]) for s, h, k in plausible]))
    strong = (person['confidence'] >= .45 and len(ids) >= 6
              and float(np.mean(p[ids, 2])) >= .55)
    return dict(points=p, valid=valid, box=box, scale=scale, strong=bool(strong))


def match_cost(a, b):
    common = a['valid'] & b['valid']
    ids = [j for j in BODY if common[j]]
    if len(ids) < 3:
        return None
    scale = max(a['scale'], b['scale'])
    ratio = b['scale']/a['scale']
    if not b.get('support_bridge') and not .5 <= ratio <= 2:
        return None
    lo = np.maximum(a['box'][:2], b['box'][:2])
    hi = np.minimum(a['box'][2:], b['box'][2:])
    inter = float(np.prod(np.maximum(hi-lo, 0)))
    union = float(np.prod(a['box'][2:]-a['box'][:2])+np.prod(b['box'][2:]-b['box'][:2])-inter)
    iou = inter/max(union, 1)
    center = np.linalg.norm((a['box'][:2]+a['box'][2:]-b['box'][:2]-b['box'][2:])/2)/max(np.linalg.norm(a['box'][2:]-a['box'][:2]), 1)
    distance = float(np.median(np.linalg.norm(a['points'][ids, :2]-b['points'][ids, :2], axis=1))/scale)
    if b.get('folded_continuation') or b.get('support_bridge'):
        if iou < .5:
            return None
        distance = b['support_distance'] if b.get('support_bridge') else b['leg_distance']
    if distance > .8 or (iou < .15 and center > .35):
        return None
    return distance + .25*(1-iou)


class StableSubject:
    def __init__(self, folded_continuation=False):
        self.folded_continuation = folded_continuation
        self.last_t = None
        self.anchor = None
        self.last_seen = None
        self.pending = None
        self.epoch = 0
        self.support_history = deque()
        self.support_reference = None
        self.support_bridge_active = False

    def _clear_support(self):
        self.support_history.clear()
        self.support_reference = None
        self.support_bridge_active = False

    def _observe_support(self, body, t):
        if not self.folded_continuation:
            return
        scale = _reference_support(body)
        if scale is None:
            self.support_history.clear()
            return
        self.support_bridge_active = False
        self.support_history.append((t, body, scale))
        while self.support_history and t-self.support_history[0][0] > .5:
            self.support_history.popleft()
        if len(self.support_history) < 3 or t-self.support_history[0][0] < .20:
            return
        points = np.stack([row[1]['points'] for row in self.support_history])
        shin_scale = float(np.median([row[2] for row in self.support_history]))
        if np.max(np.linalg.norm(points[:, SUPPORT, :2]-points[0, SUPPORT, :2], axis=2)) / shin_scale > .20:
            return
        self.support_reference = dict(t=t, points=np.median(points, axis=0),
                                      shin_scale=shin_scale, body_scale=body['scale'])

    def update(self, persons, t, width, height):
        if not np.isfinite(t) or t < 0 or self.last_t is not None and t <= self.last_t:
            raise ValueError('摄像头时间戳必须递增，请重新启动')
        if self.last_t is not None and t-self.last_t > .35:
            self.anchor, self.pending = None, None
            self._clear_support()
        self.last_t = t
        reference = self.anchor if self.folded_continuation and self.anchor is not None and t-self.last_seen <= .5 else None
        candidates = [candidate(p, width, height, reference) for p in persons]
        support_reference = self.support_reference
        if self.folded_continuation and self.anchor is not None and t-self.last_seen <= .5:
            for i, body in enumerate(candidates):
                # Once this bridge was needed, weak torso matches cannot keep
                # it alive after the fixed reference expires. A fresh full
                # observation is required before returning to normal tracking.
                forced = self.support_bridge_active and _reference_support(body) is None
                if forced or body is None or match_cost(self.anchor, body) is None:
                    bridge = (_support_bridge(persons[i], width, height, support_reference)
                              if support_reference is not None and t-support_reference['t'] <= 1.5 else None)
                    if bridge is not None or forced:
                        candidates[i] = bridge
        status = [dict(index=i, eligible=c is not None, acquisition_ready=bool(c and c['strong']),
                       reason='骨架支持关联' if c else '当前肩髋膝链或几何证据不足') for i, c in enumerate(candidates)]
        def result(index, state, reason):
            return dict(target_index=index, target_epoch=self.epoch, state=state, reason=reason,
                        candidates=status, raw_count=len(persons))
        if self.anchor is not None:
            # Expire before considering a new image; a stale anchor cannot be
            # revived by a late match after the retention interval.
            if t-self.last_seen > .5:
                self.anchor, self.pending = None, None
                self._clear_support()
            else:
                matches = [(match_cost(self.anchor, c), i) for i, c in enumerate(candidates) if c]
                matches = sorted((cost, i) for cost, i in matches if cost is not None)
                if matches and (len(matches) == 1 or matches[1][0]-matches[0][0] >= .20):
                    cost, index = matches[0]
                    self.anchor, self.last_seen = candidates[index], t
                    if self.anchor.get('support_bridge'):
                        self.support_bridge_active = True
                    self._observe_support(self.anchor, t)
                    return result(index, 'locked', '当前骨架和位置连续匹配已锁定目标；其他原始候选保留显示')
                self.support_history.clear()
                return result(None, 'ambiguous' if matches else 'missing',
                              '附近候选竞争，暂不切换目标' if matches else '已锁定目标暂时缺少匹配；不借用其他人形候选')
        strong = [i for i, c in enumerate(candidates) if c and c['strong']]
        if len(strong) != 1:
            self.pending = None
            return result(None, 'ambiguous' if len(strong)>1 else 'searching',
                          '多个候选均有完整人体证据，暂不能确认本人' if len(strong)>1 else '等待检测置信度与可见骨架共同支持的候选')
        index = strong[0]
        c = candidates[index]
        if self.pending is None or match_cost(self.pending['body'], c) is None:
            self.pending = dict(body=c, since=t, count=1)
        else:
            self.pending.update(body=c, count=self.pending['count']+1)
        if self.pending['count'] < 4 or t-self.pending['since'] < .30-1e-7:
            return result(None, 'confirming', '唯一可靠候选正在连续确认（至少 0.30 秒、4 次观测）')
        self.anchor, self.last_seen, self.pending = c, t, None
        self.epoch += 1
        self._clear_support()
        self._observe_support(c, t)
        return result(index, 'locked', '连续观测确认目标；依据骨架与位置，不依据框面积')
