"""Read-only temporal fall evidence with explicit pose/RGB association.

The ST window belongs to the ACTION track namespace. An RGB fall track has a
separate epoch, so equal epoch values never establish identity. Classification,
observation quality and the existing event manager remain separate. Neither a
positive nor an absent ST classification creates, cancels or rearms an event.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .stgcn_temporal import MODEL_ID, TemporalConfig, TemporalPoseBuffer

VERSION = 'constrained-fall-temporal-evidence-v1'
CORE = (5, 6, 11, 12)
BODY = tuple(range(5, 17))
SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))
GEOMETRY_FEATURE_NAMES = (
    'center_x_image', 'center_y_image', 'hip_x_image', 'hip_y_image',
    'center_dx_image', 'center_dy_image', 'hip_dx_image', 'hip_dy_image',
    'center_drop_reference_torso', 'hip_drop_reference_torso',
    'current_center_speed_reference_torso', 'current_hip_speed_reference_torso',
    'relative_pose_motion_rms', 'torso_tilt_deg', 'torso_length_image',
    'knee_angle_pi', 'hip_ankle_height_torso', 'leg_visibility_fraction',
    'current_body_score', 'current_visible_body_fraction', 'observed_span_s',
)


@dataclass(frozen=True)
class FallEvidenceConfig:
    minimum_score: float = .60
    minimum_margin: float = .15
    maximum_result_age_s: float = .60
    minimum_joint_score: float = .50
    minimum_rgb_score: float = .60
    association_iou: float = .50
    competitor_iou: float = .10
    geometry_window_s: float = .75
    minimum_motion_span_s: float = .20
    minimum_descent_torso: float = .15
    minimum_current_descent_speed: float = .02
    low_torso_tilt_deg: float = 55.

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or any(v > 1 for v in (self.minimum_score, self.minimum_margin,
                                      self.minimum_joint_score, self.minimum_rgb_score,
                                      self.association_iou, self.competitor_iou))
                or self.geometry_window_s < self.minimum_motion_span_s
                or self.low_torso_tilt_deg >= 90):
            raise ValueError('Invalid fall temporal evidence configuration')


def _finite(value):
    return isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    try:
        box = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    return box if box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]) else None


def _iou(a, b):
    intersection = float(np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0)))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union


def _angle(a, b, c):
    first, second = a - b, c - b
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.arccos(np.clip(np.dot(first, second) / scale, -1, 1)) / np.pi) if scale > 1e-8 else None


def fall_geometry_features(keypoints, source_times, width, height, *, window_s=.75, joint_confidence=.5):
    """Shared trailing-window geometry for replay/live and future head training.

    Input is the unmodified pixel COCO17 observations, not normalized ST tensors.
    Global image displacement and hip-relative shape change are separate feature
    groups. No future observation, interpolation, imputation or centered window
    is used. Missing quality returns ``vector=None``, never a negative example.
    """
    unavailable = dict(quality='unobservable', code='invalid_geometry', vector=None,
                       feature_names=list(GEOMETRY_FEATURE_NAMES), causal=True)
    try:
        p, times = np.asarray(keypoints, dtype=float), np.asarray(source_times, dtype=float)
    except (TypeError, ValueError):
        return unavailable
    if (not _finite(width) or not _finite(height) or width <= 0 or height <= 0
            or not _finite(window_s) or window_s <= 0
            or not _finite(joint_confidence) or not 0 < joint_confidence <= 1
            or times.ndim != 1 or len(times) < 1 or p.shape != (len(times), 17, 3)
            or not np.isfinite(times).all() or not np.isfinite(p).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)
            or np.any((p[:, :, 2] < 0) | (p[:, :, 2] > 1))):
        return unavailable
    retained = times >= times[-1] - window_s - 1e-9
    p, times = p[retained], times[retained]
    valid = ((p[:, :, 2] >= joint_confidence) & (p[:, :, 0] > 0) & (p[:, :, 0] < width)
             & (p[:, :, 1] > 0) & (p[:, :, 1] < height))
    if not valid[:, list(CORE)].all() or np.any(valid[:, list(BODY)].sum(1) < 8):
        return dict(unavailable, code='insufficient_visible_joints')
    if len(times) < 3 or times[-1] - times[0] < .20 - 1e-9:
        return dict(unavailable, quality='insufficient', code='geometry_warmup')
    if np.max(np.diff(times)) > .45 + 1e-9:
        return dict(unavailable, code='geometry_gap')
    hips = p[:, [11, 12], :2].mean(1)
    shoulders = p[:, [5, 6], :2].mean(1)
    centers = (hips + shoulders) / 2
    torsos = hips - shoulders
    lengths = np.linalg.norm(torsos, axis=1)
    if np.any(lengths < 20):
        return dict(unavailable, code='torso_projection_unreliable')
    common = valid[0] & valid[-1]
    common[:5] = False
    relative_start = (p[0, common, :2] - hips[0]) / lengths[0]
    relative_end = (p[-1, common, :2] - hips[-1]) / lengths[-1]
    knees, heights = [], []
    for _, h, k, a in SIDES:
        if valid[-1, [h, k, a]].all():
            angle = _angle(p[-1, h, :2], p[-1, k, :2], p[-1, a, :2])
            if angle is not None:
                knees.append(angle)
                heights.append(float((p[-1, a, 1] - hips[-1, 1]) / lengths[-1]))
    knee = float(np.mean(knees)) if knees else 0.
    hip_ankle = float(np.mean(heights)) if heights else 0.
    tilt = float(np.degrees(np.arctan2(abs(torsos[-1, 0]), abs(torsos[-1, 1]))))
    scale, dt = lengths[0], times[-1] - times[-2]
    metrics = dict(
        center_x_image=float(centers[-1, 0] / width), center_y_image=float(centers[-1, 1] / height),
        hip_x_image=float(hips[-1, 0] / width), hip_y_image=float(hips[-1, 1] / height),
        center_dx_image=float((centers[-1, 0] - centers[0, 0]) / width),
        center_dy_image=float((centers[-1, 1] - centers[0, 1]) / height),
        hip_dx_image=float((hips[-1, 0] - hips[0, 0]) / width),
        hip_dy_image=float((hips[-1, 1] - hips[0, 1]) / height),
        center_drop_reference_torso=float((centers[-1, 1] - centers[0, 1]) / scale),
        hip_drop_reference_torso=float((hips[-1, 1] - hips[0, 1]) / scale),
        current_center_speed_reference_torso=float((centers[-1, 1] - centers[-2, 1]) / scale / dt),
        current_hip_speed_reference_torso=float((hips[-1, 1] - hips[-2, 1]) / scale / dt),
        relative_pose_motion_rms=float(np.sqrt(np.mean((relative_end - relative_start) ** 2))),
        torso_tilt_deg=tilt, torso_length_image=float(lengths[-1] / height),
        knee_angle_pi=knee, hip_ankle_height_torso=hip_ankle, leg_visibility_fraction=len(knees) / 2,
        current_body_score=float(p[-1, list(BODY), 2].mean()),
        current_visible_body_fraction=float(valid[-1, list(BODY)].mean()),
        observed_span_s=float(times[-1] - times[0]))
    return dict(quality='observable', code='observed', metrics=metrics,
                vector=[metrics[name] for name in GEOMETRY_FEATURE_NAMES],
                feature_names=list(GEOMETRY_FEATURE_NAMES), causal=True,
                window_start_s=float(times[0]), window_end_s=float(times[-1]), observations=len(times),
                geometry_squat=bool(len(knees) == 2 and min(knees) < .45
                                    and min(heights) > -.15 and max(heights) < 1.0 and tilt < 35),
                feature_semantics='pixel/image global motion plus hip-relative shape; no risk probability')


def _associate_rgb(result, person, action_epoch, config):
    """Associate namespaces using CURRENT boxes and visible torso, not epoch equality."""
    evidence = dict(associated=False, action_target_epoch=action_epoch,
                    rgb_target_epoch=None, namespaces_distinct=True, method='current_unique_box_and_torso',
                    fall_pose_reference_epoch=(result.get('rgb_diagnostics') or {}).get('reference_pose_epoch'))

    def reject(code):
        return dict(evidence, code=code)

    observation = result.get('fall_observation') or {}
    selection, fall = observation.get('selection') or {}, result.get('fall') or {}
    rgb = observation.get('person')
    rgb_epoch = selection.get('target_epoch')
    evidence['rgb_target_epoch'] = rgb_epoch
    if (result.get('fall_mode') != 'rgb' or observation.get('source') != 'rgb_fall'
            or selection.get('state') != 'locked' or rgb_epoch is None
            or fall.get('target_epoch') != rgb_epoch or fall.get('observed') is not True
            or not isinstance(rgb, dict)):
        return reject('rgb_target_unavailable')
    a, b = _box(person.get('box')), _box(rgb.get('box'))
    if a is None or b is None:
        return reject('association_box_missing')
    if (not _finite(person.get('confidence')) or person['confidence'] < .5
            or not _finite(rgb.get('confidence')) or rgb['confidence'] < config.minimum_rgb_score):
        return reject('association_detection_weak')
    overlap = _iou(a, b)
    evidence['pose_rgb_iou'] = overlap
    points = np.asarray(person['keypoints'], dtype=float)[list(CORE), :2]
    if overlap < config.association_iou or not ((points >= b[:2]) & (points <= b[2:])).all():
        return reject('association_geometry_mismatch')
    diagnostics = result.get('rgb_diagnostics') or {}
    if diagnostics.get('ambiguity_retained') or diagnostics.get('person_ambiguity_retained'):
        return reject('rgb_association_ambiguous')
    pose_matches = []
    for other in result.get('persons') or []:
        box = _box(other.get('box')) if isinstance(other, dict) else None
        if box is not None and _finite(other.get('confidence')) and other['confidence'] >= .25:
            if _iou(box, b) >= config.competitor_iou:
                pose_matches.append(other)
    if len(pose_matches) != 1 or pose_matches[0] is not person:
        return reject('multiple_pose_matches')
    raw = result.get('rgb_raw_candidates') or {}
    if not isinstance(raw.get('persons'), list) or not isinstance(raw.get('fallen'), list):
        return reject('rgb_current_candidates_missing')
    nearby = {}
    for category in ('persons', 'fallen'):
        nearby[category] = []
        for proposal in raw[category]:
            box = _box(proposal.get('box')) if isinstance(proposal, dict) else None
            score = proposal.get('confidence') if isinstance(proposal, dict) else None
            if box is not None and _finite(score) and .25 <= score <= 1 and _iou(box, b) >= config.competitor_iou:
                nearby[category].append((box, float(score)))
        if len(nearby[category]) > 1:
            return reject('multiple_rgb_matches')
    if not nearby['fallen']:
        return reject('rgb_fallen_evidence_missing')
    fallen_box, fallen_score = nearby['fallen'][0]
    evidence.update(rgb_falling_score=fallen_score, pose_fallen_iou=_iou(a, fallen_box))
    if fallen_score < config.minimum_rgb_score or min(_iou(a, fallen_box), _iou(b, fallen_box)) < config.association_iou:
        return reject('rgb_fallen_evidence_weak_or_mismatched')
    return dict(evidence, associated=True, code='associated_current_rgb')


class FallTemporalEvidence:
    """One per source stream. Call for every observation, including missing ones."""

    def __init__(self, config=None):
        self.config = config or FallEvidenceConfig()
        self.buffer = TemporalPoseBuffer(TemporalConfig())
        self.geometry_rows = []
        self.generation = None
        self.association_pair = None

    def update(self, result, t, temporal=None):
        out = dict(schema_version=1, version=VERSION, mode='shadow', status='abstain',
                   support=False, code='evidence_unavailable', reason='证据不足',
                   source_time_s=float(t) if _finite(t) else None,
                   observation_quality=dict(status='unobservable'), model_evidence=dict(status='unavailable'),
                   association=dict(associated=False), geometry={}, conflicts=[],
                   changes_primary_decision=False, changes_alarm=False, changes_event=False,
                   event_management='existing primary event manager only; no event emitted',
                   negative_example=False, shared_pose_evidence=True)

        def stop(code, reason, quality=None):
            out.update(code=code, reason=reason)
            if quality:
                out['observation_quality']['status'] = quality
            return out

        if not _finite(t) or t < 0:
            self.geometry_rows.clear()
            self.association_pair = None
            self.buffer.reset('无效源时间')
            return stop('invalid_timestamp', '源时间必须是有限非负数')
        t = float(t)
        action = result.get('action') or {}
        selection = action.get('selection') or {}
        index, epoch = action.get('target_index'), selection.get('target_epoch')
        persons = result.get('persons') or []
        person = (persons[index] if isinstance(index, int) and not isinstance(index, bool)
                  and 0 <= index < len(persons) else None)
        camera = (result.get('quality') or {}).get('camera_stable') is True
        try:
            state = self.buffer.observe(person, t, result.get('width'), result.get('height'), epoch, camera)
        except (TypeError, ValueError):
            self.geometry_rows.clear()
            self.association_pair = None
            return stop('invalid_timestamp', '源时间无效或不递增；不复用旧结果')
        if self.generation != state['generation']:
            self.geometry_rows.clear()
            self.association_pair = None
            self.generation = state['generation']
        if (not camera or not self.buffer.current_valid or person is None or epoch is None
                or selection.get('state') != 'locked'
                or selection.get('target_index', index) != index):
            self.geometry_rows.clear()
            self.association_pair = None
            return stop('current_observation_unavailable', '机位、目标身份或当前骨架不足；输出未知')
        geometry = fall_geometry_features(
            np.stack([row[1] for row in self.buffer.samples]), [row[0] for row in self.buffer.samples],
            result['width'], result['height'], window_s=self.config.geometry_window_s,
            joint_confidence=self.config.minimum_joint_score)
        out['geometry'] = geometry
        out['observation_quality'] = dict(status=geometry['quality'], code=geometry['code'])
        if geometry['quality'] != 'observable':
            self.association_pair = None
            self.geometry_rows.clear()
            return stop('current_geometry_unavailable', '近期连续可见骨架不足；不作为非跌倒样本')
        association = _associate_rgb(result, person, epoch, self.config)
        out['association'] = association
        pair = (epoch, association.get('rgb_target_epoch')) if association['associated'] else None
        if pair != self.association_pair or not association['associated']:
            self.geometry_rows.clear()
            self.association_pair = pair
        if pair is not None:
            self.geometry_rows.append(float(t))
            self.geometry_rows = [v for v in self.geometry_rows if t - v <= self.config.geometry_window_s + 1e-9]
        temporal = result.get('temporal') if temporal is None else temporal
        if not isinstance(temporal, dict) or temporal.get('status') != 'ready':
            return stop('temporal_unavailable', '尚无可用时序分类结果')
        window = temporal.get('window') or {}
        start, end = window.get('start_s'), window.get('end_s')
        cfg = self.buffer.config
        if (temporal.get('model_id') != MODEL_ID or window.get('causal') is not True
                or window.get('temporal_wrap') is not False or not _finite(start) or not _finite(end)
                or not 0 <= start < end or not cfg.min_span_s - 1e-6 <= end - start <= cfg.window_s + 1e-6
                or window.get('requested_window_s') != cfg.window_s
                or window.get('sample_count') != cfg.sample_count or window.get('resampling') != cfg.resampling):
            return stop('temporal_protocol_mismatch', '时序结果与冻结的因果输入协议不符')
        if temporal.get('target_epoch') != epoch or temporal.get('generation') != state['generation']:
            return stop('temporal_identity_mismatch', 'ST结果不属于当前action目标或连续观测代次')
        age = float(t - end)
        if not self.buffer.is_result_current(temporal) or age > self.config.maximum_result_age_s + 1e-9:
            return stop('temporal_stale_or_future', 'ST结果过期或引用了未来窗口终点')
        if state['status'] != 'ready':
            return stop('temporal_history_insufficient', '连续骨架未达到冻结历史要求')
        try:
            scores = np.asarray(temporal.get('all_scores'), dtype=float)
        except (ValueError, TypeError):
            scores = np.empty(0)
        if (scores.shape != (60,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or not np.isclose(scores.sum(), 1., atol=1e-5)):
            return stop('invalid_model_scores', '闭集分类输出无效')
        top = int(np.argmax(scores))
        falling_score = float(scores[42])
        margin = falling_score - float(np.max(np.delete(scores, 42)))
        out['model_evidence'] = dict(status='observed', model_id=MODEL_ID, top_class_index=top,
                                     falling_score=falling_score, falling_margin=margin,
                                     window_start_s=start, window_end_s=end, result_age_s=age,
                                     score_semantics='closed-set past-action classification; not fall probability')
        squat = (geometry['geometry_squat'] or action.get('label') == 'squatting'
                 or (action.get('low_squat') or {}).get('candidate') is True)
        if squat and falling_score >= .5:
            out['conflicts'].append(dict(code='squat_vs_st_falling', falling_score=falling_score,
                                          resolution='keep event manager unchanged; review observation'))
        if not association['associated']:
            return stop(association['code'], '当前action目标与RGB倒地证据未唯一关联；不融合两个身份空间')
        if out['conflicts']:
            out['status'] = 'conflict'
            return stop('squat_fall_conflict', '蹲姿与ST跌倒分类冲突；不触发或取消现有事件')
        if top != 42 or falling_score < self.config.minimum_score or margin < self.config.minimum_margin:
            return stop('model_fall_support_insufficient', '时序跌倒分类证据不足；这不等于非跌倒')
        if (len(self.geometry_rows) < 3
                or t - self.geometry_rows[0] < self.config.minimum_motion_span_s - 1e-9):
            return stop('association_warmup', '当前两路目标尚未持续唯一关联')
        m = geometry['metrics']
        descending = (m['center_drop_reference_torso'] >= self.config.minimum_descent_torso
                      and m['hip_drop_reference_torso'] >= self.config.minimum_descent_torso
                      and m['current_center_speed_reference_torso'] > self.config.minimum_current_descent_speed
                      and m['current_hip_speed_reference_torso'] > self.config.minimum_current_descent_speed)
        low = m['torso_tilt_deg'] >= self.config.low_torso_tilt_deg
        out['geometry'].update(descending=descending, low_pose=low)
        if not descending or not low:
            return stop('current_fall_geometry_insufficient', '需近期真实下降及当前低姿；持续倒卧不重复视为新跌倒')
        out.update(status='support', support=True, code='constrained_fall_support',
                   reason='ST跌倒分类、当前下降低姿与唯一关联的RGB倒地证据一致；仅记录旁路支持')
        return out
