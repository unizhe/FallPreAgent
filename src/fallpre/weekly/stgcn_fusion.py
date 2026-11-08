"""Constrained, causal ST-GCN++ candidate alongside the established decision.

B uses the verified NTU60 model, not a calibrated risk estimator. The two
branches share YOLO joints: agreement is a consistency check, not independent
evidence. Only unknown action outputs can receive a seat transition proposal.
Alarm state and every existing decision remain owned by the primary pipeline.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .stgcn_temporal import MODEL_ID, TemporalConfig, TemporalPoseBuffer

FUSION_VERSION = 'constrained-ntu60-seat-transitions-v1'
TRANSITIONS = {7: 'sitting_down', 8: 'standing_up'}
TEXT = {'sitting_down': '坐下', 'standing_up': '起身'}
SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))
JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)


@dataclass(frozen=True)
class FusionConfig:
    """Conservative predeclared B policy; not fitted to the ten development clips."""

    minimum_score: float = .60
    minimum_margin: float = .15
    maximum_result_age_s: float = .60
    minimum_joint_score: float = .50
    maximum_anchor_age_s: float = 2.5
    geometry_window_s: float = .45
    minimum_motion_span_s: float = .20
    minimum_hip_speed: float = .08
    minimum_joint_change: float = .015
    maximum_feet_motion: float = .12

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or self.minimum_score > 1 or self.minimum_margin > 1
                or self.minimum_joint_score > 1
                or self.geometry_window_s < self.minimum_motion_span_s):
            raise ValueError('Invalid constrained fusion configuration')


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _angle(a, b, c):
    first, second = a - b, c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return None
    return float(np.arccos(np.clip(np.dot(first, second) / denominator, -1, 1)) / np.pi)


def _geometry(person, width, height, confidence):
    try:
        p = np.asarray(person['keypoints'], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if p.shape != (17, 3) or not np.isfinite(p).all():
        return None
    points = p[list(JOINTS)]
    if (not np.all(points[:, 2] >= confidence) or np.any(points[:, 2] > 1)
            or not np.all((points[:, 0] > 0) & (points[:, 0] < width)
                          & (points[:, 1] > 0) & (points[:, 1] < height))):
        return None
    lengths = [float(np.linalg.norm(p[s, :2] - p[h, :2])) for s, h, _, _ in SIDES]
    scale = float(np.median(lengths))
    if min(lengths) < 12:
        return None
    knees = [_angle(p[h, :2], p[k, :2], p[a, :2]) for _, h, k, a in SIDES]
    if any(k is None for k in knees):
        return None
    heights = [(p[a, 1] - p[h, 1]) / scale for _, h, _, a in SIDES]
    upright = all((p[h, 1] - p[s, 1]) / length > .70
                  for (s, h, _, _), length in zip(SIDES, lengths))
    squat = min(knees) < .45 and max(heights) < 1.0
    return dict(points=p, scale=scale, knee=float(np.mean(knees)), upright=upright,
                squat=bool(squat), minimum_hip_height=float(min(heights)),
                maximum_hip_height=float(max(heights)))


class ConstrainedSTGCNFusion:
    """One instance per stream; ``update`` returns an independent action dict.

    ``result`` is the unchanged live result (persons/action/quality/fall/etc.).
    ``temporal`` is the LiveTemporalBranch output or a cached runtime prediction;
    when omitted, result['temporal'] is used. Call on every observation, even
    missing ones. Do not pass centered windows or carry state between videos.
    """

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.buffer = TemporalPoseBuffer(TemporalConfig())
        self.history = deque()
        self.anchor = None
        self.generation = None

    def reset(self):
        self.buffer = TemporalPoseBuffer(TemporalConfig())
        self.history.clear()
        self.anchor = None
        self.generation = None

    def _clear_motion(self):
        self.history.clear()
        self.anchor = None

    def update(self, result, t, temporal=None):
        action = result.get('action') or {}
        baseline = action.get('label', 'unknown')
        behavior = result.get('behavior') or {}
        output = dict(schema_version=1, version=FUSION_VERSION, mode='shadow',
                      label=baseline, text=action.get('text', baseline),
                      baseline_label=baseline, changed=False, status='retained',
                      code='baseline_retained', reason='保留主识别；融合仅提供有约束的旁路候选',
                      changes_primary_decision=False, changes_alarm=False,
                      source_time_s=t if _finite(t) else None,
                      conflicts=[], evidence={},
                      axes={k: dict(behavior[k]) for k in ('posture', 'transition', 'locomotion', 'turning')
                            if isinstance(behavior.get(k), dict)})

        def finish(code, reason, status='abstained'):
            output.update(code=code, reason=reason, status=status)
            return output

        selection = action.get('selection') or {}
        index = action.get('target_index')
        epoch = selection.get('target_epoch')
        persons = result.get('persons') or []
        person = (persons[index] if isinstance(index, int) and not isinstance(index, bool)
                  and 0 <= index < len(persons) else None)
        camera = (result.get('quality') or {}).get('camera_stable') is True
        width, height = result.get('width'), result.get('height')
        try:
            state = self.buffer.observe(person, t, width, height, epoch, camera_stable=camera)
        except (ValueError, TypeError):
            # A malformed timestamp must not make the optional branch crash the
            # primary pipeline, nor become a new source clock for old evidence.
            self._clear_motion()
            self.generation = self.buffer.generation
            return finish('invalid_timestamp', '时间戳无效或不递增，候选证据已清空')
        if self.generation != state['generation']:
            self._clear_motion()
            self.generation = state['generation']
        if person is None or epoch is None or selection.get('state') != 'locked':
            self._clear_motion()
            return finish('target_unavailable', '当前目标缺失或有歧义；不借用其他人体')
        if not camera:
            self._clear_motion()
            return finish('camera_unstable', '机位证据不稳定，暂停融合')
        geometry = (_geometry(person, width, height, self.config.minimum_joint_score)
                    if self.buffer.current_valid else None)
        if geometry is None:
            self._clear_motion()
            return finish('pose_unreliable', '当前双侧肩髋膝踝证据不足')
        self.history.append((float(t), geometry))
        while self.history and t - self.history[0][0] > self.config.geometry_window_s + 1e-9:
            self.history.popleft()
        if (baseline in {'bending', 'squatting', 'walking', 'falling'} or geometry['squat']
                or not geometry['upright']):
            self.anchor = None
        elif baseline == 'seated' and .45 < geometry['knee'] < .88 and geometry['minimum_hip_height'] > .65:
            self.anchor = dict(t=float(t), label='seated')
        elif baseline == 'standing' and geometry['knee'] > .88 and geometry['minimum_hip_height'] > 1.2:
            self.anchor = dict(t=float(t), label='standing')
        if self.anchor and t - self.anchor['t'] > self.config.maximum_anchor_age_s:
            self.anchor = None
        output['evidence'].update(target_epoch=epoch, generation=state['generation'],
                                  geometry_knee=round(geometry['knee'], 5),
                                  geometry_upright=geometry['upright'], geometry_squat=geometry['squat'],
                                  anchor=dict(self.anchor) if self.anchor else None)
        temporal = temporal if temporal is not None else result.get('temporal')
        if not isinstance(temporal, dict) or temporal.get('status') != 'ready':
            return finish('temporal_unavailable', '尚无可用时序结果')
        window = temporal.get('window') or {}
        end, start = window.get('end_s'), window.get('start_s')
        if (temporal.get('model_id') != MODEL_ID or window.get('causal') is not True
                or window.get('temporal_wrap') is not False
                or not _finite(end) or not _finite(start) or not 0 <= start < end
                or end - start < self.buffer.config.min_span_s - 1e-6
                or end - start > self.buffer.config.window_s + 1e-6
                or window.get('requested_window_s') != self.buffer.config.window_s
                or window.get('resampling') != self.buffer.config.resampling
                or window.get('sample_count') != self.buffer.config.sample_count):
            return finish('protocol_mismatch', '时序模型或因果输入协议与冻结方案不符')
        age = t - end
        output['evidence'].update(window_start_s=start, window_end_s=end, result_age_s=round(age, 6))
        if temporal.get('target_epoch') != epoch or temporal.get('generation') != state['generation']:
            return finish('identity_mismatch', '结果来自其他目标或中断前的骨架历史')
        if (not self.buffer.is_result_current(temporal) or age > self.config.maximum_result_age_s + 1e-9):
            return finish('temporal_stale_or_future', '结果过期或窗口终点晚于当前观测')
        if state['status'] != 'ready':
            return finish('history_insufficient', '当前连续骨架尚未达到冻结窗口要求')
        scores = temporal.get('all_scores')
        try:
            scores = np.asarray(scores, dtype=float)
        except (ValueError, TypeError):
            scores = np.empty(0)
        if (scores.shape != (60,) or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1)) or not np.isclose(scores.sum(), 1., atol=1e-5)):
            return finish('invalid_scores', '时序分类分数格式无效')
        order = np.argsort(-scores, kind='stable')
        top, score = int(order[0]), float(scores[order[0]])
        margin = score - float(scores[order[1]])
        output['evidence'].update(stgcn_class_index=top, stgcn_score=score, stgcn_margin=margin,
                                  score_semantics='closed-set action classification; not fall risk',
                                  shared_pose_evidence=True)
        squat = (baseline == 'squatting' or geometry['squat']
                 or (action.get('low_squat') or {}).get('candidate') is True)
        if scores[42] >= .5 and squat:
            output['conflicts'].append(dict(code='squat_vs_falling', stgcn_falling_score=float(scores[42]),
                                             baseline_label=baseline,
                                             resolution='retain primary action and alarm; manual review'))
        fall = result.get('fall') or {}
        if (fall.get('state') in {'candidate', 'suspected', 'confirmed', 'recovering'} or fall.get('current_target_confirmed')
                or fall.get('new_event') or action.get('raw_label') == 'falling'
                or baseline == 'falling'):
            self.anchor = None
            return finish('fall_state_protected', '跌倒及恢复由现有事件链处理；时序候选不能触发或取消报警', 'protected')
        if output['conflicts']:
            return finish('squat_fall_conflict', '蹲姿与时序跌倒候选冲突，保留原判断并记录复核', 'conflict')
        if top not in TRANSITIONS:
            return finish('unsupported_class', '预训练类别不能直接映射为当前家庭姿态')
        proposed = TRANSITIONS[top]
        output['evidence']['proposed_label'] = proposed
        if score < self.config.minimum_score or margin < self.config.minimum_margin:
            return finish('score_insufficient', '起坐候选分数或类别间隔不足')
        if baseline != 'unknown':
            return finish('established_action_protected', '当前已有确认动作或姿态，保留主识别', 'retained')
        if action.get('decision_stage') in {'quality', 'target', 'input', 'history'}:
            return finish('primary_evidence_unavailable', '主路径缺少当前观测或历史证据，不以分类分数补全')
        if action.get('decision_code') in {'bend_recovery', 'rise_motion_stopped', 'step_departure',
                                           'daily_pose_pending', 'motion_direction_conflict'}:
            return finish('primary_geometry_veto', '当前几何或动作完成约束禁止补全起坐')
        if squat or not geometry['upright']:
            return finish('geometry_conflict', '当前低位或躯干几何不支持起坐')
        if (action.get('static_evidence') or {}).get('stable'):
            return finish('static_completion', '当前已稳定，历史动作窗口不能延长转换过程')
        phase = action.get('resolved_phase', action.get('phase', 'unknown'))
        allowed = {'rising', 'forward_lean', 'unknown'} if proposed == 'standing_up' else {'descending', 'unknown'}
        if phase not in allowed:
            return finish('phase_conflict', '当前动作阶段与历史起坐候选冲突')
        origin = 'seated' if proposed == 'standing_up' else 'standing'
        if not self.anchor or self.anchor['label'] != origin:
            return finish('origin_unconfirmed', '缺少近期已确认的起坐起点，不能把蹲起或低位活动认作起坐')
        if len(self.history) < 3 or t - self.history[0][0] < self.config.minimum_motion_span_s - 1e-9:
            return finish('motion_history_insufficient', '尚无足够的近期真实位移观测')
        first_t, first = self.history[0]
        previous_t, previous = self.history[-2]
        p, q = geometry['points'], first['points']
        scale = first['scale']
        span = t - first_t
        hip_speed = float(np.mean(q[[11, 12], 1] - p[[11, 12], 1]) / scale / span)
        shoulder_speed = float(np.mean(q[[5, 6], 1] - p[[5, 6], 1]) / scale / span)
        current_speed = float(np.mean(previous['points'][[11, 12], 1] - p[[11, 12], 1])
                              / previous['scale'] / (t - previous_t))
        feet_motion = float(np.max(np.linalg.norm(p[[15, 16], :2] - q[[15, 16], :2], axis=1)) / scale)
        knee_change = geometry['knee'] - first['knee']
        direction = 1 if proposed == 'standing_up' else -1
        output['evidence'].update(hip_speed=hip_speed, shoulder_speed=shoulder_speed,
                                  current_hip_speed=current_speed, feet_motion=feet_motion,
                                  knee_change=knee_change, motion_span_s=span)
        if (direction * hip_speed < self.config.minimum_hip_speed
                or direction * shoulder_speed < self.config.minimum_hip_speed / 2
                or direction * current_speed <= .02
                or direction * knee_change < self.config.minimum_joint_change
                or feet_motion > self.config.maximum_feet_motion):
            return finish('motion_geometry_conflict', '当前髋肩方向、膝角变化或脚部稳定性不支持起坐')
        output.update(label=proposed, text=TEXT[proposed], changed=True, status='candidate',
                      code='constrained_transition',
                      reason='历史时序候选与近期起点、当前阶段及真实位移一致；仅作为旁路动作候选')
        output['axes']['transition'] = dict(label=proposed, text=TEXT[proposed], source=FUSION_VERSION)
        return output
