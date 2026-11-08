"""Per-session observation counts; no pixels, keypoint coordinates or recordings."""
from collections import Counter

import numpy as np

REASONS = {
    'descent_shape_pending': '头肩先下探但屈膝证据不足，暂缓确认坐下',
    'daily_pose_pending': '当前姿态与起坐不符，等待连续日常姿态确认',
    'step_departure': '髋、膝、脚正在同向移步，退出站立确认',
    'target_unconfirmed': '目标尚未确认或暂时丢失',
    'multiple_people': '多个候选尚未明确目标',
    'joint_chain_missing': '完整肩髋膝踝链不足',
    'history_warmup': '连续姿态仍在积累',
    'pose_unreliable': '当前姿态几何不可靠',
    'model_unknown': '原动作模型未知',
    'phase_unknown': '阶段模型未确认状态',
    'other_action': '原模型判断为其他动作',
    'action_phase_conflict': '动作与阶段不一致',
    'motion_direction_conflict': '动作与当前运动方向不一致',
    'static_motion': '静态状态仍有运动',
    'static_geometry': '静态姿势与几何证据不一致',
    'static_phase_unsupported': '模型阶段尚不支持静态姿态',
    'static_confirmation': '静态状态等待持续确认',
    'seat_support_veto': '坐姿支持层暂不接受',
    'reconciliation_pending': '动作协调等待连续证据',
    'rise_pending': '已离座，等待连续起身运动确认',
    'rise_motion_stopped': '离座后上升已停止，等待稳定姿态',
}
RESET_REASONS = {
    'descent_shape_pending': '头肩先下探但屈膝证据不足，暂缓确认坐下',
    'daily_pose_pending': '当前姿态与起坐不符，等待连续日常姿态确认',
    'source_gap': '输入间隔超过 0.35 秒',
    'target_changed': '目标身份建立或变化',
    'target_missing': '当前目标无法关联',
    'joint_chain_missing': '完整关节链不足',
    'multiple_people': '全部候选流程出现多人',
    'seat_joints_missing': '坐姿支持层缺少双肩、双髋或可见膝',
    'seat_anchor_invalidated': '其他动作或站立使坐姿参考失效',
    'seat_anchor_expired': '坐姿参考超过有效期',
}
RISE_EVIDENCE_FIELDS = (
    'code', 'reason', 'rising', 'veto_seated', 'stop_rise', 'applied',
    'anchor_available', 'anchor_kind', 'anchor_age_s', 'hip_up', 'shoulder_up',
    'hip_absolute_up', 'shoulder_absolute_up', 'horizontal',
    'thigh_extension', 'knee_opening', 'motion_duration_s', 'positive_steps', 'additional_up',
    'current_continues', 'current_feet_stable', 'foot_motion', 'reset_reason', 'standing_complete',
    'window_extended', 'support_foot_count', 'walking_tail',
)


def scalar_rise_evidence(evidence):
    """Keep decision provenance, without copying pose coordinates into exports."""
    if not isinstance(evidence, dict):
        return None
    return {key: value for key in RISE_EVIDENCE_FIELDS
            if (value := evidence.get(key)) is None or isinstance(value, (str, bool))
            or isinstance(value, (int, float)) and np.isfinite(value)}


def pose_quality(person, width, height):
    valid = np.zeros(17, dtype=bool)
    if person is not None:
        points = np.asarray(person['keypoints'])
        valid = (np.isfinite(points).all(1) & (points[:, 2] >= .30)
                 & (points[:, 0] > 0) & (points[:, 0] < width)
                 & (points[:, 1] > 0) & (points[:, 1] < height))
    return dict(shoulders=int(valid[[5, 6]].sum()), hips=int(valid[[11, 12]].sum()),
                knees=int(valid[[13, 14]].sum()), ankles=int(valid[[15, 16]].sum()),
                complete_side=bool(any(valid[ids].all() for ids in ([5, 11, 13, 15], [6, 12, 14, 16]))))


def has_conflict(label, phase):
    """Disagreement is diagnostic, not proof that either model is wrong."""
    if label in {'unknown', 'unavailable'} or phase in {'unknown', 'other'}:
        return False
    compatible = {'seated': {'background'}, 'standing': {'background'},
                  'rising': {'standing_up'}, 'descending': {'sitting_down'},
                  'forward_lean': {'background', 'bending'}}
    return phase in compatible and label not in compatible[phase]


class SessionDiagnostics:
    def __init__(self):
        self.labels = Counter()
        self.unknown_reasons = Counter()
        self.resets = Counter()
        self.first_t = self.last_t = None
        self.max_gap = 0.
        self.raw_unknown = self.raw_unavailable = 0
        self.conflicts = self.resolved = self.long_gaps = 0

    def update(self, t, action, persons, width, height):
        interval = None if self.last_t is None else t - self.last_t
        if interval is not None and interval <= 0:
            raise ValueError('诊断时间戳必须递增')
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        self.max_gap = max(self.max_gap, interval or 0.)
        self.long_gaps += int(interval is not None and interval > .35)
        self.labels[action['label']] += 1
        self.raw_unknown += int(action['raw_label'] == 'unknown')
        self.raw_unavailable += int(action['raw_label'] == 'unavailable')
        conflict = has_conflict(action['raw_label'], action['phase'])
        self.conflicts += int(conflict)
        reconciliation = action.get('reconciliation', {})
        changed = bool(reconciliation.get('changed'))
        self.resolved += int(changed)
        code = action['decision_code']
        if action['label'] == 'unknown':
            self.unknown_reasons[code] += 1
        resets = sorted(set(action.get('reset_reasons', [])))
        self.resets.update(resets)
        index = action['target_index']
        quality = pose_quality(persons[index] if index is not None else None, width, height)
        total = sum(self.labels.values())
        unknown = self.labels['unknown']
        sample = dict(t=round(t, 4), interval_s=None if interval is None else round(interval, 4),
                      label=action['label'], raw_label=action['raw_label'], phase=action['phase'],
                      resolved_label=action.get('resolved_label', action['raw_label']),
                      resolved_phase=action.get('resolved_phase', action['phase']),
                      reason_code=code, reason=action['reason'], stage=action['decision_stage'],
                      reset_reasons=resets, conflict=conflict,
                      resolution=reconciliation.get('code') if changed else None,
                      reconciliation_reason=reconciliation.get('reason', ''),
                      reconciliation_code=reconciliation.get('code'),
                      reconciliation_evidence=reconciliation.get('evidence', {}),
                      rise_evidence=scalar_rise_evidence(action.get('rise_evidence')),
                      seat_rise_handoff={k:v for k,v in action.get('seat_rise_handoff', {}).items()
                                         if k in {'code','applied','support_feet','positive_steps','hip_up','extension','lean_ascent','terminal_extension','reference_age_s','span_hip_up','knee_opening','lower_seat_refreshed'}
                                         and isinstance(v,(str,int,float,bool))},
                      static_evidence={k:v for k,v in action.get('static_evidence', {}).items() if isinstance(v,(str,int,float,bool)) or v is None},
                      target_continuity={k:v for k,v in action.get('target_continuity', {}).items() if isinstance(v,(str,int,float,bool)) or v is None},
                      evidence_s=action['evidence_s'], selection_state=action['selection']['state'],
                      target_index=index, quality=quality, metrics=action.get('metrics', {}))
        seat = action.get('visible_seat', {})
        sample.update(seat_recovery_s=seat.get('seat_recovery_s', 0.),
                      seat_recovered=bool(seat.get('seat_recovered')),
                      seat_veto=bool(seat.get('veto')), seat_reason=seat.get('reason', ''))
        summary = dict(total_frames=total, known_frames=total-unknown, unknown_frames=unknown,
                       unknown_rate=unknown/total, raw_unknown_frames=self.raw_unknown,
                       raw_unavailable_frames=self.raw_unavailable, conflict_frames=self.conflicts,
                       resolved_frames=self.resolved, long_gap_frames=self.long_gaps,
                       max_gap_s=round(self.max_gap, 4), elapsed_s=round(t-self.first_t, 3),
                       labels=dict(self.labels),
                       unknown_reasons=[dict(code=k, reason=REASONS.get(k, k), count=v, rate=v/unknown)
                                        for k, v in self.unknown_reasons.most_common()],
                       resets=[dict(code=k, reason=RESET_REASONS.get(k, k), count=v)
                               for k, v in self.resets.most_common()])
        return dict(schema='fallpre.webcam_diagnostics.v1', sample=sample, summary=summary)
