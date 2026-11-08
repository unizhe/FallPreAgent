"""Conservative live display adapter for the frozen V7 action/phase model.

Only current/past observations are used. No weights, cached poses or source-specific
rules are changed. Seated persistence is bounded and rechecked against current geometry.
"""
from collections import deque

import numpy as np

from .action_v2 import PersonTracker
from .action_v4 import phase_features
from .action_v7 import OnlineCompletion
from .observed_rise import ObservedRise
from .webcam_daily_activity import DailyActivity
from .seat_rise_handoff import SeatRiseHandoff
from .visible_seat import VisibleSeat
from .visible_seat_previous import VisibleSeat as PreviousVisibleSeat
from .webcam_diagnostics import SessionDiagnostics
from .webcam_reconcile import ReconcileActions

LABELS = {"standing": "站立", "seated": "坐着", "sitting_down": "坐下",
          "standing_up": "起身", "unknown": "未知", "walking": "行走 / 转身", "squatting": "蹲着", "bending": "弯腰"}


def seated_geometry(info):
    return bool(info and info["reliable"] and info["knee"] < .83
                and info["hip_angle"] < .82 and info["tilt"] < .45)


class SeatEvidence:
    """Causal stability evidence. Never changes the frozen model or creates a rise.

    A forward-lean phase alone cannot initialize a seat anchor. At least 0.16 s
    of explicit seated phases is required, followed by stable current geometry.
    Hip positions are measured from observations, not interpolated skeletons.
    """
    def __init__(self):
        self.samples = deque()
        self.anchor = None

    def update(self, t, prediction, info):
        if info is None or not info["reliable"]:
            self.samples.clear()
            self.anchor = None
            return dict(accepted=False, reason="连续可靠姿态不足")
        phase, label = prediction.get("phase"), prediction.get("label")
        # Do not carry a seat across an observed transition or another action.
        if phase in {"rising", "standing", "descending", "other"} or label != "background":
            self.samples.clear()
            self.anchor = None
            return dict(accepted=False, reason="原始模型处于动作过程或其他状态")
        self.samples.append(dict(t=t, phase=phase, **{k: info[k] for k in
                            ("hip_y", "scale", "knee", "hip_angle", "tilt", "reliable")}))
        while self.samples and self.samples[0]["t"] < t - .65:
            self.samples.popleft()
        samples = list(self.samples)
        explicit = [s for s in samples if s["phase"] == "seated" and seated_geometry(s)]
        if len(explicit) >= 2 and explicit[-1]["t"] - explicit[0]["t"] >= .16:
            self.anchor = dict(t=t, hip_y=float(np.median([s["hip_y"] for s in explicit])),
                               knee=float(np.median([s["knee"] for s in explicit])),
                               scale=float(np.median([s["scale"] for s in explicit])))
        if self.anchor and t - self.anchor["t"] > 10:
            self.anchor = None
        details = dict(accepted=False, reason="等待至少 0.4 秒的稳定坐姿证据",
                       anchor_available=self.anchor is not None)
        if len(samples) < 4 or t - samples[0]["t"] < .4:
            return details
        scale = max(float(np.median([s["scale"] for s in samples])), 1)
        ys = np.array([s["hip_y"] for s in samples])
        times = np.array([s["t"] for s in samples])
        n = max(1, len(samples) // 3)
        speed = float((np.median(ys[:n]) - np.median(ys[-n:])) /
                      scale / max(np.median(times[-n:]) - np.median(times[:n]), .05))
        spread = float((np.quantile(ys, .9) - np.quantile(ys, .1)) / scale)
        current_offset = float(abs(ys[-1] - np.median(ys)) / scale)
        geometry_fraction = sum(seated_geometry(s) for s in samples) / len(samples)
        stable = abs(speed) <= .10 and spread <= .10 and current_offset <= .08
        details.update(trend_speed=round(speed, 3), hip_spread=round(spread, 3),
                       geometry_fraction=round(geometry_fraction, 3), stable=stable)
        if not seated_geometry(info) or geometry_fraction < .8:
            return dict(details, reason="屈髋、屈膝坐姿证据不足")
        if not stable:
            return dict(details, reason="髋部仍有持续位移，等待坐稳")
        if phase == "seated":
            return dict(details, accepted=True, reason="原始坐姿阶段与短窗口稳定姿态一致")
        if phase != "forward_lean" or self.anchor is None:
            return dict(details, reason="模型未确认坐姿；前倾本身不足以判为坐着")
        up = (self.anchor["hip_y"] - info["hip_y"]) / self.anchor["scale"]
        opening = info["knee"] - self.anchor["knee"]
        details.update(anchor_up=round(up, 3), knee_opening=round(opening, 3))
        if abs(up) > .08 or opening > .06:
            return dict(details, reason="相对坐姿锚点已抬髋或伸腿，不能继续确认坐着")
        return dict(details, accepted=True, reason="此前已确认坐姿；当前前倾但屈膝、髋位稳定，无起身证据")


def decision_details(prediction, info):
    """Static phases require compatible geometry; transitions require real motion."""
    if info is None or not info["reliable"]:
        return "unknown", "正在积累连续姿态，或髋、膝、踝证据不足", (
            'history_warmup' if info is None else 'pose_unreliable')
    label, phase = prediction.get("label"), prediction.get("phase")
    speed = info["hip_speed"]  # positive = upward, torso lengths / second
    if label == "standing_up" and phase == "rising" and speed > .015:
        return "standing_up", "时序模型确认起身，髋部仍在上升", 'accepted'
    if label == 'sitting_down' and phase in {'standing', 'rising'}:
        return 'unknown', '坐下动作与当前站立或上升阶段冲突', 'action_phase_conflict'
    if label == "sitting_down" and speed < -.03:
        return "sitting_down", "时序模型判断坐下，髋部正在下降", 'accepted'
    if label not in {"background"}:
        code = ('model_unknown' if label == 'unknown' else
                'action_phase_conflict' if label == 'standing_up' and phase != 'rising' else
                'motion_direction_conflict' if label in {'standing_up', 'sitting_down'} else 'other_action')
        return "unknown", "动作与当前运动证据未达成一致，或不属于本次四类", code
    recovery = prediction.get('static_recovery') or {}
    if recovery.get('stable') and recovery.get('phase') == 'standing':
        return 'standing', '当前躯干与腿部已连续站稳', 'accepted'
    if (phase == 'seated' and recovery.get('phase') == 'seated' and recovery.get('stable')
            and recovery.get('support') in {'observed_descent', 'verified_seat'}
            and recovery.get('duration_s', 0) >= .35 and recovery.get('samples', 0) >= 4):
        return 'seated', '观察到从站立下降后坐稳，连续关节与坐姿支持证据一致', 'accepted'
    if abs(speed) > .10:
        return "unknown", "身体仍在移动，静态状态证据不足", 'static_motion'
    if phase == "standing" and info["knee"] > .90 and info["hip_angle"] > .80 and info["tilt"] < .25:
        return "standing", "站立阶段与腿部伸展、躯干姿态一致", 'accepted'
    if phase == "seated" and info["knee"] < .83 and info["hip_angle"] < .82 and info["tilt"] < .45:
        return "seated", "坐姿阶段与屈髋、屈膝姿态一致", 'accepted'
    if phase not in {'standing', 'seated'}:
        return 'unknown', '当前阶段尚未支持静态姿态，等待独立稳定证据恢复', (
            'phase_unknown' if phase == 'unknown' else 'static_phase_unsupported')
    return 'unknown', '静态阶段与当前腿部或躯干几何不符', 'static_geometry'


def display_decision(prediction, info):
    return decision_details(prediction, info)[:2]


class BaselineLiveActions:
    def __init__(self, model):
        self.model = model
        self.tracker = PersonTracker(30)
        self.last_t = None
        self.epoch = None
        self.reset_evidence()

    def reset_evidence(self):
        self.online = OnlineCompletion(self.model, live_timing=True)
        self.reconciler = ReconcileActions()
        self.history = deque()
        self.pending = None
        self.pending_since = None
        self.seat_evidence = SeatEvidence()

    def update(self, persons, t, width, height):
        self.current_prediction = None
        self.current_raw_prediction = None
        resets = []
        if not np.isfinite(t) or t < 0 or self.last_t is not None and t <= self.last_t:
            raise ValueError("摄像头时间戳必须递增，请重新启动")
        if self.last_t is not None and t - self.last_t > .35:
            self.reset_evidence()
            resets.append('source_gap')
        self.last_t = t
        instances = []
        for p in persons:
            points = np.asarray(p["keypoints"], dtype=float).copy()
            valid = (np.isfinite(points).all(axis=1) & (points[:, 0] > 0) &
                     (points[:, 0] < width) & (points[:, 1] > 0) & (points[:, 1] < height) &
                     (points[:, 2] >= .30))
            points[~valid] = 0
            instances.append(dict(box=p["box"], det=p["confidence"], kpts=points.tolist()))
        tracked = self.tracker.update(dict(t=t, instances=instances))
        self.current_tracked = tracked
        target = tracked["target_index"]
        if tracked["target_epoch"] != self.epoch:
            self.reset_evidence()
            if self.epoch is not None:
                resets.append('target_changed')
            self.epoch = tracked["target_epoch"]
        # Deliberately a single-person demo: no guessing who should be classified.
        reason = None
        if len(persons) > 1:
            reason = "多人或额外人形候选，目标不明确；请保持单人入镜"
            code = 'multiple_people'
        elif target is None:
            reason = "暂未关联到当前人体，等待目标确认"
            code = 'target_unconfirmed'
        elif not any(all(np.asarray(instances[target]["kpts"])[ids, 2] >= .30)
                     for ids in ([5, 11, 13, 15], [6, 12, 14, 16])):
            reason = "至少需要一侧肩、髋、膝、踝清晰入镜；当前关节不完整"
            code = 'joint_chain_missing'
        if reason:
            self.reset_evidence()
            resets.append('target_missing' if code == 'target_unconfirmed' else code)
            return dict(label="unknown", text="未知", reason=reason,
                        target_index=None if len(persons) > 1 else target,
                        raw_label="unavailable", phase="unknown", evidence_s=0,
                        resolved_label='unavailable', resolved_phase='unknown',
                        decision_code=code, decision_stage=('quality' if code == 'joint_chain_missing' else 'target'),
                        reset_reasons=resets, metrics={},
                        reconciliation=dict(changed=False, code='model_unavailable', reason=reason),
                        previous_label="unknown", previous_reason=reason,
                        seat_evidence=dict(accepted=False, reason=reason))
        raw_prediction = self.online.update(tracked)
        self.current_raw_prediction = raw_prediction
        self.history.append(tracked)
        while self.history and self.history[0]["t"] < t - 4.05:
            self.history.popleft()
        _, info = phase_features(list(self.history))
        reconciliation = self.reconciler.update(t, raw_prediction, info, tracked['instances'][0])
        prediction = reconciliation['prediction']
        self.current_prediction = prediction
        label, reason, code = decision_details(prediction, info)
        if label == 'unknown' and reconciliation['changed'] and prediction.get('label') == 'unknown':
            reason = reconciliation['reason']
            phase = raw_prediction.get('phase')
            opposite = (phase in {'seated', 'standing'} or
                        raw_prediction.get('label') == 'sitting_down' and phase == 'rising' or
                        raw_prediction.get('label') == 'standing_up' and phase == 'descending')
            code = ('step_departure' if reconciliation['code'] == 'observed_step' else
                    'action_phase_conflict' if opposite else
                    'reconciliation_pending' if reconciliation['code'] == 'warming_up' else
                    'motion_direction_conflict')
        if label != self.pending:
            self.pending, self.pending_since = label, t
        sustained_static = reconciliation['code'] in {'static_recovered', 'static_confirmed'}
        if label in {"standing", "seated"} and not sustained_static and t - self.pending_since < .25:
            label, reason = "unknown", "等待静态姿态持续稳定（至少 0.25 秒）"
            code = 'static_confirmation'
        previous_label, previous_reason = label, reason
        seat = self.seat_evidence.update(t, prediction, info)
        # Only replace an uncertain static result. Rise/down outputs remain exact.
        if label == "unknown" and seat["accepted"]:
            label, reason = "seated", seat["reason"]
            code = 'accepted'
        return dict(label=label, text=LABELS[label], reason=reason, target_index=target,
                    raw_label=raw_prediction.get("label"), phase=raw_prediction.get("phase", "unknown"),
                    resolved_label=prediction.get('label'), resolved_phase=prediction.get('phase', 'unknown'),
                    reconciliation={k: v for k, v in reconciliation.items() if k != 'prediction'},
                    decision_code=code, decision_stage=(
                        'accepted' if label != 'unknown' else 'history' if code == 'history_warmup' else
                        'quality' if code == 'pose_unreliable' else
                        'reconciliation' if reconciliation['changed'] else
                        'model' if code in {'model_unknown', 'phase_unknown'} else 'display'),
                    reset_reasons=resets,
                    metrics={} if info is None else {k: round(info[k], 4) for k in ('hip_speed', 'knee', 'hip_angle', 'tilt')},
                    evidence_s=round(t - self.history[0]["t"], 2),
                    previous_label=previous_label, previous_reason=previous_reason,
                    seat_evidence=seat,
                    hip_speed=None if info is None else round(info["hip_speed"], 3))


class SinglePersonActions(BaselineLiveActions):
    """Reconcile current evidence, then verify visible seated/rising support."""
    def __init__(self, model):
        super().__init__(model)
        self.visible_seat = VisibleSeat(model.base.base.base.config['seat_probability'])
        self.previous_visible_seat = PreviousVisibleSeat(model.base.base.base.config['seat_probability'])
        self.observed_rise = ObservedRise()
        self.daily_activity = DailyActivity()
        self.seat_rise_handoff = SeatRiseHandoff()

    def update(self, persons, t, width, height, camera_stable=True):
        result = super().update(persons, t, width, height)
        previous_label, previous_reason = result['label'], result['reason']
        tr = self.current_tracked
        person = tr['instances'][0] if len(persons) == 1 and tr['instances'] else None
        old = self.previous_visible_seat.update(t, tr['target_epoch'], person, self.current_prediction,
                                                result['label'], self.current_prediction is not None)
        if previous_label == 'seated' and old['veto']:
            previous_label, previous_reason = 'unknown', old['reason']
        elif previous_label == 'unknown' and old['accepted']:
            previous_label, previous_reason = 'seated', old['reason']
        support_prediction = (dict(self.current_prediction,
                                   source_label=self.current_raw_prediction['label'],
                                   observed_hip_speed=result.get('hip_speed'))
                              if self.current_prediction is not None else None)
        support = self.visible_seat.update(t, tr['target_epoch'], person, support_prediction,
                                          result['label'], self.current_prediction is not None)
        result['reset_reasons'].extend(support.get('reset_reasons', []))
        if result['label'] in {'unknown', 'seated'} and support.get('rising'):
            result.update(label='standing_up', text='起身', reason=support['reason'], decision_code='accepted', decision_stage='visible_support')
        elif result['label'] == 'seated' and support['veto']:
            result.update(label='unknown', text='未知', reason=support['reason'], decision_code='seat_support_veto', decision_stage='visible_support')
        elif result['label'] == 'unknown' and support['accepted']:
            result.update(label='seated', text='坐着', reason=support['reason'], decision_code='accepted', decision_stage='visible_support')
        verified_recovery = dict((self.current_prediction or {}).get('static_recovery') or {})
        descent = self.reconciler.static_recovery.descent
        if descent is not None:
            verified_recovery['origin_time_s'] = descent['origin']['t']
        rise = self.observed_rise.update(t, tr['target_epoch'],
                                        person if self.current_raw_prediction is not None else None,
                                        self.current_raw_prediction,
                                        verified_recovery)
        if result['label'] in {'unknown', 'seated'} and rise['rising']:
            result.update(label='standing_up', text='起身', reason=rise['reason'],
                          decision_code='accepted', decision_stage='rise_evidence')
            rise['applied'] = True
        elif ((result['label'] == 'seated' and rise['veto_seated'])
              or (result['label'] == 'standing_up' and rise['stop_rise'])):
            result.update(label='unknown', text='未知', reason=rise['reason'],
                          decision_code='rise_motion_stopped' if rise['stop_rise'] else 'rise_pending',
                          decision_stage='rise_evidence')
            rise['applied'] = True
        # A current, independently rechecked static window owns the handoff.
        # The rise completion latch alone is never enough to assert standing.
        recovery = (self.current_prediction or {}).get('static_recovery') or {}
        if recovery.get('stable') and recovery.get('phase') in {'standing', 'seated'}:
            label = recovery['phase']
            if label == 'standing' or not rise['rising'] and not rise['veto_seated']:
                result.update(label=label, text=LABELS[label], reason='当前躯干与腿部连续稳定，确认'+LABELS[label],
                              decision_code='accepted', decision_stage='static_handoff')
        result['static_evidence'] = dict(self.reconciler.static_recovery.details)
        if (result['label'] == 'seated' and not result['static_evidence'].get('stable')
                and (result['static_evidence'].get('current_motion')
                     or result['static_evidence'].get('code') == 'core_motion')):
            result.update(label='unknown',text='未知',reason='躯干或腿部已出现运动，退出静态坐姿并等待动作确认',
                          decision_code='static_motion',decision_stage='static_handoff')
        result.update(previous_label=previous_label, previous_reason=previous_reason,
                      visible_seat=support, previous_visible_seat=old, rise_evidence=rise,
                      phase_scores=(self.current_prediction or {}).get('phase_scores', {}))
        result = self.daily_activity.update(t, tr['target_epoch'], person, result, camera_stable)
        return self.seat_rise_handoff.update(t, tr['target_epoch'], person, result)


class LiveActions:
    """Compare all candidates with the selected subject under the same live rules."""
    def __init__(self, model):
        from .webcam_target import StableSubject
        self.model = model
        self.before = SinglePersonActions(model)
        self.selected = SinglePersonActions(model)
        self.selector = StableSubject()
        self.geometry_selector = StableSubject(folded_continuation=True)
        from .webcam_daily_activity import DailyActivity
        self.geometry_activity = DailyActivity()
        self.epoch = None
        self.last_t = None
        self.diagnostics = SessionDiagnostics()
        self.last_selected_t = None

    def update(self, persons, t, width, height, camera_stable=True):
        selection = self.selector.update(persons, t, width, height)
        previous = self.before.update(persons, t, width, height, camera_stable)
        self.last_t = t
        if selection['target_epoch'] != self.epoch:
            self.selected = SinglePersonActions(self.model)
            target_changed = self.epoch is not None
            self.epoch = selection['target_epoch']
        else:
            target_changed = False
        index = selection['target_index']
        retained = (index is None and not target_changed and self.last_selected_t is not None
                    and t-self.last_selected_t <= .30 and self.selector.anchor is not None)
        if retained:
            # Preserve history for a brief unresolved association, but output
            # unknown and feed NO invented observation into any action model.
            result = dict(label='unknown', text='未知', reason=selection['reason'],
                          raw_label='unavailable', phase='unknown', evidence_s=0,
                          reset_reasons=[], metrics={}, decision_code='target_unconfirmed', decision_stage='input')
        else:
            result = self.selected.update([] if index is None else [persons[index]], t, width, height, camera_stable)
        if index is not None:
            self.last_selected_t = t
        result['target_continuity'] = dict(history_retained=retained,
            gap_s=None if self.last_selected_t is None else round(t-self.last_selected_t,4))
        result['target_index'] = index
        if index is None:
            result.update(label='unknown', text='未知', reason=selection['reason'],
                          decision_code='target_unconfirmed', decision_stage='input')
        if target_changed:
            result['reset_reasons'].append('target_changed')
        result.update(previous_label=previous['label'], previous_reason=previous['reason'],
                      previous_raw_label=previous['raw_label'], previous_phase=previous['phase'],
                      previous_target_index=previous['target_index'], selection=selection)
        geometry_selection = self.geometry_selector.update(persons, t, width, height)
        geometry_index = geometry_selection['target_index']
        geometry_person = None
        if geometry_index is not None:
            raw = persons[geometry_index]
            geometry_person = dict(kpts=raw['keypoints'], box=raw['box'], det=raw['confidence'])
        geometry = self.geometry_activity.update(t, geometry_selection['target_epoch'],
                                                geometry_person, dict(result), camera_stable)
        # The auxiliary branch only recovers an explicit bend during unresolved
        # association, or shares confirmation with the same reacquired target.
        # It never injects folded poses into the four-action model,
        # invents keypoints, or changes raw votes / fall-target routing.
        if (geometry.get('label') == 'bending' and result.get('raw_label') != 'falling'
                and (index is None or (geometry_index == index
                     and result['label'] == 'unknown'
                     and result.get('decision_code') == 'daily_pose_pending'))):
            for key in ['label', 'text', 'reason', 'decision_code', 'decision_stage']:
                result[key] = geometry[key]
            result['reason'] = (geometry['reason'] if geometry.get('decision_stage') == 'frontal_bend'
                                else '腿部位置连续且直腿屈髋证据持续，独立姿态分支确认弯腰')
            result['view_bend'] = geometry.get('view_bend', {})
            result['auxiliary_posture'] = dict(label='bending', target_index=geometry_index,
                target_epoch=geometry_selection['target_epoch'], source_time=t)
        result['diagnostics'] = self.diagnostics.update(t, result, persons, width, height)
        return result
