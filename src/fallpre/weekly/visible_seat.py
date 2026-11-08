"""Observed sit-down and visible-body support, separate from frozen V7 outputs."""
from collections import deque

import numpy as np


def visible_body(person):
    """No fallback coordinates: both shoulders/hips and a current knee are needed."""
    if person is None:
        return None
    p = np.asarray(person['kpts'], dtype=float)
    if not np.all(p[[5, 6, 11, 12], 2] >= .30):
        return None
    knees = [i for i in (13, 14) if p[i, 2] >= .30]
    if not knees:
        return None
    shoulder, hip = p[5:7, :2].mean(0), p[11:13, :2].mean(0)
    scale = float(np.linalg.norm(hip - shoulder))
    if scale < 12 or hip[1] <= shoulder[1]:
        return None
    return dict(hip=hip, shoulder=shoulder, scale=scale,
                thighs={i: p[i, :2] - p[i-2, :2] for i in knees})


class VisibleSeat:
    """Causal visible-body evidence; frozen model scores are never modified.

    A fixed confirmed seat reference survives small measurement noise. Rising
    requires sustained shoulder/hip ascent AND relative thigh extension, plus
    a prior seat or repeated raw-model rise votes. No missing joint is filled.
    """
    def __init__(self, seat_probability):
        self.seat_probability = seat_probability
        self.reset()

    def reset(self):
        self.history = deque()
        self.descent = None
        self.anchor = None
        self.departed = False
        self.seat_recovery = None
        self.last_t = None
        self.epoch = None

    def update(self, t, epoch, person, prediction, baseline_label, full_pose):
        reset_reasons = []
        if self.epoch is not None and epoch != self.epoch:
            reset_reasons.append('target_changed')
        if self.last_t is not None and t - self.last_t > .35:
            reset_reasons.append('source_gap')
        if epoch != self.epoch or self.last_t is not None and t - self.last_t > .35:
            self.reset()
        self.last_t, self.epoch = t, epoch
        current = visible_body(person)
        if current is None:
            self.reset()
            return dict(accepted=False, veto=False, rising=False,
                        reset_reasons=reset_reasons + ['seat_joints_missing'],
                        reason='双肩、双髋或可见膝不足，不能验证坐姿或起身')
        label = (prediction or {}).get('label', 'unavailable')
        source_label = (prediction or {}).get('source_label', label)
        phase = (prediction or {}).get('phase', 'unknown')
        scores = (prediction or {}).get('phase_scores', {})
        emission = bool(scores and max(scores, key=scores.get) == 'seated'
                        and scores['seated'] >= self.seat_probability)
        recovered = (prediction or {}).get('static_recovery') or {}
        observed_recovery = bool(recovered.get('phase') == 'seated' and recovered.get('stable')
                                 and recovered.get('support') == 'observed_descent')
        recovery_supported = (full_pose and baseline_label == 'seated' and label == 'background'
                              and phase == 'seated' and (emission or observed_recovery))
        if not recovery_supported:
            self.seat_recovery = None
        current.update(t=t, raw_label=source_label)
        self.history.append(current)
        while self.history and self.history[0]['t'] < t - .65:
            self.history.popleft()
        details = dict(accepted=False, veto=False, rising=False, motion='insufficient',
                       reset_reasons=reset_reasons,
                       reason='等待连续可见关节证据')
        if label in {'walking', 'bending', 'falling'} or phase == 'standing':
            if self.anchor is not None:
                reset_reasons.append('seat_anchor_invalidated')
            self.anchor, self.descent, self.departed = None, None, True
            return dict(details, veto=True, reason='原始模型已有站立或其他动作证据')
        if label == 'sitting_down' or source_label == 'sitting_down':
            if self.descent is None or t - self.descent['last_t'] > .35:
                self.descent = dict(origin=current, last_t=t, observed=False)
            self.descent['last_t'] = t
            start = self.descent['origin']
            if (t - start['t'] >= .3
                    and (current['hip'][1] - start['hip'][1]) / start['scale'] >= .12
                    and (current['shoulder'][1] - start['shoulder'][1]) / start['scale'] >= .12):
                self.descent['observed'] = True
                self.departed = False
        details['observed_descent'] = bool(self.descent and self.descent['observed'])
        if self.anchor is not None and t - self.anchor['confirmed_t'] > 10:
            reset_reasons.append('seat_anchor_expired')
            self.anchor, self.departed = None, True
            return dict(details, veto=True, reason='坐姿参考已过期，需要重新观察完整坐姿或坐下过程')
        if len(self.history) < 4 or t - self.history[0]['t'] < .4:
            self.seat_recovery = None
            return dict(details, veto=label == 'standing_up' or phase == 'rising')

        samples = list(self.history)
        scale = float(np.median([s['scale'] for s in samples]))
        n = max(1, len(samples) // 3)
        first, last = samples[:n], samples[-n:]
        def median(group, key):
            return np.median([s[key] for s in group], axis=0)
        dt = max(float(median(last, 't') - median(first, 't')), .05)
        hip_delta = float(median(first, 'hip')[1] - median(last, 'hip')[1])
        shoulder_delta = float(median(first, 'shoulder')[1] - median(last, 'shoulder')[1])
        hip_speed, shoulder_speed = hip_delta / scale / dt, shoulder_delta / scale / dt
        hip_spread = float(np.ptp(np.quantile([s['hip'][1] for s in samples], [.1, .9])) / scale)
        stable = abs(hip_speed) <= .10 and abs(shoulder_speed) <= .15 and hip_spread <= .10
        if not stable:
            self.seat_recovery = None
        # Relative knee-to-hip motion rejects whole-body/image translation. Use
        # every currently visible side; losing one side cannot fabricate a knee.
        common = set(current['thighs'])
        for s in first + last:
            common &= set(s['thighs'])
        extensions = [float((np.median([s['thighs'][i][1] for s in last])
                             - np.median([s['thighs'][i][1] for s in first])) / scale) for i in common]
        extension = min(extensions, default=0.)
        # A two-pixel endpoint floor prevents low-resolution point jitter being
        # amplified by torso normalization; it does not relax pose confidence.
        motion = (hip_delta > max(2., .06*scale) and shoulder_delta > max(2., .06*scale)
                  and hip_speed > .10 and shoulder_speed > .10 and extension > .06)
        # At least three recent samples must support displacement, not one jump.
        origin_hip, origin_shoulder = median(first, 'hip'), median(first, 'shoulder')
        support = [s for s in samples if origin_hip[1]-s['hip'][1] > max(2., .06*scale)
                   and origin_shoulder[1]-s['shoulder'][1] > max(2., .06*scale)
                   and common and common.issubset(s['thighs']) and all((s['thighs'][i][1] - np.median([x['thighs'][i][1] for x in first])) / scale > .06
                                      for i in common)]
        sustained = len(support) >= 3 and support[-1]['t'] - support[0]['t'] >= .20-1e-7
        # Reject a transient spike whose current observation has already returned.
        current_continues = (bool(support) and support[-1]['t'] == t
                             and origin_hip[1]-current['hip'][1] > max(2., .06*scale)
                             and origin_shoulder[1]-current['shoulder'][1] > max(2., .06*scale))
        raw_rise = [s for s in samples if s['raw_label'] == 'standing_up']
        raw_context = source_label == 'standing_up' and len(raw_rise) >= 2 and raw_rise[-1]['t']-raw_rise[0]['t'] >= .1
        seat_context = False
        if self.anchor is not None:
            a = self.anchor
            up = float((a['hip'][1]-current['hip'][1]) / a['scale'])
            shoulder_up = float((a['shoulder'][1]-current['shoulder'][1]) / a['scale'])
            seat_context = up > .10 and shoulder_up > .10
            details.update(hip_up=round(up, 3), shoulder_up=round(shoulder_up, 3),
                           anchor_age_s=round(t-a['confirmed_t'], 3), anchor_kind=a['kind'])
            # The long robust seat window deliberately filters jitter, but must
            # not keep a seat once the current full-pose estimate has begun a
            # supported lift. Abstain immediately; emitting rise still needs
            # the stronger sustained motion rule below.
            observed_speed = (prediction or {}).get('observed_hip_speed')
            common_anchor = set(current['thighs']) & set(a['thighs'])
            current_extension = min(((current['thighs'][i][1] - a['thighs'][i][1]) / a['scale']
                                     for i in common_anchor), default=0.)
            early_lift = (full_pose and observed_speed is not None and observed_speed > .10
                          and up > .015 and shoulder_up > .04 and current_extension > .04)
        else:
            early_lift = False
        details.update(trend_speed=round(hip_speed, 3), shoulder_speed=round(shoulder_speed, 3),
                       thigh_extension=round(extension, 3), stable=stable, hip_spread=round(hip_spread, 3),
                       rise_support_s=round(support[-1]['t']-support[0]['t'], 3) if support else 0,
                       rise_source='seat_history' if seat_context else 'raw_model' if raw_context else 'none')
        if motion and sustained and current_continues and (seat_context or raw_context):
            # Keep the fixed reference as context for the remainder of ascent,
            # but latch departure so a slow rise can never renew its own seat.
            self.departed = True
            return dict(details, veto=True, rising=True, motion='sustained_lift',
                        reason='持续抬髋、肩部同步上升且可见大腿伸展；坐姿历史或原模型起身判断支持')
        if early_lift:
            return dict(details, veto=True, motion='lift_pending',
                        reason='当前已观察到抬髋、肩部上升和大腿伸展，暂停坐姿保持并等待起身确认')
        if label == 'standing_up' or phase == 'rising':
            return dict(details, veto=True, motion='rise_pending', reason='原模型提示起身，等待持续抬髋和大腿伸展确认')

        # Departure vetoes persistence, but must not permanently veto a new,
        # independently confirmed full-body seat. Reacquire only after sustained
        # model, geometry and stability agreement; a partial pose cannot rearm it.
        if self.departed and recovery_supported and stable:
            if self.seat_recovery is None:
                self.seat_recovery = dict(since=t, count=0)
            self.seat_recovery['count'] += 1
            recovery_s = t - self.seat_recovery['since']
            details['seat_recovery_s'] = round(recovery_s, 3)
            independently_stable = (recovered.get('phase') == 'seated' and recovered.get('stable')
                                     and recovered.get('support') in {'observed_descent', 'seated_emission'}
                                     and recovered.get('duration_s', 0) >= .35
                                     and recovered.get('samples', 0) >= 4)
            if independently_stable or recovery_s >= .4 - 1e-7 and self.seat_recovery['count'] >= 4:
                self.anchor = dict(current, kind='reconfirmed_seat', confirmed_t=t)
                self.descent = None
                self.departed = False
                self.seat_recovery = None
                details['seat_recovered'] = True
            else:
                return dict(details, veto=True, reason='完整坐姿已重新出现，等待至少 0.4 秒连续稳定确认')
        descent_support = self.descent is not None and self.descent['observed'] and t-self.descent['last_t'] <= 2
        if descent_support:
            start = self.descent['origin']
            descent_support = ((current['hip'][1]-start['hip'][1]) / start['scale'] >= .12
                               and (current['shoulder'][1]-start['shoulder'][1]) / start['scale'] >= .12)
        frontal = full_pose and label == 'background' and emission and descent_support
        if self.anchor is None and not self.departed and stable and (baseline_label == 'seated' or frontal):
            self.anchor = dict(current, kind='front' if frontal else 'previous_seat', confirmed_t=t)
        if self.anchor is None or self.departed:
            return dict(details, veto=self.departed, reason='没有可用坐姿参考，或已确认离座', seated_emission=emission)
        a = self.anchor
        # Recent medians filter single-frame jitter, while current large geometry
        # changes still immediately abstain. Coordinates in the reference NEVER
        # follow the subject upward, even during prolonged partial occlusion.
        recent = [s for s in samples if t-s['t'] <= .2+1e-7]
        common = set(current['thighs']) & set(a['thighs'])
        for s in recent:
            common &= set(s['thighs'])
        thigh_change = min((float(np.linalg.norm(np.median([s['thighs'][i] for s in recent], axis=0)-a['thighs'][i]))
                            for i in common), default=float('inf'))
        # Check horizontal adjustment and vertical lift separately; diagonal
        # pixel noise must not count twice toward a Euclidean-distance gate.
        hip_shift = float(np.max(np.abs(median(recent, 'hip')-a['hip'])))
        current_shift = float(np.max(np.abs(current['hip']-a['hip'])))
        current_thigh = min((float(np.linalg.norm(current['thighs'][i]-a['thighs'][i])) for i in common), default=float('inf'))
        details.update(hip_shift=round(hip_shift/a['scale'], 3), thigh_change=round(thigh_change/a['scale'], 3) if np.isfinite(thigh_change) else None)
        compatible = hip_shift <= max(3., .08*a['scale']) and thigh_change <= max(3., .10*a['scale'])
        abrupt = current_shift > max(5., .12*a['scale']) or current_thigh > max(5., .16*a['scale'])
        if not compatible and not abrupt and stable and full_pose and baseline_label == 'seated':
            a['confirmed_t'] = t
            return dict(details, reason='完整骨架独立确认坐姿；保留固定位置参考')
        if not compatible or abrupt:
            if (a['hip'][1]-current['hip'][1])/a['scale'] > .12 and (a['shoulder'][1]-current['shoulder'][1])/a['scale'] > .10:
                self.departed = True
            return dict(details, veto=True, motion='ambiguous_motion', reason='髋位或可见大腿偏离坐姿，尚不足以确认起身')
        # A partial pose may maintain an established seat for the original 10 s
        # lifetime; full poses without fresh score support have a short 1 s grace.
        if full_pose and label not in {'background', 'unknown'}:
            return dict(details, reason='原模型动作尚未结束')
        if full_pose and (baseline_label == 'seated' or label == 'background' and emission):
            a['confirmed_t'] = t
        if full_pose and t-a['confirmed_t'] > 1:
            return dict(details, reason='当前坐姿模型证据持续不足，停止短暂保持')
        kind = 'leaning' if abs(shoulder_speed) > .15 and abs(hip_speed) <= .10 else 'seat_adjustment'
        return dict(details, accepted=True, motion=kind, reason=(
            '髋位和大腿仍支持坐姿；肩部前倾不等于起身' if kind == 'leaning' else
            '近期髋位和大腿保持坐姿；短时点位抖动或轻微调整不触发离座'))
