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
    """Uses measured positions and a fixed seat reference; cannot emit standing_up.

    Visibility confidence and V4 seated-emission threshold are unchanged. An
    observed descent plus a current seated emission can replace unreliable 2-D
    knee angles. Partial observations can support an existing seat, not create it.
    A departure latch prevents partial observations from renewing a seat anchor
    while rising slowly; a fresh full-body confirmation may re-establish it.
    """
    def __init__(self, seat_probability):
        self.seat_probability = seat_probability
        self.reset()

    def reset(self):
        self.history = deque()
        self.descent = None
        self.anchor = None
        self.departed = False
        self.last_t = None
        self.epoch = None

    def update(self, t, epoch, person, prediction, baseline_label, full_pose):
        if epoch != self.epoch or self.last_t is not None and t - self.last_t > .35:
            self.reset()
        self.last_t, self.epoch = t, epoch
        current = visible_body(person)
        if current is None:
            self.reset()
            return dict(accepted=False, veto=False, reason='双肩、双髋或可见膝不足，不能验证坐姿')
        current['t'] = t
        self.history.append(current)
        while self.history and self.history[0]['t'] < t - .65:
            self.history.popleft()
        label = (prediction or {}).get('label', 'unavailable')
        phase = (prediction or {}).get('phase', 'unknown')
        if label in {'walking', 'bending', 'falling', 'standing_up'} or phase in {'rising', 'standing'}:
            self.anchor, self.descent = None, None
            self.departed = True
            return dict(accepted=False, veto=True, reason='原始模型已有离座或其他动作证据')
        # Record the start of a model-supported sit-down. A label alone is not
        # sufficient: later hips AND shoulders must have descended measurably.
        if label == 'sitting_down':
            if self.descent is None or t - self.descent['last_t'] > .35:
                self.descent = dict(origin=current, last_t=t, observed=False)
            self.descent['last_t'] = t
            start = self.descent['origin']
            down = (current['hip'][1] - start['hip'][1]) / start['scale']
            shoulders_down = (current['shoulder'][1] - start['shoulder'][1]) / start['scale']
            if t - start['t'] >= .3 and down >= .12 and shoulders_down >= .12:
                self.descent['observed'] = True
                self.departed = False
        details = dict(accepted=False, veto=False, reason='等待可见身体的稳定证据',
                       observed_descent=bool(self.descent and self.descent['observed']))
        if self.anchor is not None:
            a = self.anchor
            up = float((a['hip'][1] - current['hip'][1]) / a['scale'])
            shoulders_up = float((a['shoulder'][1] - current['shoulder'][1]) / a['scale'])
            departure = up > .06 or (up > .025 and shoulders_up > .025)
            if departure:
                self.anchor, self.descent, self.departed = None, None, True
                return dict(details, veto=True, reason='相对固定坐姿位置已抬髋，立即退出坐着',
                            hip_up=round(up, 3), shoulder_up=round(shoulders_up, 3))
            if t - a['confirmed_t'] > 10:
                self.anchor, self.departed = None, True
                return dict(details, veto=True, reason='坐姿参考已过期，需要重新观察完整坐姿或坐下过程')
        if len(self.history) < 4 or t - self.history[0]['t'] < .4:
            return details
        samples = list(self.history)
        scale = float(np.median([s['scale'] for s in samples]))
        n = max(1, len(samples) // 3)
        times = np.array([s['t'] for s in samples])
        hips = np.array([s['hip'] for s in samples])
        shoulders = np.array([s['shoulder'] for s in samples])
        dt = max(float(np.median(times[-n:]) - np.median(times[:n])), .05)
        hip_speed = float((np.median(hips[:n, 1]) - np.median(hips[-n:, 1])) / scale / dt)
        shoulder_speed = float((np.median(shoulders[:n, 1]) - np.median(shoulders[-n:, 1])) / scale / dt)
        hip_spread = float((np.quantile(hips[:, 1], .9) - np.quantile(hips[:, 1], .1)) / scale)
        stable = abs(hip_speed) <= .10 and abs(shoulder_speed) <= .15 and hip_spread <= .10
        details.update(trend_speed=round(hip_speed, 3), shoulder_speed=round(shoulder_speed, 3),
                       stable=stable, hip_spread=round(hip_spread, 3))
        # Require current evidence too, not just a lagging median window.
        if hip_speed > .10 and shoulder_speed > .10:
            # A short trend veto does not erase the reference. The fixed-position
            # departure check above latches a supported displacement instead.
            return dict(details, veto=True, reason='肩髋持续上升，不能继续显示坐着')
        if not stable:
            return dict(details, reason='肩髋仍在移动，尚未坐稳')
        scores = (prediction or {}).get('phase_scores', {})
        emission = bool(scores and max(scores, key=scores.get) == 'seated'
                        and scores['seated'] >= self.seat_probability)
        descent_support = (self.descent is not None and self.descent['observed']
                           and t - self.descent['last_t'] <= 2)
        if descent_support:
            start = self.descent['origin']
            descent_support = ((current['hip'][1] - start['hip'][1]) / start['scale'] >= .12
                               and (current['shoulder'][1] - start['shoulder'][1]) / start['scale'] >= .12)
        frontal = (full_pose and label == 'background' and emission and descent_support)
        if full_pose and baseline_label == 'seated':
            # A new complete observation independently accepted by the previous
            # pipeline may re-establish a seat after posture adjustment. Partial
            # observations can NEVER re-arm themselves or move their reference.
            self.departed = False
        if self.anchor is None and not self.departed and (baseline_label == 'seated' or frontal):
            self.anchor = dict(current)  # immutable: never follow slow ascent
            self.anchor['kind'] = 'front' if frontal else 'previous_seat'
            self.anchor['confirmed_t'] = t
        if self.anchor is None or self.departed:
            return dict(details, reason='没有完整坐姿或已观察坐下后的坐稳证据', seated_emission=emission)
        a = self.anchor
        common = set(current['thighs']) & set(a['thighs'])
        thigh_change = min((float(np.linalg.norm(current['thighs'][i] - a['thighs'][i]) / a['scale'])
                            for i in common), default=float('inf'))
        hip_shift = float(np.linalg.norm(current['hip'] - a['hip']) / a['scale'])
        details.update(anchor_age_s=round(t-a['confirmed_t'], 3), anchor_kind=a['kind'])
        if thigh_change > .10 or hip_shift > .08:
            return dict(details, reason='可见大腿或髋位已偏离坐姿参考')
        if not full_pose:
            return dict(details, accepted=True, reason='局部遮挡：已有坐姿证据，当前髋位和可见大腿仍稳定')
        if baseline_label == 'seated' or label == 'background' and a['kind'] == 'front' and emission:
            # Renew confidence from a full observation, NEVER move the position
            # reference. Partial observations cannot renew their own anchor.
            a['confirmed_t'] = t
        if label != 'background':
            return dict(details, reason='原始模型尚未结束动作或证据不足')
        if a['kind'] == 'front' and emission:
            return dict(details, accepted=True, reason='已观察坐下及坐稳，坐姿分数支持正面坐姿；不依赖二维膝角')
        return dict(details, reason='当前模型坐姿证据不足，保留原判断')
