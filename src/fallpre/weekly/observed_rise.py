"""Causal sit-to-stand onset from observed articulation, independent of phase lag.

This adapter does not feed its decisions back into the learned action or fall
models. Its seat reference is fixed: a pause partway up cannot become a new seat.
Every displacement uses the same visible shoulder/hip/knee/ankle chain.
"""
from collections import deque

import numpy as np

from .action_v3 import signals

SIDES = ((5, 11, 13, 15), (6, 12, 14, 16))


def observation(t, person):
    if person is None:
        return None
    points = np.asarray(person['kpts'], dtype=float)
    if (points.shape != (17, 3) or not np.isfinite(points).all()
            or not any(np.all(points[list(s), 2] >= .30) for s in SIDES)):
        return None
    shape, _, scale = signals(person)
    if scale < 12 or shape[0] >= .45:
        return None
    return dict(t=t, points=points.copy(), scale=float(scale), tilt=float(shape[0]),
                knee=float(shape[1]), hip_angle=float(shape[2]))


def movement(first, last, scale=None, fixed_sides=None):
    """Return current measured displacement; never switch sides within a pair."""
    p, q = first['points'], last['points']
    sides = [s for s in (SIDES if fixed_sides is None else fixed_sides) if np.all(p[list(s), 2] >= .30)
             and np.all(q[list(s), 2] >= .30)]
    if not sides:
        return None
    scale = max(float(scale or first['scale']), 1.)
    d = [(q[[s[i] for s in sides], :2].mean(0)
          - p[[s[i] for s in sides], :2].mean(0)) / scale for i in range(4)]
    shoulder, hip, knee, foot = d
    # Every available side must articulate; a jittering/occluded opposite side
    # must not be silently discarded to select the most favorable displacement.
    extension = min(float(((q[s[2], 1]-q[s[1], 1])
                           - (p[s[2], 1]-p[s[1], 1])) / scale) for s in sides)
    per_side = [(q[list(s), :2]-p[list(s), :2])/scale for s in sides]
    return dict(hip_up=min(float(d[3,1]-d[1,1]) for d in per_side),
                shoulder_up=min(float(d[3,1]-d[0,1]) for d in per_side),
                hip_absolute_up=min(float(-d[1,1]) for d in per_side),
                shoulder_absolute_up=min(float(-d[0,1]) for d in per_side),
                thigh_extension=extension, foot_motion=max(float(np.linalg.norm(d[3])) for d in per_side),
                horizontal=float(max(abs(hip[0]-foot[0]), abs(shoulder[0]-foot[0]))),
                knee_opening=last['knee']-first['knee'])


def upright(row):
    return row['knee'] > .90 and row['hip_angle'] > .80 and row['tilt'] < .25


class ObservedRise:
    """Fixed independently observed seat + two actual upward motion increments.

    Thresholds use torso lengths and elapsed source time, not frame counts alone.
    A short onset window supports a 7.5 Hz camera without waiting for the old
    0.65 s median to catch up. Current motion is still required after onset.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.anchor = None
        self.descent = None
        self.departed = False
        self.completed = False
        self.last_seat_t = None
        self.last_t = None
        self.epoch = None
        self.last_upright_t = None
        self.conflict_since = None
        self.last_rise_t = None

    def update(self, t, epoch, person, prediction, verified_recovery=None):
        reset_reason = None
        if self.epoch is not None and epoch != self.epoch:
            reset_reason = 'target_changed'
        elif self.last_t is not None and (t <= self.last_t or t-self.last_t > .35):
            reset_reason = 'source_gap'
        if reset_reason:
            self.reset()
        self.epoch, self.last_t = epoch, t
        details = dict(code='no_seat_reference', rising=False, veto_seated=False,
                       stop_rise=False, applied=False, anchor_available=False,
                       reset_reason=reset_reason, reason='等待独立坐姿或可见坐下后坐稳的参考')
        original = prediction or {}
        row = observation(t, person)
        if row is None:
            self.reset()
            self.epoch, self.last_t = epoch, t
            return dict(details, code='missing_pose', reason='缺少同侧肩髋膝踝，起身分支重新积累证据')
        if original.get('label') == 'falling':
            delta = movement(self.anchor, row) if self.anchor else None
            self.conflict_since = t if self.conflict_since is None else self.conflict_since
            retain = bool(delta and not self.departed and t-self.conflict_since <= .25
                          and abs(delta['hip_up']) <= .10 and -.06 <= delta['shoulder_up'] <= .35
                          and abs(delta['thigh_extension']) <= .10 and delta['knee_opening'] <= .06
                          and delta['foot_motion'] <= .08 and delta['horizontal'] <= .08)
            self.history.clear()  # A protected sample cannot contribute onset increments.
            if not retain:
                self.reset()
                self.epoch, self.last_t = epoch, t
            return dict(details, code='action_conflict_held' if retain else 'protected_action',
                        anchor_available=retain, reason='原始跌倒预测暂停起身输出；仅短时保留仍被观察到的坐姿参考')
        self.conflict_since = None
        if upright(row):
            self.last_upright_t = t
        scores = original.get('phase_scores') or {}
        row['seat_top'] = bool(scores and max(scores, key=scores.get) == 'seated')
        row['seat_emission'] = bool(scores and scores.get('seated', 0) >= .45
                                    and max(scores, key=scores.get) == 'seated')
        self.history.append(row)
        while self.history and t-self.history[0]['t'] > 2.:
            self.history.popleft()
        continuing_rise = bool(original.get('label') == 'walking' and self.last_rise_t is not None
                               and t-self.last_rise_t <= .35 and self.anchor is not None
                               and self.departed and not self.completed)
        if original.get('label') in {'walking', 'bending'} and not continuing_rise:
            # These actions cannot arm or emit a rise. Keep observation-only
            # history so a subsequent measured descent can still use its actual
            # upright origin when the frozen walking vote finishes late.
            self.anchor, self.descent, self.departed = None, None, True
            self.completed = False
            return dict(details, code='protected_action', reason='当前原始行走或弯腰动作不建立起身参考')
        # A new descent must begin at an independently observed upright body.
        # It is useful even when frontal sitting projects almost straight angles.
        if not upright(row):
            for origin in self.history:
                if not upright(origin) or not .2 <= t-origin['t'] <= 1.5:
                    continue
                m = movement(origin, row)
                if (m and -.65 <= m['hip_up'] <= -.12 and m['shoulder_up'] <= -.12
                        and m['hip_absolute_up'] <= -.12 and m['shoulder_absolute_up'] <= -.12
                        and m['foot_motion'] <= .12
                        and m['thigh_extension'] < -.06 and m['knee_opening'] < -.04):
                    if self.descent is None or row['points'][11:13, 1].mean() >= self.descent['points'][11:13, 1].mean():
                        self.descent = dict(row, origin_t=origin['t'])
                    break
        if self.descent and (t-self.descent['t'] > 2. or
                             (movement(self.descent, row) or {}).get('hip_up', 1) > .08):
            self.descent = None

        # At least three observations over .20 s must agree on a stationary seat.
        recent = [s for s in self.history if t-s['t'] <= .45+1e-8]
        stable = False
        if len(recent) >= 3 and t-recent[0]['t'] >= .20-1e-8:
            moves = [movement(recent[0], s) for s in recent]
            stable = all(m and abs(m['hip_up']) <= .035 and abs(m['shoulder_up']) <= .05
                         and m['foot_motion'] <= .08 and m['horizontal'] <= .05 for m in moves)
        supported_seat = (all(s['seat_emission'] and s['knee'] < .83 and s['hip_angle'] < .82
                              for s in recent) if recent else False)
        top_seated = []
        for s in reversed(recent):
            if not s['seat_top']:
                break
            top_seated.append(s)
        sustained_top = len(top_seated) >= 2 and t-top_seated[-1]['t'] >= .16-1e-8
        observed_descent = bool(self.descent and self.descent['origin_t'] < recent[0]['t']
                                and sustained_top)
        new_descent = bool(observed_descent and (self.last_seat_t is None or
                           self.descent['origin_t'] > self.last_seat_t))
        verified = verified_recovery or {}
        origin_t = verified.get('origin_time_s')
        recovered_descent = bool(verified.get('phase') == 'seated' and verified.get('stable')
                                 and verified.get('support') in {'observed_descent', 'verified_seat'}
                                 and verified.get('duration_s', 0) >= .35
                                 and verified.get('samples', 0) >= 4
                                 and isinstance(origin_t, (int, float)) and 0 <= origin_t <= t
                                 and (self.last_seat_t is None or origin_t > self.last_seat_t))
        seat_time = verified.get('seat_time_s')
        verified_seat = bool(verified.get('phase') == 'seated' and verified.get('stable')
                             and verified.get('support') in {'seated_emission', 'verified_seat'}
                             and verified.get('duration_s', 0) >= .35 and verified.get('samples', 0) >= 4
                             and isinstance(seat_time, (int, float)) and 0 <= seat_time <= t
                             and self.anchor is None
                             and (self.last_seat_t is None or
                                  (self.last_upright_t is not None
                                   and self.last_seat_t < self.last_upright_t < seat_time)))
        if recovered_descent or verified_seat or stable and (new_descent
                       or supported_seat and self.anchor is None and not self.departed):
            # An independently verified static window already establishes stability.
            # Preserve its descent origin across seat continuity so a completed
            # previous cycle cannot veto a newly verified seated cycle.
            # A new descent may rearm a completed cycle. A high seat score alone
            # cannot turn the elevated midpoint of a stopped rise into a seat.
            self.anchor = dict(row, confirmed_t=t,
                               kind='observed_descent' if new_descent else
                               'verified_descent' if recovered_descent else
                               'shared_verified_seat' if verified_seat else 'seated_geometry')
            self.departed = False
            self.completed = False
            self.last_seat_t = t
        if self.anchor is None:
            return details
        a = self.anchor
        if t-a['confirmed_t'] > 10.:
            self.anchor = None
            self.departed = True
            return dict(details, code='seat_reference_expired', reason='坐姿参考过期，等待新的完整坐下过程')
        short = [s for s in self.history if t-s['t'] <= .30+1e-8]
        # At camera rates a single unresolved association can leave only two
        # observed samples in .30 s. Use three REAL samples, with bounded gaps;
        # do not interpolate the missing observation or relax the two-step rule.
        extended = False
        if len(short) < 3 and len(self.history) >= 3:
            last_three = list(self.history)[-3:]
            if (t-last_three[0]['t'] <= .50+1e-8
                    and all(0 < y['t']-x['t'] <= .30+1e-8 for x,y in zip(last_three,last_three[1:]))):
                short = last_three
                extended = True
        common_sides = [s for s in SIDES if all(np.all(r['points'][list(s), 2] >= .30)
                                               for r in [a, *short])]
        m = movement(a, row, fixed_sides=common_sides)
        details.update(anchor_available=True, anchor_kind=a['kind'], anchor_age_s=round(t-a['t'], 4))
        if m is None:
            return dict(details, code='missing_common_chain', reason='当前关节与坐姿参考缺少同一侧完整关节链')
        details.update({k: round(v, 4) for k, v in m.items()})
        if (stable and all(upright(s) for s in recent)
                and m['hip_absolute_up'] > .12 and m['shoulder_absolute_up'] > .12):
            self.completed = True
            self.departed = True
        if self.completed:
            # Keep a stop latch for the older median-window rise tail. A stretch
            # after completed standing cannot reuse the previous seat anchor.
            return dict(details, code='standing_complete', standing_complete=True,
                        stop_rise=True, veto_seated=True,
                        reason='已连续站稳，上一轮起身结束；等待新的坐下过程')
        if (not self.departed and stable and sustained_top and abs(m['hip_up']) <= .04
                and abs(m['shoulder_up']) <= .06 and abs(m['thigh_extension']) <= .06):
            # Refresh the lifetime only, never drag the spatial reference up.
            a['confirmed_t'] = t
        # Preserve common joints over the complete short window, including its
        # earliest observation; alternating visible sides cannot invent motion.
        steps = [movement(x, y, a['scale'], common_sides) for x, y in zip(short, short[1:])]
        positive = []
        motion_start = 0
        for i, d in enumerate(steps):
            if d is None or d['hip_up'] < -.015 or d['shoulder_up'] < -.025:
                positive.clear()  # returning after a spike breaks the sequence
                motion_start = i+1
            elif (d['hip_up'] > .005 and d['shoulder_up'] > .005
                  and d['hip_absolute_up'] > .003 and d['shoulder_absolute_up'] > .003):
                positive.append(d)
        current = steps[-1] if steps else None
        dt = row['t']-short[-2]['t'] if len(short) >= 2 else 1.
        continues = bool(current and current['hip_up'] / dt > .04
                         and current['shoulder_up'] / dt > .05
                         and current['hip_absolute_up'] / dt > .025
                         and current['shoulder_absolute_up'] / dt > .025)
        duration = t-short[0]['t'] if short else 0.
        span = movement(short[motion_start], row, a['scale'], common_sides)
        window_moves = [movement(short[0], s, a['scale'], common_sides) for s in short]
        # A natural rise can transfer weight and move the free foot. Choose a
        # support foot by its stability across the entire window, never by the
        # most favorable rise displacement. BOTH visible chains must still rise
        # and extend; the free foot's movement is bounded as well.
        supporting_sides = [side for side in common_sides if all(
            (movement(short[0], s, a['scale'], [side]) or {}).get('foot_motion', float('inf')) <= .08
            for s in short)]
        current_feet_stable = bool(supporting_sides and window_moves and all(
            m and m['foot_motion'] <= .20 for m in window_moves))
        # After removing the largest step, at least one actual pixel (and a
        # scale-relative floor) of additional net ascent must remain. Thus a
        # jump plus subpixel plateau jitter cannot masquerade as two movements;
        # genuine small increments at 30 Hz may accumulate over the window.
        remainder = (min(span['hip_up']-max((d['hip_up'] for d in positive), default=0.),
                         span['shoulder_up']-max((d['shoulder_up'] for d in positive), default=0.))
                     if span else 0.)
        lift = (m['hip_up'] > .06 and m['shoulder_up'] > .06
                and m['hip_absolute_up'] > .04 and m['shoulder_absolute_up'] > .04
                and m['thigh_extension'] > .06 and current_feet_stable)
        details.update(motion_duration_s=round(duration, 4), positive_steps=len(positive),
                       additional_up=round(remainder, 4),
                       current_feet_stable=current_feet_stable,
                       current_continues=continues, window_extended=extended,
                       support_foot_count=len(supporting_sides), walking_tail=continuing_rise)
        sustained = bool(common_sides and len(positive) >= 2 and duration >= .10-1e-8
                         and remainder >= max(1./a['scale'], .01)-1e-8)
        if lift and sustained and continues:
            self.last_rise_t = t
            self.departed = True
            return dict(details, code='rise_confirmed', rising=True, veto_seated=True,
                        reason='从已观察坐姿连续抬髋、抬肩并伸腿；当前仍相对双脚上升')
        if self.departed:
            return dict(details, code='rise_stopped' if not continues else 'lift_pending',
                        veto_seated=True, stop_rise=not continues,
                        reason='已离开固定坐姿参考；当前上升停止，等待稳定站立或新的坐下证据'
                        if not continues else '离座后仍在移动，等待当前连续上升证据')
        if lift and continues:
            return dict(details, code='lift_pending', veto_seated=True,
                        reason='已观察抬髋和伸腿，等待第二次实际上升增量确认')
        return dict(details, code='seat_reference', reason='保留固定坐姿参考，尚无连续起身证据')
