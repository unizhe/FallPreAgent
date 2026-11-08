"""Causal reconciliation of frozen action and phase outputs for the live adapter.

This module never modifies the frozen prediction. A phase alone is insufficient:
transitions need observed articulation relative to the feet, and static states
need a stable pose window. It cannot distinguish a chair from a squat by itself.
"""
import math
from collections import deque

import numpy as np


def stopped_upright(samples):
    """Short walking-exit proof from body position and a fixed support foot.

    This stricter proof can supersede a stale walking vote before the ordinary
    .60 s window. It cannot be established by posture alone or switching sides.
    """
    if len(samples) < 4 or samples[-1]['t']-samples[0]['t'] < .35-1e-8:
        return False
    if not all(s['knee'] > .90 and s['hip_angle'] > .80 and s['tilt'] < .25 for s in samples):
        return False
    p = np.stack([s['points'] for s in samples])
    sides = [s for s in ((5,11,13,15), (6,12,14,16)) if np.all(p[:,list(s),2] >= .30)]
    if not sides:
        return False
    scale = max(float(np.median([s['scale'] for s in samples])), 1.)
    feet = [float(np.linalg.norm(np.ptp(p[:,s[3],:2],axis=0))/scale) for s in sides]
    core = [j for s in sides for j in s[:2]]
    return bool(min(feet) <= .06 and max(feet) <= .18
                and np.max(np.ptp(p[:,core,0],axis=0))/scale <= .10
                and np.max(np.ptp(p[:,core,1],axis=0))/scale <= .06
                and np.ptp([s['knee'] for s in samples]) <= .07
                and np.ptp([s['hip_angle'] for s in samples]) <= .08)


def brief_settle(samples, candidate, supported_seat=False):
    """A short, stricter proof from real observations, including the stop anchor.

    The first observation may end the preceding motion window; only increments
    after it matter here. Upright weight shifts may move the torso over planted
    feet, but steps, vertical transitions and changing leg shape cannot qualify.
    A new seat additionally needs a previously observed descent or fixed seat.
    """
    if len(samples) < 3 or samples[-1]['t']-samples[0]['t'] < .22-1e-8:
        return False
    if any(not 0 < b['t']-a['t'] <= .20 for a,b in zip(samples,samples[1:])):
        return False
    if any(s.get('candidate') != candidate for s in samples):
        return False
    if any(current_step(a,b) for a,b in zip(samples,samples[1:])):
        return False
    p = np.stack([s['points'] for s in samples])
    sides = [side for side in ((5,11,13,15),(6,12,14,16))
             if np.all(p[:,list(side),2] >= .30)]
    if not sides:
        return False
    scale = max(float(np.median([s['scale'] for s in samples])),1.)
    spread = np.ptp(p[:,:,:2],axis=0)/scale
    feet = [float(np.linalg.norm(spread[side[3]])) for side in sides]
    core = [j for side in sides for j in side[:2]]
    if candidate == 'standing':
        return bool(all(s['knee'] > .94 and s['hip_angle'] > .90 and s['tilt'] < .15 for s in samples)
                    and min(feet) <= .05 and max(feet) <= .20
                    and np.max(spread[core,1]) <= .08
                    and np.max(spread[core,0]) <= (.35 if max(feet) <= .12 else .10)
                    and np.ptp([s['knee'] for s in samples]) <= .035
                    and np.ptp([s['hip_angle'] for s in samples]) <= .04)
    return bool(candidate == 'seated' and supported_seat
                and min(feet) <= .04 and max(feet) <= .10
                and all(spread[side[1],1] <= .035 and spread[side[1],0] <= .05
                        and np.max(spread[side[2]]) <= (.12 if all(s['knee'] < .80 and s['hip_angle'] < .80 for s in samples) else .08)
                        and spread[side[0],1] <= .06 and spread[side[0],0] <= .12 for side in sides)
                and np.ptp([s['knee'] for s in samples]) <= .06
                and np.ptp([s['hip_angle'] for s in samples]) <= .05)


def articulated_descent(samples):
    """Independent descent proof when the frozen phase has not caught up.

    Three real downward increments, fixed feet and shortening thighs are
    required on every common side. This only supplies a descent reference;
    seat height, stability and departure checks still decide whether to sit.
    """
    if len(samples) < 4 or samples[-1]['t']-samples[0]['t'] < .25:
        return False
    p = np.stack([s['points'] for s in samples])
    sides = [s for s in ((5,11,13,15),(6,12,14,16)) if np.all(p[:,list(s),2] >= .30)]
    scale = max(samples[0]['scale'], 1.)
    if not sides or samples[0]['knee']-samples[-1]['knee'] < .06:
        return False
    for side in sides:
        q = p[:,list(side),:2]/scale
        relative = q[:,:2,1]-q[:,3:4,1]
        increments = np.diff(relative,axis=0)
        if (np.min(relative[-1]-relative[0]) < .18
                or np.count_nonzero(np.all(increments > .008,axis=1)) < 3
                or np.any(increments < -.05)
                or np.max(np.linalg.norm(q[:,3]-q[0,3],axis=1)) > .08
                or (q[0,2,1]-q[0,1,1])-(q[-1,2,1]-q[-1,1,1]) < .12):
            return False
    return True


def current_step(previous, current):
    """A planted opposite foot must not hide a new, coherent stepping leg."""
    dt = current['t']-previous['t']
    if not 0 < dt <= .30:
        return False
    p, q = previous['points'], current['points']
    for side in ((5,11,13,15),(6,12,14,16)):
        if not np.all(p[list(side),2] >= .30) or not np.all(q[list(side),2] >= .30):
            continue
        d = (q[list(side),0]-p[list(side),0])/max(previous['scale'],1.)
        hip, knee, foot = d[1:]
        if (abs(foot) > max(.03,.50*dt) and abs(hip) > max(.02,.45*dt)
                and abs(knee) > max(.02,.25*dt) and hip*foot > 0 and knee*foot > 0):
            return True
    return False


def anchored_upright(old, current):
    """Continue an established upright body despite one unstable ankle.

    All common torso chains stay near the fixed standing reference. At least
    one of those same feet must remain planted; the opposite ankle alone cannot
    manufacture vertical motion. A coherent step is rejected by the caller.
    """
    if (old['phase'] != 'standing' or not 0 < current['t']-old['t'] <= .20
            or current['knee'] <= .94 or current['hip_angle'] <= .90 or current['tilt'] >= .15):
        return False
    p, q, fixed = old['points'], current['points'], old['fixed_points']
    sides = [side for side in ((5,11,13,15),(6,12,14,16))
             if all(np.all(a[list(side),2] >= .30) for a in (p,q,fixed))]
    if not sides:
        return False
    scale = max(old['scale'],1.)
    planted = False
    for side in sides:
        step = (q[list(side),:2]-p[list(side),:2])/scale
        offset = (q[list(side),:2]-fixed[list(side),:2])/scale
        planted |= bool(np.linalg.norm(offset[3]) <= .08 and np.linalg.norm(step[3]) <= .05)
        if (np.max(np.abs(offset[:2,1])) > .10 or np.max(np.abs(offset[:2,0])) > .20
                or np.max(np.abs(step[:2,1])) > .09
                or np.max(np.abs(step[:2,0])) > .08):
            return False
    return planted


def anchored_seat(anchor, current):
    """A verified planted descent can tolerate foot repositioning.

    This does not identify a chair from a bent pose: the anchor must come from
    the independently measured descent, and all visible hips remain fixed.
    Shoulder rotation is constrained by current tilt/hip angle, not displacement
    from the initially forward-leaning seated frame.
    """
    if (not anchor.get('planted_descent') or not 0 <= current['t']-anchor['last_seen'] <= .35
            or current['knee'] >= .80 or current['hip_angle'] >= .80 or current['tilt'] >= .25):
        return False
    p, q = anchor['points'], current['points']
    sides = [side for side in ((5,11,13,15),(6,12,14,16))
             if all(np.all(a[list(side),2] >= .30) for a in (p,q))]
    if not sides:
        return False
    for side in sides:
        d = (q[list(side),:2]-p[list(side),:2])/max(anchor['scale'],1.)
        if (abs(d[1,0]) > .10 or not -.04 <= d[1,1] <= .10 or np.max(np.abs(d[2])) > .15
                or np.linalg.norm(d[3]) > .25):
            return False
    return True


class StaticRecovery:
    """Independent observed stability can repair a stale action/phase pair.

    The window contains actual observations only. Upright geometry is useful even
    when the phase forest says ``other``; a bent pose needs additional seated or
    observed-descent evidence, since bending alone cannot identify a seat.
    """

    def __init__(self):
        self.history = deque()
        self.stable = deque()
        self.descent = None
        self.seat = None
        self.posture = None
        self.details = {}
        self.conflict_since = None

    def clear(self):
        self.history.clear()
        self.stable.clear()
        self.descent = None
        self.seat = None
        self.posture = None
        self.details = {}
        self.conflict_since = None

    def update(self, t, prediction, values, points):
        self.details = dict(code='static_geometry', stable=False)
        label, phase = prediction.get('label'), prediction.get('phase')
        if label == 'falling':
            # A raw action vote is not the independent fall alarm. Quarantine a
            # brief contradiction only while the fixed verified seat is still
            # observed; never emit a static state on the conflicting frame.
            from .observed_rise import movement
            current = dict(points=points, knee=values['knee'])
            anchor = self.seat
            delta = movement(anchor, current) if anchor and current else None
            self.conflict_since = t if self.conflict_since is None else self.conflict_since
            retain = bool(delta and t-anchor['last_seen'] <= .25
                          and t-self.conflict_since <= .25 and values['tilt'] < .25
                          and abs(delta['hip_up']) <= .10 and -.06 <= delta['shoulder_up'] <= .35
                          and abs(delta['thigh_extension']) <= .10 and delta['knee_opening'] <= .06
                          and delta['foot_motion'] <= .08 and delta['horizontal'] <= .08)
            # A single action-classifier conflict during a visible upright
            # descent must not erase its measured origin. Retain observations
            # only, never a static output/confirmation, for at most .25 s.
            descent_history = bool(self.history and t-self.history[-1]['t'] <= .25
                and t-self.conflict_since <= .25 and values['tilt'] < .25
                and values['knee'] > .65 and values['hip_angle'] > .65
                and sum(s['descent_vote'] for s in self.history) >= 2
                and any(s['knee'] > .90 and s['hip_angle'] > .80 for s in self.history))
            # Quarantine a short classifier conflict without discarding an
            # already measured descent. This preserves evidence, never a label:
            # subsequent non-conflicting frames must still prove a stable seat.
            # Use fixed image geometry, not the stale normalized speed vote.
            descent_hold = False
            if self.descent is not None and t-self.descent['t'] <= .40 and t-self.conflict_since <= .25:
                origin = self.descent['origin']
                scale = max(origin['scale'], 1.)
                sides = [s for s in ((5,11,13,15),(6,12,14,16))
                         if np.all(points[list(s),2] >= .50)
                         and np.all(origin['points'][list(s),2] >= .50)]
                descent_hold = bool(sides and values['tilt'] < .25
                    and .40 < values['knee'] < .90 and .40 < values['hip_angle'] < .90
                    and all(np.linalg.norm(points[a,:2]-self.descent.get('conflict_points', self.descent['points'])[a,:2])/scale <= .05
                        and np.linalg.norm(points[a,:2]-self.descent['points'][a,:2])/scale <= .08
                        and (points[h,1]-origin['points'][h,1])/scale >= .12
                        and (points[s,1]-origin['points'][s,1])/scale >= .12
                        and (points[a,1]-points[h,1])/scale >= .75
                        for s,h,k,a in sides))
            if descent_hold:
                # Keep the last measured foot positions for the next adjacent
                # observation, while bounding total drift against the fixed
                # descent reference. Neither descent time nor origin advances.
                self.descent['conflict_points'] = points.copy()
                self.stable.clear()
                self.posture = None
                self.details = dict(code='descent_conflict_held', stable=False, reference_retained=True)
                return None
            if not retain and not descent_history:
                self.clear()
            elif descent_history and not retain:
                self.stable.clear()
                self.posture = self.seat = self.descent = None
            retain = retain or descent_history
            self.details = dict(code='action_conflict_held' if retain else 'protected_action',
                                stable=False, reference_retained=retain)
            return None
        self.conflict_since = None
        scores = prediction.get('phase_scores') or {}
        try:
            seated_score = float(scores.get('seated', 0))
            score_values = [float(v) for v in scores.values()]
        except (TypeError, ValueError):
            seated_score, score_values = 0., []
        seated_top = (math.isfinite(seated_score) and seated_score > 0 and bool(score_values)
                      and all(math.isfinite(v) and seated_score >= v for v in score_values))
        row = dict(t=t, points=points.copy(), **values)
        row['descent_vote'] = label == 'sitting_down' or phase == 'descending'
        row['seated_emission'] = seated_top and seated_score >= .45
        self.history.append(row)
        while self.history and t - self.history[0]['t'] > 1.5:
            self.history.popleft()
        standing = values['knee'] > .90 and values['hip_angle'] > .80 and values['tilt'] < .25
        seated = values['knee'] < .83 and values['hip_angle'] < .82 and values['tilt'] < .45
        # A fixed, previously verified seat may persist, but never follow a
        # rising/descending person or bridge a missing common leg chain.
        seat_continues = False
        pelvis_seat = False
        if self.seat is not None:
            common = [s for s in ((5, 11, 13, 15), (6, 12, 14, 16))
                      if all(points[j, 2] >= .30 and self.seat['points'][j, 2] >= .30 for j in s)]
            if common and t-self.seat['last_seen'] <= .35:
                delta = [(points[list(s), :2]-self.seat['points'][list(s), :2]) / self.seat['scale'] for s in common]
                seat_continues = (values['tilt'] < .45 and all(
                    -.10 <= (d[1, 1]-d[3, 1]) <= .10 and np.linalg.norm(d[3]) <= .12
                    and abs(d[1, 0]-d[3, 0]) <= .12 for d in delta))
            pelvis_seat = bool(label in {'background','unknown'} and anchored_seat(self.seat,row))
            seat_continues = seat_continues or pelvis_seat
            if seat_continues:
                self.seat['last_seen'] = t  # Current fixed-position evidence, even while output waits.
            else:
                self.seat = None
        departed_descent = False
        if self.descent is not None:
            anchor = self.descent
            common_hips = [j for j in (11, 12) if points[j, 2] >= .30 and anchor['points'][j, 2] >= .30]
            if not common_hips or (anchor['points'][common_hips, 1].mean()
                                   - points[common_hips, 1].mean()) / max(anchor['scale'], 1.) > .08:
                self.descent = None
                departed_descent = True
                # Departure invalidates the descent's *origin and votes*, not
                # just its current anchor. A later pause halfway up must not
                # reconstruct a higher seat from the old upright observation.
                self.history.clear()
                self.history.append(row)
                self.stable.clear()
        # A supported descent requires observed flexion plus torso displacement
        # relative to the same visible feet, not a model vote or camera pan alone.
        if not standing and values['tilt'] < .45 and not departed_descent and values['hip_speed'] <= .03:
            for origin in self.history:
                votes = [s for s in self.history if s['t'] >= origin['t'] and s['descent_vote']]
                upright_origin = origin['knee'] > .90 and origin['hip_angle'] > .80 and origin['tilt'] < .25
                model_descent = len(votes) >= 2 and votes[-1]['t']-votes[0]['t'] >= .1
                if t-origin['t'] < .2:
                    continue
                physical_descent = articulated_descent([s for s in self.history if s['t'] >= origin['t']])
                # A person may plant their feet only after turning toward a bed.
                # Retain the real upright context, but measure flexion/descent
                # from the later planted observation rather than the moving foot.
                planted_descent = bool(physical_descent and model_descent and origin['descent_vote']
                    and any(s['t'] < origin['t'] and s['knee'] > .90 and s['hip_angle'] > .80
                            and s['tilt'] < .25 for s in self.history))
                if not (upright_origin or planted_descent):
                    continue
                if not (model_descent or physical_descent):
                    continue
                pair = np.stack([origin['points'], points])
                common = np.all(pair[:, :, 2] >= .30, axis=0)
                sides = [ids for ids in ((5, 11, 13, 15), (6, 12, 14, 16)) if common[list(ids)].all()]
                if not sides:
                    continue
                scale = max(origin['scale'], 1.)
                delta = [(points[[s[i] for s in sides], :2].mean(0)
                          - origin['points'][[s[i] for s in sides], :2].mean(0)) / scale for i in (0, 1, 3)]
                shoulder, hip, foot = delta
                if (shoulder[1] >= .12 and hip[1] >= .12
                        and shoulder[1] - foot[1] >= .12 and hip[1] - foot[1] >= .12
                        and np.linalg.norm(foot) <= .08 and values['knee'] - origin['knee'] <= -.04):
                    # Keep the lowest observed reference, never drag it upward
                    # during a slow/partial rise that still projects as bent.
                    common_hips = ([j for j in (11, 12) if points[j, 2] >= .30
                                    and self.descent['points'][j, 2] >= .30] if self.descent else [])
                    if self.descent is None or (common_hips and points[common_hips, 1].mean()
                                                >= self.descent['points'][common_hips, 1].mean()):
                        self.descent = dict(t=t, points=points.copy(), scale=values['scale'], origin=origin,
                                            physical=physical_descent, planted_descent=planted_descent and not upright_origin)
                    break
        if self.descent is not None and (t - self.descent['t'] > 2. or standing):
            self.descent = None
        # Frontal foreshortening can keep projected hip/knee angles almost
        # straight while sitting. A measured descent from upright plus the
        # current seated emission or supported seat height is a separate route; never relax the
        # global seated-angle or score gates for an isolated bent pose.
        sides_now = [s for s in ((5, 11, 13, 15), (6, 12, 14, 16)) if all(points[j, 2] >= .30 for j in s)]
        # Observed flexion/descent can support a foreshortened seat without a
        # phase vote. Exclude deep squats and floor-level hips explicitly.
        seat_level = bool(sides_now and all(
            (points[s[3], 1]-points[s[1], 1])/values['scale'] >= .75
            and -.15 <= (points[s[2], 1]-points[s[1], 1])/values['scale'] <= .65
            for s in sides_now))
        perspective_seat = self.descent is not None and (seated_top or seat_level) and values['tilt'] < .25
        candidate = 'seated' if seat_continues else 'standing' if standing else 'seated' if seated or perspective_seat else None
        row['candidate'] = candidate
        self.details.update(candidate=candidate, seat_level=seat_level, seat_continues=seat_continues,
                            descent_support=('observed_geometry' if self.descent.get('physical') else 'model_and_geometry')
                            if self.descent else None)
        stepping = bool(candidate == 'standing' and len(self.history) >= 2
                        and current_step(self.history[-2], row))
        self.details['step_departure'] = stepping
        if stepping:
            self.posture = None
            # Keep the actual moving sample in the stability window. Clearing
            # the window here imposed a second warm-up after a short step;
            # dropping only this sample would hide its motion from the window.
            if self.stable and self.stable[-1]['candidate'] != candidate:
                self.stable.clear()
            row['candidate'] = candidate
            self.stable.append(row)
            required = .60 if label == 'walking' else .35
            while len(self.stable) > 4 and t-self.stable[1]['t'] >= required:
                self.stable.popleft()
            self.details['code'] = 'step_departure'
            return None
        recent = [s for s in self.history if t-s['t'] <= .30+1e-8]
        moving = False
        body_moving = False
        for side in sides_now:
            if len(recent) < 3 or not all(np.all(s['points'][list(side), 2] >= .30) for s in recent):
                continue
            q = np.stack([s['points'][list(side), :2] for s in recent])
            relative = (q[:, :2, 1]-q[:, 3:4, 1])/values['scale']
            steps = np.diff(relative, axis=0)
            span = relative[-1]-relative[0]
            absolute = q[:, :2, 1]/values['scale']
            absolute_steps = np.diff(absolute,axis=0)
            absolute_span = absolute[-1]-absolute[0]
            body_moving |= bool((np.all(absolute_steps > .003) and np.all(absolute_span > .04))
                                or (np.all(absolute_steps < -.003) and np.all(absolute_span < -.04)))
            moving |= bool((np.all(steps > .003) and np.all(span > .06))
                           or (np.all(steps < -.003) and np.all(span < -.06)))
        self.details.update(current_motion=moving, body_motion=body_moving, legacy_hip_speed=values['hip_speed'])
        # Standing is an upright posture, not a requirement to freeze every
        # joint. A torso shift without leg flexion is not a sit/rise transition.
        leg_articulation = bool(recent and np.ptp([s['knee'] for s in recent]) > .035)
        motion_blocks = moving and (candidate != 'standing' or leg_articulation and body_moving)
        self.details['anchored_seat'] = pelvis_seat
        if pelvis_seat:
            motion_blocks = False
        if self.posture is not None:
            old = self.posture
            same = [s for s in sides_now if all(old['points'][j,2] >= .30 for j in s)]
            continues = (candidate == old['phase'] and t-old['t'] <= .35 and not motion_blocks and bool(same)
                         and abs(values['knee']-old['knee']) <= (.07 if candidate=='standing' else .06)
                         and abs(values['hip_angle']-old['hip_angle']) <= .06)
            if continues:
                foot_steps = []
                for side in same:
                    delta = (points[list(side),:2]-old['points'][list(side),:2])/old['scale']
                    foot_steps.append(float(np.linalg.norm(delta[3])))
                    if seat_continues:
                        # Arm raises move the shoulders while the established
                        # seat, hips and legs remain fixed. Never relax the hip
                        # or leg motion gates with the shoulder tolerance.
                        ordinary = np.max(np.abs(delta[:2,1])) <= .06
                        shoulder_only = (abs(delta[0,1]) <= .18 and abs(delta[1,1]) <= .025
                                         and np.max(np.abs(delta[2])) <= .06)
                        continues &= bool((ordinary or shoulder_only)
                                          and np.max(np.abs(delta[:2,0]-delta[3,0])) <= .06)
                    else:
                        continues &= bool(np.max(np.abs(delta[:2,1])) <= .04
                                          and np.max(np.abs(delta[:2,0]-delta[3,0])) <= .06)
                # Upright support can survive one uncertain ankle while the
                # other foot and both torso chains remain fixed. A seated
                # reference still requires every visible foot to stay put.
                continues &= bool(min(foot_steps) <= .05 and max(foot_steps) <=
                                  (.18 if candidate == 'standing' else .05))
            anchored = bool(candidate == 'standing' and label in {'background','unknown'}
                            and anchored_upright(old,row))
            continues = continues or anchored or (candidate == 'seated' and pelvis_seat)
            self.details['anchored_upright'] = anchored
            if continues and label == 'walking':
                # A sequence of small steps cannot drift away from a stationary
                # reference while retaining the same static label indefinitely.
                planted = False
                for side in same:
                    fixed = (points[list(side),:2]-old['fixed_points'][list(side),:2])/old['scale']
                    planted |= bool(np.linalg.norm(fixed[3]) <= .10)
                    continues &= bool(np.max(np.abs(fixed[:2,0])) <= .20)
                continues &= planted
            if continues:
                old.update(t=t, points=points.copy(), knee=values['knee'], hip_angle=values['hip_angle'])
                if self.seat is not None:self.seat['last_seen']=t
                row['candidate'] = candidate
                self.stable.append(row)
                required = .60 if label == 'walking' else .35
                while len(self.stable) > 4 and t-self.stable[1]['t'] >= required:
                    self.stable.popleft()
                # A short confirmed seat must mature with subsequent real
                # observations so the existing rise branch can adopt its fixed
                # reference after the ordinary .35 s / four-observation gate.
                old['duration_s'] = round(t-self.stable[0]['t'],4)
                old['samples'] = len(self.stable)
                self.details.update(code='static_continuity',stable=True,phase=candidate)
                return dict(phase=candidate,stable=True,duration_s=old['duration_s'],samples=old['samples'],
                            support='upright_geometry' if candidate=='standing' else 'verified_seat',
                            origin_time_s=(self.seat or {}).get('origin_time_s'),
                            seat_time_s=(self.seat or {}).get('confirmed_t'))
            self.posture = None
        if candidate is None or candidate != 'standing' and not seat_continues and self.descent is None and abs(values['hip_speed']) > .10 or motion_blocks:
            self.stable.clear()
            self.details['code'] = 'static_motion' if moving or abs(values['hip_speed']) > .10 else 'static_geometry'
            return None
        if candidate == 'seated' and self.descent is None and not seat_continues:
            # Shape-only history cannot turn a single new high phase score into
            # a sustained seat. Model-only recovery needs the same independent
            # observation count and duration as its geometric proof.
            if not row['seated_emission']:
                self.stable.clear()
                self.details['code'] = 'seat_support_missing'
                return None
            if any(not s['seated_emission'] for s in self.stable):
                self.stable.clear()
        if self.stable and self.stable[-1]['candidate'] != candidate:
            self.stable.clear()
        row['candidate'] = candidate
        self.stable.append(row)
        required_duration = .60 if label == 'walking' else .35
        # Keep the shortest real observation window meeting duration/count.
        # This adapts to 5–30 Hz without duplicating samples or accumulating sway.
        while len(self.stable) > 4 and t-self.stable[1]['t'] >= required_duration:
            self.stable.popleft()
        samples = list(self.stable)
        fast_stop = False
        if label == 'walking' and candidate == 'standing':
            fast_samples = samples[:]
            while len(fast_samples) > 4 and t-fast_samples[1]['t'] >= .35:
                fast_samples.pop(0)
            fast_stop = stopped_upright(fast_samples)
            if fast_stop:
                samples, required_duration = fast_samples, .35
        brief_samples = list(self.history)
        while len(brief_samples) > 3 and t-brief_samples[1]['t'] >= .22:
            brief_samples.pop(0)
        # Fast establishment is for a completed rise or a measured descent,
        # not a shortcut for startup/walking or re-anchoring an existing seat.
        earlier = [s for s in self.history if s['t'] < brief_samples[0]['t']]
        rise_origin = False
        if candidate == 'standing':
            from .observed_rise import movement
            for origin in earlier:
                delta = movement(origin, brief_samples[0])
                if (delta and delta['hip_absolute_up'] >= .12 and delta['shoulder_absolute_up'] >= .12
                        and delta['thigh_extension'] >= .06 and delta['knee_opening'] >= .04
                        and any(all(origin['points'][j,2] >= .30 and brief_samples[0]['points'][j,2] >= .30 for j in side)
                                and np.linalg.norm(origin['points'][side[3],:2]-brief_samples[0]['points'][side[3],:2])/max(origin['scale'],1.) <= .08
                                for side in ((5,11,13,15),(6,12,14,16)))
                        and delta['foot_motion'] <= .50):
                    rise_origin = True
                    break
        recent_short_seat = bool(seat_continues and self.seat.get('brief')
                                 and t-self.seat['confirmed_t'] <= .60)
        brief = bool((rise_origin or candidate == 'seated' and (self.seat is None or recent_short_seat)
                      and label in {'unknown','background','sitting_down'})
                     and brief_settle(brief_samples, candidate, self.descent is not None or recent_short_seat))
        self.details['brief_settle_confirmed'] = brief
        if brief:
            samples, required_duration = brief_samples, .22
        self.details['walking_stop_confirmed'] = fast_stop
        # Four independent observations and a duration gate support both 30 Hz
        # replay and a 6–10 Hz camera. Walking needs the longer counterevidence.
        duration = t - samples[0]['t']
        self.details.update(duration_s=round(duration,4), samples=len(samples), code='static_confirmation')
        if duration + 1e-8 < required_duration or len(samples) < (3 if brief else 4):
            return None
        p = np.stack([s['points'] for s in samples])
        common = np.all(p[:, :, 2] >= .30, axis=0)
        sides = [ids for ids in ((5, 11, 13, 15), (6, 12, 14, 16)) if common[list(ids)].all()]
        if not sides:
            self.details['code'] = 'common_chain_missing'
            return None
        scale = max(float(np.median([s['scale'] for s in samples])), 1.)
        shoulder, hip = [p[:, [side[i] for side in sides], :2].mean(axis=1) for i in (0, 1)]
        joint_spread = float(np.max(np.linalg.norm(np.ptp(p[:, common, :2], axis=0), axis=1)) / scale)
        core_ids = sorted({j for s in sides for j in s})
        core_spread = float(np.max(np.linalg.norm(np.ptp(p[:, core_ids, :2], axis=0), axis=1)) / scale)
        horizontal = float(max(np.ptp(hip[:, 0]), np.ptp(shoulder[:, 0])) / scale)
        vertical = float(max(np.ptp(hip[:, 1]), np.ptp(shoulder[:, 1])) / scale)
        foot_spread = min(float(np.linalg.norm(np.ptp(p[:,s[3],:2],axis=0))/scale) for s in sides)
        if candidate == 'standing':
            stable = (horizontal <= .25 and vertical <= .08
                      and foot_spread <= .10
                      and (label != 'walking' or horizontal <= .20)
                      and np.ptp([s['knee'] for s in samples]) <= .10
                      and np.ptp([s['hip_angle'] for s in samples]) <= .12)
        else:
            stable = (core_spread <= .12 and horizontal <= .10 and vertical <= .08
                      and np.ptp([s['knee'] for s in samples]) <= .07
                      and np.ptp([s['hip_angle'] for s in samples]) <= .07)
        stable = stable or brief
        self.details.update(code='static_ready' if stable else 'core_motion', stable=bool(stable),
                            joint_spread=round(joint_spread,4), core_spread=round(core_spread,4),
                            foot_spread=round(foot_spread,4),
                            horizontal_spread=round(horizontal,4), vertical_spread=round(vertical,4))
        if not stable:
            return None
        seated_emission = seated_top and seated_score >= .45
        if candidate == 'seated' and not (seated_emission or self.descent is not None or seat_continues):
            self.details['code'] = 'seat_support_missing'
            return None
        if candidate == 'seated':
            if self.seat is None:
                self.seat = dict(points=points.copy(), scale=scale, last_seen=t, confirmed_t=t, knee=values['knee'], brief=brief,
                                 planted_descent=bool(self.descent and self.descent.get('planted_descent')),
                                 origin_time_s=self.descent['origin']['t'] if self.descent else None)
            self.seat['last_seen'] = t
        self.posture = dict(t=t,phase=candidate,points=points.copy(),fixed_points=points.copy(),scale=scale,
                            knee=values['knee'],hip_angle=values['hip_angle'],duration_s=round(duration,4),samples=len(samples))
        return dict(phase=candidate, duration_s=round(duration, 4), samples=len(samples),
                    stable=True, origin_time_s=(self.seat or {}).get('origin_time_s'),
                    seat_time_s=(self.seat or {}).get('confirmed_t'), joint_spread=round(joint_spread, 4), core_spread=round(core_spread,4),
                    horizontal_spread=round(horizontal, 4), vertical_spread=round(vertical, 4),
                    support='upright_geometry' if candidate == 'standing' else
                    'observed_descent' if self.descent is not None else 'seated_emission' if seated_emission else 'verified_seat')


class ReconcileActions:
    """Per-target evidence; the caller recreates this on an identity boundary."""

    def __init__(self):
        self.samples = deque()
        self.last_t = None
        self.static_recovery = StaticRecovery()

    def update(self, t, prediction, info, tracked_person):
        original = dict(prediction or {})
        evidence = {}

        def result(code, reason, label=None, resolved_phase=None):
            # Block an observed contradiction, not merely missing evidence. The
            # legacy partial-seat path needs the original rising context when its
            # phase is unavailable; same-phase warm-up must not add a global gate.
            expected_phase = {"sitting_down": "descending", "standing_up": "rising"}.get(
                original.get("label"))
            phase_conflict = (expected_phase is not None
                              and original.get("phase") in {"seated", "standing", "descending", "rising"}
                              and original.get("phase") != expected_phase)
            # Failing a stricter *new-action* gate is not evidence that an
            # already consistent frozen transition is wrong. Only independent
            # affirmative counterevidence can veto that existing decision.
            observed_conflict = code in {"rigid_translation_confirmed", "opposite_motion_confirmed"}
            if label is None and expected_phase and (phase_conflict or observed_conflict):
                label = "unknown"
                evidence["transition_blocked"] = True
                reason += "；原动态标签与当前阶段或运动证据矛盾"
            coordinated = dict(original)
            changed = ((label is not None and label != original.get("label"))
                       or (resolved_phase is not None and resolved_phase != original.get('phase')))
            if label is not None:
                coordinated["label"] = label
            if resolved_phase is not None:
                coordinated['phase'] = resolved_phase
                coordinated['static_recovery'] = dict(evidence)
            return dict(prediction=coordinated, changed=changed, reason=reason,
                        code=code, evidence=dict(evidence))

        if not isinstance(t, (int, float)) or not math.isfinite(t):
            self.samples.clear()
            self.static_recovery.clear()
            self.last_t = None
            return result("invalid_time", "时间无效，重新积累连续证据")
        if self.last_t is not None and (t <= self.last_t or t - self.last_t > .35):
            self.samples.clear()
            self.static_recovery.clear()
        self.last_t = t
        required = ("knee", "hip_angle", "tilt", "scale", "hip_speed")
        try:
            values = {k: float(info[k]) for k in required}
            points = np.asarray(tracked_person["kpts"], dtype=float)
            valid = (bool(info["reliable"]) and points.shape == (17, 3)
                     and np.isfinite(points).all()
                     and all(math.isfinite(v) for v in values.values())
                     and values["scale"] > 0)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self.samples.clear()
            self.static_recovery.clear()
            return result("insufficient_pose", "缺少连续可靠关节，无法协调动作与阶段")
        recovery = self.static_recovery.update(t, original, values, points)
        if self.static_recovery.details.get('step_departure'):
            self.samples.clear()
            return result('observed_step', '当前髋、膝与脚发生同向移步，退出站立确认', 'unknown')
        if original.get('label') == 'falling':
            self.samples.clear()
            return result('protected_action', '保留原模型的跌倒判断；独立跌倒分支持续检查')
        if recovery is not None:
            evidence.update(recovery)
            code = ('static_confirmed' if original.get('phase') == recovery['phase']
                    and recovery['support'] not in {'observed_descent', 'verified_seat'} else 'static_recovered')
            return result(code, '连续可靠关节已稳定，几何与独立支持证据确认静态姿态',
                          'background', recovery['phase'])
        if original.get('label') == 'walking':
            self.samples.clear()
            return result('protected_action', '尚无持续静止反证，保留原模型的行走判断')
        phase = original.get("phase")
        if phase not in {"seated", "standing", "descending", "rising"}:
            self.samples.clear()
            return result("unsupported_phase", "当前阶段不足以确认四类动作")
        if self.samples and self.samples[-1]["phase"] != phase:
            self.samples.clear()
        scores = original.get("phase_scores") or {}
        try:
            score = float(scores.get(phase, 0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        self.samples.append(dict(t=t, phase=phase, points=points.copy(), score=score, **values))
        while self.samples and t - self.samples[0]["t"] > .5:
            self.samples.popleft()
        samples = list(self.samples)
        duration = t - samples[0]["t"]
        evidence.update(duration_s=round(duration, 4), samples=len(samples))
        needed = .25 if phase in {"seated", "standing"} else .15
        minimum_samples = 4 if phase in {"seated", "standing"} else 3
        if duration + 1e-8 < needed or len(samples) < minimum_samples:
            return result("warming_up", "等待同一阶段的连续运动或稳定证据")

        p = np.stack([s["points"] for s in samples])
        common = np.all(p[:, :, 2] >= .30, axis=0)
        # Use the same actual joints throughout; switching visible sides must not
        # manufacture motion. Feet provide the reference for camera translation.
        sides = [ids for ids in [(5, 11, 13, 15), (6, 12, 14, 16)] if common[list(ids)].all()]
        if not sides:
            return result("insufficient_common_joints", "连续窗口缺少同侧肩髋膝踝，不能确认相对运动")
        scale = max(float(np.median([s["scale"] for s in samples])), 1.)
        shoulder, hip, knee, ankle = [p[:, [side[i] for side in sides], :2].mean(axis=1)
                                      for i in range(4)]
        hip_delta = (hip[-1] - hip[0]) / scale
        shoulder_delta = (shoulder[-1] - shoulder[0]) / scale
        foot_delta = (ankle[-1] - ankle[0]) / scale
        knee_change = samples[-1]["knee"] - samples[0]["knee"]
        evidence.update(hip_dy=round(float(hip_delta[1]), 4),
                        shoulder_dy=round(float(shoulder_delta[1]), 4),
                        hip_relative_foot_dy=round(float(hip_delta[1] - foot_delta[1]), 4),
                        shoulder_relative_foot_dy=round(float(shoulder_delta[1] - foot_delta[1]), 4),
                        knee_change=round(float(knee_change), 4),
                        phase_score=round(score, 4))

        if phase in {"seated", "standing"}:
            seated = phase == "seated"
            geometry = [s["knee"] < .83 and s["hip_angle"] < .82 and s["tilt"] < .45
                        if seated else s["knee"] > .90 and s["hip_angle"] > .80 and s["tilt"] < .25
                        for s in samples]
            threshold = .45 if seated else .50
            if not all(geometry) or not all(s["score"] >= threshold for s in samples):
                return result("static_geometry_or_score", "静态阶段分数或关节姿态不足，不能确认静态状态")
            # Absolute motion rejects both walking drift and camera panning;
            # per-joint motion also rejects arm/leg activity hidden by a still hip.
            common_points = p[:, common, :2]
            joint_spread = float(np.max(np.linalg.norm(np.ptp(common_points, axis=0), axis=1)) / scale)
            horizontal = float(max(np.ptp(hip[:, 0]), np.ptp(shoulder[:, 0])) / scale)
            vertical = float(max(np.ptp(hip[:, 1]), np.ptp(shoulder[:, 1])) / scale)
            stable = (joint_spread <= .10 and horizontal <= .05 and vertical <= .05
                      and all(abs(s["hip_speed"]) <= .10 for s in samples)
                      and np.ptp([s["knee"] for s in samples]) <= .04
                      and np.ptp([s["hip_angle"] for s in samples]) <= .04)
            evidence.update(joint_spread=round(joint_spread, 4), horizontal_spread=round(horizontal, 4),
                            vertical_spread=round(vertical, 4), stable=bool(stable))
            if not stable:
                return result("static_motion_conflict", "仍有身体位移或关节活动，不能覆盖为静态状态")
            return result("static_confirmed", "阶段、姿态与持续稳定证据一致，解除滞留动作", "background")

        # A rising phase is the frozen V4 seat-anchor transition; no other phase
        # can create a rise here. A descending phase still needs independent motion.
        direction = 1. if phase == "descending" else -1.
        core_ids = sorted({joint for side in sides for joint in side})
        core = p[:, core_ids, :2]
        relative_core = (core - ankle[:, None, :]) / scale
        relative_spread = float(np.max(np.linalg.norm(np.ptp(relative_core, axis=0), axis=1)))
        common_travel = float(np.min(np.linalg.norm(core[-1] - core[0], axis=1)) / scale)
        evidence.update(relative_core_spread=round(relative_spread, 4),
                        common_core_travel=round(common_travel, 4))
        if relative_spread <= .015 and common_travel >= .035:
            return result("rigid_translation_confirmed", "肩髋膝踝共同平移且相对形状不变，缺少人体起坐运动")
        relative_hip = (hip[:, 1] - ankle[:, 1]) / scale
        relative_shoulder = (shoulder[:, 1] - ankle[:, 1]) / scale
        relative_up_down = direction * np.array([relative_hip[-1] - relative_hip[0],
                                               relative_shoulder[-1] - relative_shoulder[0]])
        absolute_up_down = direction * np.array([hip_delta[1], shoulder_delta[1]])
        progression = np.minimum(np.diff(relative_hip), np.diff(relative_shoulder)) if direction > 0 else \
            np.minimum(-np.diff(relative_hip), -np.diff(relative_shoulder))
        monotonic_fraction = float(np.mean(progression >= -.005))
        moving_steps = int(np.sum(progression > .001))
        current_direction = min(direction * (relative_hip[-1] - relative_hip[-2]),
                                direction * (relative_shoulder[-1] - relative_shoulder[-2])) >= -.005
        evidence["motion_agreement"] = round(monotonic_fraction, 4)
        evidence["moving_steps"] = moving_steps
        reverse_steps = int(np.sum(progression < -.001))
        if (np.all(absolute_up_down <= -.025) and np.all(relative_up_down <= -.025)
                and reverse_steps >= 2 and float(np.mean(progression <= .005)) >= .75):
            return result("opposite_motion_confirmed", "肩髋相对双脚的持续运动方向与原动态标签相反")
        if not (np.all(absolute_up_down >= .025) and np.all(relative_up_down >= .025)
                and monotonic_fraction >= .75 and current_direction and moving_steps >= 2):
            return result("transition_motion_conflict", "阶段与肩髋相对双脚的持续运动不一致")
        if phase == "descending":
            flexion = knee_change <= -.025
            bent = (values["knee"] < .83 and values["hip_angle"] < .82 and values["tilt"] < .45
                    and knee_change <= -.01)
            if not (flexion or bent) or values["hip_speed"] >= -.03:
                return result("descent_without_flexion", "缺少同步屈膝下降证据，不能把弯腰或平移当作坐下")
            if original.get("label") not in {"background", "bending", "unknown", "sitting_down"}:
                return result("opposed_transition", "原动作方向冲突，等待进一步证据")
            return result("descent_confirmed", "持续下降阶段与肩髋下降、屈膝证据一致", "sitting_down")
        if knee_change < .025 or values["hip_speed"] <= .015:
            return result("rise_without_extension", "起身阶段缺少持续抬髋伸腿证据，不能确认起身")
        if original.get("label") not in {"background", "bending", "unknown", "standing_up"}:
            return result("opposed_transition", "原动作方向冲突，等待进一步证据")
        return result("rise_confirmed", "原起身阶段与持续肩髋上升、伸腿证据一致", "standing_up")
