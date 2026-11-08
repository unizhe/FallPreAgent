"""One-way handoff from verified visible sitting to measured pelvis ascent.

A forward lean need not lift the shoulders above the original sitting height.
This fallback only fills an unknown; it never vetoes a known action or supplies
fall evidence. It retains a fixed seat, not a moving or inferred skeleton.
"""
from collections import deque
import numpy as np
from .action_v3 import signals
from .observed_rise import SIDES, movement


class SeatRiseHandoff:
    def __init__(self):
        self.history = deque()
        self.seats = deque()
        self.anchor = None
        self.epoch = None
        self.last_t = None
        self.last_supported = None

    def clear(self):
        self.history.clear()
        self.seats.clear()
        self.anchor = None
        self.last_supported = None

    def update(self, t, epoch, person, result):
        if epoch != self.epoch or self.last_t is not None and not 0 < t-self.last_t <= .35:
            self.clear()
        self.epoch, self.last_t = epoch, t
        details = dict(code='no_verified_seat', applied=False, lower_seat_refreshed=False)
        result['seat_rise_handoff'] = details
        if person is None or result.get('raw_label') in {'falling','walking','bending'} or result['label'] in {'standing','squatting','bending'}:
            self.clear()
            details['code']='protected_or_missing'
            return result
        p = np.asarray(person['kpts'],float)
        if p.shape != (17,3) or not np.isfinite(p).all():
            self.clear()
            return result
        sides=[s for s in SIDES if np.all(p[list(s),2] >= .30)]
        shape,_,scale=signals(person)
        if not sides or scale < 12 or shape[0] >= .65:
            self.clear()
            details['code']='pose_unavailable'
            return result
        row=dict(t=t,points=p.copy(),scale=float(scale),knee=float(shape[1]))
        self.history.append(row)
        while self.history and t-self.history[0]['t'] > .5:
            self.history.popleft()
        verified = (result['label']=='seated' and .40 < shape[1] < .85 and shape[2] < .85 and shape[0] < .30
                    and min((p[a,1]-p[h,1])/scale for _,h,k,a in sides) > .75)
        if verified:
            self.seats.append(row)
            while self.seats and t-self.seats[0]['t'] > .6:
                self.seats.popleft()
            moves=[movement(self.seats[0],s) for s in self.seats]
            stable=(len(moves)>=4 and t-self.seats[0]['t'] >= .35 and all(
                m and abs(m['hip_absolute_up']) < .035 and abs(m['thigh_extension']) < .04
                and m['foot_motion'] < .10 and m['horizontal'] < .10 for m in moves))
            if stable:
                if self.anchor is None:
                    self.anchor=self.seats[0]
                fixed=movement(self.anchor,row)
                # A newly stable, confirmed LOWER seat may replace the old
                # reference. Never move it upward to follow a partial rise.
                lower=movement(self.anchor,self.seats[0])
                if (fixed and lower and fixed['hip_absolute_up'] < -.04
                        and lower['hip_absolute_up'] < -.04
                        and fixed['thigh_extension'] < .04
                        and row['knee'] <= self.anchor['knee']+.03):
                    self.anchor=self.seats[0]
                    details['lower_seat_refreshed']=True
                    fixed=movement(self.anchor,row)
                if fixed and abs(fixed['hip_absolute_up']) < .04 and abs(fixed['thigh_extension']) < .05:
                    self.last_supported=t
        else:
            self.seats.clear()
        if self.anchor is None or self.last_supported is None:
            return result
        if t-self.last_supported > 2.:
            self.clear()
            details['code']='seat_expired'
            return result
        details['code']='seat_reference'
        short=list(self.history)
        while len(short)>3 and t-short[1]['t'] >= .20:
            short.pop(0)
        common=[s for s in SIDES if all(np.all(x['points'][list(s),2]>=.30) for x in [self.anchor,*short])]
        if len(short)<3 or t-short[0]['t'] < .20 or not common:
            return result
        m=movement(self.anchor,row,fixed_sides=common)
        steps=[movement(a,b,self.anchor['scale'],common) for a,b in zip(short,short[1:])]
        span=movement(short[0],row,self.anchor['scale'],common)
        support=[s for s in common if all((movement(short[0],x,self.anchor['scale'],[s]) or {}).get('foot_motion',1) <= .12 for x in short)]
        feet=all((movement(short[0],x,self.anchor['scale'],common) or {}).get('foot_motion',1) <= .35 for x in short)
        positive=all(d and d['hip_absolute_up'] > .008 and d['thigh_extension'] > -.015
                     and d['shoulder_absolute_up'] > -.015 for d in steps)
        remainder=span['hip_absolute_up']-max(d['hip_absolute_up'] for d in steps)
        accepted=bool(m and support and feet and positive and remainder >= max(.015,1/self.anchor['scale'])
            and m['hip_absolute_up'] > .08 and m['hip_up'] > .04
            and ((m['thigh_extension'] > .12 and m['knee_opening'] > .06)
                 or (m['thigh_extension'] > .10 and m['knee_opening'] > .12))
            and span['hip_absolute_up'] > .06 and span['thigh_extension'] > .06
            and steps[-1]['thigh_extension'] > .008 and steps[-1]['shoulder_absolute_up'] > .005)
        # Strong pelvis/leg ascent can precede shoulder recovery from a lean.
        # This route demands more ascent and extension than the ordinary gate.
        lean_ascent=bool(m and support and feet and remainder >= max(.015,1/self.anchor['scale'])
            and all(d and d['hip_absolute_up'] > .008 and d['thigh_extension'] > .008
                    and d['shoulder_absolute_up'] > -.04 for d in steps)
            and m['hip_absolute_up'] > .18 and m['hip_up'] > .04
            and m['thigh_extension'] > .20 and m['knee_opening'] > .04
            and span['hip_absolute_up'] > .12 and steps[-1]['shoulder_absolute_up'] > .005)
        # At the end of a real rise the knees may already be straight while
        # the torso is still unfolding. Require the original rise vote and its
        # stopped-hip veto, plus continuing shoulder ascent over planted feet.
        terminal_extension=bool(result.get('raw_label')=='standing_up'
            and result.get('decision_code')=='rise_motion_stopped'
            and support and feet and shape[1] > .92 and shape[2] > .80 and shape[0] < .25
            and m['hip_absolute_up'] > .25 and m['thigh_extension'] > .25
            and all(d and -.03 < d['hip_absolute_up'] < .05
                    and d['thigh_extension'] > -.03 and d['shoulder_absolute_up'] > .02 for d in steps)
            and span['shoulder_absolute_up'] > .08)
        accepted = accepted or lean_ascent or terminal_extension
        details.update(code='pelvis_ascent' if accepted else 'ascent_pending',
                       support_feet=len(support),positive_steps=positive,lean_ascent=lean_ascent,terminal_extension=terminal_extension,
                       reference_age_s=round(t-self.last_supported,4),span_hip_up=round(span['hip_absolute_up'],4),
                       knee_opening=round(m['knee_opening'],4),
                       hip_up=round(m['hip_absolute_up'],4),extension=round(m['thigh_extension'],4))
        if accepted and result['label']=='unknown':
            result.update(label='standing_up',text='起身',reason=('离座后腿已伸直，躯干仍持续向上展开；保留起身直到实际停稳' if terminal_extension else
                                  '已确认坐姿后，连续抬髋伸膝且支撑脚稳定；前倾不等于骨架缺失'),
                          decision_code='accepted',decision_stage='seat_rise_handoff')
            details['applied']=True
        return result
