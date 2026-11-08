"""Current visible geometry vetoes incompatible seats; never changes fall evidence.

These are conservative 2D posture descriptions, not contact/scene recognition.
All sides that are visible must agree; missing evidence cannot retain a label.
"""
from collections import deque
import numpy as np
from .action_v3 import signals
from .action_v2 import angle
from .webcam_frontal_bend import FrontalBend
from .webcam_low_squat import ProjectedLowSquat


def sequential_hinge(history):
    """Confirm an existing bend candidate from three real, unambiguous poses.

    Two independent head-first steps avoid starting another timer after motion
    has already been observed. Never infer support from a single displacement.
    """
    if len(history) < 3:
        return False
    rows = list(history)[-3:]
    if not .20 <= rows[-1][0]-rows[0][0] <= .35:
        return False
    sides = ((5,11,13,15),(6,12,14,16))
    scale = rows[0][2]
    if scale < 12 or any(not np.all(p[list(side),2] >= .85)
                        or angle(p[h,:2],p[k,:2],p[a,:2]) <= .94
                        for _,p,_,_ in rows for side in sides for s,h,k,a in [side]):
        return False
    for left,right in zip(rows,rows[1:]):
        if not 0 < right[0]-left[0] <= .20:
            return False
        delta = (right[1][:,:2]-left[1][:,:2])/scale
        for shoulder,hip,knee,ankle in sides:
            sd,hd = delta[shoulder,1],delta[hip,1]
            if (sd <= .05 or sd <= 2*max(hd,.02) or hd < -.03
                    or np.linalg.norm(delta[ankle]) >= .05):
                return False
    return True


class DailyActivity:
    def __init__(self):
        self.history = deque()
        self.epoch = None
        self.motion_history = deque()
        self.last_t = None
        self.bend_reference = None
        self.bend_exit_reference = None
        self.frontal_bend = FrontalBend()
        self.low_squat = ProjectedLowSquat()

    def update(self, t, epoch, person, result, camera_stable=True):
        low_squat = self.low_squat.update(t, epoch, person, result, camera_stable)
        result['low_squat'] = low_squat
        front = self.frontal_bend.update(t, epoch, person, result, camera_stable)
        result["view_bend"] = front
        if epoch != self.epoch or self.last_t is not None and not 0 < t-self.last_t <= .35:
            self.history.clear()
            self.motion_history.clear()
            self.bend_reference = None
            self.bend_exit_reference = None
        self.epoch, self.last_t = epoch, t
        candidate = None
        strong_bend = False
        observed = False
        continued_bend = False
        protected = result.get('raw_label') == 'falling'
        if person is not None:
            p = np.asarray(person['kpts'], dtype=float)
            if p.shape == (17, 3) and np.isfinite(p).all():
                sides = [s for s in ((5,11,13,15),(6,12,14,16)) if np.all(p[list(s),2] >= .30)]
                shape, _, scale = signals(person)
                # Midpoint shoulder/hip scale collapses when shoulders overlap hips.
                # Use the observed side lengths for geometry only; motion/model
                # normalization elsewhere remains unchanged.
                if sides:
                    scale = max(scale, float(np.median([np.linalg.norm(p[s,:2]-p[h,:2]) for s,h,k,a in sides])))
                if sides and scale >= 12:
                    observed = True
                    prior = [r for r in self.motion_history if .25 <= t-r[0] <= .60]
                    hinge = False
                    if prior:
                        _, q, old_scale, old_knee = prior[0]
                        common = [side for side in sides if np.all(q[list(side),2] >= .30)]
                        if common:
                            delta = (p[:,:2]-q[:,:2])/old_scale
                            shoulder_drop = float(np.mean([delta[side[0],1] for side in common]))
                            hip_drop = float(np.mean([delta[side[1],1] for side in common]))
                            feet = max(float(np.linalg.norm(delta[side[3]])) for side in common)
                            hinge = (shoulder_drop > .15 and shoulder_drop > 2*max(hip_drop,.03)
                                     and feet < .12 and shape[1]-old_knee > -.03)
                    self.motion_history.append((t,p.copy(),float(scale),float(shape[1])))
                    while self.motion_history and t-self.motion_history[0][0] > .65:
                        self.motion_history.popleft()
                    heights = [(p[a,1]-p[h,1])/scale for _,h,k,a in sides]
                    if not protected and shape[1] < .40 and max(heights) < 1.0 and min(heights) > .15:
                        candidate = 'squatting'
                    elif (shape[1] > .90 and shape[2] < .40 and min(heights) > 1.3
                          or hinge and shape[1] > .92
                          and min(heights) > 1.1):
                        candidate = 'bending'
                        strong_bend = bool(shape[1] > .94 and shape[2] < .20 and min(heights) > 1.5
                            and any(np.all(p[list(side),2] >= .60) for side in sides))
                        # Head-first descent with both confident straight legs
                        # is already dynamic evidence; do not discard it as the
                        # projected hip angle passes through its folded middle.
                        strong_bend = strong_bend or bool(hinge and shape[1] > .94
                            and len(sides) == 2 and all(np.all(p[list(side),2] >= .85)
                                and angle(p[h,:2],p[k,:2],p[a,:2]) > .94 for side in sides for s,h,k,a in [side]))
                    elif (not protected and len(sides) == 2 and self.bend_reference is not None
                          and 0 < t-self.bend_reference[0] <= .30):
                        # A confirmed bend can briefly project shoulder onto hip.
                        # Continue only with current real, upright straight legs,
                        # measured against a fixed independently confirmed pose.
                        # Weak continuation frames must not renew this reference.
                        _, q = self.bend_reference
                        leg_lengths = [float(np.linalg.norm(q[h,:2]-q[a,:2])) for _,h,k,a in sides]
                        continued_bend = all(
                            length >= 24 and np.all(p[list(side),2] >= .85)
                            and angle(p[h,:2],p[k,:2],p[a,:2]) > .94
                            and .75 <= np.linalg.norm(p[h,:2]-p[a,:2])/length <= 1.3
                            and p[a,1]-p[h,1] > .75*length
                            and 8 <= np.linalg.norm(p[s,:2]-p[h,:2]) < .25*length
                            and p[s,1]-q[s,1] >= -.03*length
                            and np.linalg.norm(p[a,:2]-q[a,:2]) < .10*length
                            and np.linalg.norm(p[h,:2]-q[h,:2]) < .12*length
                            for side,length in zip(sides,leg_lengths) for s,h,k,a in [side])
                        if continued_bend:
                            candidate = 'bending'
                    if candidate is None and not protected and result['label'] == 'sitting_down' and hinge and shape[1] > .92 and shape[2] > .85:
                        result.update(label='unknown',text='未知',reason='头肩明显先下探而腿部仍接近伸直，暂不能把弯腰趋势确认成坐下',
                                      decision_code='descent_shape_pending',decision_stage='daily_geometry')
        if not observed:
            self.motion_history.clear()
            self.bend_exit_reference = None
        if candidate is None:
            self.history.clear()
            self.bend_reference = None
        else:
            if self.history and self.history[-1][1] != candidate:
                self.history.clear()
            self.history.append((t,candidate,strong_bend))
            while self.history and t-self.history[0][0] > .5:
                self.history.popleft()
            stable = len(self.history) >= 3 and t-self.history[0][0] >= .20
            # Two high-confidence full chains with a strongly folded hip and
            # straight knees can confirm a clear bend without a third request.
            if candidate == 'bending' and len(self.history) >= 2:
                last_two=list(self.history)[-2:]
                stable = stable or (all(r[2] for r in last_two) and .10 <= t-last_two[0][0] <= .20)
            if candidate == 'bending' and not continued_bend:
                stable = stable or sequential_hinge(self.motion_history)
            label = candidate if stable else 'unknown'
            if protected or candidate != 'bending':
                self.bend_reference = None
            elif stable and not continued_bend:
                self.bend_reference = (t, p.copy())
                if not protected:
                    self.bend_exit_reference = (t, p.copy())
            # Geometry may be observed during a conflicting raw vote, but
            # never replace a protected falling output on that observation.
            if not protected:
                result.update(label=label, text={'squatting':'蹲着','bending':'弯腰','unknown':'未知'}[label],
                              reason='连续可见关节支持'+('深屈膝低髋姿态' if candidate=='squatting' else '直腿屈髋或头肩先下探、腿部保持伸直') if stable else '当前姿态与起坐不符，等待连续姿态确认',
                              decision_code='accepted' if stable else 'daily_pose_pending', decision_stage='daily_geometry')
        # A folded torso changing projection can make the old normalized hip
        # speed appear downward during straightening. Reject a seat transition
        # only with current straight legs, planted feet and actual shoulder
        # ascent from a recently CONFIRMED bend. Do not manufacture standing.
        reference = self.bend_exit_reference
        if protected or reference is not None and t-reference[0] > .80:
            self.bend_exit_reference = None
        elif (candidate is None and observed and reference is not None
              and result['label'] == 'sitting_down'):
            _, q = reference
            common = [side for side in ((5,11,13,15),(6,12,14,16))
                      if np.all(q[list(side),2] >= .50) and np.all(p[list(side),2] >= .50)]
            recovering = bool(common and all(
                np.linalg.norm(q[h,:2]-q[a,:2]) >= 40
                and angle(p[h,:2],p[k,:2],p[a,:2]) > .94
                and (q[s,1]-p[s,1])/np.linalg.norm(q[h,:2]-q[a,:2]) > .10
                and np.linalg.norm(p[a,:2]-q[a,:2])/np.linalg.norm(q[h,:2]-q[a,:2]) < .08
                and (p[a,1]-p[h,1])/np.linalg.norm(q[h,:2]-q[a,:2]) > .65
                for s,h,k,a in common))
            if recovering:
                result.update(label='unknown', text='恢复直立', activity_hint='straightening',
                              reason='已确认弯腰后肩部实际上升，双脚稳定且可见腿仍伸直，不能判为坐下',
                              decision_code='bend_recovery', decision_stage='daily_geometry')
        if (candidate is None and person is not None and result['label']=='unknown'
                and result.get('raw_label')=='walking'
                and result.get('decision_code') in {'other_action','step_departure'}
                and ((result.get('static_evidence') or {}).get('current_motion')
                     or (result.get('static_evidence') or {}).get('step_departure'))):
            result.update(text='移动待确认',reason='模型提示行走 / 转身，当前仍有运动；起坐状态尚未确认',
                          activity_hint='walking',decision_stage='daily_activity')
        if front['confirmed'] and not protected and result['label'] in {'unknown', 'standing', 'sitting_down'}:
            result.update(label='bending', text='弯腰', reason='本次站立参照下肩部持续下探、躯干投影缩短，双腿保持伸展且髋脚稳定',
                          decision_code='accepted', decision_stage='frontal_bend')
        if (low_squat['confirmed'] and not protected
                and result['label'] in {'unknown', 'sitting_down'}):
            result.update(label='squatting', text='蹲着',
                          reason='双侧髋位持续接近足部，躯干直立且至少一腿明确屈膝；连续可见关节保持稳定',
                          decision_code='accepted', decision_stage='projected_low_squat')
        return result
