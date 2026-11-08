"""Current-image fallen detector with an explicitly witnessed detector handoff.

Only the person detector establishes support geometry and descent. A fallen
track takes over through measured co-motion, or recent independently measured
descent plus a current low torso/pelvis witness and repeated matching proposals.
Its boxes never substitute for the person detector's reference or pose joints.
"""
from __future__ import annotations

import math
from copy import deepcopy
from statistics import median

from .webcam_fall_geometry import containing_competitor, deduplicate_parts, pose_geometry
from .webcam_rgb_fall import LiveRgbFall, _box, _overlap


class LiveFallHandoff(LiveRgbFall):
    person_confidence = .35
    admission_confidence = .60
    fallen_confidence = .60
    handshake_observations = 3
    handshake_motion_norm = .15

    def _confirmation_duration(self, t):
        return 1.0

    def _clear(self):
        super()._clear()
        self._person_anchor = None
        self._person_seen = None
        self._fall_anchor = None
        self._fall_seen = None
        self._pairs = []
        self._descent = None
        self._handoff = False
        self._ambiguity_since = None
        self._person_ambiguity_since = None
        self._measured_descent = None
        self._person_history = []
        self._support_history = []
        self._reference_kind = None
        self._step_diagnostics = {}
        self._low_motion = []
        self._low_pairs = []
        self._recovery_progress = None

    def _continuing_recovery(self, person, pose_person, pose_epoch, geometry, t, width, height):
        """Brief ongoing rising motion is not a stable recovery or a new fall.

        The support classifier may stop matching while someone gathers their
        feet. Require current reliable torso pixels and upward motion from a
        recent multi-frame recovery witness; never extend the witness here.
        """
        r = self._recovery_progress
        if (not r or not person or not pose_person or not self._handoff
                or self._epoch != r['event_epoch'] or pose_epoch != r['pose_epoch']
                or not 0 < t-r['t'] <= .35 or person['confidence'] < self.admission_confidence
                or not self._continuous(r['box'], person['box'])
                or _box(pose_person.get('box')) is None
                or _overlap(pose_person['box'], person['box']) < .5):
            return None
        try:
            points = pose_person['keypoints']
            if len(points) != 17 or not all(
                len(points[j]) == 3 and all(math.isfinite(float(v)) for v in points[j])
                and points[j][2] >= .5 and 0 <= points[j][0] < width and 0 <= points[j][1] < height
                for j in (5, 6, 11, 12)):
                return None
            shoulder = tuple((points[5][i]+points[6][i])/2 for i in (0, 1))
            hip = tuple((points[11][i]+points[12][i])/2 for i in (0, 1))
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        scale = r['geometry']['length']
        shoulder_drop = (shoulder[1]-r['geometry']['shoulder'][1])/scale
        hip_drop = (hip[1]-r['geometry']['hip'][1])/scale
        # Short torso projection is not reconstructed or assigned an angle.
        # It may only support a brief uncertainty interval while actually rising.
        not_lying = bool(geometry and geometry['tilt'] < 45 or
                         geometry is None and math.dist(shoulder, hip) < .35*scale)
        if (not_lying and max(shoulder_drop, hip_drop) <= .10
                and min(shoulder_drop, hip_drop) <= -.10):
            return dict(witness_time_s=r['t'], age_s=t-r['t'], pose_epoch=pose_epoch,
                        shoulder_change_torso=shoulder_drop, hip_change_torso=hip_drop,
                        short_torso_projection=geometry is None)
        return None

    @staticmethod
    def _valid(items):
        valid = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            box = _box(item.get('box'))
            try:
                score = float(item.get('confidence', 0))
            except (TypeError, ValueError):
                continue
            if box is not None and math.isfinite(score) and .25 <= score <= 1:
                valid.append((i, dict(box=list(box), confidence=score, keypoints=[])))
        return valid

    @staticmethod
    def _near(a, b):
        # Spatial ambiguity uses actual overlap; distant low-score furniture is
        # not allowed to veto a person merely because its center is nearby.
        return _overlap(a, b) >= .05

    def _finish(self, state, reason, code, t, start, detection=None, index=None, metrics=None):
        result = self._output(state, reason, code, t, start, detection, index, metrics)
        result['diagnostics'].update(handoff_ready=self._handoff,
                                     handoff_pairs=len(self._pairs),
                                     ambiguity_retained=self._ambiguity_since is not None,
                                     ambiguity_since_s=self._ambiguity_since,
                                     descent_witness=deepcopy(self._descent),
                                     geometry_space='person_detector_then_verified_fallen_track')
        result['diagnostics'].update(self._step_diagnostics)
        result['diagnostics'].update(reference_kind=self._reference_kind,
                                     measured_descent=deepcopy(self._measured_descent),
                                     person_ambiguity_retained=self._person_ambiguity_since is not None)
        fall = result['fall']
        current = next((e for e in reversed(self.confirmed_events) if e['target_epoch'] == self._epoch), None)
        fresh = any(e['kind'] == 'confirmed' for e in fall['transitions'])
        lifecycle = ('new_confirmed' if fresh else 'ongoing_confirmed' if state == 'confirmed'
                     else 'new_candidate' if state == 'suspected'
                     else 'low_posture_ready' if self._reference_kind == 'low_support' and not self._handoff and detection
                     else 'recovering' if code in {'recovery_pending', 'recovery_transition'} else 'observing')
        fall.update(lifecycle=lifecycle, new_event=fresh,
                    current_event_id=current['event_id'] if current and state == 'confirmed' and detection else None,
                    last_confirmed_event_id=self.confirmed_events[-1]['event_id'] if self.confirmed_events else None,
                    event_type=current.get('event_type', 'fall') if current else
                    'low_support_to_lying' if self._reference_kind == 'low_support' and self._handoff else None)
        if code == 'recovery_transition':
            fall['text'] = '恢复中，尚未确认稳定'
        if self._reference_kind == 'low_support' and state == 'suspected':
            fall['text'] = '低位倒卧待确认'
            fall['reason'] = '低位参考后已见新的下降与侧倾，正在累计本次倒卧证据'
        if current and current.get('event_type') == 'low_support_to_lying' and state == 'confirmed':
            fall['text'] = '低位倒卧事件已确认'
            fall['reason'] = '低位参考后已见新的下降与侧倾；倒卧证据累计满一秒，原因待核对'
        if detection:
            result['observation']['selection']['detector'] = detection['detector']
        return result

    def update(self, raw, t, width, height, pose_person=None, pose_epoch=None,
               camera_stable=True, quality_reason=None):
        if not math.isfinite(t) or t < 0 or self._last_t is not None and t <= self._last_t:
            raise ValueError('RGB 跌倒分支源时间必须有限、非负并严格递增')
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError('图像尺寸必须有限且为正')
        if not isinstance(raw, dict) or not isinstance(raw.get('persons'), list) or not isinstance(raw.get('fallen'), list):
            raise ValueError('缺少当帧人体与倒地检测候选')
        self._step_diagnostics = {}
        start = len(self.events)
        self.statistics['updates'] += 1
        if self._shape is not None and self._shape != (width, height):
            self._reset('image_size_changed', t)
        if self._last_seen is not None and t - self._last_seen > self.config.max_gap_s + 1e-9:
            self._reset('observation_gap', t)
        self._shape, self._last_t = (width, height), t

        def missing(code, reason, reset=False):
            if reset:
                self._reset(code, t)
            self._previous_qualifying = False
            self._pending = []
            self._pending_pose_epoch = None
            self._support_history = []
            self._recovery_progress = None
            if not self._handoff:
                self._low_pairs = []
                self._low_motion = []
            return self._finish('unobservable', reason, code, t, start)

        if camera_stable is not True:
            return missing('camera_quality_unavailable', quality_reason or '相机不稳定，已清除身份与下降证据', True)
        persons, falls = self._valid(raw['persons']), self._valid(raw['fallen'])
        recent_person_anchor = (self._person_anchor if self._person_seen is not None
                                and t-self._person_seen <= self.config.max_gap_s else None)
        persons, removed = deduplicate_parts(persons, recent_person_anchor, pose_person, width, height)
        self._step_diagnostics['person_parts_removed'] = removed
        self._step_diagnostics['person_candidate_count'] = len(persons)
        people = [(i, p) for i, p in persons if p['confidence'] >= self.person_confidence]
        if len(people) > 1:
            elapsed = t-self._person_ambiguity_since if self._person_ambiguity_since is not None else 0.
            if (self._reference and elapsed <= .30 and pose_epoch == self._reference_pose_epoch
                    and containing_competitor(people, recent_person_anchor, pose_person, width, height)):
                self._person_ambiguity_since = t if self._person_ambiguity_since is None else self._person_ambiguity_since
                return missing('expanded_person_ambiguity', '当前骨架与近期人体一致，但有扩张大框；暂停确认并限时保留事件')
            return missing('multiple_person_candidates', '多个人体候选，无法证明同一目标', True)
        person_index, person = people[0] if people else (None, None)
        if person and any(i != person_index and self._near(p['box'], person['box']) for i, p in persons):
            elapsed = t-self._person_ambiguity_since if self._person_ambiguity_since is not None else 0.
            retain = bool(self._reference and self._person_anchor and self._last_seen is not None
                          and t-self._last_seen <= .30 and elapsed <= .30
                          and self._continuous(self._person_anchor,person['box'])
                          and pose_epoch == self._reference_pose_epoch
                          and pose_geometry(pose_person,person['box'],width,height) is not None)
            if retain:
                self._person_ambiguity_since = t if self._person_ambiguity_since is None else self._person_ambiguity_since
                return missing('ambiguous_person_candidate', '当前人体仍有姿态支持；短时保留事件，歧义不计入确认')
            return missing('ambiguous_person_candidate', '弱人体框与当前目标重叠，停止关联', True)
        if self._person_ambiguity_since is not None:
            if (t-self._person_ambiguity_since > .30 or person is None
                    or pose_epoch != self._reference_pose_epoch):
                self._reset('person_ambiguity_expired',t)
            self._person_ambiguity_since = None
        top_clipped_support = False
        geometry_box_valid = True
        if person:
            b = person['box']
            geometry_box_valid = b[0] > 2 and b[1] > 2 and b[2] < width-2 and b[3] <= height
            # A visible standing skeleton can witness recovery even when the
            # head touches the top border. It cannot measure full body height.
            same_support_epoch = bool(pose_epoch is not None and (
                self._reference_pose_epoch == pose_epoch or
                self._support_history and self._support_history[-1]['epoch'] == pose_epoch
                and self._support_history[-1]['kind'] == 'standing'))
            top_clipped_support = bool(0 <= b[1] <= 2 and b[0] > 2 and b[2] < width-2
                and b[3] < height-2 and person['confidence'] >= self.admission_confidence
                and same_support_epoch and recent_person_anchor
                and self._continuous(recent_person_anchor, b)
                and pose_geometry(pose_person, b, width, height)
                and self._pose_upright(pose_person, b, width, height))
            if not geometry_box_valid and not top_clipped_support:
                return missing('unreliable_clipped_box', '人体顶部或侧边出画，停止几何判断')
            if self._person_anchor is not None and not self._handoff and not self._continuous(self._person_anchor, b):
                self._reset('person_identity_discontinuity', t)
            recovery_geometry = pose_geometry(pose_person, b, width, height)
            recovery_continuity = bool(recent_person_anchor and self._continuous(recent_person_anchor, b)
                and person['confidence'] >= self.admission_confidence and recovery_geometry
                and (recovery_geometry['seated'] or recovery_geometry['low_support']
                     or self._pose_upright(pose_person, b, width, height)))
            if (self._handoff and self._fall_anchor is not None and _overlap(b, self._fall_anchor) < .3
                    and not recovery_continuity):
                return missing('handoff_person_conflict', '重新出现的人体与倒地目标不一致', True)
            self._person_anchor = tuple(b)
            self._person_seen = t

        # Before handoff, proposals must overlap the independently detected
        # person. Afterwards they must continue the witnessed fallen track.
        anchor = person['box'] if person is not None else self._fall_anchor if self._handoff else None
        matches = [(i, f) for i, f in falls if anchor is not None and
                   (_overlap(anchor, f['box']) >= .5 if person else self._continuous(anchor, f['box']))]
        def ambiguous_fall(code, reason):
            # Retain a reference for at most .30 s only while the independent
            # person detector still observes the same admitted subject. Nothing
            # on an ambiguous frame counts toward descent, handshake or alarm.
            elapsed = t-self._ambiguity_since if self._ambiguity_since is not None else 0.
            retain = bool(person and self._reference and self._last_seen is not None
                          and t-self._last_seen <= .30+1e-9 and elapsed <= .30+1e-9
                          and person['confidence'] >= self.admission_confidence
                          and pose_epoch == self._reference_pose_epoch)
            if retain:
                self._ambiguity_since = t if self._ambiguity_since is None else self._ambiguity_since
                return missing(code, reason+'；同一人体仍可见，限时保留参考')
            return missing(code, reason, True)

        if len(matches) > 1:
            return ambiguous_fall('ambiguous_fallen_candidates', '多个倒地框对应当前目标，暂停交接')
        fall_index, fall = matches[0] if matches else (None, None)
        if fall and any(i != fall_index and self._near(f['box'], fall['box']) for i, f in falls):
            return ambiguous_fall('overlapping_fallen_competitor', '倒地框旁存在重叠候选，暂停关联')
        if self._ambiguity_since is not None:
            if (t-self._ambiguity_since > .30+1e-9 or self._last_seen is None
                    or t-self._last_seen > .35+1e-9 or person is None
                    or person['confidence'] < self.admission_confidence
                    or pose_epoch != self._reference_pose_epoch):
                self._reset('fallen_ambiguity_expired', t)
                self._person_anchor = tuple(person['box']) if person else None
            self._ambiguity_since = None
        strong_fall = fall is not None and fall['confidence'] >= self.fallen_confidence
        if person is None and not (self._handoff and strong_fall):
            return missing('no_current_observation', '没有已关联的当前图像目标；缺失时间不计入确认')

        upright = bool(person and person['confidence'] >= self.admission_confidence
                       and person['box'][3] < height - 2
                       and self._pose_upright(pose_person, person['box'], width, height))
        geometry = pose_geometry(pose_person, person['box'], width, height) if person else None
        low_support = bool(person and geometry and geometry['low_support']
                           and person['confidence'] >= self.admission_confidence)
        support = ('low_support' if low_support else 'standing' if upright else 'seated' if person and geometry and geometry['seated']
                   and person['confidence'] >= self.admission_confidence and not strong_fall else None)
        # A camera-facing reclining body may project as upright. Record this
        # ambiguity without vetoing admission: regression shows that a hard
        # RGB-label veto also removes witnessed real fall events.
        admission_conflict = bool(support == 'standing' and strong_fall
                                  and self._reference is None and not self._handoff)
        self._step_diagnostics.update(initial_support_conflict=admission_conflict)
        self._step_diagnostics.update(support_candidate=support, pose_geometry=geometry)
        if support:
            if self._support_history and (self._support_history[-1]['kind'] != support
                    or self._support_history[-1]['epoch'] != pose_epoch
                    or t-self._support_history[-1]['t'] > .25):
                self._support_history = []
            b=person['box']
            self._support_history.append(dict(t=t,kind=support,epoch=pose_epoch,box=list(b),geometry=geometry,
                                             geometry_box_valid=geometry_box_valid))
            while len(self._support_history) > 4 and t-self._support_history[1]['t'] >= .5:
                self._support_history.pop(0)
            ready = len(self._support_history) >= 4 and t-self._support_history[0]['t'] >= .5-1e-9
            if support in {'seated', 'low_support'} and ready:
                g=[x['geometry'] for x in self._support_history]
                scale=max(1.,median(x['length'] for x in g))
                ready = (max(math.dist(x['hip'],g[0]['hip']) for x in g)/scale <= (.25 if low_support else .10)
                         and max(math.dist(x['shoulder'],g[0]['shoulder']) for x in g)/scale <= (.25 if low_support else .20))
            self._pending = [(x['t'],x['box'][3]-x['box'][1],(x['box'][1]+x['box'][3])/2,x['box'][1]) for x in self._support_history]
            self._pending_pose_epoch = pose_epoch
            geometry_history = [x for x in self._support_history if x['geometry_box_valid']]
            # Initial admission needs several complete boxes. Once standing is
            # established, a current complete standing box can refresh the
            # reference after top-clipped frames without another admission wait.
            refresh_standing = bool(support == 'standing' and self._reference_kind == 'standing'
                and self._reference_pose_epoch == pose_epoch and not self._handoff and geometry_box_valid)
            if ready and (len(geometry_history) >= 4 or refresh_standing):
                pending=[(x['t'],x['box'][3]-x['box'][1],(x['box'][1]+x['box'][3])/2,x['box'][1])
                         for x in geometry_history]
                previous = next((e for e in reversed(self.confirmed_events) if e['target_epoch']==self._epoch),None)
                rearming = self._handoff
                low_boundary = support == 'low_support' and self._reference_kind != 'low_support'
                support_geometry = [x['geometry'] for x in self._support_history if x['geometry']]
                support_history = list(self._support_history)
                if rearming or low_boundary:
                    self._reset('support_reacquired',t)
                    self._person_anchor=tuple(b)
                    self._person_seen=t
                    if low_support or top_clipped_support:
                        # Already observed support stays continuous across the
                        # lifecycle boundary; do not impose a second wait just
                        # to refresh the new low reference on the next frame.
                        self._support_history = support_history
                self._reference = dict(height=median(x[1] for x in pending),center_y=median(x[2] for x in pending),top_y=median(x[3] for x in pending),
                                       hip_y=geometry['hip'][1] if geometry else None)
                if low_support:
                    self._reference.update(hip_y=median(g['hip'][1] for g in support_geometry),
                        shoulder_y=median(g['shoulder'][1] for g in support_geometry),
                        torso_length=median(g['length'] for g in support_geometry),
                        torso_tilt=median(g['tilt'] for g in support_geometry), recorded_at_s=t)
                    self._low_motion = []
                    self._low_pairs = []
                self._reference_pose_epoch = pose_epoch
                self._reference_kind = support
                self._measured_descent = None
                self._person_history = []
                if self._epoch is None:
                    self._next_epoch += 1
                    self._epoch = f'rgb-{self._next_epoch}'
                    self.statistics['acquisitions'] += 1
                    self.events.append(dict(kind='acquired',t_s=t,target_epoch=self._epoch,reference_kind=support))
                if rearming:
                    self.statistics['recoveries'] += 1
                    self.events.append(dict(kind='rearmed',t_s=t,target_epoch=self._epoch,
                                            previous_event_id=previous['event_id'] if previous else None,reference_kind=support,
                                            recovery_scope='low_posture_only' if low_support else 'standing_or_seated'))
        else:
            self._support_history = []
            self._pending = []
            self._pending_pose_epoch = None

        confirmed_episode = any(e['target_epoch'] == self._epoch for e in self.confirmed_events)
        if (self._handoff and confirmed_episode and support and geometry and pose_epoch is not None
                and pose_epoch == self._reference_pose_epoch and len(self._support_history) >= 4
                and t-self._support_history[0]['t'] >= .35):
            self._recovery_progress = dict(t=t, event_epoch=self._epoch, pose_epoch=pose_epoch,
                                           box=list(person['box']), geometry=deepcopy(geometry))
        recovery_transition = (self._continuing_recovery(person, pose_person, pose_epoch, geometry,
                                                        t, width, height) if support is None else None)
        if support is None and recovery_transition is None:
            self._recovery_progress = None
        self._step_diagnostics.update(recovery_transition_held=recovery_transition is not None,
                                      recovery_transition_evidence=recovery_transition)

        self._step_diagnostics.update(top_clipped_recovery_supported=top_clipped_support,
            support_geometry_usable=geometry_box_valid, support_candidate=support, pose_geometry=geometry,
            support_pending_s=(t-self._pending[0][0]) if self._pending else 0.)
        if top_clipped_support:
            # Refresh presence, never falling duration, descent or detector
            # co-motion with a clipped box. Recovery may have closed the old
            # event above; its red status is not carried into this observation.
            self._last_seen = t
            self._previous_qualifying = False
            return self._finish('unobservable', '当前站立骨架支持恢复；顶部贴边，不测量下降',
                                'top_clipped_recovery', t, start)

        # A completed recovery retires the old fallen track before evaluating
        # another proposal. Until then, its identity guard remains in force.
        if fall and self._fall_anchor is not None and self._fall_seen is not None:
            if t - self._fall_seen > self.config.max_gap_s + 1e-9 or not self._continuous(self._fall_anchor, fall['box']):
                if self._handoff:
                    return missing('fallen_identity_discontinuity', '倒地目标不连续，重新建立证据', True)
                self._pairs = []

        drop = top_drop = ratio = None
        descended = False
        clipped = bool(person and person['box'][3] >= height - 2)
        if person and self._reference:
            b, r = person['box'], self._reference
            drop = ((b[1] + b[3]) / 2 - r['center_y']) / r['height']
            top_drop = (b[1] - r['top_y']) / r['height']
            ratio = (b[3] - b[1]) / r['height']
            descended = drop >= .35 and (top_drop >= .35 if clipped else ratio <= .8)
        if person and self._reference:
            self._person_history.append(dict(t=t,box=list(person['box'])))
            self._person_history=[x for x in self._person_history if t-x['t'] <= 1.5]
            recent=self._person_history
            h=self._reference['height']
            measured = bool(descended and len(recent)>=3 and recent[-1]['t']-recent[0]['t']>=.20
                and (recent[-1]['box'][1]-recent[0]['box'][1])/h >= .15
                and sum(recent[j]['box'][1]-recent[j-1]['box'][1] > .02*h for j in range(1,len(recent)))>=2)
            if measured and self._measured_descent is None:
                self._measured_descent=dict(t_s=t,target_epoch=self._epoch,source='person_detector',
                    reference_kind=self._reference_kind,center_drop_norm=drop,top_drop_norm=top_drop,height_ratio=ratio,
                    end_box=list(person['box']))
        if self._measured_descent is not None and t-self._measured_descent['t_s'] > 1.5:
            self._measured_descent=None
        if person and fall:
            # Weak classifications can maintain current correspondence but do
            # not count as strong handshake observations or fallen duration.
            self._pairs = [p for p in self._pairs if t - p['t'] <= 1.0]
            if strong_fall:
                self._pairs.append(dict(t=t, person_top=person['box'][1], fall_top=fall['box'][1]))
            if self._reference and self._reference_kind != 'low_support' and len(self._pairs) >= self.handshake_observations and strong_fall and descended:
                first, last = self._pairs[0], self._pairs[-1]
                h = self._reference['height']
                pd, fd = (last['person_top'] - first['person_top']) / h, (last['fall_top'] - first['fall_top']) / h
                if min(pd, fd) >= self.handshake_motion_norm and abs(pd - fd) <= .25:
                    if not self._handoff:
                        self.events.append(dict(kind='handoff', t_s=t, target_epoch=self._epoch))
                    self._handoff = True
                    self._descent = dict(t_s=t, target_epoch=self._epoch, source='person_detector',
                                         center_drop_norm=drop, top_drop_norm=top_drop, height_ratio=ratio,
                                         bottom_clipped=clipped, person_motion_norm=pd, fallen_motion_norm=fd)
        low_geometry = bool(geometry and geometry['tilt'] >= 55 and self._reference
            and self._reference.get('hip_y') is not None
            and (geometry['hip'][1]-self._reference['hip_y'])/self._reference['height'] >= .18)
        late_ready = bool(self._reference_kind != 'low_support' and not self._handoff and self._measured_descent and person and strong_fall
            and descended and low_geometry and support is None
            and _overlap(person['box'],fall['box']) >= .60
            and len(self._pairs)>=3 and self._pairs[-1]['t']-self._pairs[-3]['t']>=.20)
        # Unlike the synchronized route, this route has an independent current
        # torso/pelvis witness. A static furniture proposal cannot replace it.
        if late_ready:
            self._handoff=True
            self._descent=dict(self._measured_descent, handoff_at_s=t,
                geometry_evidence='observed_descent_then_current_low_pose_and_fallen')
            self.events.append(dict(kind='handoff',t_s=t,target_epoch=self._epoch,
                                    geometry_evidence='observed_descent_then_current_low_pose_and_fallen'))
        if self._reference_kind == 'low_support' and not self._handoff:
            self._low_motion = [x for x in self._low_motion if t-x['t'] <= .75]
            if geometry and pose_epoch == self._reference_pose_epoch:
                self._low_motion.append(dict(t=t, hip_y=geometry['hip'][1]))
            r = self._reference
            age = t-r['recorded_at_s']
            shoulder_drop = ((geometry['shoulder'][1]-r['shoulder_y'])/r['torso_length']) if geometry else None
            hip_drop = max(((x['hip_y']-r['hip_y'])/r['torso_length'] for x in self._low_motion), default=0.)
            hip_observations = [x['t'] for x in self._low_motion
                                if (x['hip_y']-r['hip_y'])/r['torso_length'] >= .15]
            hip_span = hip_observations[-1]-hip_observations[0] if hip_observations else 0.
            low_descent = bool(geometry and person and strong_fall and age <= 2.5 and age > 0
                and pose_epoch == self._reference_pose_epoch and shoulder_drop >= .65 and hip_drop >= .20
                and len(hip_observations) >= 2 and hip_span >= .10
                and geometry['tilt'] >= 60 and geometry['tilt']-r['torso_tilt'] >= 35
                and top_drop >= .20 and ratio <= .85 and _overlap(person['box'],fall['box']) >= .60)
            self._step_diagnostics['low_transition'] = dict(reference_age_s=age, shoulder_drop_torso=shoulder_drop,
                recent_hip_drop_torso=hip_drop, hip_drop_observations=len(hip_observations), hip_drop_span_s=hip_span,
                qualifying=low_descent, reference_expires_s=2.5)
            if low_descent:
                if self._low_pairs and t-self._low_pairs[-1] > .25:
                    self._low_pairs = []
                self._low_pairs.append(t)
                if len(self._low_pairs) >= 3 and t-self._low_pairs[0] >= .20:
                    self._handoff = True
                    self._descent = dict(t_s=t,target_epoch=self._epoch,source='person_detector',
                        reference_kind='low_support',reference_at_s=r['recorded_at_s'],
                        center_drop_norm=drop,top_drop_norm=top_drop,height_ratio=ratio,
                        shoulder_drop_torso=shoulder_drop,recent_hip_drop_torso=hip_drop,
                        geometry_evidence='low_support_then_new_descent_and_side_lying')
                    self._measured_descent = deepcopy(self._descent)
                    self.events.append(dict(kind='handoff',t_s=t,target_epoch=self._epoch,
                        geometry_evidence=self._descent['geometry_evidence'], event_type='low_support_to_lying'))
                    self._candidate_since = t
                    self._low_duration = 0.
                    self._previous_qualifying = False
            else:
                self._low_pairs = []
        self._step_diagnostics.update(measured_descent_ready=self._measured_descent is not None,
            current_low_pose=low_geometry,delayed_handoff_ready=late_ready,
            support_pending_s=(t-self._pending[0][0]) if self._pending else 0.)
        if fall:
            self._fall_anchor, self._fall_seen = tuple(fall['box']), t

        dt = t - self._last_seen if self._last_seen is not None else 0.
        self._last_seen = t
        detection = deepcopy(person if person else fall)
        detection.update(label='fallen' if strong_fall else 'person', detector='person' if person else 'fallen')
        index = person_index if person else fall_index
        # An upright recovery sample contradicts the current lying episode even
        # when the RGB model continues voting fallen. It cannot finish a timer.
        qualifying = bool(strong_fall and self._handoff and self._epoch is not None
                          and support is None and recovery_transition is None)
        if strong_fall:
            if self._candidate_since is None:
                self._candidate_since = t
                self.statistics['candidates'] += 1
            if qualifying and self._previous_qualifying:
                self._low_duration += dt
        else:
            self._candidate_since = None
            self._low_duration = 0.
        self._previous_qualifying = qualifying
        current = next((e for e in reversed(self.confirmed_events) if e['target_epoch'] == self._epoch), None)
        if current is None and qualifying and self._low_duration >= self._confirmation_duration(t) - 1e-9:
            current = dict(kind='confirmed', event_id=f'rgb-fall-{len(self.confirmed_events) + 1:04d}',
                           t_s=t, target_epoch=self._epoch, candidate_since_s=self._candidate_since,
                           observed_low_duration_s=round(self._low_duration, 4),
                           event_type='low_support_to_lying' if self._reference_kind == 'low_support' else 'fall',
                           geometry_evidence=self._descent.get('geometry_evidence', 'person_descent_then_comoving_fallen_track'),
                           descent_witness=deepcopy(self._descent))
            self.confirmed_events.append(current)
            self.events.append(deepcopy(current))
            self.statistics['confirmed'] += 1
        metrics = dict(rgb_label=detection['label'], rgb_confidence=detection['confidence'],
                       fall_confidence=fall['confidence'] if fall else None,
                       had_upright=self._reference is not None, pose_upright_witness=upright,
                       center_drop_norm=drop, top_drop_norm=top_drop, height_ratio=ratio,
                       bottom_clipped=clipped, height_reduction_is_evidence=person is not None and not clipped,
                       qualifying_fallen=qualifying, handoff_ready=self._handoff,
                       current_box_detector=detection['detector'])
        if self._handoff and support:
            return self._finish('unobservable','观察到恢复姿态，等待连续支撑后重新待命','recovery_pending',t,start,metrics=metrics)
        if self._handoff and recovery_transition:
            return self._finish('unobservable', '近期连续支撑后仍在收脚起身；保留旧事件记录，尚未确认稳定恢复',
                                'recovery_transition', t, start, metrics=metrics)
        if self._handoff and not strong_fall:
            return self._finish('unobservable', '当前倒地分类证据不足，不显示为恢复正常',
                                'weak_fallen_support', t, start, metrics=metrics)
        if self._reference_kind == 'low_support' and not self._handoff:
            return self._finish('unobservable','已建立低位姿态参考；历史事件保留，等待新的下降与侧倾',
                                'low_support_ready',t,start,detection,index,metrics)
        if current:
            state, code, reason = 'confirmed', 'verified_handoff_fall', '已见同一人体下降，交接后的当前倒地分类累计满一秒'
        elif qualifying:
            state, code, reason = 'suspected', 'associated_fallen', '已验证同一人体下降与倒地交接，正在累计当前倒地证据'
        elif strong_fall:
            state, code, reason = 'unobservable', 'classification_geometry_conflict', '倒地分类尚无足够下降与共同位移支持，暂不提示跌倒'
        else:
            state, code, reason = 'normal', 'no_supported_fall', '当前人体未得到可信倒地分类支持'
        if code == 'classification_geometry_conflict':
            self._step_diagnostics['conflict_evidence'] = dict(
                current_person_observed=person is not None,
                current_torso_observed=geometry is not None,
                torso_tilt_deg=geometry['tilt'] if geometry else None,
                reference_available=self._reference is not None,
                reference_kind=self._reference_kind,
                measured_descent_available=self._measured_descent is not None,
                synchronized_pairs=len(self._pairs),
                handoff_verified=self._handoff,
                initial_support_conflict=admission_conflict,
                interpretation='fall_process_unresolved_not_nonfall')
        return self._finish(state, reason, code, t, start, detection, index, metrics)
