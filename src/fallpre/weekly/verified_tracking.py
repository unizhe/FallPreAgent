"""Causal RGB target admission and short-gap recovery, without action priors.

Pose alone may hallucinate a skeleton on furniture. A second frozen object
detector must confirm a NEW target; it is deliberately not a per-frame veto on
an existing person (lying people are often misclassified as objects).
Logical target IDs are separate from ByteTrack IDs. Long gaps start new epochs,
never establish that a newly selected person is the previously bound resident.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .action_v2 import PersonTracker


@dataclass(frozen=True)
class VerifiedTrackingConfig:
    person_conf: float = .25
    confirmation_iou: float = .30
    admission_observations: int = 2
    max_gap_s: float = .5
    max_gap_frames: int = 15
    minimum_iou: float = .10
    max_center_distance: float = .60
    min_area_ratio: float = .20
    max_area_ratio: float = 5.0
    ambiguity_margin: float = .15


def overlap(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    aa = np.maximum(0, a[2:] - a[:2]).prod()
    ab = np.maximum(0, b[2:] - b[:2]).prod()
    return float(intersection / max(aa + ab - intersection, 1e-6))


class VerifiedPersonTracker:
    """Input: pose instances plus detector_instances with class_name/box/conf.

    frame_only=True uses observation counts, never the file's unreliable FPS.
    No image coordinates, camera identities, labels or event indices are used.
    Empty observations remain empty; recovered poses always come from this frame.
    """

    def __init__(self, fps=30.0, *, frame_only=False, config=None):
        self.config = config or VerifiedTrackingConfig()
        self.frame_only = frame_only
        self.raw = PersonTracker(30 if frame_only else fps)
        self.step = 0
        self.last_t = None
        self.last_box = None
        self.confirmed_anchor = None
        self.last_seen = None
        self.last_step = None
        self.target = None
        self.epoch = 0
        self.support = {}

    def _within_gap(self, t):
        if self.last_seen is None:
            return False
        if self.frame_only:
            return self.step - self.last_step <= self.config.max_gap_frames
        return t - self.last_seen <= self.config.max_gap_s + 1e-9

    def _geometry_score(self, previous, box):
        old, new = np.asarray(previous), np.asarray(box)
        size = np.maximum(old[2:] - old[:2], 1)
        ratio = np.maximum(new[2:] - new[:2], 0).prod() / size.prod()
        distance = np.linalg.norm((new[:2]+new[2:] - old[:2]-old[2:])/2) / np.linalg.norm(size)
        iou = overlap(old, new)
        cfg = self.config
        if (iou < cfg.minimum_iou or distance > cfg.max_center_distance
                or not cfg.min_area_ratio <= ratio <= cfg.max_area_ratio):
            return None
        return iou - .35 * float(distance)

    def _continuity(self, box, t, prospective_step=None):
        # A partial-body, uncorroborated detection must not erase the recent
        # full-body anchor. Both anchors expire on the same bounded gap policy;
        # neither can synthesize a pose or recover a long-lost identity.
        anchors = [self.last_box]
        if self.confirmed_anchor is not None:
            prior, when, anchor_step = self.confirmed_anchor
            current_step = self.step if prospective_step is None else prospective_step
            recent = (current_step-anchor_step <= self.config.max_gap_frames if self.frame_only
                      else t-when <= self.config.max_gap_s + 1e-9)
            if recent:
                anchors.append(prior)
        scores = [self._geometry_score(prior, box) for prior in anchors]
        scores = [score for score in scores if score is not None]
        return max(scores) if scores else None

    def recovery_context(self, t, instances, *, allow_ambiguous=False):
        """Read-only, next-observation gate for an optional current-image recovery.

        Unambiguous native observations always have priority. An opt-in verifier
        can inspect CURRENT RGB when native matches are ambiguous; final target
        ranking still applies. Recovery cannot initialize or extend an expired target.
        """
        if not math.isfinite(t) or self.last_t is not None and t <= self.last_t:
            raise ValueError('Tracker timestamps must strictly increase')
        if self.target is None:
            return dict(allowed=False, reason='no_confirmed_target')
        recent = (self.step+1-self.last_step <= self.config.max_gap_frames if self.frame_only
                  else t-self.last_seen <= self.config.max_gap_s + 1e-9)
        if not recent:
            return dict(allowed=False, reason='target_expired')
        ranked = self._rank_continuity(instances, t, self.step+1)
        if ranked:
            if allow_ambiguous and self._ambiguous(ranked, instances):
                return dict(allowed=True, reason='native_pose_ambiguous', anchor=list(self.last_box),
                            target_epoch=self.epoch, native_matches=ranked)
            return dict(allowed=False, reason='native_pose_available')
        return dict(allowed=True, reason='current_pose_missing', anchor=list(self.last_box),
                    target_epoch=self.epoch)

    def _rank_continuity(self, persons, t, prospective_step=None):
        ranked = [(score, i) for i, p in enumerate(persons)
                  if (score := self._continuity(p['box'], t, prospective_step)) is not None]
        return sorted(ranked, reverse=True)

    def _ambiguous(self, ranked, persons):
        return (len(ranked) > 1
                and ranked[0][0] - ranked[1][0] < self.config.ambiguity_margin
                and overlap(persons[ranked[0][1]]['box'], persons[ranked[1][1]]['box']) < .5)

    def update(self, row):
        t = float(row['t'])
        if not math.isfinite(t) or self.last_t is not None and t <= self.last_t:
            raise ValueError('Tracker timestamps must strictly increase')
        self.last_t = t
        self.step += 1
        persons = row['instances']
        detections = row.get('detector_instances')
        if detections is None:
            raise ValueError('Verified tracking requires object-detector evidence (empty list is allowed)')
        confirmed = {
            i for i, p in enumerate(persons)
            if any(d['class_name'] == 'person' and d['conf'] >= self.config.person_conf
                   and overlap(p['box'], d['box']) >= self.config.confirmation_iou for d in detections)
        }
        chosen = None
        reason = 'awaiting_confirmation'
        if self.target is not None:
            if self._within_gap(t):
                ranked = self._rank_continuity(persons, t)
                ambiguous = self._ambiguous(ranked, persons)
                if ranked and not ambiguous:
                    chosen = ranked[0][1]
                    reason = 'confirmed_continuity' if chosen in confirmed else 'pose_continuity_only'
                else:
                    reason = 'ambiguous_continuity' if ambiguous else 'temporarily_unobserved'
            else:
                self.target = None
                self.last_box = None
                self.confirmed_anchor = None
                self.support.clear()
                reason = 'expired_requires_new_confirmation'

        accepted = sorted(confirmed | ({chosen} if chosen is not None else set()))
        # ByteTrack receives only admitted observations, with indices mapped back
        # to the original pose list after its high/low confidence association.
        internal = dict(row, instances=[persons[i] for i in accepted],
                        t=(self.step-1)/30 if self.frame_only else t)
        self.raw.update(internal)
        raw_ids = {accepted[int(x.idx)]: int(x.track_id)
                   for x in self.raw.tracker.tracked_stracks
                   if x.is_activated and int(x.idx) < len(accepted)}
        support = {}
        for i in confirmed:
            tid = raw_ids.get(i)
            if tid is not None:
                count, previous_step = self.support.get(tid, (0, -2))
                support[tid] = (count+1 if previous_step == self.step-1 else 1, self.step)
        self.support = support
        if self.target is None:
            candidates = [i for i in confirmed if raw_ids.get(i) in support
                          and support[raw_ids[i]][0] >= self.config.admission_observations]
            if candidates:
                chosen = max(candidates, key=lambda i: persons[i]['det'] * np.mean(np.asarray(persons[i]['kpts'])[5:, 2]))
                self.epoch += 1
                self.target = self.epoch
                reason = 'new_confirmed_epoch'
        if chosen is not None:
            self.last_box = list(persons[chosen]['box'])
            self.last_seen = t
            self.last_step = self.step
            if chosen in confirmed:
                self.confirmed_anchor = (list(persons[chosen]['box']), t, self.step)
        evidence = [dict(index=i, detector_confirmed=i in confirmed, accepted=i in accepted,
                         decision=('person_confirmation' if i in confirmed else
                                   'short_gap_continuity' if i == chosen else 'unconfirmed_pose'))
                    for i in range(len(persons))]
        return dict(row, all_instances=persons, instances=[] if chosen is None else [persons[chosen]],
                    target_index=chosen, target_id=self.target, target_epoch=self.epoch,
                    tracking_reason=reason, detection_track_ids=raw_ids,
                    target_raw_track_id=raw_ids.get(chosen), admission=evidence,
                    clock_mode='observation_count' if self.frame_only else 'source_seconds')

    def configuration(self):
        return asdict(self.config)
