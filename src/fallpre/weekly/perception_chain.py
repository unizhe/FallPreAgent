"""Explicit perception-tool contract: quality, current action, history, process, event.

The existing experimental binary heads are executable diagnostics, not approved
alarm policies. Event ownership stays with the validated handoff fusion manager.
"""
import hashlib
import time
from copy import deepcopy

import numpy as np

from fallpre.runtime_resources import resource_path

from .fall_event_fusion import TemporalFallEventFusion
from .perception_semantics import decision_context
from .stgcn_fall_evidence import FallTemporalEvidence, _box, _iou, fall_geometry_features
from .stgcn_fall_head import predict_fall_head
from .stgcn_fusion import ConstrainedSTGCNFusion
from .stgcn_home import FrozenFeatures
from .stgcn_temporal import STGCNRuntime


class PerceptionRuntime(STGCNRuntime):
    """Same pretrained input; C/D heads use their original feature definitions."""
    def __init__(self):
        super().__init__()
        self.extractor = FrozenFeatures(self)
        self.heads = {}
        for name, resource in [('st_only', 'fall-head-c'), ('st_geometry', 'fall-head-d')]:
            path = resource_path(resource)
            with np.load(path, allow_pickle=False) as values:
                parameters = {k: values[k].copy() for k in ('mean', 'scale', 'weight', 'bias')}
            self.heads[name] = parameters, hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, window):
        started = time.perf_counter()
        result = super().predict(window)
        feature, _ = self.extractor.extract(window)
        geometry = fall_geometry_features(window.keypoints, window.source_times, window.width, window.height)
        heads = {}
        for name, (parameters, digest) in self.heads.items():
            vector = feature
            if name == 'st_geometry':
                if geometry['vector'] is None:
                    heads[name] = dict(status='unavailable', reason=geometry['code'])
                    continue
                vector = np.concatenate([feature, geometry['vector']])
            scores = predict_fall_head(parameters, vector[None])[0]
            heads[name] = dict(status='ready', non_fall_score=float(scores[0]),
                               fall_process_score=float(scores[1]), weights_sha256=digest,
                               feature_dim=len(vector), training_source='URFD weak-label development',
                               deployment_status='experimental_not_alarm_authority')
        result['fall_heads'] = heads
        result['inference_ms'] = (time.perf_counter()-started)*1000
        return result


class PerceptionChain:
    def __init__(self, stream_id):
        if not stream_id:
            raise ValueError('A stream namespace is required')
        self.stream_id = stream_id
        self.actions = ConstrainedSTGCNFusion()
        self.evidence = FallTemporalEvidence()
        self.events = TemporalFallEventFusion()
        self.delivered = set()

    def update(self, result, t):
        r = deepcopy(result)
        action = self.actions.update(r, t)
        evidence = self.evidence.update(r, t)
        r['fall_temporal_evidence'] = evidence
        event = self.events.update(r, t)
        a = r['action']
        index = a.get('target_index')
        people = r.get('persons', [])
        person = people[index] if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(people) else None
        state = self.evidence.buffer.status
        observable = bool(self.evidence.buffer.current_valid and person is not None
                          and a.get('selection', {}).get('state') == 'locked'
                          and r.get('quality', {}).get('camera_stable') is True)
        current = dict(label=action['label'] if observable else 'unknown',
                       text=action['text'] if observable else '目标观测不足',
                       source='rules_with_temporal_completion' if action['changed'] else 'existing_rules',
                       known=observable and action['label'] != 'unknown',
                       conflicts=action['conflicts'], reason=action['reason'])
        temporal = r.get('temporal') or {}
        temporal_current = bool(temporal.get('status') == 'ready'
                                and self.evidence.buffer.is_result_current(temporal)
                                and 0 <= t-temporal.get('window_end_s', float('-inf')) <= .6)
        heads = temporal.get('fall_heads') or {}
        raw_head = heads.get('st_geometry') or {}
        # Process classification also applies when no fallen proposal exists.
        # Require a unique current person, not a positive fall detector label.
        rgb_people = [p for p in r.get('rgb_raw_candidates', {}).get('persons', [])
                      if _box(p.get('box')) is not None and p.get('confidence', 0) >= .25]
        current_box = _box(person.get('box')) if person else None
        unique_person = bool(observable and len(people) == 1 and len(rgb_people) == 1
                             and current_box is not None
                             and _iou(current_box, _box(rgb_people[0]['box'])) >= .5)
        usable = bool(temporal_current and unique_person
                      and evidence['observation_quality'].get('status') == 'observable'
                      and not evidence['conflicts'] and raw_head.get('status') == 'ready')
        process = dict(status='experimental' if usable else 'abstain',
                       label=('fall_process' if raw_head['fall_process_score'] >= .5 else 'non_fall_activity')
                       if usable else 'unknown',
                       heads=deepcopy(heads), reason='experimental_head_available' if usable else evidence['code'],
                       unique_current_person=unique_person, changes_alarm=False,
                       geometry=evidence['geometry'], conflicts=evidence['conflicts'],
                       score_semantics='uncalibrated observed-process class score, not future risk',
                       constrained_support=evidence['support'])
        f = event['fall']
        key = f'{self.stream_id}/{f["event_id"]}' if f.get('event_id') else None
        notify = bool(f.get('new_event') and key not in self.delivered)
        if notify:
            self.delivered.add(key)
        rearmed = next((x for x in f.get('transitions', []) if x.get('kind') == 'rearmed'), None)
        recovery = ('confirmed' if rearmed and rearmed.get('recovery_scope') == 'standing_or_seated'
                    else 'low_support_only' if rearmed else 'recovering'
                    if f['lifecycle'] == 'recovering' else 'not_established')
        event_output = dict(state=f['state'], lifecycle=f['lifecycle'], event_key=key,
                            active_event_key=key if f.get('current_event_id') else None,
                            new_event=notify, temporal_accelerated=event['temporal_accelerated'],
                            reason=f['reason'], recovery_status=recovery,
                            recovery_confirmed=True if recovery == 'confirmed' else None,
                            transitions=deepcopy(f.get('transitions', [])),
                            alarm_authority='validated_handoff_and_constrained_ST_policy')
        color = 'green' if observable else 'gray'
        box = person['box'] if observable else None
        if f['state'] in {'suspected', 'confirmed'} and event['observation'].get('person'):
            color = 'red' if f['state'] == 'confirmed' else 'yellow'
            box = event['observation']['person']['box']
        return dict(schema_version=1, stream_id=self.stream_id, source_time_s=t,
                    decision_context=decision_context(r, t, fall=f, observation=event['observation']),
                    identity=dict(action_epoch=a.get('selection', {}).get('target_epoch'),
                                  generation=state['generation']),
                    observation=dict(observable=observable, code=state['status'],
                                     event_decidable=f['state'] != 'unobservable'),
                    current_action=current,
                    temporal_action=dict(status=temporal.get('status', 'unavailable') if temporal_current
                                         else 'unavailable', result_current=temporal_current,
                                         window=temporal.get('window'), top3=temporal.get('top3', []) if temporal_current else [],
                                         all_scores=temporal.get('all_scores') if temporal_current else None,
                                         semantics='past_window_60_classes_not_current_pose'),
                    fall_process=process, event=event_output,
                    display=dict(box=box, color=color, text=current['text']),
                    notification=dict(idempotency_key=key if notify else None, emit=notify,
                                      external_action_performed=False),
                    shared_pose_evidence=True)
