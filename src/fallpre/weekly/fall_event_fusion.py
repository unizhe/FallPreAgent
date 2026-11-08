"""Causal, opt-in ST evidence acceleration of an already witnessed RGB event.

Uses the existing handoff, descent, duration, identity and recovery manager.
Temporal evidence can shorten the final confirmation dwell, never bypass those
checks, cancel a baseline alarm, or establish safety from missing observations.
"""
from copy import deepcopy

from .stgcn_fall_evidence import VERSION
from .webcam_fall_handoff import LiveFallHandoff


class _TemporalHandoff(LiveFallHandoff):
    def _clear(self):
        super()._clear()
        self.support_lease = None
        self.current_support = None

    def _confirmation_duration(self, t):
        if self.current_support is not None:
            self.support_lease = dict(at_s=t, epoch=self._epoch, evidence=deepcopy(self.current_support))
        lease = self.support_lease
        if lease and lease['epoch'] == self._epoch and 0 <= t-lease['at_s'] <= .8:
            return .6
        return 1.


class TemporalFallEventFusion:
    """Per-stream event manager; raw primary result remains available for A/B."""
    def __init__(self):
        self.manager = _TemporalHandoff()
        self.event_support = {}

    def update(self, result, t):
        evidence = result.get('fall_temporal_evidence') or {}
        association = evidence.get('association') or {}
        action = result.get('action') or {}
        model = evidence.get('model_evidence') or {}
        temporal = result.get('temporal') or {}
        identity = (association.get('action_target_epoch'), temporal.get('generation'),
                    association.get('rgb_target_epoch'))
        valid = bool(
            evidence.get('version') == VERSION and evidence.get('source_time_s') == t
            and evidence.get('support') is True and evidence.get('status') == 'support'
            and not evidence.get('conflicts') and association.get('associated') is True
            and association.get('action_target_epoch') == (action.get('selection') or {}).get('target_epoch')
            and (evidence.get('observation_quality') or {}).get('status') == 'observable'
            and model.get('status') == 'observed' and 0 <= model.get('result_age_s', -1) <= .6
            and temporal.get('status') == 'ready'
            and (result.get('quality') or {}).get('camera_stable') is True)
        lease = self.manager.support_lease
        if (temporal.get('status') != 'ready'
                or model.get('status') != 'observed'
                or not 0 <= model.get('result_age_s', -1) <= .6
                or lease and lease['evidence'].get('fusion_identity') != identity
                or evidence.get('status') == 'conflict'
                or (evidence.get('observation_quality') or {}).get('status') != 'observable'
                or not association.get('associated')):
            self.manager.support_lease = None
        self.manager.current_support = dict(evidence, fusion_identity=identity) if valid else None
        ob = result['pose_fall_observation']
        actual = self.manager.update(
            result['rgb_raw_candidates'], t, result['width'], result['height'],
            pose_person=ob['person'], pose_epoch=ob['selection']['target_epoch'],
            camera_stable=result['quality']['camera_stable'], quality_reason=result['quality']['reason'])
        fall = actual['fall']
        if fall.get('new_event') and self.manager.support_lease and fall['candidate_duration_s'] < 1.:
            self.event_support[fall['event_id']] = deepcopy(self.manager.support_lease)
        support = self.event_support.get(fall.get('current_event_id'))
        if support:
            fall['reason'] = '同一目标的下降与倒地交接已验证，ST时序证据支持缩短确认等待'
        actual.update(schema_version=1, policy='st-supported-handoff-dwell-v1',
                      temporal_accelerated=support is not None,
                      temporal_confirmation=support,
                      baseline_confirmation_s=1., supported_confirmation_s=.6,
                      shared_pose_evidence=True)
        # Agent-facing evidence, not a probability or an autonomous action.
        actual['tool_event'] = dict(
            schema_version=1, tool='fall_detection', source_time_s=t,
            observation_status='observable' if fall.get('observed') and fall['state'] != 'unobservable'
            else 'unobservable', state=fall['state'], event_id=fall.get('current_event_id'),
            new_event=fall.get('new_event', False), lifecycle=fall.get('lifecycle'),
            temporal_support=support is not None, shared_pose_evidence=True,
            evidence_code=fall['reason_code'], requires_external_action=False)
        return actual
