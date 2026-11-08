"""Event timing changes require witnessed handoff plus current temporal evidence."""
from copy import deepcopy

import pytest
from test_webcam_fall_handoff import LOW, STEPS, UP, d
from test_webcam_rgb_fall import pose

from fallpre.weekly.fall_event_fusion import TemporalFallEventFusion
from fallpre.weekly.stgcn_fall_evidence import VERSION
from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff


def observation(t, box, fallen, baseline, support=True):
    raw = dict(persons=[d(box)], fallen=[d(box)] if fallen else [])
    p = pose(UP) if box == UP else None
    original = baseline.update(raw, t, 640, 480, pose_person=p, pose_epoch=7)
    evidence = dict(version=VERSION, source_time_s=t, support=support,
                    status='support' if support else 'abstain', conflicts=[],
                    association=dict(associated=True, action_target_epoch=3, rgb_target_epoch='rgb-1'),
                    observation_quality=dict(status='observable'),
                    model_evidence=dict(status='observed', result_age_s=.1))
    return dict(width=640, height=480, quality=dict(camera_stable=True, reason='stable'),
                action=dict(selection=dict(target_epoch=3)),
                temporal=dict(status='ready', generation=1), fall_temporal_evidence=evidence,
                rgb_raw_candidates=raw, fall=original['fall'],
                pose_fall_observation=dict(person=p, selection=dict(target_epoch=7)))


def sequence():
    return [(i * .2, UP, False) for i in range(4)] + list(zip((.8, 1., 1.2), STEPS, [True]*3)) + [
        (t, LOW, True) for t in (1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6)]


def test_temporal_support_advances_one_event_without_mutating_baseline():
    fusion, base = TemporalFallEventFusion(), LiveFallHandoff()
    events, base_events = [], []
    for t, box, fallen in sequence():
        r = observation(t, box, fallen, base)
        before = deepcopy(r)
        result = fusion.update(r, t)
        assert before == r
        if result['fall']['new_event']:
            events.append(t)
            assert result['tool_event']['temporal_support']
        if r['fall']['new_event']:
            base_events.append(t)
    assert events == [1.8] and base_events == [2.2]
    assert len(fusion.manager.confirmed_events) == 1


@pytest.mark.parametrize('failure', ['missing', 'stale', 'future', 'identity', 'conflict', 'quality', 'temporal'])
def test_invalid_temporal_evidence_cannot_shorten_confirmation(failure):
    fusion, base = TemporalFallEventFusion(), LiveFallHandoff()
    events = []
    for t, box, fallen in sequence():
        r = observation(t, box, fallen, base)
        e = r['fall_temporal_evidence']
        if failure == 'missing':
            e['support'] = False
        if failure == 'stale':
            e['source_time_s'] = t - .1
        if failure == 'future':
            e['source_time_s'] = t + .1
        if failure == 'identity':
            e['association']['action_target_epoch'] = 99
        if failure == 'conflict':
            e['status'] = 'conflict'
        if failure == 'quality':
            e['observation_quality']['status'] = 'unobservable'
        if failure == 'temporal':
            r['temporal']['status'] = 'stale'
        output = fusion.update(r, t)
        if output['fall']['new_event']:
            events.append(t)
    assert events == [2.2]


def test_static_high_temporal_support_never_replaces_descent():
    fusion, base = TemporalFallEventFusion(), LiveFallHandoff()
    for i in range(30):
        t = i * .2
        r = observation(t, UP, i > 3, base)
        assert not fusion.update(r, t)['fall']['new_event']
    assert not fusion.manager.confirmed_events


def test_second_person_interrupts_supported_fall_and_clears_lease():
    fusion, base = TemporalFallEventFusion(), LiveFallHandoff()
    for t, box, fallen in sequence():
        r = observation(t, box, fallen, base)
        if t >= 1.6:
            r['rgb_raw_candidates']['persons'].append(d([450, 50, 550, 430]))
        out = fusion.update(r, t)
        if t >= 1.6:
            assert out['fall']['state'] == 'unobservable'
            assert fusion.manager.support_lease is None
    assert not fusion.manager.confirmed_events
