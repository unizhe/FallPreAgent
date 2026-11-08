"""Ongoing recovery cannot re-light historical red through a short pose gap."""
from copy import deepcopy

import pytest

from test_webcam_fall_recovery_geometry import confirmed
from test_webcam_fall_v18_boundaries import low_support_pose, side_pose, feed


def moving_support(fsm, count=5):
    # Low support is genuine but still moving, so it must not rearm yet.
    for i in range(count):
        p = low_support_pose()
        for q in p['keypoints']:
            q[0] += 12*i
        p['box'][0] += 12*i
        p['box'][2] += 12*i
        out = feed(fsm, 2.35+.13*i, p)
        assert out['fall']['reason_code'] == 'recovery_pending'
    assert not [e for e in fsm.events if e['kind'] == 'rearmed']
    return p


def rising(p, short=False):
    p = deepcopy(p)
    for j in (5, 6): p['keypoints'][j][1] = 280.
    for j in (11, 12): p['keypoints'][j][1] = 270. if short else 350.
    return p


@pytest.mark.parametrize('short', [False, True])
def test_supported_rising_gap_is_recovering_without_rearm_or_new_confirmation(short):
    f = confirmed()
    p = moving_support(f)
    previous = deepcopy(f.confirmed_events)
    epoch = f._epoch
    duration = f._low_duration
    for t in (3., 3.13):
        out = feed(f, t, rising(p, short))
        assert out['fall']['reason_code'] == 'recovery_transition'
        assert out['fall']['lifecycle'] == 'recovering'
        assert out['fall']['state'] == 'unobservable' and out['observation']['person'] is None
        assert not out['fall']['new_event'] and out['fall']['current_event_id'] is None
        assert out['fall']['last_confirmed_event_id'] == previous[0]['event_id']
        assert f._epoch == epoch and f._handoff
        assert f._low_duration == duration
    assert f.confirmed_events == previous
    assert not [e for e in f.events if e['kind'] == 'rearmed']


def test_recovery_hold_is_bounded_and_does_not_refresh_its_own_witness():
    f = confirmed()
    p = moving_support(f)
    feed(f, 3., rising(p))
    out = feed(f, 3.24, rising(p))
    assert out['fall']['reason_code'] != 'recovery_transition'
    assert not out['diagnostics']['recovery_transition_held']
    assert len(f.confirmed_events) == 1


def test_current_return_to_lying_resumes_same_episode_not_a_new_event():
    f = confirmed()
    moving_support(f)
    p = side_pose()
    for q in p['keypoints']: q[0] += 48.
    p['box'][0] += 48.; p['box'][2] += 48.
    out = feed(f, 3., p)
    assert out['fall']['state'] == 'confirmed'
    assert not out['fall']['new_event'] and len(f.confirmed_events) == 1
    assert not out['diagnostics']['recovery_transition_held']


@pytest.mark.parametrize('problem', ['one_support_frame', 'no_upward_motion', 'downward_motion',
                                      'missing_torso', 'identity', 'camera', 'observation_gap'])
def test_recovery_hold_requires_recent_same_person_and_current_upward_evidence(problem):
    f = confirmed()
    p = moving_support(f, 1 if problem == 'one_support_frame' else 5)
    q = rising(p)
    t = 2.48 if problem == 'one_support_frame' else 3.
    if problem in {'no_upward_motion', 'downward_motion'}:
        for j in (5, 6, 11, 12): q['keypoints'][j][1] = p['keypoints'][j][1] + (20 if problem == 'downward_motion' else 0)
        for j in (15, 16): q['keypoints'][j][2] = .1  # Support gate no longer matches.
    if problem == 'missing_torso': q['keypoints'][5][2] = .1
    if problem == 'observation_gap': t = 3.5
    out = feed(f, t, q, epoch=8 if problem == 'identity' else 7, camera_stable=problem != 'camera')
    assert out['fall']['reason_code'] != 'recovery_transition'
    assert not out['diagnostics'].get('recovery_transition_held')
    assert not out['fall']['new_event'] and len(f.confirmed_events) == 1


def test_ambiguous_missing_observation_clears_recovery_witness():
    f = confirmed()
    p = moving_support(f)
    feed(f, 2.94)
    out = feed(f, 3., rising(p))
    assert out['fall']['reason_code'] != 'recovery_transition'
    assert not out['fall']['new_event']
