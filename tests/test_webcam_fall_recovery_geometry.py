"""Recovery evidence does not turn clipped boxes or historical red into falls."""
from copy import deepcopy

import pytest

from test_webcam_fall_handoff import LiveFallHandoff, UP, LOW, STEPS, d, feed, handoff, pose

TOP = [200., 1., 300., 400.]


def confirmed():
    f = LiveFallHandoff()
    handoff(f)
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        feed(f, t, [d(LOW)], [d(LOW)], pose_epoch=7)
    assert len(f.confirmed_events) == 1
    return f


def test_top_clipped_standing_closes_old_event_then_requires_new_descent_and_timer():
    f = confirmed()
    old = deepcopy(f.confirmed_events[0])
    for t in (2.4, 2.52, 2.64, 2.76):
        feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    out = feed(f, 2.92, [d(TOP)], pose_person=pose(TOP), pose_epoch=7)
    assert out['fall']['reason_code'] == 'top_clipped_recovery'
    assert out['fall']['current_event_id'] is None and not out['fall']['new_event']
    assert len(f.confirmed_events) == 1 and not f._handoff
    assert f._reference['height'] == 360.  # Complete boxes only, not TOP's 399.
    assert f._epoch != old['target_epoch']
    boundary = next(e for e in f.events if e['kind'] == 'rearmed')
    assert boundary['previous_event_id'] == old['event_id']
    feed(f, 3.04, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    for t, box in zip((3.24, 3.44, 3.64), STEPS):
        out = feed(f, t, [d(box)], [d(box)], pose_epoch=7)
        assert not out['fall']['new_event']
    for t in (3.84, 4.04, 4.24, 4.44, 4.64):
        out = feed(f, t, [d(LOW)], [d(LOW)], pose_epoch=7)
    assert out['fall']['new_event'] and len(f.confirmed_events) == 2
    new = f.confirmed_events[-1]
    assert new['event_id'] != old['event_id'] and new['target_epoch'] != old['target_epoch']
    assert new['descent_witness']['t_s'] > boundary['t_s']
    for t in (4.84, 5.04):
        out = feed(f, t, [d(LOW)], [d(LOW)], pose_epoch=7)
        assert out['fall']['state'] == 'confirmed' and not out['fall']['new_event']
    assert len(f.confirmed_events) == 2


def test_recovery_is_processed_before_stale_fallen_proposal():
    f = confirmed()
    for t in (2.4, 2.6, 2.8):
        feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    # Old fallen track is stale; this current upright frame completes recovery.
    out = feed(f, 3., [d(UP)], [d(UP)], pose_person=pose(UP), pose_epoch=7)
    assert len([e for e in f.events if e['kind'] == 'rearmed']) == 1
    assert not out['diagnostics']['handoff_ready']
    assert not out['fall']['new_event'] and len(f.confirmed_events) == 1


def test_mostly_clipped_initial_standing_does_not_invent_reference_height():
    f = LiveFallHandoff()
    feed(f, 0., [d(UP)], pose_person=pose(UP), pose_epoch=7)
    for i in range(1, 12):
        out = feed(f, i*.1, [d(TOP)], pose_person=pose(TOP), pose_epoch=7)
    assert not out['diagnostics']['reference_ready']
    assert not f.confirmed_events and not f._person_history and not f._pairs


def test_established_standing_refreshes_only_from_current_complete_box_after_clipping():
    f = LiveFallHandoff()
    for t in (0., .15, .3, .5):
        feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    for t in (.65, .8, .95, 1.1, 1.25):
        out = feed(f, t, [d(TOP)], pose_person=pose(TOP), pose_epoch=7)
        assert f._reference['height'] == 360.
        assert not out['fall']['new_event'] and not f._pairs
    full = [200., 5., 300., 450.]
    feed(f, 1.4, [d(full)], pose_person=pose(full), pose_epoch=7)
    assert f._reference['height'] == 445.


@pytest.mark.parametrize('problem', ['missing_pose', 'missing_legs', 'identity', 'side_clipped', 'camera', 'competitor'])
def test_top_clipping_exception_requires_current_reliable_same_person(problem):
    f = confirmed()
    for t in (2.4, 2.52, 2.64, 2.76):
        feed(f, t, [d(UP)], pose_person=pose(UP), pose_epoch=7)
    p = pose(TOP)
    people = [d(TOP)]
    if problem == 'missing_pose': p = None
    if problem == 'missing_legs':
        for i in (13, 14, 15, 16): p['keypoints'][i][2] = 0.
    if problem == 'side_clipped': people[0]['box'][0] = 1.
    if problem == 'competitor': people.append(d([450., 20., 550., 430.]))
    out = feed(f, 2.92, people, pose_person=p, pose_epoch=8 if problem == 'identity' else 7,
               camera_stable=problem != 'camera')
    assert not out['fall']['new_event'] and len(f.confirmed_events) == 1
    assert not [e for e in f.events if e['kind'] == 'rearmed']
    assert not out['diagnostics'].get('top_clipped_recovery_supported')
