"""Independent low-origin episode contracts; no recording-specific inputs.

Synthetic image coordinates describe temporal relations, not clinical labels.
New event recognition must require a current observed transition; neither old
red state nor repeated RGB classification can supply that transition.
"""
from copy import deepcopy

import pytest

from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff


def detection(box, confidence=.9):
    return dict(box=list(box), confidence=confidence)


def low_support_pose(kind='floor_seat'):
    """A vertical torso above flexed legs, distinct from a chair-height seat."""
    points = [[240., 255., .95] for _ in range(17)]
    groups = (
        ((5, 6), ((230., 290.), (250., 290.))),
        ((11, 12), ((230., 390.), (250., 390.))),
        ((13, 14), ((300., 405.), (320., 405.))),
        ((15, 16), ((325., 455.), (345., 455.))),
    )
    for indices, coords in groups:
        for index, xy in zip(indices, coords):
            points[index] = [*xy, .95]
    if kind == 'squat':
        points[11][1] = points[12][1] = 405.
        for index, xy in zip((13, 14, 15, 16), ((300., 365.), (320., 365.), (270., 460.), (290., 460.))):
            points[index] = [*xy, .95]
    return dict(box=[180., 235., 370., 475.], keypoints=points, confidence=.95)


def side_pose():
    points = [[190., 420., .95] for _ in range(17)]
    for indices, coords in (
        ((5, 6), ((200., 420.), (210., 435.))),
        ((11, 12), ((330., 430.), (340., 445.))),
        ((13, 14), ((365., 450.), (375., 460.))),
        ((15, 16), ((390., 455.), (400., 465.))),
    ):
        for index, xy in zip(indices, coords):
            points[index] = [*xy, .95]
    return dict(box=[160., 365., 425., 478.], keypoints=points, confidence=.95)


def interpolate_pose(before, after, ratio):
    return dict(
        box=[a+(b-a)*ratio for a, b in zip(before['box'], after['box'])],
        keypoints=[[a+(b-a)*ratio for a, b in zip(left, right)]
                   for left, right in zip(before['keypoints'], after['keypoints'])],
        confidence=.95,
    )


def feed(fsm, t, pose=None, fallen=True, epoch=7, **kwargs):
    people = [detection(pose['box'])] if pose else []
    falls = [detection(pose['box'])] if pose and fallen else []
    return fsm.update(dict(persons=people, fallen=falls), t, 640, 480,
                      pose_person=pose, pose_epoch=epoch, **kwargs)


@pytest.mark.parametrize('kind', ['floor_seat', 'squat'])
def test_persistent_fallen_vote_on_static_low_support_is_not_new_fall(kind):
    fsm = LiveFallHandoff()
    pose = low_support_pose(kind)
    for i in range(20):
        out = feed(fsm, .15*i, pose)
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


def test_initial_side_lying_without_observed_transition_is_not_a_new_fall():
    fsm = LiveFallHandoff()
    for i in range(20):
        out = feed(fsm, .15*i, side_pose())
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


def test_low_support_with_missing_torso_cannot_authorize_later_lying():
    fsm = LiveFallHandoff()
    pose = low_support_pose()
    for index in (5, 6, 11, 12):
        pose['keypoints'][index][2] = .1
    for i in range(8):
        feed(fsm, .15*i, pose)
    for i in range(1, 15):
        out = feed(fsm, 1.05+.15*i, side_pose())
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


def test_empty_observations_do_not_convert_latched_red_to_a_new_event():
    # Establish a normal standing-origin fall with the unchanged legacy route.
    from test_webcam_fall_v17_boundaries import LOW, admit, descend

    fsm = LiveFallHandoff()
    admit(fsm)
    descend(fsm)
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        fsm.update(dict(persons=[detection(LOW)], fallen=[detection(LOW)]),
                   t, 640, 480, pose_epoch=7)
    original = deepcopy(fsm.confirmed_events)
    assert len(original) == 1
    for i in range(1, 20):
        out = feed(fsm, 2.2+.15*i)
        assert not out['fall']['observed']
        assert out['fall']['state'] == 'unobservable'
    assert fsm.confirmed_events == original


def admit_low_support(fsm, start=0., kind='floor_seat', epoch=7):
    pose = low_support_pose(kind)
    for i in range(7):
        out = feed(fsm, start+.15*i, pose, epoch=epoch)
    assert out['diagnostics']['reference_kind'] == 'low_support'
    assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not out['fall']['current_target_confirmed']
    return out


def transition_to_side(fsm, start=1.05, origin=None, epoch=7, end=None):
    origin = origin or low_support_pose()
    end = end or side_pose()
    outputs = []
    for i in range(1, 5):
        outputs.append(feed(fsm, start+.15*(i-1), interpolate_pose(origin, end, i/4), epoch=epoch))
    for i in range(1, 12):
        outputs.append(feed(fsm, start+.45+.15*i, end, epoch=epoch))
    return outputs


@pytest.mark.parametrize('kind', ['floor_seat', 'squat'])
def test_low_support_then_new_side_descent_confirms_one_independent_event(kind):
    fsm = LiveFallHandoff()
    admit_low_support(fsm, kind=kind)
    outputs = transition_to_side(fsm, origin=low_support_pose(kind))
    assert any(x['fall']['state'] == 'suspected' for x in outputs)
    assert outputs[-1]['fall']['state'] == 'confirmed'
    assert len(fsm.confirmed_events) == 1
    event = fsm.confirmed_events[0]
    assert event['event_type'] == 'low_support_to_lying'
    assert event['observed_low_duration_s'] >= 1.
    assert event['t_s'] > .9


def test_old_red_is_archived_before_low_origin_event_can_be_counted_again():
    from test_webcam_fall_v17_boundaries import LOW, admit, descend

    fsm = LiveFallHandoff()
    admit(fsm)
    descend(fsm)
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        fsm.update(dict(persons=[detection(LOW)], fallen=[detection(LOW)]),
                   t, 640, 480, pose_epoch=7)
    original = deepcopy(fsm.confirmed_events[0])
    out = admit_low_support(fsm, start=2.35)
    assert len(fsm.confirmed_events) == 1
    assert fsm.confirmed_events[0] == original
    assert out['fall']['target_epoch'] != original['target_epoch']
    assert out['fall']['candidate_duration_s'] == 0
    assert out['fall'].get('current_event_id') is None
    outputs = transition_to_side(fsm, start=3.4)
    assert len(fsm.confirmed_events) == 2
    assert fsm.confirmed_events[0] == original
    new = fsm.confirmed_events[1]
    assert new['event_id'] != original['event_id']
    assert new['target_epoch'] != original['target_epoch']
    assert new['candidate_since_s'] >= 3.4
    assert new['observed_low_duration_s'] >= 1.
    assert outputs[-1]['fall']['current_target_confirmed']


def test_low_support_forward_bend_with_stationary_pelvis_is_not_fall():
    fsm = LiveFallHandoff()
    admit_low_support(fsm)
    # A floor-seated reach can lower the shoulders and make the visible torso
    # horizontal while the pelvis remains supported at its original height.
    bend = low_support_pose()
    bend['box'] = [100., 350., 370., 475.]
    for index, xy in zip((5, 6), ((120., 390.), (140., 390.))):
        bend['keypoints'][index] = [*xy, .95]
    for index in (0, 1, 2, 3, 4):
        bend['keypoints'][index] = [115., 385., .95]
    outputs = transition_to_side(fsm, end=bend)
    assert all(x['fall']['state'] not in {'suspected', 'confirmed'} for x in outputs)
    assert not fsm.confirmed_events


@pytest.mark.parametrize('interrupt', ['camera', 'source_gap', 'other_pose_identity'])
def test_interrupted_low_reference_does_not_authorize_static_side_arrival(interrupt):
    fsm = LiveFallHandoff()
    admit_low_support(fsm)
    t = 1.05
    epoch = 7
    if interrupt == 'camera':
        feed(fsm, t, low_support_pose(), camera_stable=False)
        t += .15
    elif interrupt == 'source_gap':
        t += 1.
    else:
        epoch = 8
    for i in range(15):
        out = feed(fsm, t+.15*i, side_pose(), epoch=epoch)
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


def low_candidate():
    fsm = LiveFallHandoff()
    admit_low_support(fsm)
    for i in range(1, 5):
        feed(fsm, 1.05+.15*(i-1), interpolate_pose(low_support_pose(), side_pose(), i/4))
    for t in (1.65, 1.8):
        out = feed(fsm, t, side_pose())
    assert out['fall']['state'] == 'suspected'
    assert not fsm.confirmed_events
    return fsm, out


def expanded_competitor(fsm, t, *, box=None, confidence=.65, epoch=7, pose=None):
    pose = side_pose() if pose is None else pose
    person = detection(pose['box'])
    competitor = detection(box or [1., 1., 639., 479.], confidence)
    return fsm.update(dict(persons=[person, competitor], fallen=[detection(pose['box'])]),
                      t, 640, 480, pose_person=pose, pose_epoch=epoch)


def test_one_expanded_competitor_frame_pauses_but_never_adds_confirmation_time():
    fsm, before = low_candidate()
    old_duration = before['fall']['candidate_duration_s']
    old_epoch = before['fall']['target_epoch']
    out = expanded_competitor(fsm, 1.95)
    assert out['fall']['reason_code'] == 'expanded_person_ambiguity'
    assert not out['fall']['observed']
    assert out['observation']['person'] is None
    assert out['fall']['target_epoch'] == old_epoch
    assert out['fall']['candidate_duration_s'] == old_duration
    returned = feed(fsm, 2.1, side_pose())
    assert returned['fall']['target_epoch'] == old_epoch
    assert returned['fall']['candidate_duration_s'] == old_duration
    advanced = feed(fsm, 2.25, side_pose())
    assert advanced['fall']['candidate_duration_s'] == pytest.approx(old_duration+.15)


def test_persistent_expanded_competitor_cannot_keep_low_episode_alive():
    fsm, _ = low_candidate()
    for t in (1.95, 2.1, 2.25, 2.4, 2.55):
        out = expanded_competitor(fsm, t)
        assert not out['fall']['observed']
        assert out['fall']['state'] == 'unobservable'
    assert out['fall']['target_epoch'] is None
    assert not fsm.confirmed_events
    for i in range(1, 14):
        out = feed(fsm, 2.55+.15*i, side_pose())
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


@pytest.mark.parametrize('kind', ['equal_score', 'not_image_edge', 'other_pose_identity'])
def test_unsupported_competitor_must_reset_not_use_expansion_exception(kind):
    fsm, _ = low_candidate()
    kwargs = dict(confidence=.85) if kind == 'equal_score' else (
        dict(box=[10., 10., 630., 479.]) if kind == 'not_image_edge' else dict(epoch=8))
    out = expanded_competitor(fsm, 1.95, **kwargs)
    assert not out['fall']['observed']
    assert out['fall']['target_epoch'] is None
    assert not out['diagnostics']['handoff_ready']
    assert not fsm.confirmed_events


def test_expired_low_reference_cannot_be_used_for_static_late_side_pose():
    fsm = LiveFallHandoff()
    admit_low_support(fsm)
    # Continue observing this person without a valid support pose. This is
    # not a source gap, and must not renew the low support reference lifetime.
    for i in range(1, 21):
        box = low_support_pose()['box']
        fsm.update(dict(persons=[detection(box)], fallen=[]), .9+.15*i, 640, 480,
                   pose_epoch=7)
    for i in range(1, 14):
        out = feed(fsm, 3.9+.15*i, side_pose())
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events


def test_recovery_waiting_frames_cannot_finish_low_event_confirmation():
    fsm, _ = low_candidate()
    for t in (1.95, 2.1, 2.25):
        out = feed(fsm, t, side_pose())
    before = out['fall']['candidate_duration_s']
    assert before < 1.
    # The classifier remains 'fallen' after a current-pose recovery. Those
    # frames may wait for a reliable new reference, but are not lying time.
    for t in (2.4, 2.55, 2.7, 2.85, 3.):
        out = feed(fsm, t, low_support_pose())
        assert not fsm.confirmed_events
        assert not out['fall']['new_event']
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
        assert out['fall']['candidate_duration_s'] <= before


def test_one_pelvis_keypoint_spike_cannot_authorize_persistent_seated_bend():
    fsm = LiveFallHandoff()
    admit_low_support(fsm)
    bend = low_support_pose()
    bend['box'] = [100., 350., 370., 475.]
    for index, xy in zip((5, 6), ((120., 390.), (140., 390.))):
        bend['keypoints'][index] = [*xy, .95]
    for i in range(1, 5):
        pose = interpolate_pose(low_support_pose(), bend, i/4)
        if i == 4:
            # One noisy pelvis estimate is not repeated descent evidence.
            pose['keypoints'][11][1] += 30.
            pose['keypoints'][12][1] += 30.
        out = feed(fsm, 1.05+.15*(i-1), pose)
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    for i in range(1, 12):
        out = feed(fsm, 1.5+.15*i, bend)
        assert out['fall']['state'] not in {'suspected', 'confirmed'}
    assert not fsm.confirmed_events
