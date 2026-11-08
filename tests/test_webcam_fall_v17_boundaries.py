"""Adversarial causal contracts for fall handoff, not video accuracy claims.

Only public update/output data are used. No recorded-video timestamps or
sample identities belong in these synthetic checks.
"""
from copy import deepcopy

import pytest

from fallpre.weekly.webcam_fall_handoff import LiveFallHandoff

UP = [200., 40., 300., 400.]
DESCENT = [[180., 100., 310., 430.], [160., 180., 340., 460.], [140., 280., 380., 480.]]
LOW = DESCENT[-1]


def detection(box, confidence=.9):
    return dict(box=list(box), confidence=confidence)


def standing_pose(box):
    left, top, right, bottom = box
    center, height = (left + right) / 2, bottom - top
    points = [[center, top + .1 * height, .95] for _ in range(17)]
    for left_index, right_index, fraction in ((5, 6, .25), (11, 12, .5), (13, 14, .725), (15, 16, .95)):
        points[left_index] = [center - 10, top + fraction * height, .95]
        points[right_index] = [center + 10, top + fraction * height, .95]
    return dict(box=list(box), keypoints=points, confidence=.95)


def seated_pose():
    box = [180., 100., 340., 430.]
    points = [[230., 130., .95] for _ in range(17)]
    for indices, coords in (
        ((5, 6), ((220., 180.), (240., 180.))),
        ((11, 12), ((220., 265.), (240., 265.))),
        ((13, 14), ((290., 275.), (310., 275.))),
        ((15, 16), ((285., 410.), (305., 410.))),
    ):
        for index, xy in zip(indices, coords):
            points[index] = [*xy, .95]
    return dict(box=box, keypoints=points, confidence=.95)


def low_pose(box=LOW):
    points = [[170., 345., .95] for _ in range(17)]
    for indices, coords in (
        ((5, 6), ((175., 345.), (185., 355.))),
        ((11, 12), ((305., 395.), (315., 405.))),
        ((13, 14), ((345., 435.), (355., 445.))),
        ((15, 16), ((355., 460.), (365., 470.))),
    ):
        for index, xy in zip(indices, coords):
            points[index] = [*xy, .95]
    return dict(box=list(box), keypoints=points, confidence=.95)


def feed(fsm, t, people=(), fallen=(), **kwargs):
    return fsm.update(dict(persons=list(people), fallen=list(fallen)), t, 640, 480, **kwargs)


def admit(fsm, start=0., epoch=7):
    for dt in (0., .2, .4, .6):
        out = feed(fsm, start + dt, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=epoch)
    assert out['diagnostics']['reference_ready']
    return out


def descend(fsm, start=.8, epoch=7):
    for i, box in enumerate(DESCENT):
        out = feed(fsm, start + .2 * i, [detection(box)], [detection(box)], pose_epoch=epoch)
    assert out['diagnostics']['handoff_ready']
    return out


def admitted_handoff():
    fsm = LiveFallHandoff()
    admit(fsm)
    descend(fsm)
    return fsm


def assert_no_alert(out):
    assert out['fall']['state'] not in ('suspected', 'confirmed')
    assert not out['fall']['current_target_confirmed']


def test_stationary_fallen_proposal_seen_before_descent_is_not_late_evidence():
    fsm = LiveFallHandoff()
    admit(fsm)
    # Furniture-like proposal exists before the person's real descent and
    # remains stationary while the independently detected person moves.
    furniture = detection([140., 140., 380., 480.])
    for i, box in enumerate([UP, *DESCENT, *([LOW] * 12)]):
        out = feed(fsm, .7 + .15 * i, [detection(box)], [furniture], pose_epoch=7)
        assert_no_alert(out)
    assert not fsm.confirmed_events


def test_co_moving_fallen_without_vertical_descent_is_not_fall():
    fsm = LiveFallHandoff()
    admit(fsm)
    for i in range(12):
        box = [200. + 4 * i, 40., 300. + 4 * i, 400.]
        out = feed(fsm, .8 + .15 * i, [detection(box)], [detection(box)], pose_epoch=7)
        assert_no_alert(out)
    assert not fsm.confirmed_events


@pytest.mark.parametrize('boxes', [
    [[190., 90., 320., 410.], [180., 140., 330., 420.], [180., 180., 330., 425.]],
    [[160., 90., 340., 420.], [140., 130., 360., 430.], [130., 150., 370., 435.]],
    [[190., 120., 320., 420.], [180., 180., 330., 440.], [170., 240., 340., 460.]],
], ids=['sit_down', 'bend', 'squat'])
def test_normal_lowering_without_fallen_classification_does_not_alert(boxes):
    fsm = LiveFallHandoff()
    admit(fsm)
    for i, box in enumerate([*boxes, *([boxes[-1]] * 10)]):
        out = feed(fsm, .8 + .15 * i, [detection(box)], pose_epoch=7)
        assert_no_alert(out)
    assert not fsm.confirmed_events


def test_empty_frames_and_first_return_never_add_confirmation_time():
    fsm = admitted_handoff()
    before = feed(fsm, 1.4, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    duration = before['fall']['candidate_duration_s']
    for t in (1.45, 1.5, 1.55):
        out = feed(fsm, t)
        assert not out['fall']['observed']
        assert out['observation']['person'] is None
        assert out['fall']['candidate_duration_s'] == duration
    returned = feed(fsm, 1.6, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    assert returned['fall']['candidate_duration_s'] == duration
    next_frame = feed(fsm, 1.8, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    assert next_frame['fall']['candidate_duration_s'] == pytest.approx(duration + .2)
    assert not fsm.confirmed_events


def test_frequent_empty_updates_do_not_keep_identity_alive():
    fsm = admitted_handoff()
    for i in range(1, 21):
        out = feed(fsm, 1.2 + i * .1)
        assert not out['fall']['observed']
        assert_no_alert(out)
    out = feed(fsm, 3.4, fallen=[detection(LOW)])
    assert_no_alert(out)
    assert not out['diagnostics']['handoff_ready']
    assert not fsm.confirmed_events


@pytest.mark.parametrize('interruption', ['camera', 'other_person', 'image_size'])
def test_hard_interruption_prevents_old_descent_from_authorizing_new_fallen(interruption):
    fsm = admitted_handoff()
    raw = dict(persons=[detection(LOW)], fallen=[detection(LOW)])
    kwargs = {}
    width = 640
    if interruption == 'camera':
        kwargs['camera_stable'] = False
    elif interruption == 'other_person':
        raw['persons'].append(detection([440., 30., 550., 430.]))
    else:
        width = 800
    out = fsm.update(raw, 1.4, width, 480, pose_epoch=7, **kwargs)
    assert_no_alert(out)
    for i in range(12):
        out = feed(fsm, 1.6 + .15 * i, [detection(LOW)], [detection(LOW)], pose_epoch=7)
        assert_no_alert(out)
    assert not fsm.confirmed_events


def test_contained_person_with_its_own_upright_pose_must_not_be_deduplicated():
    fsm = admitted_handoff()
    # Unlike a head/torso duplicate of a lying person, this box carries an
    # independently upright full-body chain. Containment alone cannot merge it.
    other = [210., 290., 270., 470.]
    out = feed(fsm, 1.4, [detection(LOW), detection(other)], [detection(LOW)],
               pose_person=standing_pose(other), pose_epoch=8)
    assert_no_alert(out)
    assert not out['fall']['observed']
    for i in range(10):
        out = feed(fsm, 1.6 + .15 * i, fallen=[detection(LOW)])
        assert_no_alert(out)
    assert not fsm.confirmed_events


@pytest.mark.parametrize('invalid_t', [float('nan'), float('inf'), -1., 1.2])
def test_invalid_clock_cannot_mutate_descent_or_event_history(invalid_t):
    fsm = admitted_handoff()
    old = deepcopy(fsm.export())
    with pytest.raises(ValueError):
        feed(fsm, invalid_t, [detection(LOW)], [detection(LOW)])
    assert fsm.export() == old


def test_two_falls_separated_by_observed_recovery_have_distinct_events():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    first = deepcopy(fsm.confirmed_events[0])
    # A continuous source clock and a stable pose identity isolate recovery
    # from a generic long-gap reset.
    for i in range(10):
        out = feed(fsm, 2.4 + .2 * i, [detection(UP)],
                   pose_person=standing_pose(UP), pose_epoch=7)
    assert_no_alert(out)
    assert out['diagnostics']['reference_ready']
    assert out['fall']['target_epoch'] != first['target_epoch']
    assert len(fsm.confirmed_events) == 1
    descend(fsm, start=4.4)
    for i in range(1, 6):
        out = feed(fsm, 4.8 + .2 * i, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    assert len(fsm.confirmed_events) == 2
    assert fsm.confirmed_events[0] == first
    assert fsm.confirmed_events[1]['event_id'] != first['event_id']
    assert fsm.confirmed_events[1]['target_epoch'] != first['target_epoch']


def test_observed_descent_can_hand_over_to_later_current_low_pose_and_fallen():
    fsm = LiveFallHandoff()
    admit(fsm)
    for i, box in enumerate(DESCENT):
        out = feed(fsm, .8 + .2 * i, [detection(box)], pose_epoch=7)
        assert_no_alert(out)
    for t in (1.4, 1.6, 1.8):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_person=low_pose(), pose_epoch=7)
    assert out['fall']['state'] == 'suspected'
    assert out['fall']['candidate_duration_s'] == 0
    for i in range(1, 6):
        out = feed(fsm, 1.8 + .2 * i, [detection(LOW)], [detection(LOW)],
                   pose_person=low_pose(), pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    assert len(fsm.confirmed_events) == 1


def test_expired_descent_cannot_be_recycled_by_static_late_fallen_appearance():
    fsm = LiveFallHandoff()
    admit(fsm)
    for i, box in enumerate(DESCENT):
        feed(fsm, .8 + .2 * i, [detection(box)], pose_epoch=7)
    # Every frame remains observable; expiry must be about descent age, not
    # merely an observation-gap reset.
    for i in range(1, 12):
        out = feed(fsm, 1.2 + .2 * i, [detection(LOW)], pose_person=low_pose(), pose_epoch=7)
        assert_no_alert(out)
    for i in range(12):
        out = feed(fsm, 3.6 + .2 * i, [detection(LOW)], [detection(LOW)],
                   pose_person=low_pose(), pose_epoch=7)
        assert_no_alert(out)
    assert not fsm.confirmed_events


def test_stable_seated_start_admits_reference_then_independent_fall():
    fsm = LiveFallHandoff()
    pose = seated_pose()
    for t in (0., .2, .4, .6):
        out = feed(fsm, t, [detection(pose['box'])], pose_person=pose, pose_epoch=7)
        assert_no_alert(out)
    assert out['diagnostics']['reference_ready']
    assert out['diagnostics']['reference_kind'] == 'seated'
    boxes = [[170., 160., 350., 450.], [150., 240., 380., 470.], [120., 340., 440., 478.]]
    for i, box in enumerate([*boxes, *([boxes[-1]] * 8)]):
        out = feed(fsm, .8 + .2 * i, [detection(box)], [detection(box)], pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    assert len(fsm.confirmed_events) == 1


def test_one_upright_pose_glitch_cannot_erase_confirmed_fall():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    original_epoch = out['fall']['target_epoch']
    # This reliable-looking one-frame upright chain must start recovery
    # observation rather than immediately erase the existing handoff.
    recover_box = DESCENT[1]
    out = feed(fsm, 2.4, [detection(recover_box)], pose_person=standing_pose(recover_box), pose_epoch=7)
    assert out['fall']['target_epoch'] == original_epoch
    assert out['fall']['state'] != 'normal'
    out = feed(fsm, 2.6, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    assert out['fall']['target_epoch'] == original_epoch
    assert len(fsm.confirmed_events) == 1


def test_continuous_reliable_recovery_emits_rearm_without_new_fall():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    event = fsm.confirmed_events[0]
    recover_box = DESCENT[1]
    for i in range(6):
        out = feed(fsm, 2.4 + .2 * i, [detection(recover_box)],
                   pose_person=standing_pose(recover_box), pose_epoch=7)
    assert_no_alert(out)
    rearmed = [e for e in fsm.events if e['kind'] == 'rearmed']
    assert len(rearmed) == 1
    assert rearmed[0]['previous_event_id'] == event['event_id']
    assert rearmed[0]['target_epoch'] != event['target_epoch']
    assert len(fsm.confirmed_events) == 1


def test_low_overlap_recovery_requires_half_second_of_current_continuous_support():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    original_epoch = out['fall']['target_epoch']
    # UP overlaps LOW by less than .3, but is continuous with the immediately
    # preceding current person detection and carries a reliable full pose.
    for t in (2.4, 2.6, 2.8):
        out = feed(fsm, t, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=7)
        assert out['fall']['target_epoch'] == original_epoch
        assert out['fall']['state'] != 'normal'
        assert not [e for e in fsm.events if e['kind'] == 'rearmed']
    out = feed(fsm, 3., [detection(UP)], pose_person=standing_pose(UP), pose_epoch=7)
    assert_no_alert(out)
    rearmed = [e for e in fsm.events if e['kind'] == 'rearmed']
    assert len(rearmed) == 1
    assert rearmed[0]['target_epoch'] != original_epoch
    assert rearmed[0]['previous_event_id'] == fsm.confirmed_events[0]['event_id']


def test_low_overlap_far_upright_person_cannot_inherit_recovery_event():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    other = [450., 40., 550., 400.]
    out = feed(fsm, 2.4, [detection(other)], pose_person=standing_pose(other), pose_epoch=8)
    assert out['fall']['state'] == 'unobservable'
    assert out['fall']['target_epoch'] is None
    assert not out['diagnostics']['handoff_ready']
    for i in range(1, 6):
        out = feed(fsm, 2.4 + .2 * i, [detection(other)],
                   pose_person=standing_pose(other), pose_epoch=8)
    assert_no_alert(out)
    assert not [e for e in fsm.events if e['kind'] == 'rearmed']
    assert len(fsm.confirmed_events) == 1


def test_fallen_only_observations_cannot_keep_stale_person_recovery_anchor_alive():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, fallen=[detection(LOW)])
    assert out['fall']['state'] == 'confirmed'
    # The fallen track is current, but the independent person detector has
    # been absent for 1.2 seconds. A low-overlap standing arrival must not
    # borrow the old person anchor just because its geometry is nearby.
    out = feed(fsm, 2.4, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=8)
    assert out['fall']['state'] == 'unobservable'
    assert out['fall']['target_epoch'] is None
    assert not out['diagnostics']['handoff_ready']
    for i in range(1, 6):
        out = feed(fsm, 2.4 + .2 * i, [detection(UP)],
                   pose_person=standing_pose(UP), pose_epoch=8)
    assert_no_alert(out)
    assert not [e for e in fsm.events if e['kind'] == 'rearmed']
    assert len(fsm.confirmed_events) == 1


def test_missing_frame_restarts_support_confirmation_without_erasing_fall():
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)], [detection(LOW)], pose_epoch=7)
    original_epoch = out['fall']['target_epoch']
    for t in (2.4, 2.6):
        feed(fsm, t, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=7)
    missing = feed(fsm, 2.7)
    assert not missing['fall']['observed']
    assert missing['fall']['target_epoch'] == original_epoch
    for t in (2.8, 3., 3.2):
        out = feed(fsm, t, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=7)
        assert out['fall']['target_epoch'] == original_epoch
        assert not [e for e in fsm.events if e['kind'] == 'rearmed']
    out = feed(fsm, 3.4, [detection(UP)], pose_person=standing_pose(UP), pose_epoch=7)
    assert_no_alert(out)
    assert len([e for e in fsm.events if e['kind'] == 'rearmed']) == 1


@pytest.mark.parametrize('person_recent', [True, False])
def test_contained_part_deduplication_requires_recent_independent_person(person_recent):
    fsm = admitted_handoff()
    for t in (1.4, 1.6, 1.8, 2., 2.2):
        out = feed(fsm, t, [detection(LOW)] if person_recent else [], [detection(LOW)], pose_epoch=7)
    assert out['fall']['state'] == 'confirmed'
    part = [140., 300., 250., 460.]
    out = feed(fsm, 2.4, [detection(LOW), detection(part)], [detection(LOW)],
               pose_person=low_pose(), pose_epoch=7)
    if person_recent:
        assert out['fall']['state'] == 'confirmed'
        assert len(out['diagnostics']['person_parts_removed']) == 1
    else:
        assert out['fall']['state'] == 'unobservable'
        assert not out['diagnostics']['handoff_ready']
        assert out['fall']['target_epoch'] is None
    assert len(fsm.confirmed_events) == 1
