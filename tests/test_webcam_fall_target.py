"""Identity and current-frame recovery invariants, independent of learned accuracy."""
from copy import deepcopy

import pytest
from test_webcam_fall import person

from fallpre.weekly.webcam_fall_target import LiveFallSubject


def acquire(tracker, start=0.):
    for i in range(4):
        result = tracker.prepare([person()], start + i * .1, 640, 480)
    assert result['state'] == 'locked' and result['source'] == 'raw'
    return result


def moved(p, dx):
    p = deepcopy(p)
    p['box'][0] += dx
    p['box'][2] += dx
    for point in p['keypoints']:
        point[0] += dx
    return p


def recovered(kind='low'):
    return dict(person(kind), recovery_verified=True)


def test_initial_admission_requires_unique_strong_candidate_four_times_and_duration():
    tracker = LiveFallSubject()
    for i in range(3):
        out = tracker.prepare([person()], i * .1, 640, 480)
        assert out['state'] == 'confirming' and out['person'] is None
        assert not out['recovery_allowed']
    out = tracker.prepare([person()], .3, 640, 480)
    assert out['target_epoch'] == 1 and out['raw_index'] == 0
    assert out['person'] == person()
    tracker = LiveFallSubject()
    for i in range(5):
        out = tracker.prepare([person(), moved(person(), 150)], i * .1, 640, 480)
        assert out['state'] == 'ambiguous' and out['target_epoch'] is None


def test_torso_rotation_and_missing_knees_ankles_keep_established_identity():
    tracker = LiveFallSubject()
    before = acquire(tracker)
    low = person('low')
    for j in (13, 14, 15, 16):
        low['keypoints'][j] = [0, 0, 0]
    out = tracker.prepare([low], .4, 640, 480)
    assert out['state'] == 'locked' and out['target_epoch'] == before['target_epoch']
    assert out['person'] == low and out['source'] == 'raw'
    assert not out['recovery_allowed']


def test_half_second_source_ttl_does_not_inherit_actions_point35_reset():
    tracker = LiveFallSubject()
    epoch = acquire(tracker)['target_epoch']
    out = tracker.prepare([person('low')], .7, 640, 480)
    assert out['state'] == 'locked' and out['target_epoch'] == epoch


def test_box_without_torso_only_requests_recovery_never_refreshes_identity():
    tracker = LiveFallSubject()
    acquire(tracker)
    weak = person('low')
    weak['keypoints'][11] = [0, 0, 0]
    for t in (.4, .5, .6, .7, .8):
        out = tracker.prepare([weak], t, 640, 480)
        assert out['person'] is None and out['recovery_allowed']
        assert out['target_epoch'] == 1
    out = tracker.prepare([weak], .81, 640, 480)
    assert out['target_epoch'] is None and not out['recovery_allowed']


def test_recovery_is_same_frame_verified_pose_and_does_not_modify_raw_input():
    tracker = LiveFallSubject()
    acquire(tracker)
    persons = []
    out = tracker.prepare(persons, .4, 640, 480)
    assert out['recovery_allowed'] and out['anchor_box'] is not None
    proposal = recovered()
    before = deepcopy(proposal)
    out = tracker.finish([proposal], .4, 640, 480)
    assert out['source'] == 'recovered' and out['target_epoch'] == 1 and out['raw_index'] is None
    assert persons == [] and proposal == before
    out['person']['keypoints'][5][0] = 999
    assert proposal == before
    assert tracker.prepare([person('low')], .5, 640, 480)['target_epoch'] == 1


def test_unverified_or_far_recovery_cannot_become_a_current_observation():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    out = tracker.finish([person('low')], .4, 640, 480)
    assert out['person'] is None and out['reason_code'] == 'recovery_not_verified'
    tracker.prepare([], .5, 640, 480)
    out = tracker.finish([dict(moved(person(), 400), recovery_verified=True)], .5, 640, 480)
    assert out['person'] is None


def test_nearby_distinct_candidates_veto_both_raw_selection_and_recovery():
    tracker = LiveFallSubject()
    acquire(tracker)
    out = tracker.prepare([person(), moved(person(), 120)], .4, 640, 480)
    assert out['state'] == 'ambiguous' and not out['recovery_allowed'] and out['person'] is None
    with pytest.raises(ValueError, match='同一源帧'):
        tracker.finish([recovered()], .4, 640, 480)


def test_nearby_weak_second_person_also_blocks_recovery_override():
    tracker = LiveFallSubject()
    acquire(tracker)
    weak = moved(person(), 120)
    weak['keypoints'][11] = [0, 0, 0]
    out = tracker.prepare([person(), weak], .4, 640, 480)
    assert out['state'] == 'ambiguous' and not out['recovery_allowed']


def test_distinct_recovered_people_cannot_be_ranked_into_a_guessed_identity():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    out = tracker.finish([recovered('standing'), dict(moved(person(), 120), recovery_verified=True)], .4, 640, 480)
    assert out['state'] == 'ambiguous' and out['person'] is None


def test_duplicate_views_of_same_current_body_do_not_create_false_ambiguity():
    tracker = LiveFallSubject()
    acquire(tracker)
    p = person('low')
    out = tracker.prepare([p, deepcopy(p)], .4, 640, 480)
    assert out['state'] == 'locked'


def test_distant_person_cannot_replace_recently_missing_subject():
    tracker = LiveFallSubject()
    acquire(tracker)
    out = tracker.prepare([moved(person(), 400)], .4, 640, 480)
    assert out['person'] is None and out['target_epoch'] == 1
    assert out['recovery_allowed']


def test_expired_subject_requires_new_admission_not_recovery():
    tracker = LiveFallSubject()
    acquire(tracker)
    out = tracker.prepare([], .81, 640, 480)
    assert out['target_epoch'] is None and not out['recovery_allowed']
    with pytest.raises(ValueError):
        tracker.finish([recovered()], .81, 640, 480)
    assert acquire(tracker, 1.)['target_epoch'] == 2


@pytest.mark.parametrize('stable', [False, None])
def test_camera_unknown_or_moved_breaks_identity_and_disables_recovery(stable):
    tracker = LiveFallSubject()
    acquire(tracker)
    out = tracker.prepare([person()], .4, 640, 480, camera_stable=stable)
    assert out['state'] == 'unobservable' and out['target_epoch'] is None
    assert not out['recovery_allowed']
    assert acquire(tracker, .5)['target_epoch'] == 2


def test_finish_cannot_change_source_time_or_dimensions_or_run_twice():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    with pytest.raises(ValueError):
        tracker.finish([recovered()], .5, 640, 480)
    with pytest.raises(ValueError):
        tracker.finish([recovered()], .4, 800, 600)
    assert tracker.finish([recovered()], .4, 640, 480)['source'] == 'recovered'
    with pytest.raises(ValueError):
        tracker.finish([recovered()], .4, 640, 480)


@pytest.mark.parametrize('time', [.3, .2, float('nan'), float('inf'), -1])
def test_prepare_rejects_invalid_or_nonincreasing_time(time):
    tracker = LiveFallSubject()
    acquire(tracker)
    with pytest.raises(ValueError, match='源时间'):
        tracker.prepare([person()], time, 640, 480)


def test_dimension_change_discards_identity_before_new_confirmation():
    tracker = LiveFallSubject()
    acquire(tracker)
    out = tracker.prepare([person()], .4, 800, 600)
    assert out['target_epoch'] is None and out['state'] == 'confirming'


def test_new_fall_tracker_does_not_change_existing_action_selection_inputs():
    from fallpre.weekly.webcam_target import StableSubject

    control, original, fall = StableSubject(), StableSubject(), LiveFallSubject()
    for i in range(10):
        p = person('standing' if i < 4 else 'low')
        if i >= 4:
            for j in (13, 14, 15, 16):
                p['keypoints'][j] = [0, 0, 0]
        raw = [p]
        before = deepcopy(raw)
        expected = control.update(deepcopy(raw), i * .1, 640, 480)
        fall_result = fall.prepare(raw, i * .1, 640, 480)
        actual = original.update(raw, i * .1, 640, 480)
        assert raw == before and actual == expected
        if i >= 4:
            assert fall_result['source'] == 'raw' and fall_result['target_epoch'] == 1
            assert actual['target_index'] is None


def test_recovered_current_torso_can_feed_existing_fall_confirmation_without_gap_fill():
    from fallpre.weekly.webcam_fall import LiveFallDetector

    target, detector = LiveFallSubject(), LiveFallDetector()
    for i in range(19):
        t = i * .1
        raw = [person()] if i < 9 else []
        selection = target.prepare(raw, t, 640, 480)
        if i >= 9:
            assert selection['recovery_allowed']
            selection = target.finish([recovered()], t, 640, 480)
        result = detector.update(selection['person'], t, 640, 480, selection['target_epoch'])
    assert result['state'] == 'suspected' and result['candidate_duration_s'] == .9
    selection = target.prepare([], 1.9, 640, 480)
    selection = target.finish([recovered()], 1.9, 640, 480)
    result = detector.update(selection['person'], 1.9, 640, 480, selection['target_epoch'])
    assert result['state'] == 'confirmed' and result['candidate_duration_s'] == 1.


def test_unverified_current_competitor_vetoes_switch_to_a_verified_neighbour():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    original = dict(person(), recovery_verified=False, recovery_identity_competitor=True)
    neighbour = dict(moved(person(), 100), recovery_verified=True)
    result = tracker.finish([neighbour, original], .4, 640, 480)
    assert result['state'] == 'ambiguous' and result['person'] is None
    assert result['reason_code'] == 'recovered_candidates_compete'


@pytest.mark.parametrize('verified_flag', [False, True])
def test_competitor_only_never_supplies_a_pose_or_refreshes_the_identity(verified_flag):
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    competitor = dict(person(), recovery_verified=verified_flag, recovery_identity_competitor=True)
    result = tracker.finish([competitor], .4, 640, 480)
    assert result['person'] is None and result['source'] == 'none'
    assert tracker.prepare([], .81, 640, 480)['target_epoch'] is None


@pytest.mark.parametrize('competitor_first', [True, False])
def test_duplicate_competitor_cannot_replace_a_verified_observation(competitor_first):
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    verified = recovered('standing')
    competitor = dict(deepcopy(verified), recovery_verified=False, recovery_identity_competitor=True)
    proposals = [competitor, verified] if competitor_first else [verified, competitor]
    result = tracker.finish(proposals, .4, 640, 480)
    assert result['source'] == 'recovered' and result['person']['recovery_verified'] is True


def test_far_competitor_does_not_veto_verified_local_target():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    far = dict(moved(person(), 400), recovery_verified=False, recovery_identity_competitor=True)
    result = tracker.finish([recovered('standing'), far], .4, 640, 480)
    assert result['source'] == 'recovered' and result['target_epoch'] == 1


def test_same_view_nearby_people_remain_distinct_under_identity_duplicate_rules():
    tracker = LiveFallSubject()
    acquire(tracker)
    tracker.prepare([], .4, 640, 480)
    neighbour = dict(moved(person(), 25), recovery_verified=False, recovery_identity_competitor=True)
    result = tracker.finish([recovered('standing'), neighbour], .4, 640, 480)
    assert result['state'] == 'ambiguous' and result['person'] is None
