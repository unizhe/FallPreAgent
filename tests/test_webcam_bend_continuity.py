"""Behavioral checks for bounded continuation through torso foreshortening."""
import copy

import pytest

from fallpre.weekly.webcam_daily_activity import DailyActivity
from test_webcam_daily_activity import straight_pose


def unknown(raw='background'):
    return dict(label='unknown', raw_label=raw, decision_code='target_unconfirmed')


def folded_pose(compressed=False):
    p = straight_pose()
    for shoulder, hip, knee, ankle in [(5, 11, 13, 15), (6, 12, 14, 16)]:
        x = p['kpts'][hip][0]
        p['kpts'][knee][:2] = [x, 270]
        p['kpts'][ankle][:2] = [x, 380]
        # The shoulders remain at the same height; widening their projected
        # horizontal offset changes hip angle while torso stays short vs legs.
        p['kpts'][shoulder][:2] = [x+(50 if compressed else 8), 175]
    return p


def confirmed_bend():
    activity = DailyActivity()
    assert activity.update(0, 1, folded_pose(), unknown())['label'] == 'unknown'
    assert activity.update(.13, 1, folded_pose(), unknown())['label'] == 'bending'
    return activity


def test_current_complete_legs_bridge_projection_and_strong_pose_resumes_without_rewaiting():
    activity = confirmed_bend()
    p = folded_pose(compressed=True)
    original = copy.deepcopy(p)
    assert activity.update(.26, 1, p, unknown())['label'] == 'bending'
    assert p == original
    assert activity.update(.39, 1, folded_pose(), unknown())['label'] == 'bending'


def test_ambiguous_compressed_pose_cannot_initialize_bending_without_a_confirmed_reference():
    activity = DailyActivity()
    for t in [0, .13, .26, .39, .52]:
        assert activity.update(t, 1, folded_pose(compressed=True), unknown())['label'] == 'unknown'


def test_continuation_frames_do_not_renew_the_fixed_confirmation_deadline():
    activity = confirmed_bend()
    for t in [.26, .39]:
        assert activity.update(t, 1, folded_pose(compressed=True), unknown())['label'] == 'bending'
    for t in [.52, .65, .78]:
        assert activity.update(t, 1, folded_pose(compressed=True), unknown())['label'] == 'unknown'


@pytest.mark.parametrize('mode', ['missing_person', 'missing_joint', 'low_confidence', 'gap', 'epoch'])
def test_missing_or_discontinuous_evidence_cannot_resume_weak_continuation(mode):
    activity = confirmed_bend()
    p = folded_pose(compressed=True)
    if mode == 'missing_person':
        p = None
    elif mode == 'missing_joint':
        p['kpts'][15][2] = 0
    elif mode == 'low_confidence':
        p['kpts'][15][2] = .7
    t, epoch = (.6 if mode == 'gap' else .26), (2 if mode == 'epoch' else 1)
    assert activity.update(t, epoch, p, unknown())['label'] == 'unknown'
    assert activity.update(t+.13, epoch, folded_pose(compressed=True), unknown())['label'] == 'unknown'


@pytest.mark.parametrize('mode', ['one_leg_bent', 'one_foot_step', 'translation', 'horizontal_legs'])
def test_incompatible_current_legs_cannot_borrow_a_previous_bend(mode):
    activity = confirmed_bend()
    p = folded_pose(compressed=True)
    if mode == 'one_leg_bent':
        p['kpts'][13][0] -= 30
    elif mode == 'one_foot_step':
        p['kpts'][15][0] += 30
    elif mode == 'translation':
        for point in p['kpts']:
            point[0] += 30
    else:
        for shoulder, hip, knee, ankle in [(5, 11, 13, 15), (6, 12, 14, 16)]:
            x = p['kpts'][hip][0]
            p['kpts'][shoulder][:2] = [x, 142]
            p['kpts'][knee][:2] = [x+50, 160]
            p['kpts'][ankle][:2] = [x+110, 160]
    assert activity.update(.26, 1, p, unknown())['label'] == 'unknown'


@pytest.mark.parametrize('compressed', [True, False])
def test_current_raw_falling_is_never_overridden_and_clears_weak_continuation(compressed):
    activity = confirmed_bend()
    r = unknown('falling')
    assert activity.update(.26, 1, folded_pose(compressed=compressed), r)['label'] == 'unknown'
    assert r['raw_label'] == 'falling'
    assert activity.update(.39, 1, folded_pose(compressed=True), unknown())['label'] == 'unknown'


def test_visible_shoulder_unfolding_cannot_continue_a_previous_bend():
    activity = DailyActivity()
    # Independently establish a fully folded pose without motion history.
    for t in [0, .13]:
        out = activity.update(t, 1, folded_pose(), unknown())
    assert out['label'] == 'bending'
    # Both shoulders clearly rise while hips and straight legs stay put.
    # This compressed torso is now an unfolding transition, not continuing bend.
    p = folded_pose(compressed=True)
    for shoulder in [5, 6]:
        p['kpts'][shoulder][1] -= 10
    out = activity.update(.26, 1, p, unknown())
    assert out['label'] == 'unknown'
