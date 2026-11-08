"""Identity support across bends must not change ordinary/fall target routing."""
import copy

import pytest

from fallpre.weekly.webcam_target import StableSubject, candidate
from test_webcam_target import detection


def supported_subject(folded_continuation=True):
    tracker = StableSubject(folded_continuation=folded_continuation)
    for i in range(16):
        result = tracker.update([detection()], i * .1, 640, 480)
    assert result['target_index'] == 0
    return tracker, result['target_epoch']


def hip_projection_jump():
    p = detection()
    # Hip projection moves upward during a frontal fold; current knees and
    # feet remain exactly where they were. The old hip-to-ankle length and
    # torso proportions now fail, although support identity is unambiguous.
    for s, h in [(5, 11), (6, 12)]:
        p['keypoints'][h][1] -= 45
        p['keypoints'][s][:2] = [p['keypoints'][h][0] + 9, p['keypoints'][h][1] + 4]
    return p


def test_existing_identity_uses_real_stable_knees_and_feet_during_hip_projection_jump():
    tracker, epoch = supported_subject()
    p = hip_projection_jump()
    original = copy.deepcopy(p)
    assert candidate(p, 640, 480, tracker.anchor) is None
    for t in [1.6, 1.7, 1.8]:
        result = tracker.update([p], t, 640, 480)
        assert result['target_index'] == 0 and result['target_epoch'] == epoch
        assert tracker.anchor['support_bridge']
    assert p == original


def test_default_selector_keeps_old_behavior_and_cannot_use_support_bridge():
    tracker, _ = supported_subject(folded_continuation=False)
    assert tracker.update([hip_projection_jump()], 1.6, 640, 480)['target_index'] is None
    assert tracker.support_reference is None


def test_support_bridge_cannot_acquire_a_new_person():
    tracker = StableSubject(folded_continuation=True)
    for i in range(20):
        assert tracker.update([hip_projection_jump()], i * .1, 640, 480)['target_index'] is None


def test_weak_observations_do_not_extend_fixed_support_deadline():
    tracker, epoch = supported_subject()
    reference_t = tracker.support_reference['t']
    for i in range(16, 30):
        result = tracker.update([hip_projection_jump()], i * .1, 640, 480)
        assert result['target_index'] == 0 and result['target_epoch'] == epoch
        assert tracker.support_reference['t'] == reference_t
    tracker.update([hip_projection_jump()], 3.0, 640, 480)
    assert tracker.update([hip_projection_jump()], 3.1, 640, 480)['target_index'] is None
    assert tracker.support_reference['t'] == reference_t


@pytest.mark.parametrize('mode', ['low_leg_confidence', 'missing_foot', 'translation',
                                 'flexed_knee', 'horizontal_support', 'weak_detection',
                                 'competition', 'gap'])
def test_support_bridge_abstains_without_current_complete_unique_planted_support(mode):
    tracker, _ = supported_subject()
    p = hip_projection_jump()
    if mode == 'low_leg_confidence':
        p['keypoints'][13][2] = .84
    elif mode == 'missing_foot':
        p['keypoints'][15][2] = 0
    elif mode == 'translation':
        for point in p['keypoints']:
            point[0] += 20
    elif mode == 'flexed_knee':
        p['keypoints'][13][0] += 20
    elif mode == 'horizontal_support':
        for h, k, a in [(11, 13, 15), (12, 14, 16)]:
            p['keypoints'][k][:2] = [p['keypoints'][h][0]+35, p['keypoints'][h][1]]
            p['keypoints'][a][:2] = [p['keypoints'][h][0]+70, p['keypoints'][h][1]]
    elif mode == 'weak_detection':
        p['confidence'] = .69
    items = [p, copy.deepcopy(p)] if mode == 'competition' else [p]
    result = tracker.update(items, 2.0 if mode == 'gap' else 1.6, 640, 480)
    assert result['target_index'] is None


def test_fresh_full_pose_can_end_the_bridge_without_new_identity_or_stale_imputation():
    tracker, epoch = supported_subject()
    tracker.update([hip_projection_jump()], 1.6, 640, 480)
    result = tracker.update([detection()], 1.7, 640, 480)
    assert result['target_index'] == 0 and result['target_epoch'] == epoch
    assert not tracker.support_bridge_active
    assert not tracker.anchor.get('support_bridge')
    assert tracker.update([], 1.8, 640, 480)['target_index'] is None
