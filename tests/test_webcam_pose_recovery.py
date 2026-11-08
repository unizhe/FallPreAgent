from copy import deepcopy

import numpy as np
import pytest
from test_webcam_actions import detection

from fallpre.weekly.webcam_pose_recovery import LivePoseRecovery, restore_pose


def forward(person, width, height, turn):
    result = deepcopy(person)
    points = np.asarray(person['keypoints'])
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if turn == 1:
        points[:, 0], points[:, 1] = y, width-1-x
    elif turn == 2:
        points[:, 0], points[:, 1] = width-1-x, height-1-y
    elif turn == 3:
        points[:, 0], points[:, 1] = height-1-y, x
    x1, y1, x2, y2 = person['box']
    boxes = {0: [x1, y1, x2, y2], 1: [y1, width-x2, y2, width-x1],
             2: [width-x2, height-y2, width-x1, height-y1],
             3: [height-y2, x1, height-y1, x2]}
    return dict(result, box=boxes[turn], keypoints=points.tolist())


@pytest.mark.parametrize('turn', range(4))
def test_rotated_crop_coordinates_and_confidences_round_trip(turn):
    p = detection()
    p['keypoints'][15][2] = 0
    restored = restore_pose(forward(p, 320, 240, turn), 320, 240, turn, np.array([31, 47]))
    np.testing.assert_allclose(restored['box'], np.array(p['box'])+[31, 47, 31, 47])
    expected = np.asarray(p['keypoints'])
    expected[:, :2] += [31, 47]
    np.testing.assert_allclose(restored['keypoints'], expected)
    assert restored['keypoints'][15][2] == 0


def test_recovery_uses_current_pixels_and_two_orientations_without_joint_synthesis():
    recovery = LivePoseRecovery()
    current = detection()
    calls = []
    image = np.full((240, 320, 3), 71, np.uint8)

    def infer(view):
        assert np.all(view == 71)
        turn = recovery.config.rotations[len(calls)]
        calls.append(view.shape)
        return [forward(current, 320, 240, turn)]

    result, audit = recovery.recover(image, [80, 30, 240, 230], infer)
    assert audit['inference_passes'] == 2 and len(result) == 2
    assert result[0]['recovery_verified'] and result[0]['keypoints'] == current['keypoints']
    assert result[0]['recovery_views'] == [0, 1]
    # Recovery acceptance belongs to the tracker, not the crop verifier.
    assert not audit['accepted']


def test_missing_or_single_view_pose_never_fills_the_gap_and_calls_are_bounded():
    recovery = LivePoseRecovery()
    for mode in ('empty', 'one'):
        calls = []

        def infer(view):
            calls.append(view)
            return [detection()] if mode == 'one' and len(calls) == 1 else []

        result, audit = recovery.recover(np.zeros((240, 320, 3), np.uint8), [80, 30, 240, 230], infer)
        assert result == [] and len(calls) == audit['inference_passes'] == 4


def test_rotated_proposals_must_agree_on_each_torso_joint():
    recovery = LivePoseRecovery()
    first = dict(detection(), recovery_rotation=0)
    second = dict(deepcopy(first), recovery_rotation=1)
    second['keypoints'][11][0] += 80
    assert not recovery.agree(first, second)
    assert recovery.consensus([first, second]) == []
    assert not recovery.agree(first, dict(first))  # Same orientation cannot corroborate itself.


def test_no_missing_torso_is_invented_and_outside_image_joints_are_invalid():
    recovery = LivePoseRecovery()
    p = detection()
    assert recovery.valid(p, (240, 320, 3))
    p['keypoints'][5][2] = .29
    assert not recovery.valid(p, (240, 320, 3))
    p = detection()
    p['keypoints'][11][1] = 500
    assert not recovery.valid(p, (240, 320, 3))


def test_two_consensus_people_remain_two_for_identity_ambiguity_check():
    recovery = LivePoseRecovery()
    left, right = detection(), detection(250)
    proposals = [dict(deepcopy(p), recovery_rotation=turn) for p in (left, right) for turn in (0, 1)]
    assert {tuple(p['box']) for p in recovery.consensus(proposals)} == {tuple(left['box']), tuple(right['box'])}


def test_loose_cross_view_agreement_does_not_merge_close_people():
    recovery = LivePoseRecovery()
    first, second = detection(), detection(115)
    proposals = [dict(deepcopy(p), recovery_rotation=turn) for p in (first, second) for turn in (0, 1)]
    assert {tuple(p['box']) for p in recovery.consensus(proposals)} == {tuple(first['box']), tuple(second['box'])}


def test_single_view_competitor_is_retained_only_as_identity_veto():
    recovery = LivePoseRecovery()
    tracked, neighbor = detection(), detection(250)
    calls = []

    def infer(view):
        turn = recovery.config.rotations[len(calls)]
        calls.append(turn)
        poses = [tracked, neighbor] if turn == 0 else [neighbor]
        return [forward(p, 320, 240, turn) for p in poses]

    result, audit = recovery.recover(np.zeros((240, 320, 3), np.uint8), [80, 30, 240, 230], infer)
    competitors = [p for p in result if p.get('recovery_identity_competitor')]
    assert audit['identity_competitors'] == len(competitors) == 1
    assert competitors[0]['box'] == tracked['box'] and not competitors[0]['recovery_verified']


def test_invalid_region_does_not_invoke_model():
    def forbidden(view):
        raise AssertionError('invalid crop must not invoke inference')

    assert LivePoseRecovery().recover(np.zeros((240, 320, 3), np.uint8), [5, 5, 5, 5], forbidden)[0] == []
