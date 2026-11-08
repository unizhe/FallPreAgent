"""Synthetic RGB/pose streams verify causal semantics, not model accuracy."""
from copy import deepcopy

import pytest

from fallpre.weekly.webcam_rgb_fall import LiveRgbFall

STANDING = [100., 30., 200., 430.]
FALLEN = [70., 350., 500., 450.]


def pose(box=STANDING):
    x1, y1, x2, y2 = box
    mid, h = (x1 + x2) / 2, y2 - y1
    points = [[mid, y1 + .1 * h, .95] for _ in range(17)]
    for left, right, fraction in ((5, 6, .25), (11, 12, .5), (13, 14, .725), (15, 16, .95)):
        points[left] = [mid - 10, y1 + fraction * h, .95]
        points[right] = [mid + 10, y1 + fraction * h, .95]
    return dict(box=list(box), keypoints=points, confidence=.95)


def detection(label='standing', box=None, confidence=.95):
    return dict(box=list(box or (FALLEN if label == 'fallen' else STANDING)),
                confidence=confidence, label=label)


def feed(fsm, t, label='standing', box=None, **kwargs):
    detections = [detection(label, box)] if isinstance(label, str) else label
    return fsm.update(detections, t, 640, 480, **kwargs)


def reference(fsm, start=0., box=STANDING, pose_box=None, pose_epoch=7):
    for dt in (0., .1, .3, .5):
        result = feed(fsm, start + dt, box=box, pose_person=pose(pose_box or box), pose_epoch=pose_epoch)
    assert result['fall']['target_epoch'] is not None
    assert result['diagnostics']['reference_ready']
    return result


def confirm(fsm):
    reference(fsm)
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(fsm, t, 'fallen')
    assert result['fall']['state'] == 'confirmed'
    return result


def test_pose_witness_and_four_observations_over_half_second_establish_rgb_reference():
    fsm = LiveRgbFall()
    for t in (0., .1, .2, .3):
        result = feed(fsm, t, pose_person=pose(), pose_epoch=3)
        assert result['fall']['target_epoch'] is None
    result = feed(fsm, .5, pose_person=pose(), pose_epoch=3)
    assert result['fall']['target_epoch'] == 'rgb-1'
    assert result['diagnostics']['reference_pose_epoch'] == 3
    assert result['observation']['person']['keypoints'] == []
    assert result['observation']['source'] == 'rgb_fall'
    assert result['observation']['selection']['raw_index'] is None
    assert result['observation']['selection']['rgb_index'] == 0
    assert fsm.export()['statistics']['acquisitions'] == 1


def test_three_observations_over_half_second_are_insufficient():
    fsm = LiveRgbFall()
    for t in (0., .25, .5):
        result = feed(fsm, t, pose_person=pose())
    assert result['fall']['target_epoch'] is None
    assert feed(fsm, .6, pose_person=pose())['fall']['target_epoch'] == 'rgb-1'


@pytest.mark.parametrize('problem', ['missing', 'mismatch', 'feet_missing', 'seated', 'pose_epoch_changed'])
def test_initial_upright_requires_current_matching_reliable_pose(problem):
    fsm = LiveRgbFall()
    for i in range(8):
        current_pose = pose()
        if problem == 'missing':
            current_pose = None
        elif problem == 'mismatch':
            current_pose = pose([400, 30, 500, 430])
        elif problem == 'feet_missing':
            current_pose['keypoints'][15] = current_pose['keypoints'][16] = [0, 0, 0]
        elif problem == 'seated':
            for j in (13, 14):
                current_pose['keypoints'][j][0] += 100
        result = feed(fsm, i * .2, pose_person=current_pose,
                      pose_epoch=i if problem == 'pose_epoch_changed' else 1)
    assert not result['diagnostics']['reference_ready']
    assert result['fall']['target_epoch'] is None


def test_rgb_geometry_does_not_mix_pose_box_height_and_fallen_box_height():
    fsm = LiveRgbFall()
    reference(fsm, pose_box=[80, 10, 220, 450])
    result = feed(fsm, .75, 'fallen')['fall']
    assert result['metrics']['center_drop_norm'] == .425  # (400 - 230) / RGB h=400.
    assert result['metrics']['height_ratio'] == .25


def test_fall_survives_complete_pose_loss_and_pose_epoch_change_after_rgb_admission():
    fsm = LiveRgbFall()
    reference(fsm)
    for t in (.75, 1., 1.25, 1.5):
        result = feed(fsm, t, 'fallen', pose_epoch=99)
        assert result['fall']['state'] == 'suspected'
    result = feed(fsm, 1.75, 'fallen', pose_epoch=99)
    assert result['fall']['state'] == 'confirmed'
    assert result['fall']['candidate_duration_s'] == 1.
    assert result['fall']['target_epoch'] == result['fall']['event_target_epoch'] == 'rgb-1'
    assert result['observation']['person']['keypoints'] == []
    assert result['diagnostics']['reference_pose_epoch'] == 7


def test_initial_fallen_is_current_yellow_without_an_invented_identity_or_reference():
    fsm = LiveRgbFall()
    for i in range(16):
        result = feed(fsm, i * .25, 'fallen')
        assert result['fall']['label'] == 'LYING?' and result['fall']['observed']
        assert result['fall']['event_id'] is None and result['fall']['target_epoch'] is None
        assert result['observation']['selection']['state'] == 'provisional'
        assert result['observation']['person']['box'] == FALLEN
    assert fsm.export()['statistics']['confirmed'] == 0


@pytest.mark.parametrize('label,box', [('sitting', FALLEN), ('standing', FALLEN),
                                      ('fallen', STANDING), ('fallen', [70, 120, 500, 220]),
                                      ('fallen', [100, 150, 200, 475])])
def test_geometry_alone_or_fallen_label_alone_does_not_confirm(label, box):
    fsm = LiveRgbFall()
    reference(fsm)
    for i in range(1, 10):
        result = feed(fsm, .5 + i * .25, label, box)['fall']
        assert result['event_id'] is None
        assert result['candidate_duration_s'] == 0


def test_only_observed_fallen_with_sufficient_descent_counts_toward_one_second():
    fsm = LiveRgbFall()
    reference(fsm)
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(fsm, t, 'fallen', [70, 280, 500, 380])
    assert result['fall']['candidate_duration_s'] == 0
    for t in (2., 2.25, 2.5, 2.75):
        result = feed(fsm, t, 'fallen')
        assert result['fall']['event_id'] is None
    assert feed(fsm, 3., 'fallen')['fall']['state'] == 'confirmed'


def test_slow_continuous_descent_can_confirm_without_a_speed_or_pose_requirement():
    fsm = LiveRgbFall()
    reference(fsm)
    for i in range(1, 13):
        fraction = i / 12
        box = [a + (b - a) * fraction for a, b in zip(STANDING, FALLEN)]
        result = feed(fsm, .5 + i * .25, 'fallen', box)
        assert result['fall']['target_epoch'] == 'rgb-1'
        assert result['fall']['event_id'] is None
    for t in (3.75, 4., 4.25, 4.5):
        result = feed(fsm, t, 'fallen')
    assert result['fall']['state'] == 'confirmed'
    assert result['fall']['metrics']['center_drop_norm'] == .425


def test_missing_during_initial_admission_restarts_the_observation_window():
    fsm = LiveRgbFall()
    for t in (0., .1, .3):
        feed(fsm, t, pose_person=pose())
    feed(fsm, .4, [])
    for t in (.5, .6, .8):
        result = feed(fsm, t, pose_person=pose())
        assert result['fall']['target_epoch'] is None
    assert feed(fsm, 1., pose_person=pose())['fall']['target_epoch'] == 'rgb-1'


def test_class_confidence_boundary_is_inclusive_and_lower_confidence_is_missing():
    fsm = LiveRgbFall()
    reference(fsm)
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(fsm, t, [detection('fallen', confidence=.60)])
    assert result['fall']['state'] == 'confirmed'
    result = feed(fsm, 2., [detection('fallen', confidence=.59999)])
    assert not result['fall']['observed'] and result['observation']['person'] is None


def test_missing_short_interval_preserves_reference_but_never_counts_or_draws_old_box():
    fsm = LiveRgbFall()
    reference(fsm)
    feed(fsm, .75, 'fallen')
    feed(fsm, 1., 'fallen')
    missing = feed(fsm, 1.25, [])
    assert missing['observation']['person'] is None and missing['observation']['source'] == 'none'
    assert not missing['fall']['observed']
    resumed = feed(fsm, 1.5, 'fallen')['fall']
    assert resumed['candidate_duration_s'] == .25 and resumed['metrics']['had_upright']
    for t in (1.75, 2.):
        assert feed(fsm, t, 'fallen')['fall']['state'] == 'suspected'
    assert feed(fsm, 2.25, 'fallen')['fall']['state'] == 'confirmed'


@pytest.mark.parametrize('missing_frame', [True, False])
def test_long_source_gap_forgets_reference_even_when_wall_clock_is_fast(missing_frame):
    fsm = LiveRgbFall()
    reference(fsm)
    feed(fsm, .75, 'fallen')
    if missing_frame:
        feed(fsm, 1.3, [])
    for i in range(8):
        result = feed(fsm, 1.5 + i * .25, 'fallen')['fall']
    assert result['target_epoch'] is None and result['event_id'] is None
    assert result['state'] == 'suspected' and not result['metrics']['had_upright']


@pytest.mark.parametrize('stable', [False, None])
def test_unknown_or_moving_camera_clears_reference(stable):
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, 'fallen', camera_stable=stable, quality_reason='测试相机未知')
    assert result['fall']['reason'] == '测试相机未知' and not result['fall']['observed']
    assert result['observation']['person'] is None
    for t in (1., 1.25, 1.5, 1.75, 2.):
        result = feed(fsm, t, 'fallen')
    assert result['fall']['state'] == 'suspected' and not result['diagnostics']['reference_ready']


def test_multiple_people_reset_identity_without_selecting_highest_score():
    fsm = LiveRgbFall()
    reference(fsm)
    feed(fsm, .75, 'fallen')
    result = feed(fsm, 1., [detection('fallen'), detection('standing', [530, 30, 620, 430], .60)])
    assert result['fall']['reason_code'] == 'multiple_rgb_candidates'
    assert result['fall']['target_epoch'] is None and result['observation']['person'] is None
    for t in (1.25, 1.5, 1.75, 2., 2.25):
        result = feed(fsm, t, 'fallen')
    assert result['fall']['event_id'] is None


def test_near_duplicate_different_class_is_ambiguous_not_a_vote():
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, [detection('fallen'), detection('sitting', FALLEN)])
    assert result['fall']['reason_code'] == 'multiple_rgb_candidates'
    assert not result['fall']['observed']


@pytest.mark.parametrize('confidence', [.25, .59999])
def test_weak_nearby_person_vetoes_strong_neighbor_and_clears_reference(confidence):
    fsm = LiveRgbFall()
    reference(fsm)
    feed(fsm, .75, 'fallen')
    result = feed(fsm, 1., [detection('fallen'), detection('sitting', [100, 330, 240, 450], confidence)])
    assert result['fall']['reason_code'] == 'weak_rgb_identity_competitor'
    assert not result['fall']['observed'] and result['observation']['person'] is None
    assert result['fall']['target_epoch'] is None and not result['diagnostics']['reference_ready']
    for t in (1.25, 1.5, 1.75, 2., 2.25):
        result = feed(fsm, t, 'fallen')
    assert result['fall']['event_id'] is None


def test_weak_current_target_prevents_a_strong_new_person_taking_its_reference():
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, [detection('fallen', [350, 350, 600, 450]),
                           detection('standing', confidence=.30)])
    assert result['fall']['reason_code'] == 'weak_rgb_identity_competitor'
    assert not result['diagnostics']['reference_ready']


def test_weak_proposals_cannot_establish_identity_or_extend_ttl():
    fsm = LiveRgbFall()
    for i in range(4):
        result = feed(fsm, i * .2, [detection('standing', confidence=.59)], pose_person=pose())
    assert result['fall']['target_epoch'] is None and not result['fall']['observed']
    reference(fsm, start=1.)
    for t in (1.75, 2., 2.25):
        result = feed(fsm, t, [detection('fallen', confidence=.59)])
        assert not result['fall']['observed']
    assert not result['diagnostics']['reference_ready']


def test_two_weak_nearby_proposals_clear_reference_without_selecting_either():
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, [detection('standing', confidence=.59),
                           detection('sitting', [180, 100, 300, 430], .3)])
    assert result['fall']['reason_code'] == 'weak_rgb_identity_competitor'
    assert result['fall']['target_epoch'] is None and result['observation']['person'] is None


def test_far_weak_proposal_does_not_replace_or_hide_the_current_strong_observation():
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, [detection('standing', [530, 30, 620, 430], .3), detection()])
    assert result['fall']['target_epoch'] == 'rgb-1' and result['fall']['observed']
    assert result['observation']['selection']['raw_index'] is None
    assert result['observation']['selection']['rgb_index'] == 1
    assert result['observation']['person']['box'] == STANDING


def test_nonoverlapping_person_never_inherits_reference():
    fsm = LiveRgbFall()
    reference(fsm)
    for i in range(6):
        result = feed(fsm, .75 + i * .25, 'fallen', [350, 350, 600, 450])
    assert result['fall']['target_epoch'] is None and result['fall']['event_id'] is None


def test_bottom_clipping_uses_visible_top_descent_and_not_height_shrinkage():
    fsm = LiveRgbFall()
    reference(fsm, box=[100, 100, 200, 400])
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(fsm, t, 'fallen', [100, 230, 250, 480])
    fall = result['fall']
    assert fall['state'] == 'confirmed'
    assert fall['reason_code'] == 'bottom_clipped_fallen_with_visible_top_descent'
    assert fall['metrics']['height_ratio'] > .8
    assert fall['metrics']['top_drop_norm'] >= .35 and fall['metrics']['center_drop_norm'] >= .35
    assert not fall['metrics']['height_reduction_is_evidence']
    assert fsm.export()['confirmed_events'][0]['geometry_evidence'] == fall['reason_code']


def test_bottom_clipped_classification_without_visible_descent_does_not_confirm():
    fsm = LiveRgbFall()
    reference(fsm)
    for i in range(6):
        result = feed(fsm, .75 + i * .25, 'fallen', [100, 80, 250, 480])
    assert result['fall']['state'] == 'suspected' and result['fall']['event_id'] is None
    assert result['fall']['metrics']['bottom_clipped']
    assert result['fall']['candidate_duration_s'] == 0


def test_bottom_clipped_standing_cannot_establish_whole_body_reference():
    fsm = LiveRgbFall()
    for i in range(6):
        result = feed(fsm, i * .2, box=[100, 30, 200, 480], pose_person=pose())
    assert result['fall']['target_epoch'] is None
    assert not result['diagnostics']['reference_ready']


@pytest.mark.parametrize('box', [[0, 350, 500, 450], [70, 0, 500, 450], [70, 350, 640, 450]])
def test_top_and_side_clipping_do_not_supply_geometry(box):
    fsm = LiveRgbFall()
    reference(fsm)
    result = feed(fsm, .75, 'fallen', box)
    assert result['fall']['reason_code'] == 'unreliable_clipped_box'
    assert not result['fall']['observed'] and result['observation']['person'] is None


def test_sitting_interrupts_candidate_and_repeated_fallen_must_start_duration_again():
    fsm = LiveRgbFall()
    reference(fsm)
    feed(fsm, .75, 'fallen')
    feed(fsm, 1., 'fallen')
    assert feed(fsm, 1.25, 'sitting', FALLEN)['fall']['state'] == 'normal'
    for t in (1.5, 1.75, 2., 2.25):
        assert feed(fsm, t, 'fallen')['fall']['event_id'] is None
    assert feed(fsm, 2.5, 'fallen')['fall']['state'] == 'confirmed'


def test_historical_event_latches_but_missing_or_new_target_never_gets_old_red_box():
    fsm = LiveRgbFall()
    old = confirm(fsm)['fall']
    current = feed(fsm, 2., 'sitting', FALLEN)['fall']
    assert current['state'] == 'confirmed' and current['current_target_confirmed']
    missing = feed(fsm, 2.25, [])
    assert missing['fall']['state'] == 'unobservable' and missing['observation']['person'] is None
    new = reference(fsm, start=3., box=[400, 30, 500, 430])['fall']
    assert new['target_epoch'] == 'rgb-2' and new['state'] == 'normal'
    assert not new['current_target_confirmed'] and new['event_id'] == old['event_id']
    assert new['event_target_epoch'] == 'rgb-1'
    assert len(fsm.export()['confirmed_events']) == 1


@pytest.mark.parametrize('bad_t', [0., -.1, float('nan'), float('inf')])
def test_duplicate_reverse_or_nonfinite_time_is_rejected_without_state_mutation(bad_t):
    fsm = LiveRgbFall()
    feed(fsm, 0., pose_person=pose())
    before = fsm.export()
    with pytest.raises(ValueError):
        feed(fsm, bad_t, 'fallen')
    assert fsm.export() == before


def test_low_confidence_and_malformed_inputs_are_missing_and_inputs_are_never_mutated():
    fsm = LiveRgbFall()
    reference(fsm)
    inputs = [detection('fallen', confidence=.59), dict(box=[0, 0, 0, 0], label='fallen', confidence=.99),
              dict(box=FALLEN, label='fallen', confidence=float('nan'))]
    original = deepcopy(inputs)
    result = feed(fsm, .75, inputs)
    assert result['observation']['person'] is None
    assert not result['fall']['observed']
    assert inputs[0] == original[0] and inputs[1] == original[1]
    result['fall']['metrics']['changed'] = True
    report = fsm.export()
    report['statistics']['confirmed'] = 100
    assert fsm.export()['statistics']['confirmed'] == 0


def test_dimension_change_clears_reference_without_new_pose_witness():
    fsm = LiveRgbFall()
    reference(fsm)
    result = fsm.update([detection('fallen')], .75, 800, 600)
    assert result['fall']['target_epoch'] is None
    assert not result['diagnostics']['reference_ready']
