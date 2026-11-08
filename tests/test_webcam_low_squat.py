import numpy as np
import pytest

from fallpre.weekly.webcam_daily_activity import DailyActivity
from fallpre.weekly.webcam_low_squat import ProjectedLowSquat


def frontal_low_squat():
    points = np.zeros((17, 3))
    for side, coordinates in [
        ((5, 11, 13, 15), ((150, 100), (140, 190), (175, 195), (135, 235))),
        ((6, 12, 14, 16), ((55, 105), (75, 195), (70, 210), (80, 235))),
    ]:
        for joint, xy in zip(side, coordinates):
            points[joint] = (*xy, .95)
    return dict(kpts=points.tolist(), box=[40, 50, 200, 250], det=.95)


def result(label='unknown', raw='background'):
    return dict(label=label, raw_label=raw)


def test_frontal_low_squat_requires_real_stationary_window_without_relaxing_knee_gate():
    daily = DailyActivity()
    outputs = [daily.update(t, 1, frontal_low_squat(), result()) for t in (0, .13, .26)]
    assert [r['label'] for r in outputs] == ['unknown', 'unknown', 'squatting']
    assert outputs[-1]['decision_stage'] == 'projected_low_squat'
    assert outputs[-1]['low_squat']['confirmed']


@pytest.mark.parametrize('scale,mirror', [(0.6, False), (2., False), (1., True)])
def test_decision_uses_body_ratios_and_is_invariant_to_image_mirroring(scale, mirror):
    detector = ProjectedLowSquat()
    person = frontal_low_squat()
    points = np.asarray(person['kpts'])
    points[:, :2] *= scale
    if mirror:
        points[:, 0] = 400 - points[:, 0]
    person['kpts'] = points.tolist()
    for t in (0, .13, .26):
        info = detector.update(t, 1, person, result())
    assert info['confirmed']


@pytest.mark.parametrize('mode', ['seat_height', 'horizontal_torso', 'straight_legs',
                                 'missing_leg', 'weak_leg', 'moving_feet', 'moving_body'])
def test_seated_floored_missing_and_moving_counterexamples_do_not_confirm(mode):
    detector = ProjectedLowSquat()
    outputs = []
    for i, t in enumerate((0, .13, .26, .39)):
        person = frontal_low_squat()
        p = np.asarray(person['kpts'])
        if mode == 'seat_height':
            p[[15, 16], 1] += 40  # overlap with seated support remains unclassified
        elif mode == 'horizontal_torso':
            p[[5, 6], :2] = p[[11, 12], :2] + [90, -10]
        elif mode == 'straight_legs':
            p[[13, 14], :2] = (p[[11, 12], :2] + p[[15, 16], :2]) / 2
        elif mode in {'missing_leg', 'weak_leg'}:
            p[16, 2] = 0 if mode == 'missing_leg' else .79
        elif mode == 'moving_feet':
            p[[15, 16], 0] += i * 15
        elif mode == 'moving_body':
            p[[5, 6, 11, 12], 0] += i * 12
        person['kpts'] = p.tolist()
        outputs.append(detector.update(t, 1, person, result()))
    assert not any(o['confirmed'] for o in outputs)


@pytest.mark.parametrize('mode', ['falling', 'rising', 'camera', 'missing', 'epoch', 'gap'])
def test_confirmed_squat_cannot_persist_through_conflicts_or_discontinuity(mode):
    detector = ProjectedLowSquat()
    for t in (0, .13, .26):
        before = detector.update(t, 1, frontal_low_squat(), result())
    assert before['confirmed']
    after = detector.update(.9 if mode == 'gap' else .39,
                            2 if mode == 'epoch' else 1,
                            None if mode == 'missing' else frontal_low_squat(),
                            result('standing_up' if mode == 'rising' else 'unknown',
                                   'falling' if mode == 'falling' else 'background'),
                            camera_stable=mode != 'camera')
    assert not after['confirmed']


def test_daily_branch_keeps_current_falling_vote_and_original_seat_height_output():
    for mode in ('falling', 'seat_height'):
        daily = DailyActivity()
        for t in (0, .13, .26, .39):
            person = frontal_low_squat()
            if mode == 'seat_height':
                for joint in (15, 16):
                    person['kpts'][joint][1] += 40
            out = daily.update(t, 1, person,
                               result('seated', 'falling' if mode == 'falling' else 'background'))
            assert out['label'] == 'seated'


def test_confirmed_low_seat_is_not_overridden_by_projected_squat_geometry():
    # A low stool can produce the same short leg projection. The new route
    # cannot establish ground contact or overturn an already confirmed seat.
    daily = DailyActivity()
    for t in (0, .13, .26, .39):
        out = daily.update(t, 1, frontal_low_squat(), result('seated'))
        assert out['label'] == 'seated'
        assert not out['low_squat']['confirmed']


def test_confirmed_seat_clears_prior_projected_squat_evidence():
    detector = ProjectedLowSquat()
    for t in (0, .13, .26):
        info = detector.update(t, 1, frontal_low_squat(), result())
    assert info['confirmed']
    assert not detector.update(.39, 1, frontal_low_squat(), result('seated'))['confirmed']
    assert not detector.update(.52, 1, frontal_low_squat(), result())['confirmed']
