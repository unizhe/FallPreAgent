"""Synthetic observations check causal event semantics, not detection accuracy."""
from copy import deepcopy

import pytest

from fallpre.weekly.webcam_fall import LiveFallDetector


def person(kind='standing'):
    points = [[150., 80., .95] for _ in range(17)]
    box = [100., 30., 200., 430.]
    for left, right, y in ((5, 6, 130), (11, 12, 230), (13, 14, 320), (15, 16, 410)):
        points[left], points[right] = [140., float(y), .95], [160., float(y), .95]
    if kind == 'low':
        box = [70., 350., 500., 450.]
        points[5], points[6] = [100., 380., .95], [100., 400., .95]
        points[11], points[12] = [290., 380., .95], [290., 400., .95]
    elif kind == 'seated':
        box = [100., 150., 270., 450.]
        points[5], points[6] = [140., 220., .95], [160., 220., .95]
        points[11], points[12] = [140., 330., .95], [160., 330., .95]
        points[13], points[14] = [240., 350., .95], [260., 350., .95]
        points[15], points[16] = [240., 430., .95], [260., 430., .95]
    elif kind == 'bending':
        box = [100., 130., 320., 430.]
        points[5], points[6] = [290., 250., .95], [310., 250., .95]
        points[11], points[12] = [140., 280., .95], [160., 280., .95]
    return dict(box=box, confidence=.95, keypoints=points)


def feed(detector, t, kind='standing', epoch='person-a', **kwargs):
    detection = person(kind) if isinstance(kind, str) else kind
    return detector.update(detection, t, 640, 480, epoch, **kwargs)


def standing_reference(detector):
    for t in (0., .25, .5):
        result = feed(detector, t)
    assert result['metrics']['had_upright']
    return result


def confirmed(detector):
    standing_reference(detector)
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(detector, t, 'low')
    assert result['state'] == 'confirmed'
    return result


def test_fast_descent_requires_observed_low_duration_and_emits_one_event():
    detector = LiveFallDetector()
    standing_reference(detector)
    first = feed(detector, .75, 'low')
    assert first['label'] == 'LYING?' and first['candidate_duration_s'] == 0
    for t in (1., 1.25, 1.5):
        assert feed(detector, t, 'low')['state'] == 'suspected'
    result = feed(detector, 1.75, 'low')
    assert result['label'] == 'FALL' and result['confirmed_at_s'] == 1.75
    assert result['candidate_duration_s'] == 1
    assert result['event_target_epoch'] == result['target_epoch'] == 'person-a'
    for t in (2., 2.25, 2.5):
        feed(detector, t, 'low')
    report = detector.export()
    assert report['statistics']['confirmed'] == len(report['confirmed_events']) == 1
    assert [x['state'] for x in report['events'] if x['kind'] == 'state'] == ['normal', 'suspected', 'confirmed']


def test_slow_descent_has_no_speed_requirement_but_needs_low_posture():
    detector = LiveFallDetector()
    standing_reference(detector)
    # The descent is deliberately spread over multiple seconds. It never
    # crosses a rapid-change threshold because this branch does not need one.
    for i in range(1, 13):
        p = person('low')
        fraction = i / 12
        p['box'] = [70., 200. + 150 * fraction, 500., 300. + 150 * fraction]
        for j in (5, 6, 11, 12):
            p['keypoints'][j][1] -= 150 * (1 - fraction)
        result = feed(detector, .5 + i * .25, p)
    assert result['state'] == 'confirmed'
    assert result['metrics']['vertical_drop_norm'] >= .35


def test_startup_lying_and_falling_action_alone_cannot_confirm():
    detector = LiveFallDetector()
    for i in range(25):
        result = feed(detector, i * .25, 'low', raw_label='falling')
    assert result['state'] == 'suspected'
    assert not result['metrics']['had_upright'] and result['event_id'] is None
    detector = LiveFallDetector()
    for i in range(25):
        result = feed(detector, i * .25, raw_label='falling')
        assert result['state'] == 'normal' and result['event_id'] is None


@pytest.mark.parametrize('kind', ['seated', 'bending'])
def test_sitting_and_bending_are_not_falls_even_with_wrong_action_label(kind):
    detector = LiveFallDetector()
    standing_reference(detector)
    for i in range(1, 17):
        result = feed(detector, .5 + i * .25, kind, raw_label='falling')
        assert result['state'] == 'normal' and not result['metrics']['low_pose']


def test_missing_ankles_allow_low_pose_but_missing_torso_is_unobservable():
    detector = LiveFallDetector()
    standing_reference(detector)
    p = person('low')
    p['keypoints'][15] = p['keypoints'][16] = [0., 0., 0.]
    for t in (.75, 1., 1.25, 1.5, 1.75):
        result = feed(detector, t, p)
    assert result['state'] == 'confirmed'
    missing = deepcopy(p)
    missing['keypoints'][11] = [0., 0., 0.]
    result = feed(detector, 2., missing)
    assert result['state'] == 'unobservable' and not result['observed']
    assert result['event_id'] is not None  # Historical event, not a current visible red box.


def test_short_missing_interval_preserves_reference_without_counting_gap():
    detector = LiveFallDetector()
    standing_reference(detector)
    feed(detector, .75, 'low')
    assert feed(detector, 1., 'low')['candidate_duration_s'] == .25
    missing = feed(detector, 1.25, None, epoch=None)
    assert missing['state'] == 'unobservable'
    resumed = feed(detector, 1.5, 'low')
    assert resumed['metrics']['had_upright'] and resumed['candidate_duration_s'] == .25
    assert resumed['event_id'] is None
    for t in (1.75, 2.):
        assert feed(detector, t, 'low')['state'] == 'suspected'
    assert feed(detector, 2.25, 'low')['state'] == 'confirmed'


@pytest.mark.parametrize('missing_frame', [True, False])
def test_long_gap_discards_reference_and_cannot_confirm(missing_frame):
    detector = LiveFallDetector()
    standing_reference(detector)
    feed(detector, .75, 'low')
    feed(detector, 1., 'low')
    if missing_frame:
        feed(detector, 1.6, None)
    for t in (1.75, 2., 2.25, 2.5, 2.75, 3.):
        result = feed(detector, t, 'low')
        assert result['state'] == 'suspected' and not result['metrics']['had_upright']
        assert result['event_id'] is None


@pytest.mark.parametrize('stable', [False, None])
def test_camera_motion_and_unknown_stability_clear_reference(stable):
    detector = LiveFallDetector()
    standing_reference(detector)
    feed(detector, .75, 'low')
    out = feed(detector, 1., 'low', camera_stable=stable, quality_reason='相机质量测试')
    assert out['state'] == 'unobservable' and out['reason'] == '相机质量测试'
    for t in (1.25, 1.5, 1.75, 2., 2.25, 2.5):
        result = feed(detector, t, 'low')
    assert result['state'] == 'suspected' and result['event_id'] is None


def test_target_change_does_not_inherit_candidate_or_red_box():
    detector = LiveFallDetector()
    old = confirmed(detector)
    new = feed(detector, 2., 'standing', epoch='person-b')
    assert new['state'] == 'normal' and not new['current_target_confirmed']
    assert not new['metrics']['had_upright']
    assert new['event_id'] == old['event_id'] and new['event_target_epoch'] == 'person-a'
    assert new['target_epoch'] == 'person-b'
    assert feed(detector, 2.25, 'low', epoch='person-b')['state'] == 'suspected'


def test_candidate_recovers_only_after_sustained_upright():
    detector = LiveFallDetector()
    standing_reference(detector)
    feed(detector, .75, 'low')
    for t in (1., 1.25, 1.5, 1.75):
        assert feed(detector, t)['state'] == 'suspected'
    result = feed(detector, 2.)
    assert result['state'] == 'normal' and result['candidate_since_s'] is None
    assert detector.export()['statistics']['recoveries'] == 1


def test_confirmed_event_survives_missing_frames_and_upright_recovery():
    detector = LiveFallDetector()
    event = confirmed(detector)
    out = feed(detector, 2., None)
    assert out['state'] == 'unobservable' and out['event_id'] == event['event_id']
    for i in range(1, 9):
        out = feed(detector, 2. + i * .25)
    assert out['state'] == 'confirmed' and out['event_id'] == event['event_id']
    assert len(detector.export()['confirmed_events']) == 1


@pytest.mark.parametrize('bad_t', [.5, .25, float('nan'), float('inf'), -1])
def test_repeated_reversed_and_invalid_times_rejected_without_mutation(bad_t):
    detector = LiveFallDetector()
    standing_reference(detector)
    before = detector.export()
    with pytest.raises(ValueError, match='源时间'):
        feed(detector, bad_t)
    assert detector.export() == before


def test_export_is_isolated_from_internal_event_state():
    detector = LiveFallDetector()
    confirmed(detector)
    exported = detector.export()
    exported['confirmed_events'][0]['target_epoch'] = 'corrupted'
    assert detector.export()['confirmed_events'][0]['target_epoch'] == 'person-a'


def test_valid_camera_reason_does_not_hide_a_missing_target():
    detector = LiveFallDetector()
    result = feed(detector, 0., None, epoch=None, quality_reason='可见背景稳定')
    assert result['state'] == 'unobservable'
    assert '目标不可见' in result['reason'] and '背景稳定' not in result['reason']


def test_changed_image_dimensions_clear_old_reference():
    detector = LiveFallDetector()
    standing_reference(detector)
    feed(detector, .75, 'low')
    result = detector.update(person('low'), 1., 800, 600, 'person-a')
    assert result['state'] == 'suspected' and not result['metrics']['had_upright']
    assert any(e.get('reason') == 'image_size_changed' for e in result['transitions'])


@pytest.mark.parametrize('joint_value', [[float('nan'), 10, .9], [-1, 10, .9], [10, 600, .9]])
def test_nonfinite_and_out_of_frame_torso_points_are_unobservable(joint_value):
    detector = LiveFallDetector()
    standing_reference(detector)
    p = person('low')
    p['keypoints'][5] = joint_value
    result = feed(detector, .75, p)
    assert result['state'] == 'unobservable' and result['metrics'] == {}
