import copy
import json

import numpy as np
import pytest

from fallpre.weekly.visible_seat import VisibleSeat, visible_body


def body(down=0, missing=()):
    p = np.zeros((17, 3))
    for i, x, y in [(5, 85, 100), (6, 115, 100), (11, 88, 160), (12, 112, 160),
                    (13, 88, 205), (14, 112, 205), (15, 88, 250), (16, 112, 250)]:
        p[i] = [x, y + down, .9]
    for i in missing:
        p[i, 2] = 0
    return dict(kpts=p.tolist())


def pred(label='background', phase='unknown', seat=.8):
    return dict(label=label, phase=phase, phase_scores={'seated': seat, 'standing': 1-seat})


def anchored():
    s = VisibleSeat(.45)
    for i in range(20):
        s.update(i/30, 1, body(), pred(phase='seated'), 'seated', True)
    assert s.anchor is not None
    return s


def test_front_facing_pose_needs_observed_descent_and_stable_end():
    s = VisibleSeat(.45)
    for i in range(45):
        offset = min(i, 15) * 1.2
        label = 'sitting_down' if i < 20 else 'background'
        out = s.update(i/30, 1, body(offset), pred(label), 'unknown', True)
    assert out['accepted'] and s.anchor['kind'] == 'front'
    assert s.seat_probability == .45


def test_high_seated_emission_without_history_cannot_create_front_seat():
    s = VisibleSeat(.45)
    for i in range(60):
        assert not s.update(i/30, 1, body(), pred(seat=.99), 'unknown', True)['accepted']


def test_sit_down_label_without_actual_descent_is_not_enough():
    s = VisibleSeat(.45)
    for i in range(60):
        out = s.update(i/30, 1, body(), pred('sitting_down' if i < 30 else 'background'), 'unknown', True)
    assert not out['accepted']


def test_display_veto_does_not_erase_observed_raw_descent_context():
    s = VisibleSeat(.45)
    for i in range(45):
        offset = min(i, 15) * 1.2
        prediction = pred('unknown' if i < 20 else 'background')
        prediction['source_label'] = 'sitting_down' if i < 20 else 'background'
        out = s.update(i/30, 1, body(offset), prediction, 'unknown', True)
    assert out['accepted'] and s.anchor['kind'] == 'front'


@pytest.mark.parametrize('missing', [(15, 16), (13, 15, 16)])
def test_partial_pose_uses_current_observations_without_filling_joints(missing):
    s = anchored()
    p = body(missing=missing)
    snapshot = copy.deepcopy(p)
    out = s.update(20/30, 1, p, None, 'unknown', False)
    assert out['accepted'] and p == snapshot


@pytest.mark.parametrize('missing', [(13, 14, 15, 16), (5,), (11,)])
def test_missing_torso_or_both_knees_abstains_and_clears_anchor(missing):
    s = anchored()
    assert not s.update(.7, 1, body(missing=missing), None, 'unknown', False)['accepted']
    assert s.anchor is None


def test_partial_from_start_never_infers_seat():
    s = VisibleSeat(.45)
    for i in range(60):
        assert not s.update(i/30, 1, body(missing=(15,16)), None, 'unknown', False)['accepted']


def test_slow_rise_does_not_drag_anchor_upward_or_rearm_during_occlusion():
    s = anchored()
    reference = s.anchor['hip'].copy()
    exited = False
    for i in range(20, 100):
        # Each frame moves very little, but displacement accumulates.
        out = s.update(i/30, 1, body(-(i-19)*.15, missing=(15,16)), None, 'unknown', False)
        if s.anchor is not None:
            assert np.array_equal(reference, s.anchor['hip'])
        if out['veto']:
            exited = True
        if exited:
            assert not out['accepted']
    assert exited and s.departed


def test_single_small_translation_does_not_erase_confirmed_seat():
    s = anchored()
    out = s.update(20/30, 1, body(-3, missing=(15,16)), None, 'unknown', False)
    assert out['accepted'] and not out['rising'] and s.anchor is not None


@pytest.mark.parametrize('change', ['identity', 'gap', 'missing', 'expiry'])
def test_reference_cannot_cross_evidence_boundaries(change):
    s = anchored()
    if change == 'expiry':
        for i in range(20, 345):
            out = s.update(i/30, 1, body(missing=(15,16)), None, 'unknown', False)
    else:
        out = s.update(2 if change == 'gap' else .7, 2 if change == 'identity' else 1,
                       None if change == 'missing' else body(), None, 'unknown', False)
    assert not out['accepted']


def test_standing_bending_walking_and_falling_cannot_use_seated_support():
    for label in ['standing_up', 'bending', 'walking', 'falling']:
        s = anchored()
        out = s.update(.7, 1, body(), pred(label), 'unknown', True)
        assert not out['accepted'] and not out['rising']
        if label != 'standing_up':
            assert s.anchor is None


def test_visible_body_never_estimates_missing_joint_positions():
    assert visible_body(body(missing=(13,14))) is None
    current = visible_body(body(missing=(13,15,16)))
    assert set(current['thighs']) == {14}


def lift(up, missing=(15, 16)):
    p = body(missing=missing)
    # Knees stay observed in place while the torso lifts: relative thigh extension.
    for j in (5, 6, 11, 12):
        p['kpts'][j][1] -= up
    return p


@pytest.mark.parametrize('missing', [(15,16), (13,15,16), ()])
def test_sustained_lift_with_seat_context_emits_rise_and_exits_seat(missing):
    s = anchored()
    outs = [s.update(i/30, 1, lift((i-19)*.8, missing), None, 'unknown', False)
            for i in range(20, 50)]
    assert any(o['rising'] for o in outs)
    assert all(not o['accepted'] for o in outs if o['rising'])
    assert s.departed
    # Once ascent stops, never keep displaying rise just from the old event.
    for i in range(50, 80):
        out = s.update(i/30, 1, lift(24, missing), None, 'unknown', False)
    assert not out['accepted'] and not out['rising']


def test_repeated_raw_rise_votes_require_observed_extension():
    for extension in [False, True]:
        s = VisibleSeat(.45)
        outs = [s.update(i/30, 1, lift(i*.8) if extension else body(-i*.8),
                         pred('standing_up'), 'unknown', True) for i in range(40)]
        assert any(o['rising'] for o in outs) == extension


def test_ascent_without_seat_or_raw_model_context_remains_unknown():
    s = VisibleSeat(.45)
    for i in range(40):
        out = s.update(i/30, 1, lift(i*.8), pred(), 'unknown', True)
        assert not out['rising'] and not out['accepted']


def test_isolated_large_keypoint_spike_abstains_but_never_emits_rise():
    s = anchored()
    for i in range(20, 45):
        out = s.update(i/30, 1, lift(6 if i == 23 else 0), None, 'unknown', False)
        assert not out['rising']
    assert out['accepted']


def test_seated_noise_and_lean_do_not_create_rise():
    for lean in [False, True]:
        s = anchored()
        for i in range(20, 100):
            p = body(missing=(15,16))
            for j in (11,12):
                p['kpts'][j][1] += 1.5*np.sin(i)
            if lean:
                for j in (5,6):
                    p['kpts'][j][1] += min(i-20,20)*.4
            out = s.update(i/30, 1, p, None, 'unknown', False)
            assert out['accepted'] and not out['rising']


def test_shared_image_translation_is_not_rise_even_after_seat():
    s = anchored()
    for i in range(20, 60):
        out = s.update(i/30, 1, body(-(i-19)*.8, missing=(15,16)), None, 'unknown', False)
        assert not out['rising']
    assert not out['accepted']


def test_current_reversal_cannot_emit_rise_from_lagging_window():
    s = anchored()
    for i in range(20, 40):
        s.update(i/30, 1, lift((i-19)*.8), None, 'unknown', False)
    out = s.update(40/30, 1, lift(0), None, 'unknown', False)
    assert not out['rising']


def test_current_full_body_lift_stops_seat_before_long_window_confirms_rise():
    s = anchored()
    prediction = pred(phase='seated')
    prediction['observed_hip_speed'] = .18
    p = lift(3.5, ())
    out = s.update(20/30, 1, p, prediction, 'seated', True)
    assert out['veto'] and not out['accepted'] and not out['rising']
    assert out['motion'] == 'lift_pending'


def test_current_hip_noise_without_joint_extension_does_not_veto_seat():
    s = anchored()
    prediction = pred(phase='seated')
    prediction['observed_hip_speed'] = .18
    out = s.update(20/30, 1, body(-3), prediction, 'seated', True)
    assert out['accepted'] and not out['rising']


def test_visible_geometry_can_change_which_knee_is_available_without_exception():
    s = anchored()
    for i in range(20, 50):
        out = s.update(i/30, 1, lift((i-19)*.8, (13,15,16) if i%3 else (14,15,16)), None, 'unknown', False)
        assert isinstance(out['rising'], bool)
        json.dumps(out, allow_nan=False)


def test_current_thigh_evidence_must_not_be_supplied_by_previous_frames():
    s = anchored()
    for i in range(20, 40):
        s.update(i/30, 1, lift((i-19)*.8), None, 'unknown', False)
    # Torso remains lifted, but current knees shift with it: no current extension.
    out = s.update(40/30, 1, body(-16, missing=(15,16)), None, 'unknown', False)
    assert not out['rising']


def seated_body():
    p = body()
    for knee, ankle in [(13, 15), (14, 16)]:
        p['kpts'][knee][0] += 50
        p['kpts'][knee][1] = 165
        p['kpts'][ankle][0] += 50
        p['kpts'][ankle][1] = 220
    return p


def test_standing_then_complete_stable_seat_recovers_without_descent_label():
    s = VisibleSeat(.45)
    s.update(0, 1, body(), pred(phase='standing', seat=.01), 'standing', True)
    accepted_at = None
    for i in range(1, 40):
        out = s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
        if out['accepted'] and accepted_at is None:
            accepted_at = i/30
        if i <= 12:
            assert not out['accepted']
    assert accepted_at is not None and accepted_at >= .4
    assert not s.departed and s.anchor['kind'] == 'reconfirmed_seat'


@pytest.mark.parametrize('support', ['seated_emission', 'observed_descent'])
def test_independently_verified_static_window_does_not_wait_twice(support):
    s = VisibleSeat(.45)
    s.update(0, 1, body(), pred(phase='standing', seat=.01), 'standing', True)
    for i in range(1, 6):
        prediction = pred(phase='other', seat=.8 if support == 'seated_emission' else .1)
        out = s.update(i * .133, 1, seated_body(), prediction, 'unknown', True)
        assert not out['accepted']
    prediction['phase'] = 'seated'
    prediction['static_recovery'] = dict(phase='seated', stable=True, support=support,
                                         duration_s=.399, samples=4)
    out = s.update(6 * .133, 1, seated_body(), prediction, 'seated', True)
    assert out['accepted'] and out['seat_recovered']


def test_one_walking_false_positive_does_not_permanently_veto_seat():
    s = VisibleSeat(.45)
    for i in range(20):
        s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
    assert s.anchor is not None
    out = s.update(20/30, 1, seated_body(), pred('walking', phase='seated'), 'unknown', True)
    assert out['veto'] and s.departed
    for i in range(21, 51):
        out = s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
        if i <= 32:
            assert not out['accepted']
    assert out['accepted'] and not s.departed


@pytest.mark.parametrize('unsupported', ['partial', 'baseline_unknown', 'low_score', 'wrong_phase', 'rising'])
def test_seat_recovery_requires_complete_independent_stationary_support(unsupported):
    s = VisibleSeat(.45)
    s.update(0, 1, body(), pred(phase='standing'), 'standing', True)
    for i in range(1, 90):
        p = seated_body()
        prediction = pred(phase='seated')
        baseline, full_pose = 'seated', True
        if unsupported == 'partial':
            p['kpts'][15][2] = p['kpts'][16][2] = 0
            full_pose = False
        elif unsupported == 'baseline_unknown':
            baseline = 'unknown'
        elif unsupported == 'low_score':
            prediction = pred(phase='seated', seat=.3)
        elif unsupported == 'wrong_phase':
            prediction = pred(phase='forward_lean')
        elif unsupported == 'rising':
            for joint in [5, 6, 11, 12]:
                p['kpts'][joint][1] -= i * .8
        out = s.update(i/30, 1, p, prediction, baseline, full_pose)
        assert not out['accepted']
    assert s.departed


def test_partial_frame_interrupts_seat_recovery_confirmation():
    s = VisibleSeat(.45)
    s.update(0, 1, body(), pred(phase='standing'), 'standing', True)
    for i in range(1, 20):
        s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
    assert s.seat_recovery is not None
    s.update(20/30, 1, seated_body(), None, 'unknown', False)
    for i in range(21, 33):
        out = s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
        assert not out['accepted']
    for i in range(33, 40):
        out = s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
    assert out['accepted']


@pytest.mark.parametrize('boundary', ['identity', 'gap'])
def test_new_evidence_boundary_drops_pending_seat_recovery(boundary):
    s = VisibleSeat(.45)
    s.update(0, 1, body(), pred(phase='standing'), 'standing', True)
    for i in range(1, 20):
        s.update(i/30, 1, seated_body(), pred(phase='seated'), 'seated', True)
    assert s.seat_recovery is not None
    start, epoch = (1.2, 1) if boundary == 'gap' else (20/30, 2)
    for i in range(12):
        out = s.update(start+i/30, epoch, seated_body(), pred(phase='seated'), 'seated', True)
        assert not out['accepted']
    assert s.seat_recovery is None
