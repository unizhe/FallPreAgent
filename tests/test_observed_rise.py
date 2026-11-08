"""Causal onset, negative motion controls and evidence-boundary regressions."""
import copy

import numpy as np
import pytest

from fallpre.weekly.observed_rise import ObservedRise


def pose(up=0, mode='lift', missing=()):
    points = np.zeros((17, 3))
    for j, x, y in [(5,85,100), (6,115,100), (11,88,160), (12,112,160),
                    (13,138,165), (14,162,165), (15,138,220), (16,162,220)]:
        dy = up if (mode == 'translation' or mode == 'lift' and j in (5,6,11,12)
                    or mode == 'lean' and j in (5,6)) else 0
        points[j] = [x, y-dy, .9 if j not in missing else 0]
    return dict(kpts=points.tolist(), box=[50,50,200,250], det=.9)


def prediction(label='background', phase='seated'):
    return dict(label=label, phase=phase, phase_scores={'seated': .8, 'standing': .1, 'other': .1})


def anchored(hz):
    rise = ObservedRise()
    times = np.arange(0, 1., 1/hz)
    for t in times:
        out = rise.update(float(t), 1, pose(), prediction())
    assert rise.anchor is not None and not out['rising']
    return rise, float(times[-1]+1/hz)


@pytest.mark.parametrize('hz', [7.5, 10, 30])
@pytest.mark.parametrize('phase', ['seated', 'unknown', 'other'])
def test_early_lift_uses_observed_seat_and_current_motion_not_stale_phase(hz, phase):
    rise, start = anchored(hz)
    outs = []
    for t in np.arange(start, start+.55, 1/hz):
        # 0.6 torso lengths/s, stationary feet, continuous leg opening.
        out = rise.update(float(t), 1, pose((t-start)*36), prediction(phase=phase))
        outs.append((t, out))
    recognized = [(t, o) for t, o in outs if o['rising']]
    assert recognized and recognized[0][0]-start <= .28+1e-8
    assert all(o['veto_seated'] and o['positive_steps'] >= 2 for _, o in recognized)


@pytest.mark.parametrize('hz', [7.5, 10, 30])
@pytest.mark.parametrize('mode', ['translation', 'lean'])
def test_rigid_translation_or_seated_lean_never_create_rise(hz, mode):
    rise, start = anchored(hz)
    for t in np.arange(start, start+.8, 1/hz):
        assert not rise.update(float(t), 1, pose((t-start)*36, mode), prediction())['rising']


@pytest.mark.parametrize('hz', [7.5, 10, 30])
@pytest.mark.parametrize('values', [[0,15,15,15,15,15], [0,15,0,0,15,0],
                                  [0,15,15.7,15.3,15.8,15.4]])
def test_one_jump_or_separate_spikes_are_not_continuous_rise(hz, values):
    rise, start = anchored(hz)
    for i, up in enumerate(values):
        out = rise.update(start+i/hz, 1, pose(up), prediction())
        assert not out['rising']


@pytest.mark.parametrize('hz', [7.5, 10, 30])
def test_partial_rise_pause_never_reanchors_or_repeats_rise(hz):
    rise, start = anchored(hz)
    original = rise.anchor['points'].copy()
    for t in np.arange(start, start+.6, 1/hz):
        rise.update(float(t), 1, pose((t-start)*36), prediction())
    assert rise.departed
    last = float(t)
    up = (last-start)*36
    for t in np.arange(last+1/hz, last+2., 1/hz):
        out = rise.update(float(t), 1, pose(up), prediction())
        assert not out['rising'] and out['veto_seated'] and out['stop_rise']
        assert np.array_equal(original, rise.anchor['points'])


def test_long_confirmed_seat_refreshes_age_without_moving_fixed_coordinates():
    rise, start = anchored(10)
    original = rise.anchor['points'].copy()
    for t in np.arange(start, 20., .1):
        rise.update(float(t), 1, pose(), prediction())
    assert rise.anchor is not None
    assert np.array_equal(original, rise.anchor['points'])
    outs = [rise.update(float(20+i*.1), 1, pose(i*4), prediction()) for i in range(6)]
    assert any(o['rising'] for o in outs)


@pytest.mark.parametrize('label', ['walking', 'bending', 'falling'])
def test_protected_raw_actions_never_emit_rise(label):
    rise, start = anchored(10)
    for i in range(5):
        out = rise.update(start+i*.1, 1, pose(i*5), prediction(label))
        assert not out['rising']
        if label != 'falling' or i >= 2:
            assert rise.anchor is None


def test_single_conflicting_vote_keeps_fixed_seat_but_contributes_no_rise_motion():
    rise, start = anchored(10)
    anchor = rise.anchor['points'].copy()
    out = rise.update(start, 1, pose(), prediction('falling'))
    assert out['code'] == 'action_conflict_held' and not out['rising']
    assert not rise.history
    outputs = [rise.update(start+(i+1)*.1, 1, pose(i*4), prediction()) for i in range(5)]
    assert any(o['rising'] for o in outputs)
    assert np.array_equal(anchor, rise.anchor['points'])


def test_sustained_conflicting_votes_expire_even_with_stationary_geometry():
    rise, start = anchored(10)
    for i in range(5):
        out = rise.update(start+i*.1, 1, pose(), prediction('falling'))
        assert not out['rising']
    assert rise.anchor is None


@pytest.mark.parametrize('free_foot', [15,16])
def test_one_missing_observation_and_weight_transfer_still_need_two_real_rise_steps(free_foot):
    rise, start = anchored(10)
    for dt,up,shift in [(0.,0.,0.),(.26,10.,6.),(.39,16.,9.)]:
        p=pose(up)
        p['kpts'][free_foot][0] += shift
        out=rise.update(start+dt,1,p,prediction())
        if dt < .39:
            assert not out['rising']
    assert out['rising'] and out['positive_steps']==2
    assert out['window_extended'] and out['support_foot_count']==1


def test_missing_sample_does_not_turn_one_jump_and_plateau_into_two_steps():
    rise,start=anchored(10)
    for dt,up in [(0.,0.),(.26,16.),(.39,16.3),(.49,16.1)]:
        assert not rise.update(start+dt,1,pose(up),prediction())['rising']


def test_weight_transfer_requires_one_fixed_foot_throughout_the_window():
    rise,start=anchored(10)
    for i in range(7):
        p=pose(i*5)
        for joint in (15,16):p['kpts'][joint][0]+=i*7
        assert not rise.update(start+i*.1,1,p,prediction())['rising']


def test_confirmed_rise_can_finish_despite_walking_tail_but_cannot_extend_after_stop():
    rise,start=anchored(10)
    for i in range(5):out=rise.update(start+i*.1,1,pose(i*5),prediction())
    assert out['rising']
    out=rise.update(start+.5,1,pose(25),prediction('walking',phase='rising'))
    assert out['rising'] and out['walking_tail']
    for i in range(6,11):
        out=rise.update(start+i*.1,1,pose(25),prediction('walking'))
        assert not out['rising']
    assert rise.anchor is None


def test_shared_current_seat_can_arm_after_walking_without_a_descent_vote():
    rise = ObservedRise()
    for i in range(5):
        rise.update(i*.1, 1, pose(), prediction('walking'))
    verified = dict(phase='seated', stable=True, support='verified_seat',
                    duration_s=.4, samples=4, seat_time_s=.8, origin_time_s=None)
    for i in range(5, 10):
        rise.update(i*.1, 1, pose(), weak_seated(), verified if i >= 8 else None)
    assert rise.anchor['kind'] == 'shared_verified_seat'
    assert any(rise.update(1.+i*.1, 1, pose(i*4), weak_seated())['rising'] for i in range(5))


def test_shared_seat_without_new_upright_cannot_reanchor_partial_rise():
    rise, start = anchored(10)
    for i in range(5):
        rise.update(start+i*.1, 1, pose(i*4), prediction())
    assert rise.departed
    original = rise.anchor['points'].copy()
    old_seat_time = rise.last_seat_t
    rise.update(start+.5, 1, pose(16), prediction('walking'))
    verified = dict(phase='seated', stable=True, support='verified_seat',
                    duration_s=.4, samples=4, seat_time_s=start+.6)
    for i in range(6, 15):
        out = rise.update(start+i*.1, 1, pose(16), prediction(), verified)
        assert not out['rising'] and rise.last_seat_t == old_seat_time
        if rise.anchor is not None:
            assert np.array_equal(rise.anchor['points'], original) and out['veto_seated']


@pytest.mark.parametrize('boundary', ['gap', 'epoch', 'missing', 'side_switch'])
def test_missing_or_changed_evidence_cannot_borrow_old_seat(boundary):
    rise, start = anchored(10)
    if boundary == 'side_switch':
        # Each individual image has a complete side, but no common chain over
        # the onset window. Interleaving sides must not fabricate an increment.
        for i in range(5):
            missing = (5,11,13,15) if i%2 else (6,12,14,16)
            out = rise.update(start+i*.1, 1, pose(i*5, missing=missing), prediction())
            assert not out['rising']
        return
    args = (start+.6 if boundary == 'gap' else start,
            2 if boundary == 'epoch' else 1,
            None if boundary == 'missing' else pose(15), prediction())
    assert not rise.update(*args)['rising'] and rise.anchor is None


def test_inputs_are_immutable_and_unanchored_ascent_does_not_create_rise():
    rise = ObservedRise()
    for i in range(10):
        person, pred = pose(i*4), prediction(phase='rising')
        original = copy.deepcopy((person, pred))
        out = rise.update(i*.1, 1, person, pred)
        assert not out['rising'] and (person, pred) == original


def test_newly_visible_moving_side_cannot_overrule_stationary_common_chain():
    rise, start = anchored(10)
    for i in range(5):
        p = pose(missing=(6,12,14,16) if i == 0 else ())
        for j in (6, 12):
            p['kpts'][j][1] -= i*20
        out = rise.update(start+i*.1, 1, p, prediction())
        assert not out['rising']


def transition(fraction):
    """Side-view pelvis moves over planted feet during a real stand-up."""
    seat = pose()
    stand = copy.deepcopy(seat)
    for j in (5, 6, 11, 12):
        stand['kpts'][j][0] += 50
        stand['kpts'][j][1] -= 30
    for j in (13, 14):
        stand['kpts'][j][1] = 175
    seat['kpts'] = (np.asarray(seat['kpts'])*(1-fraction)
                    + np.asarray(stand['kpts'])*fraction).tolist()
    return seat


def test_pelvis_shifts_over_planted_feet_and_completion_stops_standing_stretch():
    rise, t = anchored(10)
    outs = []
    for f in [.25, .5, .75, 1.]:
        outs.append(rise.update(t, 1, transition(f), prediction()))
        t += .1
    assert any(o['rising'] for o in outs)
    for _ in range(10):
        out = rise.update(t, 1, transition(1.), prediction(phase='standing'))
        t += .1
    assert out['code'] == 'standing_complete' and out['stop_rise']
    for up in [2, 4, 6, 8]:
        p = transition(1.)
        for j in (5, 6, 11, 12):
            p['kpts'][j][1] -= up
        out = rise.update(t, 1, p, prediction(phase='standing'))
        t += .1
        assert not out['rising'] and out['stop_rise']
    # A genuine new descent and stable bottom may rearm the next cycle.
    for f in [.75, .5, .25, 0.]:
        rise.update(t, 1, transition(f), prediction())
        t += .1
    for _ in range(10):
        rise.update(t, 1, transition(0.), prediction())
        t += .1
    outs = []
    for f in [.25, .5, .75, 1.]:
        outs.append(rise.update(t, 1, transition(f), prediction()))
        t += .1
    assert any(o['rising'] for o in outs)


def test_foot_reposition_while_seated_requires_planted_current_feet_before_rise():
    rise, t = anchored(10)
    for i in range(10):
        p = pose()
        for j in (15,16):
            p['kpts'][j][0] += 20
        rise.update(t, 1, p, prediction())
        t += .1
    outs = []
    for i in range(6):
        p = pose(i*4)
        for j in (15,16):
            p['kpts'][j][0] += 20
        outs.append(rise.update(t, 1, p, prediction()))
        t += .1
    assert any(o['rising'] for o in outs)


def test_moving_current_feet_cannot_support_new_rise():
    rise, t = anchored(10)
    for i in range(6):
        p = pose(i*4)
        for j in (15,16):
            p['kpts'][j][0] += i*8
        assert not rise.update(t, 1, p, prediction())['rising']
        t += .1


def test_completed_standing_retires_context_even_if_early_rise_was_missed():
    rise, t = anchored(10)
    out = rise.update(t, 1, transition(1.), prediction())
    assert not out['rising']  # one jump is deliberately not onset evidence
    for _ in range(10):
        t += .1
        out = rise.update(t, 1, transition(1.), prediction(phase='standing'))
    assert out['standing_complete'] and out['stop_rise']
    for up in [2,4,6,8]:
        p = transition(1.)
        for j in (5,6,11,12):
            p['kpts'][j][1] -= up
        t += .1
        out = rise.update(t, 1, p, prediction(phase='standing'))
        assert not out['rising'] and out['stop_rise']


def recovery(origin=0.):
    return dict(phase='seated', stable=True, support='observed_descent',
                duration_s=.4, samples=5, origin_time_s=origin)


def weak_seated():
    return dict(label='background', phase='unknown',
                phase_scores={'seated': .35, 'other': .3, 'standing': .2, 'rising': .15})


def test_verified_descent_needs_real_origin_and_cannot_rearm_partial_after_walking():
    rise = ObservedRise()
    for i in range(10):
        out = rise.update(1+i*.1, 1, pose(), weak_seated(), recovery())
    assert rise.anchor is not None and rise.anchor['kind'] == 'verified_descent'
    old_reference_time = rise.last_seat_t
    for t, up in [(2., 1.95), (2.1, 3.9)]:
        out = rise.update(t, 1, pose(up), weak_seated(), recovery())
    assert out['rising']
    # The old recovery's .08-torso departure threshold can still retain a .065
    # lift. Its origin identifies the old descent, not a genuinely new seat.
    for i in range(8):
        out = rise.update(2.2+i*.1, 1, pose(3.9), prediction('walking'), recovery())
    for i in range(10):
        out = rise.update(3.+i*.1, 1, pose(3.9), weak_seated(), recovery())
        assert not out['rising'] and rise.anchor is None
        assert rise.last_seat_t == old_reference_time


@pytest.mark.parametrize('proof', [{}, recovery(None), recovery(-1),
                                  dict(recovery(), stable=False),
                                  dict(recovery(), support='seated_emission')])
def test_unverified_static_claim_cannot_create_a_seat_reference(proof):
    rise = ObservedRise()
    for i in range(10):
        rise.update(1+i*.1, 1, pose(), weak_seated(), proof)
    assert rise.anchor is None


@pytest.mark.parametrize('support', ['observed_descent', 'verified_seat'])
def test_new_verified_descent_rearms_completed_cycle_without_reusing_old_origin(support):
    rise, start = anchored(10)
    rise.completed = rise.departed = True
    previous = rise.last_seat_t
    # A separate real-observation static verifier established a new descent
    # and stable seat; its origin is newer than the completed cycle's seat.
    verified = dict(phase='seated', stable=True, support=support,
                    samples=4, duration_s=.4, origin_time_s=start)
    for i in range(4):
        out = rise.update(start+i*.1, 1, pose(), prediction(),
                          verified if i == 3 else None)
    assert not rise.completed and not out['veto_seated']
    assert rise.last_seat_t > previous
    confirmed = rise.last_seat_t
    # Retaining the same lineage must not move or refresh the spatial anchor.
    rise.completed = rise.departed = True
    out = rise.update(start+.4, 1, pose(12), prediction(), verified)
    assert rise.last_seat_t == confirmed and out['code'] == 'standing_complete'
