"""Static recovery from stale action state, with motion and identity boundaries."""
import copy

import pytest

from fallpre.weekly.webcam_actions import decision_details
from fallpre.weekly.webcam_reconcile import ReconcileActions
from test_webcam_reconcile import observation


def sample(standing=True, label='background', phase='other'):
    prediction, features, person = observation(0, 'standing' if standing else 'seated', label)
    prediction['phase'] = phase
    prediction['phase_scores'] = {'other': .9, 'seated': .05, 'standing': .05}
    return prediction, features, person


@pytest.mark.parametrize('label', ['background', 'unknown', 'standing_up', 'sitting_down', 'walking'])
@pytest.mark.parametrize('dt', [1 / 30, .133, .20])
def test_sustained_current_standing_repairs_stale_phase_and_action(label, dt):
    stream = ReconcileActions()
    prediction, features, person = sample(label=label)
    original = copy.deepcopy((prediction, features, person))
    outputs = [stream.update(i * dt, prediction, features, person) for i in range(round(1. / dt) + 1)]
    assert outputs[-1]['prediction']['label'] == 'background'
    assert outputs[-1]['prediction']['phase'] == 'standing'
    assert outputs[-1]['code'] == 'static_recovered'
    assert decision_details(outputs[-1]['prediction'], features)[0] == 'standing'
    first = next(i * dt for i, out in enumerate(outputs) if out['code'] == 'static_recovered')
    assert first >= .35 - 1e-8
    assert (prediction, features, person) == original


@pytest.mark.parametrize('mode', ['translation', 'arm_motion', 'alternating_side', 'bent', 'instantaneous_speed'])
def test_walking_needs_actual_full_chain_stillness_before_correction(mode):
    stream = ReconcileActions()
    for i in range(15):
        prediction, features, person = sample(label='walking')
        if mode == 'translation':
            for point in person['kpts']:
                point[0] += i * 4
        elif mode == 'arm_motion':
            person['kpts'][9][0] += i * 4
        elif mode == 'alternating_side':
            for joint in ((5, 11, 13, 15) if i % 2 else (6, 12, 14, 16)):
                person['kpts'][joint][2] = 0
        elif mode == 'bent':
            features.update(knee=.70, hip_angle=.65)
        else:
            features['hip_speed'] = .11
            for point in person['kpts']:point[0] += i*4
        out = stream.update(i * .133, prediction, features, person)
        if mode != 'arm_motion' or i*.133 < .35:
            assert out['prediction']['label'] == 'walking'
            assert out['code'] == 'protected_action'
        else:
            assert out['prediction']['label'] == 'background'
            assert out['prediction']['phase'] == 'standing'


def test_falling_is_never_replaced_even_by_long_stable_standing_geometry():
    stream = ReconcileActions()
    prediction, features, person = sample(label='falling')
    for i in range(30):
        out = stream.update(i * .133, prediction, features, person)
        assert not out['changed'] and out['prediction']['label'] == 'falling'
        assert not stream.static_recovery.history


def test_verified_seat_survives_one_raw_conflict_without_output_on_that_frame():
    stream = ReconcileActions()
    pred, features, person = sample(standing=False)
    pred['phase_scores'] = {'seated': .8, 'other': .2}
    for i in range(6):
        stream.update(i*.1, pred, features, person)
    seat = stream.static_recovery.seat
    assert seat is not None
    pred['label'] = 'falling'
    out = stream.update(.6, pred, features, person)
    assert out['prediction']['label'] == 'falling' and not out['changed']
    assert stream.static_recovery.details['reference_retained']
    pred['label'] = 'background'
    out = stream.update(.7, pred, features, person)
    assert out['prediction']['phase'] == 'seated'
    assert stream.static_recovery.seat is seat


def test_fixed_seat_uses_observed_motion_instead_of_stale_median_speed():
    stream = ReconcileActions()
    pred, features, person = sample(standing=False)
    pred['phase_scores'] = {'seated': .8, 'other': .2}
    for i in range(6):
        stream.update(i*.1, pred, features, person)
    features['hip_speed'] = .2
    for i in range(6, 12):
        out = stream.update(i*.1, pred, features, person)
        assert out['prediction']['phase'] == 'seated'
    # Real sustained displacement still exits the seat despite its score.
    for j in (5, 6, 11, 12):
        person['kpts'][j][1] -= 20
    out = stream.update(1.2, pred, features, person)
    assert not out['evidence'].get('stable')


def test_standing_continuity_cannot_follow_many_small_walking_steps():
    stream = ReconcileActions()
    pred, features, person = sample(label='walking')
    for i in range(8):
        out = stream.update(i*.1, pred, features, person)
    assert out['evidence']['stable']
    rejected = False
    for i in range(8, 25):
        for point in person['kpts']:
            point[0] += 3
        out = stream.update(i*.1, pred, features, person)
        rejected |= not out['evidence'].get('stable', False)
    assert rejected and out['prediction']['label'] == 'walking'


def test_continuity_refreshes_real_window_instead_of_leaving_sparse_old_samples():
    stream = ReconcileActions()
    pred, features, person = sample(standing=False)
    pred['phase_scores'] = {'seated': .8, 'other': .2}
    for i in range(50):
        stream.update(i*.1, pred, features, person)
    window = stream.static_recovery.stable
    assert window[-1]['t'] == pytest.approx(4.9)
    assert window[-1]['t']-window[0]['t'] <= .45
    assert all(b['t']-a['t'] <= .11 for a,b in zip(window,list(window)[1:]))


def test_standing_body_does_not_inherit_false_vertical_motion_from_one_ankle():
    stream=ReconcileActions()
    for i in range(12):
        pred,features,person=sample(label='walking')
        # The torso and one complete support chain stay in place; an ankle
        # estimate drifts vertically while projected knee angle changes a little.
        person['kpts'][16][1]-=(i%4)*4
        features['knee']=[.98,.96,.94,.93][i%4]
        out=stream.update(i*.133,pred,features,person)
    assert out['prediction']['phase']=='standing'
    assert not stream.static_recovery.details['body_motion']


@pytest.mark.parametrize('mode',['moving_feet','translation','descent'])
def test_fast_walking_exit_does_not_accept_actual_motion(mode):
    stream=ReconcileActions()
    for i in range(4):
        pred,features,person=sample(label='walking')
        if mode=='translation':
            for point in person['kpts']:point[0]+=i*5
        elif mode=='moving_feet':
            for joint in (15,16):person['kpts'][joint][0]+=i*5
        else:
            for joint in (5,6,11,12):person['kpts'][joint][1]+=i*5
            features['knee']=.99-i*.025
        out=stream.update(i*.133,pred,features,person)
        assert out['prediction']['label']=='walking'
        assert not stream.static_recovery.details.get('walking_stop_confirmed')


@pytest.mark.parametrize('boundary', ['gap', 'missing', 'movement'])
def test_static_recovery_cannot_cross_unreliable_or_moving_interval(boundary):
    stream = ReconcileActions()
    prediction, features, person = sample(label='walking')
    for i in range(7):
        out = stream.update(i * .133, prediction, features, person)
    assert out['code'] == 'static_recovered'
    if boundary == 'gap':
        out = stream.update(2., prediction, features, person)
    elif boundary == 'missing':
        stream.update(.931, prediction, None, None)
        out = stream.update(1.064, prediction, features, person)
    else:
        features['hip_speed'] = .2
        for j in (5,6,11,12):person['kpts'][j][1] -= 20
        out = stream.update(.931, prediction, features, person)
    assert out['code'] != 'static_recovered'
    assert out['prediction']['label'] == 'walking'


def test_still_bent_body_without_seat_or_descent_evidence_stays_unknown():
    stream = ReconcileActions()
    prediction, features, person = sample(standing=False)
    for i in range(30):
        out = stream.update(i * .133, prediction, features, person)
        assert out['code'] == 'unsupported_phase'
        assert decision_details(out['prediction'], features)[0] == 'unknown'


def test_stable_seat_can_recover_phase_with_independent_seated_emission():
    stream = ReconcileActions()
    prediction, features, person = sample(standing=False)
    prediction['phase_scores'] = {'seated': .8, 'other': .2}
    for i in range(6):
        out = stream.update(i * .133, prediction, features, person)
    assert out['code'] == 'static_recovered'
    assert out['prediction']['phase'] == 'seated'
    assert out['evidence']['support'] in {'seated_emission', 'verified_seat'}


@pytest.mark.parametrize('translate_feet', [False, True])
def test_descent_support_requires_articulation_relative_to_observed_feet(translate_feet):
    stream = ReconcileActions()
    for i in range(12):
        step = min(i, 4)
        prediction, features, person = sample(standing=False)
        if i < 5:
            prediction['label'] = 'sitting_down'
            prediction['phase'] = 'descending'
        features.update(knee=.95 - step * .06, hip_angle=.95 - step * .065,
                        hip_speed=-.3 if i < 5 else 0)
        for joint in (5, 6, 11, 12):
            person['kpts'][joint][1] += step * 5
        if translate_feet:
            for joint in (15, 16):
                person['kpts'][joint][1] += step * 5
        out = stream.update(i * .133, prediction, features, person)
    if translate_feet:
        assert out['code'] != 'static_recovered'
    else:
        assert out['code'] == 'static_recovered'
        assert out['prediction']['phase'] == 'seated'
        assert out['evidence']['support'] in {'observed_descent','verified_seat'}


def test_unsupported_static_phase_is_not_mislabeled_geometry_failure():
    prediction, features, _ = sample()
    assert decision_details(prediction, features)[2] == 'static_phase_unsupported'
    prediction['phase'] = 'standing'
    features['knee'] = .6
    assert decision_details(prediction, features)[2] == 'static_geometry'


@pytest.mark.parametrize('missing_evidence', [None, 'upright_origin', 'seated_emission', 'foot_reference'])
def test_foreshortened_seat_needs_observed_descent_and_model_support(missing_evidence):
    stream = ReconcileActions()
    for i in range(12):
        step = min(i, 4)
        prediction, features, person = sample()
        if i < 5:
            prediction.update(label='sitting_down', phase='descending')
        elif missing_evidence != 'seated_emission':
            prediction['phase_scores'] = {'seated': .4, 'descending': .3, 'other': .3}
        features.update(knee=.98 - step * .025, hip_angle=.98 - step * .025,
                        hip_speed=-.3 if i < 5 else 0)
        if missing_evidence == 'upright_origin':
            features['knee'] = .88
        for joint in (5, 6, 11, 12):
            person['kpts'][joint][1] += step * 5
        if missing_evidence == 'foot_reference':
            for joint in (15, 16):
                person['kpts'][joint][1] += step * 5
        out = stream.update(i * .133, prediction, features, person)
    if missing_evidence in (None,'seated_emission'):
        assert out['prediction']['phase'] == 'seated'
        assert out['evidence']['support'] in {'observed_descent','verified_seat'}
        assert decision_details(out['prediction'], features)[0] == 'seated'
    else:
        assert out['code'] != 'static_recovered'


def test_same_seated_phase_can_release_stale_action_with_descent_proof():
    stream = ReconcileActions()
    for i in range(12):
        step = min(i, 4)
        prediction, features, person = sample(standing=False, label='sitting_down',
                                               phase='descending' if i < 5 else 'seated')
        features.update(knee=.95 - step * .06, hip_angle=.95 - step * .065,
                        hip_speed=-.3 if i < 5 else 0)
        for joint in (5, 6, 11, 12):
            person['kpts'][joint][1] += step * 5
        out = stream.update(i * .133, prediction, features, person)
    assert out['prediction']['label'] == 'background'
    assert out['prediction']['phase'] == 'seated'
    assert out['code'] == 'static_recovered'


def test_one_seated_score_spike_cannot_reuse_unsupported_bending_history():
    stream = ReconcileActions()
    prediction, features, person = sample(standing=False, label='bending')
    for i in range(12):
        if i == 11:
            prediction['phase_scores'] = {'seated': .8, 'other': .2}
        out = stream.update(i * .133, prediction, features, person)
        assert out['code'] != 'static_recovered'
        assert decision_details(out['prediction'], features)[0] == 'unknown'
    # Sustained model support can recover, but its time starts with the first
    # actual supporting score, not with the preceding stable bent geometry.
    for i in range(12, 15):
        out = stream.update(i * .133, prediction, features, person)
        if i < 14:
            assert out['code'] != 'static_recovered'
    assert out['code'] == 'static_recovered'
    assert out['evidence']['duration_s'] == pytest.approx(.399)


def test_partial_rise_cannot_reconstruct_seat_from_predeparture_descent_votes():
    stream = ReconcileActions()
    had_descent = False
    for i in range(20):
        step = min(i, 4)
        prediction, features, person = sample()
        prediction.update(label='sitting_down' if i < 5 else 'background' if i < 9 else 'standing_up',
                          phase='descending' if i < 5 else 'other',
                          phase_scores={'seated': .4, 'other': .3, 'descending': .3})
        features.update(knee=.98 - step * .06, hip_angle=.98 - step * .065,
                        hip_speed=-.3 if i < 5 else .2 if i == 9 else 0)
        for joint in (5, 6, 11, 12):
            person['kpts'][joint][1] += step * 8 if i < 9 else 20
        out = stream.update(i * .1, prediction, features, person)
        if i < 9:
            had_descent |= stream.static_recovery.descent is not None
        else:
            assert stream.static_recovery.descent is None
            assert out['code'] != 'static_recovered'
            assert decision_details(out['prediction'], features)[0] != 'seated'
    assert had_descent
    # Invalidation must not permanently forbid a genuinely new cycle.
    for i in range(20, 33):
        step = min(i - 20, 4)
        prediction, features, person = sample(standing=False, label='sitting_down' if step < 4 else 'background',
                                               phase='descending' if step < 4 else 'other')
        features.update(knee=.98 - step * .06, hip_angle=.98 - step * .065,
                        hip_speed=-.3 if i < 25 else 0)
        for joint in (5, 6, 11, 12):
            person['kpts'][joint][1] += step * 8
        out = stream.update(i * .1, prediction, features, person)
    assert out['code'] == 'static_recovered'
    assert out['prediction']['phase'] == 'seated'


@pytest.mark.parametrize('mode', ['one_ankle', 'both_feet', 'body_shift'])
def test_verified_standing_tolerates_one_ankle_but_not_actual_displacement(mode):
    stream = ReconcileActions()
    pred, features, person = sample(label='walking')
    for i in range(5):
        out = stream.update(i * .133, pred, features, person)
    assert out['prediction']['phase'] == 'standing'
    if mode == 'one_ankle':
        person['kpts'][16][1] -= 12
    elif mode == 'both_feet':
        for j in (15, 16):
            person['kpts'][j][1] -= 12
    else:
        for point in person['kpts']:
            point[0] += 15
    out = stream.update(5 * .133, pred, features, person)
    assert (out['prediction']['phase'] == 'standing') == (mode == 'one_ankle')
    if mode == 'one_ankle':
        assert stream.static_recovery.details['code'] == 'static_continuity'


@pytest.mark.parametrize('mode', ['shoulders_only', 'hip_rise', 'forward_shift'])
def test_verified_seat_allows_arm_raise_but_not_lifting_or_forward_departure(mode):
    stream = ReconcileActions()
    pred, features, person = sample(standing=False)
    pred['phase_scores'] = {'seated': .8, 'other': .2}
    for i in range(6):
        stream.update(i*.133,pred,features,person)
    for joint in (5,6):
        person['kpts'][joint][1] -= 12
    if mode == 'hip_rise':
        for joint in (11,12):person['kpts'][joint][1] -= 7
        features['knee'] += .08
    elif mode == 'forward_shift':
        for joint in (5,6):person['kpts'][joint][0] += 10
    out=stream.update(6*.133,pred,features,person)
    assert (out['code'] == 'static_recovered') == (mode == 'shoulders_only')


@pytest.mark.parametrize('label', ['background','walking'])
@pytest.mark.parametrize('dt', [.05,.133,.2])
def test_planted_opposite_foot_cannot_hide_coherent_step(label,dt):
    stream=ReconcileActions()
    pred,features,person=sample(label=label,phase='standing')
    for i in range(round(1/dt)+1):stream.update(i*dt,pred,features,person)
    t=(round(1/dt)+1)*dt
    # Other leg stays planted while hip, knee and ankle on one side move.
    for j,speed in [(11,70),(13,80),(15,110)]:person['kpts'][j][0]+=speed*dt
    out=stream.update(t,pred,features,person)
    assert out['prediction']['label']=='unknown'
    assert out['code']=='observed_step'
    assert stream.static_recovery.posture is None


@pytest.mark.parametrize('mode',['ankle_only','arms_only'])
def test_step_exit_needs_coherent_hip_knee_foot_not_one_joint(mode):
    stream=ReconcileActions();pred,features,person=sample(label='background',phase='standing')
    for i in range(8):stream.update(i*.133,pred,features,person)
    for j in ([15] if mode=='ankle_only' else [7,9]):person['kpts'][j][0]+=15
    stream.update(8*.133,pred,features,person)
    assert not stream.static_recovery.details.get('step_departure')


@pytest.mark.parametrize('mode',['articulated','rigid','shoulders_only','one_jump','floor'])
def test_geometry_descent_without_model_votes_requires_articulation_and_seat_height(mode):
    stream=ReconcileActions()
    for i in range(14):
        step=min(i,4)
        pred,features,person=sample()
        pred.update(label='background',phase='unknown',phase_scores={'other':.9,'seated':.05})
        features.update(knee=.98-step*.025,hip_angle=.98-step*.025,hip_speed=-.2 if i<5 else 0)
        dy=step*8 if mode!='one_jump' else (32 if i>=4 else 0)
        for j in (5,6,11,12):person['kpts'][j][1]+=dy
        if mode=='rigid':
            for j in (13,14,15,16):person['kpts'][j][1]+=dy
        if mode=='shoulders_only':
            for j in (11,12):person['kpts'][j][1]-=dy
        if mode=='floor':
            for j in (5,6,11,12):person['kpts'][j][1]+=step*22
        out=stream.update(i*.133,pred,features,person)
    assert (out['code']=='static_recovered') == (mode=='articulated')
    if mode=='articulated':assert out['prediction']['phase']=='seated'


def test_step_rejection_retains_real_moving_sample_in_the_window():
    stream=ReconcileActions();pred,features,person=sample(label='background',phase='standing')
    for i in range(8):stream.update(i*.133,pred,features,person)
    for j,dx in [(11,10),(13,12),(15,18)]:person['kpts'][j][0]+=dx
    out=stream.update(8*.133,pred,features,person)
    assert out['code']=='observed_step'
    window=stream.static_recovery.stable
    assert len(window)>=4
    assert window[-1]['t']==pytest.approx(8*.133)
    assert window[-1]['points'][15,0]==person['kpts'][15][0]
