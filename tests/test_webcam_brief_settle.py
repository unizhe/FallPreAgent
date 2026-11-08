"""Short stops must be causal and must not weaken motion/identity protection."""
import copy
import numpy as np
import pytest
from fallpre.weekly.webcam_reconcile import ReconcileActions, brief_settle
from test_webcam_static_recovery import sample


def rows(candidate='standing', dt=.133):
    _, f, p = sample(standing=candidate=='standing')
    return [dict(t=i*dt, candidate=candidate, points=np.array(p['kpts']), **f) for i in range(3)]


@pytest.mark.parametrize('candidate',['standing','seated'])
@pytest.mark.parametrize('mode',['one_frame','too_short','gap','missing_chain','changing_side','motion','step'])
def test_short_proof_rejects_missing_or_moving_evidence(candidate,mode):
    rr=rows(candidate)
    if mode=='one_frame':rr=rr[-1:]
    elif mode=='too_short':rr=rows(candidate,.04)
    elif mode=='gap':rr[-1]['t']+=.4
    elif mode=='missing_chain':rr[1]['points'][[15,16],2]=0
    elif mode=='changing_side':
        rr[0]['points'][[5,11,13,15],2]=0
        rr[-1]['points'][[6,12,14,16],2]=0
    elif mode=='motion':
        for i,r in enumerate(rr):r['points'][[5,6,11,12],1]+=i*7
    elif mode=='step':
        for j in [11,13,15]:rr[-1]['points'][j,0]+=15
    assert not brief_settle(rr,candidate,True)


def test_short_seat_needs_independent_descent_or_verified_support():
    rr=rows('seated')
    assert not brief_settle(rr,'seated',False)
    assert brief_settle(rr,'seated',True)


def test_upright_shift_with_planted_feet_is_not_a_step():
    rr=rows()
    for i,r in enumerate(rr):r['points'][[5,6,11,12],0]+=i*10
    assert brief_settle(rr,'standing')
    for i,r in enumerate(rr):r['points'][[13,14,15,16],0]+=i*10
    assert not brief_settle(rr,'standing')


@pytest.mark.parametrize('mode',['rise_then_stop','still_startup','continuing_rise','missing'])
def test_short_stop_uses_last_motion_endpoint_without_restarting_window(mode):
    stream=ReconcileActions();out=None
    for i in range(5):
        pred,f,p=sample()
        if mode!='still_startup':
            dy=max(0,2-i)*15 if mode!='continuing_rise' else (4-i)*15
            for j in [5,6,11,12]:p['kpts'][j][1]+=dy
            f.update(knee=.98-max(0,2-i)*.08,hip_angle=.98-max(0,2-i)*.08)
        if mode=='missing' and i==3:
            stream.update(i*.133,pred,None,None)
            continue
        out=stream.update(i*.133,pred,f,p)
        if mode=='still_startup' and i==2:
            assert out['code']!='static_recovered'
    assert (stream.static_recovery.details.get('brief_settle_confirmed',False))==(mode=='rise_then_stop')
    if mode=='rise_then_stop':
        assert out['prediction']['phase']=='standing'
        # A new step immediately exits the newly confirmed state.
        for j in [11,13,15]:p['kpts'][j][0]+=15
        out=stream.update(5*.133,pred,f,p)
        assert out['code']=='observed_step'


def test_quick_seat_matures_for_rise_reference_without_moving_anchor():
    stream=ReconcileActions();first=None
    for i in range(9):
        step=min(i,4);pred,f,p=sample()
        pred.update(label='background',phase='unknown',phase_scores={'other':.9})
        f.update(knee=.98-step*.025,hip_angle=.98-step*.025,hip_speed=-.2 if i<5 else 0)
        for j in [5,6,11,12]:p['kpts'][j][1]+=step*8
        out=stream.update(i*.133,pred,f,p)
        if stream.static_recovery.details.get('brief_settle_confirmed'):
            first=copy.deepcopy(stream.static_recovery.seat)
            assert out['evidence']['samples']==3
    assert first is not None
    assert out['evidence']['duration_s']>=.35 and out['evidence']['samples']>=4
    assert np.array_equal(first['points'],stream.static_recovery.seat['points'])


def test_short_seat_requires_one_fixed_foot_even_with_small_ankle_drift():
    rr=rows('seated')
    for i,r in enumerate(rr):r['points'][16,0]+=i*4.5
    assert brief_settle(rr,'seated',True)
    for i,r in enumerate(rr):r['points'][15,0]+=i*3
    assert not brief_settle(rr,'seated',True)
