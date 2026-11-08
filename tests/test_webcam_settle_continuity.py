"""Fixed spatial references tolerate local pose jitter, never follow a departure."""
import numpy as np
import pytest
from fallpre.weekly.webcam_reconcile import ReconcileActions, anchored_upright, anchored_seat
from test_webcam_static_recovery import sample


def upright_pair():
    _,f,p=sample();points=np.array(p['kpts'])
    old=dict(t=0.,phase='standing',points=points.copy(),fixed_points=points.copy(),**f)
    current=dict(t=.133,points=points.copy(),**f)
    return old,current


def test_fixed_upright_tolerates_one_ankle_with_bounded_torso_jitter():
    old,cur=upright_pair()
    cur['points'][16,:2]+=[12,-19]
    cur['points'][[11,12],1]-=7
    cur['points'][[5,6],1]-=2
    assert anchored_upright(old,cur)


@pytest.mark.parametrize('mode',['gap','both_feet','vertical_departure','horizontal_departure','bent','chain_loss','accumulated_drift'])
def test_upright_continuity_cannot_cross_departure_or_loss(mode):
    old,cur=upright_pair()
    if mode=='gap':cur['t']=.4
    elif mode=='both_feet':cur['points'][[15,16],0]+=10
    elif mode=='vertical_departure':cur['points'][[5,6,11,12],1]+=14
    elif mode=='horizontal_departure':cur['points'][[5,6,11,12],0]+=25
    elif mode=='bent':cur['knee']=.87
    elif mode=='chain_loss':cur['points'][[15,16],2]=0
    else:
        old['points'][[5,6,11,12],1]+=9
        cur['points'][[5,6,11,12],1]+=12
    assert not anchored_upright(old,cur)


@pytest.mark.parametrize('label',['walking','standing_up','sitting_down','falling'])
def test_anchored_upright_does_not_override_original_dynamic_action(label):
    stream=ReconcileActions();pred,f,p=sample()
    for i in range(6):stream.update(i*.133,pred,f,p)
    pred['label']=label
    for j in [5,6,11,12]:p['kpts'][j][1]+=7
    stream.update(6*.133,pred,f,p)
    assert not stream.static_recovery.details.get('anchored_upright')


def seat_pair():
    _,f,p=sample(standing=False);points=np.array(p['kpts'])
    anchor=dict(confirmed_t=0.,last_seen=0.,planted_descent=True,points=points.copy(),**f)
    cur=dict(t=.133,points=points.copy(),**f)
    return anchor,cur


def test_verified_descent_seat_keeps_fixed_hips_while_feet_adjust():
    a,r=seat_pair();r['points'][15,0]+=15;r['points'][16,1]-=12
    assert anchored_seat(a,r)
    assert np.array_equal(a['points'],seat_pair()[0]['points'])


@pytest.mark.parametrize('mode',['no_descent','old','rise','sideways','knee_extension','missing','large_foot_motion'])
def test_seat_continuity_needs_recent_descent_and_fixed_bent_hips(mode):
    a,r=seat_pair()
    if mode=='no_descent':a['planted_descent']=False
    elif mode=='old':r['t']=1.
    elif mode=='rise':r['points'][[11,12],1]-=9
    elif mode=='sideways':r['points'][[11,12],0]+=12
    elif mode=='knee_extension':r['knee']=.85
    elif mode=='missing':r['points'][[15,16],2]=0
    else:r['points'][15,0]+=35
    assert not anchored_seat(a,r)


@pytest.mark.parametrize('mode',['planted_descent','no_upright','no_descent_votes','feet_still_moving'])
def test_turn_then_sit_measures_descent_after_planting_feet(mode):
    stream=ReconcileActions();seen=False
    for i in range(15):
        step=min(i,7);pred,f,p=sample()
        pred.update(label='sitting_down' if step<7 else 'background',phase='descending' if step<7 else 'other')
        if mode=='no_descent_votes':pred.update(label='background',phase='other')
        f.update(knee=.98-step*.045,hip_angle=.98-step*.05,hip_speed=-.3 if i<8 else 0.)
        if mode=='no_upright':f.update(knee=.85-step*.025,hip_angle=.79-step*.025)
        for j in [5,6,11,12]:p['kpts'][j][1]+=step*8
        for j in [15,16]:p['kpts'][j][0]+=(step if mode=='feet_still_moving' else min(step,3))*12
        stream.update(i*.133,pred,f,p)
        descent=stream.static_recovery.descent
        seen |= bool(descent and descent.get('planted_descent'))
    assert seen==(mode=='planted_descent')
    if mode=='planted_descent':assert stream.static_recovery.seat is not None


def test_fixed_seat_allows_straightening_upper_body_without_reanchoring_hips():
    a,r=seat_pair();r['points'][[5,6],0]+=35
    r['points'][15,0]+=15
    assert anchored_seat(a,r)
    r['points'][[11,12],1]-=5
    assert not anchored_seat(a,r)


def test_seat_reference_does_not_expire_while_current_fixed_hips_remain_visible():
    a,r=seat_pair();a['last_seen']=4.;r['t']=4.133
    assert anchored_seat(a,r)
    r['t']=4.5
    assert not anchored_seat(a,r)
