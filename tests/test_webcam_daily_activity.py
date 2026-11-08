import copy
import numpy as np
import pytest
from fallpre.weekly.webcam_daily_activity import DailyActivity
from fallpre.weekly.observed_rise import observation


def pose(mode='squat'):
    p=np.zeros((17,3))
    for shoulder,hip,knee,ankle,x in [(5,11,13,15,100),(6,12,14,16,140)]:
        for j,xy in [(shoulder,(x,100)),(hip,(x,160)),(knee,(x+35,145)),(ankle,(x+15,190))]:p[j]=[*xy,.95]
        if mode=='seat':p[knee]=[x+50,165,.95];p[ankle]=[x+50,220,.95]
    return dict(kpts=p.tolist(),box=[50,50,240,260],det=.95)


def result(raw='background',code='accepted'):
    return dict(label='seated',raw_label=raw,decision_code=code)


def test_squat_vetoes_seat_immediately_but_needs_three_real_observations_to_confirm():
    d=DailyActivity()
    assert d.update(0,1,pose(),result())['label']=='unknown'
    assert d.update(.13,1,pose(),result())['label']=='unknown'
    assert d.update(.26,1,pose(),result())['label']=='squatting'


@pytest.mark.parametrize('mode',['seat','missing','falling','gap','epoch'])
def test_geometry_does_not_persist_through_missing_conflicting_or_changed_evidence(mode):
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,pose(),result())
    p=pose('seat') if mode=='seat' else None if mode=='missing' else pose()
    r=result('falling' if mode=='falling' else 'background')
    out=d.update(.7 if mode=='gap' else .39,2 if mode=='epoch' else 1,p,r)
    assert out['label']!='squatting'
    if mode in ['seat','falling']:assert out['label']=='seated' # no override of protected raw vote


def test_walking_mapping_does_not_hide_input_unknowns_or_override_verified_standing():
    d=DailyActivity()
    for code in ['history_warmup','pose_unreliable','target_unconfirmed']:
        r=dict(label='unknown',raw_label='walking',decision_code=code)
        assert d.update(0,1,pose('seat'),r)['label']=='unknown'
    r=dict(label='standing',raw_label='walking',decision_code='accepted')
    assert d.update(0,1,pose('seat'),r)['label']=='standing'
    r=dict(label='unknown',raw_label='walking',decision_code='other_action',static_evidence={'current_motion':True})
    out=d.update(0,1,pose('seat'),r)
    assert out['label']=='unknown' and out['activity_hint']=='walking'



def test_single_descent_conflict_preserves_observations_but_never_static_output():
    from fallpre.weekly.webcam_reconcile import StaticRecovery
    from test_webcam_static_recovery import sample
    recovery=StaticRecovery();pred,f,p=sample()
    points=np.asarray(p['kpts'])
    for t in [0,.13,.26]:
        recovery.history.append(dict(t=t,points=points.copy(),descent_vote=t>0,**f))
    pred['label']='falling'
    for t in [.39,.52]:
        assert recovery.update(t,pred,f,points) is None
        assert not recovery.details['stable']
    assert not recovery.history # no indefinite retention of a falling vote
    assert recovery.seat is None and recovery.posture is None


def straight_pose(shoulder_drop=0,hip_drop=0,foot_shift=0):
    p=np.zeros((17,3))
    for s,h,k,a,x in [(5,11,13,15,100),(6,12,14,16,140)]:
        for j,y in [(s,100+shoulder_drop),(h,160+hip_drop),(k,210),(a,270)]:
            p[j]=[x+foot_shift,y,.95]
    return dict(kpts=p.tolist(),box=[50,50,200,300],det=.95)


@pytest.mark.parametrize('mode',['hinge','sit','missing','gap','falling'])
def test_descent_veto_needs_head_first_motion_not_just_straight_legs(mode):
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,straight_pose(),dict(label='standing',raw_label='background'))
    p=straight_pose(shoulder_drop=24,hip_drop=8) if mode!='sit' else straight_pose(16,16)
    if mode=='missing':
        for j in [15,16]:p['kpts'][j][2]=0
    r=dict(label='sitting_down',raw_label='falling' if mode=='falling' else 'sitting_down')
    out=d.update(1. if mode=='gap' else .39,1,p,r)
    assert out['label']==('unknown' if mode=='hinge' else 'sitting_down')


def test_sustained_head_first_motion_is_explicit_bending_not_only_a_veto():
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,straight_pose(),dict(label='standing',raw_label='background'))
    outs=[]
    for i,(shoulder,hip) in enumerate([(24,8),(36,10),(48,12)]):
        outs.append(d.update(.39+i*.13,1,straight_pose(shoulder,hip),dict(label='unknown',raw_label='background')))
    assert outs[0]['label']=='unknown'
    assert outs[-1]['label']=='bending'


def test_bending_geometry_does_not_override_current_falling_vote():
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,straight_pose(),dict(label='standing',raw_label='background'))
    for i,(shoulder,hip) in enumerate([(24,8),(36,10),(48,12)]):
        out=d.update(.39+i*.13,1,straight_pose(shoulder,hip),dict(label='unknown',raw_label='falling'))
        assert out['label']=='unknown'


@pytest.mark.parametrize('confidence,expected',[(.95,'bending'),(.5,'unknown')])
def test_two_observations_only_fast_confirm_strong_complete_bend(confidence,expected):
    d=DailyActivity();p=straight_pose()
    for j in [5,6]:p['kpts'][j][1]=200
    for j in [5,6,11,12,13,14,15,16]:p['kpts'][j][2]=confidence
    assert d.update(0,1,p,dict(label='unknown',raw_label='background'))['label']=='unknown'
    assert d.update(.13,1,p,dict(label='unknown',raw_label='background'))['label']==expected


def test_confident_head_first_descent_confirms_on_second_observation_across_fold():
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,straight_pose(),dict(label='standing',raw_label='background'))
    a=d.update(.39,1,straight_pose(24,8),dict(label='unknown',raw_label='background'))
    b=d.update(.52,1,straight_pose(36,10),dict(label='unknown',raw_label='background'))
    assert a['label']=='unknown' and b['label']=='bending'
    c=d.update(.65,1,straight_pose(48,12),dict(label='unknown',raw_label='falling'))
    assert c['label']=='unknown'


def test_low_confidence_head_first_motion_keeps_three_observation_confirmation():
    d=DailyActivity()
    for t in [0,.13,.26]:d.update(t,1,straight_pose(),dict(label='standing',raw_label='background'))
    for t,drop in [(.39,24),(.52,36)]:
        p=straight_pose(drop,8)
        for j in [5,6,11,12,13,14,15,16]:p['kpts'][j][2]=.7
        assert d.update(t,1,p,dict(label='unknown',raw_label='background'))['label']=='unknown'
