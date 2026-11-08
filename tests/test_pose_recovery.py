import numpy as np

from fallpre.weekly.pose_recovery import PoseRecovery, restore_points, restore_instance
from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from test_verified_tracking import observation
from test_action_v2 import person


def initialized():
    t=VerifiedPersonTracker()
    for i in range(2):t.update(observation(i,[person()],(0,)))
    return t


def test_recovery_cannot_create_new_person_from_chair_or_extend_expired_identity():
    def forbidden(_):raise AssertionError('Must not run without recent confirmed target')
    r=PoseRecovery(forbidden);image=np.zeros((300,600,3),np.uint8)
    row=observation(0,[])
    out,a=r.augment(image,row,VerifiedPersonTracker());assert out is row and not a['executed']
    out,a=r.augment(image,observation(30,[]),initialized());assert not a['executed']


def test_native_observation_is_never_replaced():
    def forbidden(_):raise AssertionError('Native pose has priority')
    row=observation(2,[person()],())
    out,a=PoseRecovery(forbidden).augment(np.zeros((300,600,3),np.uint8),row,initialized())
    assert out is row and not a['executed']


def test_recovered_person_is_from_current_inference_not_old_coordinates():
    current=person(112)
    row=observation(2,[person(500)],())
    t=initialized()
    out,a=PoseRecovery(lambda _: [person(500),current]).augment(np.zeros((300,600,3),np.uint8),row,t)
    assert a['accepted'] and len(out['instances'])==2 and len(row['instances'])==1
    assert out['instances'][1]['box']==current['box']
    result=t.update(out)
    assert result['target_index']==1 and result['target_epoch']==1


def test_empty_or_low_quality_recovery_does_not_fill_missing_person():
    image=np.zeros((300,600,3),np.uint8)
    poor=person();poor['kpts']=[[x,y,0] for x,y,c in poor['kpts']]
    for backend in [lambda _:[],lambda _:[poor]]:
        row=observation(2,[])
        out,a=PoseRecovery(backend).augment(image,row,initialized())
        assert not a['accepted'] and out is row and out['instances']==[]


def test_quarter_turn_inverse_preserves_points_and_confidences():
    original=np.array([[10,20,.8],[90,60,.9]])
    w,h=120,80
    for turn in range(4):
        q=original.copy();x,y=q[:,0].copy(),q[:,1].copy()
        if turn==1:q[:,0],q[:,1]=y,w-1-x
        if turn==2:q[:,0],q[:,1]=w-1-x,h-1-y
        if turn==3:q[:,0],q[:,1]=h-1-y,x
        np.testing.assert_allclose(restore_points(q,w,h,turn),original)


def test_rotated_box_restores_all_four_corners_and_crop_offset():
    p=dict(box=[20,29,60,109],det=.9,kpts=[[20,109,.7]]*17)
    out=restore_instance(p,120,80,1,(100,200))
    np.testing.assert_allclose(out['box'],[110,220,190,260])
    np.testing.assert_allclose(out['kpts'][0],[110,220,.7])


def test_inconsistent_torso_proposals_cannot_supply_rotation_consensus():
    r=PoseRecovery();a=person();b=person()
    for j in [5,6,11,12]:b['kpts'][j][0]+=100
    assert not r._agreement(a,b)


def test_frame_clock_gate_is_independent_of_physical_fps():
    t=VerifiedPersonTracker(frame_only=True)
    for i in range(2):t.update(dict(observation(i,[person()],(0,)),t=i/120))
    row=dict(observation(2,[]),t=100)
    out,a=PoseRecovery(lambda _:[person()]).augment(np.zeros((300,600,3),np.uint8),row,t)
    assert a['accepted']
