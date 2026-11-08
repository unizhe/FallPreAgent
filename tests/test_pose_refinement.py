import numpy as np

from fallpre.weekly.pose_refinement import (
    RTMPoseBackend, RefinedPoseRecovery, consensus_joints, rotate_points, visible_joints,
)
from fallpre.weekly.pose_recovery import PoseRecovery, restore_points
from test_action_v2 import person
from test_pose_recovery import initialized
from test_verified_tracking import observation


def test_rotation_round_trip_on_non_square_image():
    pts=np.array([[12,31,.8],[61,90,.5]])
    for turn in range(4):
        np.testing.assert_allclose(restore_points(rotate_points(pts,120,100,turn),120,100,turn),pts)


def test_outside_or_weak_joints_cannot_be_reported_as_supported():
    pts=np.array(person()['kpts'],float)
    pts[0]=[-1,50,.99];pts[1]=[50,300,.99];pts[2]=[50,50,.29]
    k,s=consensus_joints([pts,pts],(300,600,3),200)
    assert not k[:3,2].any() and s[:3]==[[],[],[]]


def test_competing_joint_locations_remain_unknown():
    a=np.array(person()['kpts'],float);b=a.copy();b[5,0]+=100
    k,s=consensus_joints([a,a,b,b],(300,600,3),200)
    assert k[5,2]==0 and s[5]==[]
    assert k[6,2]>.3


def test_sdk_preprocess_rgb_and_simcc_decode_coordinates():
    class Session:
        def run(self,_,inputs):
            tensor=inputs['input']
            np.testing.assert_allclose(tensor[0,:,128,96],(np.array([30,20,10])-[123.675,116.28,103.53])/[58.395,57.12,57.375],rtol=1e-6)
            x=np.zeros((1,17,384),np.float32);y=np.zeros((1,17,512),np.float32)
            x[:,:,192]=.8;y[:,:,256]=.6
            return x,y
    backend=RTMPoseBackend.__new__(RTMPoseBackend);backend.session=Session();backend.input_name='input'
    image=np.full((300,600,3),[10,20,30],np.uint8)
    k=backend(image,[100,100,200,200])
    np.testing.assert_allclose(k,np.tile([150,150,.6],(17,1)),rtol=1e-6)


def test_refiner_cannot_initialize_target_or_replace_native_pose():
    def forbidden(*_):raise AssertionError('No refinement expected')
    r=RefinedPoseRecovery(backend=forbidden,refiner=forbidden)
    image=np.zeros((300,600,3),np.uint8)
    row=observation(2,[person()],())
    out,a=r.augment(image,row,initialized())
    assert out is row and not a['refinement']['executed']


def test_topdown_pose_without_current_rtmo_evidence_is_not_a_person(monkeypatch):
    row=observation(2,[])
    monkeypatch.setattr(PoseRecovery,'augment',lambda self,image,row,tracker:(row,
        dict(executed=True,accepted=False,candidates=[],inference_passes=8)))
    def forbidden(*_):raise AssertionError('No current proposal')
    out,a=RefinedPoseRecovery(refiner=forbidden).augment(np.zeros((300,600,3),np.uint8),row,initialized())
    assert out is row and not a['refinement']['executed']


def test_current_lower_body_can_be_observed_without_inventing_shoulders(monkeypatch):
    p=person();pts=np.array(p['kpts'],float);pts[:11,2]=0
    # Six supported lower-body joints including a connected hip/knee/ankle chain.
    p.update(kpts=pts.tolist(),continuity_score=.9,recovery_view='full_2')
    row=observation(2,[])
    monkeypatch.setattr(PoseRecovery,'augment',lambda self,image,row,tracker:(row,
        dict(executed=True,accepted=False,candidates=[p],inference_passes=8)))
    calls=[]
    def current(image,box,turn,expansion):
        calls.append((image.shape,list(box),turn,expansion));return pts.copy()
    out,a=RefinedPoseRecovery(refiner=current).augment(np.zeros((300,600,3),np.uint8),row,initialized())
    assert a['accepted'] and len(calls)==8
    result=out['instances'][0]
    assert result['pose_coverage']=='partial' and not np.array(result['kpts'])[:11,2].any()
    assert visible_joints(result['kpts'],(300,600,3)).sum()==6


def test_unstable_single_view_joint_cannot_be_counted_as_completion():
    p=np.array(person()['kpts'],float);q=p.copy();q[15,2]=0
    k,s=consensus_joints([p,q],(300,600,3),200)
    assert k[15,2]==0 and not s[15]
