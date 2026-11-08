import numpy as np

from fallpre.weekly.pose_refinement import partial_body_evidence, predict_with_pose_quality, RefinedPoseRecovery
from fallpre.weekly.pose_recovery import PoseRecovery
from test_pose_recovery import initialized
from test_verified_tracking import observation
from test_action_v2 import person


def partial_points():
    k=np.zeros((17,3))
    for j,xy in {7:(120,160),9:(130,180),10:(140,180),13:(150,200),14:(160,190),15:(155,220)}.items():
        k[j]=[*xy,.9]
    reliable=k[:,2]>=.3;agreed=np.zeros(17,bool);agreed[[7,9,13,15]]=True
    return k,reliable,agreed


def test_disconnected_supported_limb_segments_are_presence_evidence():
    k,r,a=partial_points();e=partial_body_evidence(k,r,a,160)
    assert e['eligible'] and ((7,9),(13,15)) in e['disjoint_pairs']
    assert not r[5] and not r[11]  # no shoulder or hip is invented


def test_uncorroborated_joints_cannot_make_a_second_segment():
    k,r,a=partial_points();a[15]=False
    assert not partial_body_evidence(k,r,a,160)['eligible']


def test_collapsed_or_excessively_long_segments_are_not_presence():
    k,r,a=partial_points();k[15,:2]=k[13,:2]
    assert not partial_body_evidence(k,r,a,160)['eligible']
    k[15,0]=1000
    assert not partial_body_evidence(k,r,a,160)['eligible']


def test_partial_presence_is_scale_and_translation_invariant():
    k,r,a=partial_points();first=partial_body_evidence(k,r,a,160)
    k[:,:2]=k[:,:2]*3+[200,-50]
    assert partial_body_evidence(k,r,a,480)==first


def test_action_model_receives_missing_pose_while_tracker_keeps_person():
    class Online:
        def update(self,row):
            assert row['instances']==[] and row['target_id']==2
            return dict(label='background',phase='other')
    p=dict(person(),action_eligible=False);row=dict(instances=[p],target_id=2,t=1)
    pred=predict_with_pose_quality(Online(),row)
    assert pred['label']=='unknown' and pred['phase']=='unknown' and pred['pose_input_withheld']
    assert row['instances']==[p]


def test_regular_pose_and_frame_only_behavior_are_unchanged():
    row=dict(instances=[person()]);expected=dict(label='walking',phase='other')
    class Online:
        def update(self,r):assert r is row;return expected
    assert predict_with_pose_quality(Online(),row) is expected
    assert predict_with_pose_quality(None,row)['label']=='disabled_frame_only'


def test_partial_admission_can_be_disabled_and_uses_current_points(monkeypatch):
    k,r,a=partial_points();p=dict(person(),box=[100,140,190,240],kpts=k.tolist(),continuity_score=.8,recovery_view='full_2')
    row=observation(2,[])
    monkeypatch.setattr(PoseRecovery,'augment',lambda self,image,row,tracker:(row,
        dict(executed=True,accepted=False,candidates=[p],inference_passes=8)))
    for enabled in [False,True]:
        model=RefinedPoseRecovery(refiner=lambda *args:k.copy(),allow_partial_observation=enabled)
        out,audit=model.augment(np.zeros((300,600,3),np.uint8),row,initialized())
        assert audit['accepted']==enabled
        if enabled:
            result=out['instances'][0]
            assert result['action_eligible'] is False and result['observation_quality']=='partial_person'
            np.testing.assert_allclose(result['kpts'],k)
        else:assert out is row
