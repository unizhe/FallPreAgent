import copy

import numpy as np
import pytest

from fallpre.weekly.pose_recovery import PoseRecovery
from fallpre.weekly.pose_refinement import RefinedPoseRecovery
from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from test_action_v2 import person
from test_pose_recovery import initialized
from test_verified_tracking import observation


def test_ambiguous_native_matches_trigger_inspection_but_not_forced_selection():
    tracker=initialized()
    row=observation(2,[person(80),person(120)])
    assert not tracker.recovery_context(row['t'],row['instances'])['allowed']
    context=tracker.recovery_context(row['t'],row['instances'],allow_ambiguous=True)
    assert context['allowed'] and context['reason']=='native_pose_ambiguous'
    assert tracker.step==2  # read-only inspection
    out,audit=PoseRecovery(lambda image:[],resolve_native_ambiguity=True).augment(
        np.zeros((300,600,3),np.uint8),row,tracker)
    assert not audit['accepted'] and out is row
    assert tracker.update(out)['tracking_reason']=='ambiguous_continuity'


def test_current_complete_candidate_can_resolve_split_native_detections():
    tracker=initialized();row=observation(2,[person(80),person(120)])
    current=person(102)
    out,audit=PoseRecovery(lambda image:[current],resolve_native_ambiguity=True).augment(
        np.zeros((300,600,3),np.uint8),row,tracker)
    result=tracker.update(out)
    assert audit['trigger']=='native_pose_ambiguous'
    assert result['target_index']==2 and result['target_epoch']==1
    assert result['instances'][0]['kpts']==current['kpts']


def test_two_current_people_still_ambiguous_after_recovery():
    tracker=initialized();row=observation(2,[person(80),person(120)])
    recovery=PoseRecovery(lambda image:[person(80),person(120)],resolve_native_ambiguity=True)
    # Test selection independently of rotated-model coordinate transforms.
    candidates=[dict(p,continuity_score=tracker._continuity(p['box'],row['t']))
                for p in row['instances']]
    assert recovery._choose(candidates) is None
    assert tracker.update(row)['instances']==[]


def test_opt_in_verification_keeps_clear_native_matches_and_expired_targets():
    def forbidden(*args):raise AssertionError('No current verification expected')
    recovery=RefinedPoseRecovery(backend=forbidden,refiner=forbidden,
        allow_detector_proposals=True,resolve_native_ambiguity=True)
    image=np.zeros((300,600,3),np.uint8)
    for tracker,row in [(initialized(),observation(2,[person()])),
                        (initialized(),observation(30,[person(80),person(120)])),
                        (VerifiedPersonTracker(),observation(0,[]))]:
        out,audit=recovery.augment(image,row,tracker)
        assert out is row and not audit['executed']


def weak_proposal():
    p=person(score=.45)
    k=np.asarray(p['kpts'],float)
    k[:5,2]=0
    p.update(kpts=k.tolist(),continuity_score=.8,recovery_view='full_3')
    return p


@pytest.mark.parametrize('confirmation,accepted',[
    ('person',True),('chair',False),('distant_person',False),('too_weak',False)])
def test_weak_detector_support_only_triggers_current_pose_verification(monkeypatch,confirmation,accepted):
    proposal=weak_proposal()
    row=observation(2,[])
    row['detector_instances']=[dict(class_name='chair' if confirmation=='chair' else 'person',
        conf=.14 if confirmation=='too_weak' else .19,
        box=person(500)['box'] if confirmation=='distant_person' else proposal['box'])]
    monkeypatch.setattr(PoseRecovery,'augment',lambda self,image,row,tracker:(row,
        dict(executed=True,accepted=False,candidates=[proposal],inference_passes=8)))
    calls=[]
    def refine(image,box,turn,expansion):
        calls.append(copy.deepcopy(box));return np.array(proposal['kpts'])
    recovery=RefinedPoseRecovery(refiner=refine,allow_detector_proposals=True)
    out,audit=recovery.augment(np.zeros((300,600,3),np.uint8),row,initialized())
    assert audit['accepted']==accepted
    assert len(calls)==(8 if accepted else 0)
    if accepted:
        assert audit['refinement']['detector_supported_proposal']
        result=np.asarray(out['instances'][0]['kpts'])
        np.testing.assert_allclose(result[5:],np.asarray(proposal['kpts'])[5:])
        assert not result[:5].any()  # unsupported coordinates are not fabricated
    else:assert out is row


def test_weak_box_agreement_cannot_bypass_joint_agreement(monkeypatch):
    proposal=weak_proposal();row=observation(2,[])
    row['detector_instances']=[dict(class_name='person',conf=.19,box=proposal['box'])]
    monkeypatch.setattr(PoseRecovery,'augment',lambda self,image,row,tracker:(row,
        dict(executed=True,accepted=False,candidates=[proposal],inference_passes=8)))
    wrong=np.array(proposal['kpts']);wrong[:,:2]+=[100,0]
    out,audit=RefinedPoseRecovery(refiner=lambda *args:wrong,
        allow_detector_proposals=True).augment(np.zeros((300,600,3),np.uint8),row,initialized())
    assert audit['refinement']['executed'] and not audit['accepted'] and out is row


def test_proposal_box_support_is_translation_scale_invariant_and_can_be_disabled():
    proposal=weak_proposal()
    row=dict(detector_instances=[dict(class_name='person',conf=.19,box=proposal['box'])])
    assert not RefinedPoseRecovery()._detector_support(proposal,row)
    model=RefinedPoseRecovery(allow_detector_proposals=True)
    assert model._detector_support(proposal,row)
    proposal['box']=[v*3+200 for v in proposal['box']]
    row['detector_instances'][0]['box']=proposal['box']
    assert model._detector_support(proposal,row)
