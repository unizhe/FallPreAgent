import copy

import pytest

from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from test_action_v2 import person, row


def observation(i, poses, confirmed=()):
    return dict(row(i, poses), detector_instances=[
        dict(class_name='person', conf=.8, box=poses[j]['box']) for j in confirmed])


def test_furniture_pose_cannot_create_target_even_with_high_pose_score():
    tracker = VerifiedPersonTracker()
    for i in range(40):
        r = tracker.update(observation(i, [person(score=.99)]))
        assert r['instances'] == [] and not r['admission'][0]['accepted']


def test_sitting_person_overlapping_chair_is_not_vetoed_by_object_label():
    tracker = VerifiedPersonTracker()
    for i in range(2):
        r = observation(i, [person()], (0,))
        r['detector_instances'].append(dict(class_name='chair', conf=.99, box=person()['box']))
        result = tracker.update(r)
    assert result['target_index'] == 0


def test_lying_pose_continues_when_second_detector_calls_person_a_suitcase():
    tracker = VerifiedPersonTracker()
    tracker.update(observation(0, [person()], (0,)))
    before = tracker.update(observation(1, [person()], (0,)))
    lying = person(score=.2)
    lying['box'] = [60, 135, 205, 200]
    r = observation(2, [lying, person(500, score=.99)])
    r['detector_instances'] = [dict(class_name='suitcase', conf=.95, box=lying['box'])]
    after = tracker.update(r)
    assert after['target_index'] == 0
    assert after['target_epoch'] == before['target_epoch']
    assert not after['admission'][1]['accepted']


def test_no_pose_means_no_synthetic_person_and_no_distant_furniture_takeover():
    tracker = VerifiedPersonTracker()
    for i in range(2):tracker.update(observation(i,[person()],(0,)))
    assert tracker.update(observation(2, []))['instances'] == []
    for i in range(3, 40):
        result=tracker.update(observation(i,[person(500,score=.99)]))
        assert result['instances'] == []


def test_long_gap_starts_new_epoch_and_requires_new_confirmation():
    tracker = VerifiedPersonTracker()
    for i in range(2):before=tracker.update(observation(i,[person()],(0,)))
    for i in range(2,25):tracker.update(observation(i,[]))
    first=tracker.update(observation(25,[person(500)],(0,)))
    second=tracker.update(observation(26,[person(500)],(0,)))
    third=tracker.update(observation(27,[person(500)],(0,)))
    assert first['instances'] == []
    assert third['target_epoch'] > before['target_epoch']
    assert third['target_index'] == 0


def test_ambiguous_close_people_do_not_force_an_identity_choice():
    tracker = VerifiedPersonTracker()
    for i in range(2):tracker.update(observation(i,[person()],(0,)))
    result=tracker.update(observation(2,[person(80),person(120)],(0,1)))
    assert result['instances'] == []
    assert result['tracking_reason'] == 'ambiguous_continuity'


def test_frame_only_gap_is_independent_of_header_clock():
    a,b=VerifiedPersonTracker(frame_only=True),VerifiedPersonTracker(frame_only=True)
    for i in range(25):
        poses=[person()] if i<2 or i>=20 else []
        r=observation(i,poses,(0,) if poses else ())
        ar=a.update(r);br=b.update(dict(r,t=i/120))
        assert (ar['target_epoch'],ar['target_index'])==(br['target_epoch'],br['target_index'])


def test_missing_confirmation_stream_fails_explicitly():
    with pytest.raises(ValueError,match='detector'):
        VerifiedPersonTracker().update(row(0,[person()]))


def test_indices_map_to_original_pose_list_after_filtering():
    tracker=VerifiedPersonTracker()
    for i in range(3):
        result=tracker.update(observation(i,[person(500),person()],(1,)))
    assert result['target_index']==1
    assert set(result['detection_track_ids'])=={1}
    assert result['instances'][0]['box']==person()['box']


def test_partial_low_confidence_box_does_not_erase_recent_full_body_anchor():
    tracker=VerifiedPersonTracker()
    for i in range(2):before=tracker.update(observation(i,[person()],(0,)))
    partial=person(score=.15);partial['box']=[70,190,120,230]
    assert tracker.update(observation(2,[partial]))['target_index']==0
    full=person(125)
    result=tracker.update(observation(3,[full],(0,)))
    assert result['target_index']==0
    assert result['target_epoch']==before['target_epoch']


def test_admission_and_recovery_are_translation_and_scale_invariant():
    original, transformed=VerifiedPersonTracker(),VerifiedPersonTracker()
    for i in range(8):
        r=observation(i,[person(100+i),person(500,score=.99)],(0,) if i<3 else ())
        shifted=copy.deepcopy(r)
        for p in shifted['instances']:
            p['box']=[v*2+300 for v in p['box']]
            p['kpts']=[[x*2+300,y*2+300,c] for x,y,c in p['kpts']]
        for d in shifted['detector_instances']:d['box']=[v*2+300 for v in d['box']]
        a,b=original.update(r),transformed.update(shifted)
        assert (a['target_index'],a['target_epoch'],a['tracking_reason']) == (b['target_index'],b['target_epoch'],b['tracking_reason'])
