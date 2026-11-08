import cv2
import numpy as np

from fallpre.weekly.webcam_quality import LiveCameraQuality


def texture():
    return np.random.default_rng(17).integers(35, 225, (360, 480, 3), dtype=np.uint8)


def test_reference_requires_a_second_observation_and_detects_movement():
    quality = LiveCameraQuality()
    frame = texture()
    assert quality.update(frame, [])['camera_stable'] is None
    assert quality.update(frame, [])['camera_stable'] is True
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 12], [0, 1, 0]]), (480, 360))
    assert quality.update(moved, [])['camera_stable'] is False
    assert quality.update(moved, [])['camera_stable'] is False
    assert quality.update(moved, [])['code'] == 'camera_moved'
    # Moving back does not silently restore the geometry baseline.
    assert quality.update(frame, [])['code'] == 'camera_moved'
    replacement = LiveCameraQuality()
    replacement.update(moved, [])
    assert replacement.update(moved, [])['camera_stable'] is True


def test_one_bad_fit_abstains_and_requires_original_reference_twice():
    q = LiveCameraQuality()
    original = texture()
    q.update(original, [], 0.)
    q.update(original, [], .1)
    moved = cv2.warpAffine(original, np.float32([[1, 0, 6], [0, 1, 0]]), (480, 360))
    result = q.update(moved, [], .2)
    assert result['camera_stable'] is False
    assert result['code'] == 'camera_motion_pending'
    assert q.update(original, [], .3)['camera_stable'] is None
    assert q.update(original, [], .4)['camera_stable'] is True
    assert q.generation == 1 and q.reference_t == 0.
    assert np.array_equal(q.reference, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))


def test_unknown_frames_and_gaps_do_not_confirm_camera_motion():
    q = LiveCameraQuality()
    original = texture()
    q.update(original, [], 0.)
    q.update(original, [], .1)
    moved = cv2.warpAffine(original, np.float32([[1, 0, 6], [0, 1, 0]]), (480, 360))
    assert q.update(moved, [], .2)['camera_stable'] is False
    assert q.update(np.zeros_like(original), [], .3)['camera_stable'] is None
    assert q.update(moved, [], .4)['code'] == 'camera_motion_pending'
    assert q.update(moved, [], 1.)['code'] == 'camera_motion_pending'
    assert not q.moved


def test_missing_background_or_lighting_is_unknown_not_stable():
    quality = LiveCameraQuality()
    assert quality.update(np.zeros((360, 480, 3), np.uint8), [])['camera_stable'] is None
    blank = np.full((360, 480, 3), 128, np.uint8)
    assert quality.update(blank, [])['code'] == 'background_features_missing'
    hidden = [{'box': [0, 0, 480, 360]}]
    assert quality.update(texture(), hidden)['code'] == 'background_hidden'


def test_person_pixels_do_not_define_camera_motion():
    quality = LiveCameraQuality()
    frame = texture()
    people = [{'box': [150, 60, 330, 300]}]
    quality.update(frame, people)
    changed = frame.copy()
    changed[60:300, 150:330] = 0
    assert quality.update(changed, people)['camera_stable'] is True


def test_resolution_change_requires_restart():
    quality = LiveCameraQuality()
    quality.update(texture(), [])
    assert quality.update(texture()[:, :400], [])['camera_stable'] is False


def other_texture():
    return np.random.default_rng(923).integers(35,225,(360,480,3),dtype=np.uint8)


def test_unverified_bad_start_reference_recovers_only_after_fixed_candidate_verification():
    q=LiveCameraQuality()
    q.update(texture(),[],0.)
    new=other_texture()
    for t in [.1,.2,.3,.4]:
        result=q.update(new,[],t)
        assert result['camera_stable'] is None
    result=q.update(new,[],.5)
    assert result['camera_stable'] is True and result['code']=='reference_recovered'
    assert result['metrics']['reference_generation']==2
    assert result['metrics']['matches']>=12
    assert q.update(new,[],.6)['code']=='stable'


def test_trusted_reference_is_never_silently_rebased_to_an_unrelated_scene():
    q=LiveCameraQuality();q.update(texture(),[],0.);q.update(texture(),[],.1)
    for i in range(2,20):
        assert q.update(other_texture(),[],i*.1)['camera_stable'] is None
    assert q.generation==1
    assert q.update(texture(),[],2.)['camera_stable'] is True


def test_startup_candidate_drift_and_source_gaps_do_not_count_as_stability():
    q=LiveCameraQuality();q.update(texture(),[],0.)
    base=other_texture()
    for i in range(1,10):
        moved=cv2.warpAffine(base,np.float32([[1,0,i*3],[0,1,0]]),(480,360))
        assert q.update(moved,[],i*.1)['camera_stable'] is None
    q=LiveCameraQuality();q.update(texture(),[],0.)
    for t in [.1,.2,.3]:q.update(base,[],t)
    assert q.update(base,[],1.)['camera_stable'] is None
    assert q.pending['count']==1
