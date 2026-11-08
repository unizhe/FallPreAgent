import pytest

from fallpre.baseline import PersonalBaseline
from fallpre.config import RuleConfig
from fallpre.demo import synthetic_human, synthetic_quality, synthetic_site
from fallpre.pipeline import ObservationPipeline
from fallpre.schemas import Behavior, FallState, MotionState, Snapshot, SpatialState, Validity


def core():
    return ObservationPipeline(synthetic_site(), RuleConfig(), track_id=1, source_type="synthetic")


def feed(pipeline, t, *, epoch="track1", low=False, quality=None):
    return pipeline.update([synthetic_human(t, round(t*10), low=low, epoch=epoch)],
                           [{"class_name": "backpack", "score": .99,
                             "bbox_px": pipeline.site.obstacle_anchor_bbox_px}],
                           quality or synthetic_quality(t), t, round(t*10), (640, 480))


def test_binding_does_not_follow_reused_track_id_epoch():
    p = core()
    first = feed(p, 0)
    assert first.binding_status == "BOUND"
    changed = feed(p, .1, epoch="track2")
    assert changed.binding_status == "SUSPENDED"
    assert changed.human is None
    assert feed(p, .2, epoch="track1").human is None


def test_long_gap_same_epoch_requires_binding_again():
    p = core()
    feed(p, 0)
    assert feed(p, 1).binding_status == "SUSPENDED"


def test_geometry_is_not_required_for_image_fall():
    site = synthetic_site().model_copy(update={"calibration_validated": False})
    p = ObservationPipeline(site, RuleConfig(), 1)
    for i in range(16):
        s = feed(p, i/10)
    for i in range(16, 35):
        s = feed(p, i/10, low=True)
    assert s.spatial.validity != Validity.VALID
    assert s.fall.had_upright and s.fall.low_pose_duration_s >= 1


def test_camera_motion_suspends_person_and_resets_image_evidence():
    p = core()
    for i in range(10):
        feed(p, i/10)
    q = synthetic_quality(1).model_copy(update={"camera_stable": False, "validity": Validity.UNKNOWN})
    s = feed(p, 1, quality=q)
    assert s.binding_status == "SUSPENDED"
    assert s.fall.phase == "UNOBSERVABLE"
    assert not s.fall.had_upright


def test_source_time_reversal_does_not_silently_join_recordings():
    p = core()
    feed(p, 1)
    with pytest.raises(ValueError):
        feed(p, .9)


def walking(t, *, speed=.5, calibration="c1", epoch="e1", walking=True):
    return Snapshot(t_s=t, frame_id=round(t*10), person_id="p", binding_id="b", track_epoch=epoch,
                    calibration_id=calibration, quality=synthetic_quality(t),
                    motion=MotionState(t_s=t, speed_mps=speed, velocity_mps=(speed, 0),
                                       validity=Validity.VALID),
                    behavior=Behavior.WALKING if walking else Behavior.STATIONARY,
                    fall=FallState(t_s=t, phase="NONE", validity=Validity.VALID),
                    spatial=SpatialState(validity=Validity.VALID, ttc_status="NO_INTERSECTION_WITHIN_HORIZON"))


def test_baseline_independent_clips_and_zero_mad_no_fake_risk_level():
    b = PersonalBaseline(RuleConfig(baseline_min_clips=2, baseline_min_duration_s=1), min_clip_s=.5)
    for start in (0, 2):
        for i in range(8):
            b.update(walking(start+i*.1))
        report = b.update(walking(start+.8, walking=False))
    assert report["baseline_status"] == "temporary"
    assert report["data_coverage"]["qualified_clip_count"] == 2
    assert report["risk_level"] is None
    for i in range(8):
        b.update(walking(4+i*.1, speed=.3))
    report = b.update(walking(4.8, walking=False))
    deviation = report["main_deviations"][0]
    assert deviation["robust_deviation"] is None
    assert deviation["speed_drop_ratio"] == pytest.approx(.4)
    assert report["reference"]["speed_median_mps"] == .5
    report = b.update(walking(5, calibration="c2"))
    assert report["baseline_status"] == "collecting"
    assert len(b.archived) == 1


def test_risk_and_intervention_discard_current_baseline_clip():
    b = PersonalBaseline(RuleConfig(), min_clip_s=.5)
    for i in range(9):
        b.update(walking(i*.1))
    b.update(walking(.9), excluded=True, intervention=True)
    for i in range(10, 30):
        report = b.update(walking(i*.1))
    assert report["data_coverage"]["qualified_clip_count"] == 0


def test_pipeline_rejects_mixed_frame_human_observations():
    p = core()
    with pytest.raises(ValueError, match="current source frame"):
        p.update([synthetic_human(.2, 2)], [], synthetic_quality(.1), .1, 1, (640, 480))


def test_future_quality_does_not_validate_current_fall_evidence():
    p = core()
    snapshot = feed(p, 0, quality=synthetic_quality(1))
    assert snapshot.fall.validity == Validity.UNKNOWN
    assert snapshot.quality.camera_stable is None


def test_report_id_does_not_change_content_without_new_qualified_clip():
    b = PersonalBaseline(RuleConfig())
    first = b.update(walking(0, walking=False))
    second = b.update(walking(.1, walking=False))
    assert first == second


def test_quality_is_expired_at_its_exact_deadline():
    p = core()
    q = synthetic_quality(0).model_copy(update={"valid_until_s": 0})
    s = feed(p, 0, quality=q)
    assert s.fall.validity == Validity.UNKNOWN
    assert s.obstacle.validity == Validity.UNKNOWN
