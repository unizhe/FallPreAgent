"""Functional boundaries for the manually registered single-camera geometry."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fallpre.config import RuleConfig, SiteConfig
from fallpre.geometry.calibration import Calibration, fit_calibration
from fallpre.geometry.motion import MotionEstimator
from fallpre.geometry.spatial import SpatialCalculator, recompute_spatial
from fallpre.perception.obstacle import ObstacleMonitor
from fallpre.perception.quality import QualityMonitor
from fallpre.schemas import (
    GroundObservation,
    HumanObservation,
    MotionState,
    ObstacleState,
    QualityState,
    Snapshot,
    SpatialState,
    Validity,
)


@pytest.fixture
def rules():
    return RuleConfig()


@pytest.fixture
def site():
    image = [(50, 250), (300, 250), (550, 250), (50, 450), (300, 450), (550, 450)]
    world = [(x / 100, y / 100) for x, y in image]
    checks = [(150, 350), (450, 400)]
    check_world = [(x / 100, y / 100) for x, y in checks]
    h, error = fit_calibration(image, world, checks, check_world, 0.1)
    return SiteConfig(image_size=(640, 480), calibration_id="cal_1", homography=h,
                      calibration_validated=True, validation_error_m=error,
                      calibration_image_points=image, calibration_world_points=world,
                      validation_image_points=checks, validation_world_points=check_world,
                      floor_polygon_px=[(10, 200), (630, 200), (630, 470), (10, 470)],
                      background_roi_px=[(10, 10), (630, 10), (630, 470), (10, 470)],
                      obstacle_footprint_m=[(2, -0.3), (2.5, -0.3), (2.5, 0.3), (2, 0.3)],
                      obstacle_anchor_bbox_px=(300, 280, 400, 400), footprint_version="fp_1")


def human(t=1.0):
    points = [(320.0, 100.0, 0.99)] * 17
    for index, xy in {5: (305, 140), 6: (335, 140), 11: (310, 220), 12: (330, 220),
                      13: (310, 290), 14: (330, 290), 15: (310, 360), 16: (330, 360)}.items():
        points[index] = (*xy, 0.99)
    return HumanObservation(t_s=t, frame_id=10, track_id=1, track_epoch="epoch_1",
                            bbox_px=(275, 80, 365, 380), keypoints=points, score=0.99)


def quality(t=1.0):
    return QualityState(t_s=t, valid_until_s=t + 1, validity=Validity.VALID, camera_stable=True)


def ground(t, x=0.0, calibration="cal_1"):
    return GroundObservation(t_s=t, xy_m=(x, 0.0), uncertainty_m=0.01,
                             calibration_id=calibration, validity=Validity.VALID)


def motion(t=0.0, xy=(0.0, 0.0), velocity=(1.0, 0.0)):
    return MotionState(t_s=t, xy_m=xy, velocity_mps=velocity, speed_mps=float(np.linalg.norm(velocity)),
                       calibration_id="cal_1", track_epoch="epoch_1", validity=Validity.VALID)


def obstacle(site, t=0.0):
    return ObstacleState(obstacle_id=site.obstacle_id, t_s=t, last_confirmed_s=t, valid_until_s=t + 2,
                         status="CONFIRMED", validity=Validity.VALID,
                         footprint_m=site.obstacle_footprint_m, footprint_version=site.footprint_version)


def test_calibration_requires_independent_validation(site):
    with pytest.raises(ValueError, match="independent"):
        fit_calibration(site.calibration_image_points, site.calibration_world_points,
                        site.calibration_image_points[:2], site.calibration_world_points[:2], 0.1)
    wrong = [(x + 0.3, y) for x, y in site.validation_world_points]
    with pytest.raises(ValueError, match="exceeds"):
        fit_calibration(site.calibration_image_points, site.calibration_world_points,
                        site.validation_image_points, wrong, 0.1)
    fake = site.model_copy(update={"validation_image_points": []})
    result = Calibration(fake).project(human(), quality(), site.image_size)
    assert result.validity == Validity.UNKNOWN
    assert result.xy_m is None


def test_upright_ankle_proxy_and_uncertainty(site):
    result = Calibration(site).project(human(), quality(), site.image_size)
    assert result.validity == Validity.VALID
    assert result.xy_m == pytest.approx((3.2, 3.6))
    assert result.uncertainty_m >= site.footpoint_error_m


@pytest.mark.parametrize("change", ["ankle", "bent_knee", "low_pose", "cropped"])
def test_unreliable_feet_never_produce_metric_coordinates(site, change):
    observation = human()
    if change == "ankle":
        observation.keypoints[16] = (330, 360, 0.1)
    elif change == "bent_knee":
        observation.keypoints[13] = (370, 270, 0.99)
    elif change == "low_pose":
        observation.keypoints[11] = (420, 180, 0.99)
        observation.keypoints[12] = (440, 180, 0.99)
    else:
        observation.bbox_px = (275, 80, 365, 480)
    result = Calibration(site).project(observation, quality(), site.image_size)
    assert result.validity == Validity.UNKNOWN
    assert result.xy_m is None


def test_projection_requires_fresh_stable_quality_matching_size_and_roi(site):
    calibration = Calibration(site)
    assert calibration.project(human(2), quality(1), site.image_size).xy_m is None
    assert calibration.project(human(), quality().model_copy(update={"camera_stable": False}), site.image_size).xy_m is None
    assert calibration.project(human(), quality(), (320, 240)).xy_m is None
    small_roi = site.model_copy(update={"floor_polygon_px": [(10, 200), (100, 200), (100, 470), (10, 470)]})
    assert Calibration(small_roi).project(human(), quality(), site.image_size).xy_m is None
    corrupt_h = np.asarray(site.homography).copy()
    corrupt_h[0, 2] += 0.3
    corrupt = site.model_copy(update={"homography": corrupt_h.tolist()})
    assert Calibration(corrupt).project(human(), quality(), site.image_size).xy_m is None


def test_variable_dt_velocity_does_not_depend_on_frame_count(rules):
    estimator = MotionEstimator(rules)
    times = np.cumsum([0.0] + [0.04, 0.13, 0.07, 0.2] * 8)
    for t in times:
        result = estimator.update(ground(float(t), 0.8 * float(t)), "epoch_1")
    assert result.validity == Validity.VALID
    assert result.velocity_mps == pytest.approx((0.8, 0), abs=0.03)
    assert result.xy_m == pytest.approx((0.8 * times[-1], 0), abs=0.02)


@pytest.mark.parametrize("reset_reason", ["epoch", "gap", "backward", "calibration", "invalid"])
def test_motion_continuity_boundaries_do_not_create_spurious_velocity(rules, reset_reason):
    estimator = MotionEstimator(rules)
    estimator.update(ground(0, 0), "epoch_1")
    estimator.update(ground(0.1, 0.1), "epoch_1")
    next_ground, epoch = ground(0.2, 5), "epoch_1"
    if reset_reason == "epoch":
        epoch = "epoch_2"
    elif reset_reason == "gap":
        next_ground = ground(2, 5)
    elif reset_reason == "backward":
        next_ground = ground(0.05, 5)
    elif reset_reason == "calibration":
        next_ground = ground(0.2, 5, "cal_2")
    else:
        next_ground = GroundObservation(t_s=0.2)
    result = estimator.update(next_ground, epoch)
    assert result.validity != Validity.VALID
    assert result.velocity_mps is None


def test_ttc_first_entry_and_age_compensation(site, rules):
    calc = SpatialCalculator(site, rules)
    current = calc.compute(motion(), obstacle(site), 0.0)
    aged = calc.compute(motion(), obstacle(site), 0.2)
    assert current.ttc_s == pytest.approx(1.6)
    assert aged.reference_time_s == 0.2
    assert aged.distance_m == pytest.approx(1.8)
    assert aged.ttc_s < current.ttc_s - 0.19
    assert aged.r_total_m > current.r_total_m
    assert aged.valid_until_s == rules.human_max_age_s
    assert calc.compute(motion(), obstacle(site), rules.human_max_age_s).ttc_status == "UNKNOWN"


def test_ttc_distinguishes_stopped_inside_from_outside_and_no_intersection(site, rules):
    calc = SpatialCalculator(site, rules)
    inside = calc.compute(motion(xy=(2.1, 0), velocity=(0, 0)), obstacle(site), 0)
    stopped = calc.compute(motion(velocity=(0, 0)), obstacle(site), 0)
    parallel = calc.compute(motion(velocity=(0, 1)), obstacle(site), 0)
    assert inside.ttc_s == 0 and inside.ttc_status == "INSIDE_RISK_REGION"
    assert stopped.ttc_s is None and stopped.ttc_status == "STATIONARY_OUTSIDE"
    assert parallel.ttc_s is None and parallel.ttc_status == "NO_INTERSECTION_WITHIN_HORIZON"


def test_unknown_obstacle_and_changed_footprint_block_geometry(site, rules):
    calc = SpatialCalculator(site, rules)
    bad = obstacle(site).model_copy(update={"validity": Validity.UNKNOWN, "status": "MOVED_OR_AMBIGUOUS"})
    assert calc.compute(motion(), bad, 0).distance_m is None
    changed = obstacle(site).model_copy(update={"footprint_version": "fp_2"})
    assert calc.compute(motion(), changed, 0).ttc_status == "UNKNOWN"
    expired = obstacle(site).model_copy(update={"valid_until_s": 0.1})
    assert calc.compute(motion(), expired, 0.2).validity == Validity.STALE


def test_consumer_recomputation_preserves_ttl_trend_and_source_snapshot(site, rules):
    original = SpatialCalculator(site, rules).compute(motion(), obstacle(site), 0)
    snapshot = Snapshot(t_s=0, person_id="p1", binding_id="binding_1", track_epoch="epoch_1",
                        calibration_id="cal_1", motion=motion(), obstacle=obstacle(site),
                        spatial=original, quality=quality(0))
    recomputed = recompute_spatial(snapshot, 0.2, rules)
    assert recomputed.ttc_s < original.ttc_s
    assert recomputed.computed_at_s == 0.2
    assert recomputed.valid_until_s == original.valid_until_s
    assert recomputed.closing_speed_mps == original.closing_speed_mps
    assert snapshot.spatial == original and snapshot.t_s == 0
    bad_quality = snapshot.model_copy(update={"quality": quality(0).model_copy(update={"valid_until_s": 0.1})})
    result = recompute_spatial(bad_quality, 0.2, rules)
    assert result.ttc_s is None and result.validity == Validity.STALE
    fake = snapshot.model_copy(update={"spatial": SpatialState()})
    assert recompute_spatial(fake, 0.2, rules).validity == Validity.UNKNOWN


def test_obstacle_confirmation_occlusion_and_expiry(site, rules):
    monitor = ObstacleMonitor(site, rules)
    detection = [{"bbox_px": site.obstacle_anchor_bbox_px, "score": 0.9, "class_name": "backpack"}]
    assert monitor.update(detection, 0).validity == Validity.UNKNOWN
    assert monitor.update(detection, 0).validity == Validity.UNKNOWN  # duplicate source frame
    monitor.update(detection, 0.5)
    confirmed = monitor.update(detection, 1.0)
    assert confirmed.status == "CONFIRMED"
    hidden = monitor.update([], 1.5)
    assert hidden.status == "TEMPORARILY_OCCLUDED" and hidden.validity == Validity.DEGRADED
    assert hidden.footprint_m == site.obstacle_footprint_m
    assert monitor.age(3.0).validity == Validity.STALE
    assert monitor.update(detection, 3.1).validity != Validity.VALID


@pytest.mark.parametrize("ambiguous", [False, True])
def test_obstacle_movement_and_ambiguity_are_latched(site, rules, ambiguous):
    monitor = ObstacleMonitor(site, rules)
    matching = {"bbox_px": site.obstacle_anchor_bbox_px, "score": 0.9, "class_name": "backpack"}
    moved = {"bbox_px": (410, 280, 510, 400), "score": 0.9, "class_name": "backpack"}
    for t in (0, 0.5, 1.0):
        monitor.update([matching], t)
    result = monitor.update([matching, moved] if ambiguous else [moved], 1.5)
    assert result.status == "MOVED_OR_AMBIGUOUS"
    for t in (2, 2.5, 3, 5):
        assert monitor.update([matching], t).validity == Validity.UNKNOWN


def textured_frame():
    # Deterministic textured reference tests geometric matching rather than mocks.
    rng = np.random.default_rng(123)
    gray = rng.integers(40, 220, size=(480, 640), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_quality_requires_reference_and_does_not_invent_camera_stability(site, rules):
    result = QualityMonitor(site, rules).update(textured_frame(), 0)
    assert result.validity == Validity.UNKNOWN
    assert result.camera_stable is None
    assert "BACKGROUND_REFERENCE_UNAVAILABLE" in result.reasons


def test_camera_motion_is_detected_and_latched(tmp_path, site, rules):
    original = textured_frame()
    path = tmp_path / "reference.png"
    assert cv2.imwrite(str(path), original)
    configured = site.model_copy(update={"background_reference": str(path)})
    monitor = QualityMonitor(configured, rules)
    stable = monitor.update(original, 0)
    assert stable.validity == Validity.VALID and stable.camera_stable is True
    shifted = cv2.warpAffine(original, np.float32([[1, 0, 10], [0, 1, 0]]), (640, 480))
    moved = monitor.update(shifted, 0.5)
    assert moved.camera_stable is False
    assert monitor.update(original, 1).camera_stable is False


def test_dark_textureless_and_occluded_background_return_unknown(tmp_path, site, rules):
    original = textured_frame()
    path = tmp_path / "reference.png"
    assert cv2.imwrite(str(path), original)
    configured = site.model_copy(update={"background_reference": str(path)})
    monitor = QualityMonitor(configured, rules)
    dark = monitor.update(np.zeros_like(original), 0)
    assert dark.validity == Validity.UNKNOWN and "LOW_LIGHT" in dark.reasons
    occluded = monitor.update(original, 0.5, [(0, 0, 640, 480)])
    assert occluded.validity == Validity.UNKNOWN
    assert occluded.camera_stable is None
    flat = monitor.update(np.full_like(original, 127), 1)
    assert flat.validity == Validity.UNKNOWN and flat.camera_stable is None
    # Low quality alone is not a confirmed camera-movement latch.
    assert monitor.update(original, 1.5).validity == Validity.VALID
