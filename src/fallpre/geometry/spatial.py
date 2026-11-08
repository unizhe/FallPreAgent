"""Conservative geometry for one registered static ground obstacle."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from ..config import RuleConfig, SiteConfig
from ..schemas import MotionState, ObstacleState, Snapshot, SpatialState, Validity

_USABLE = (Validity.VALID, Validity.DEGRADED)


def _unknown(motion, obstacle, now_s, rules, reason, stale=False):
    return SpatialState(reference_time_s=now_s, computed_at_s=now_s, valid_until_s=now_s,
                        horizon_s=rules.horizon_s, calibration_id=motion.calibration_id,
                        footprint_version=obstacle.footprint_version,
                        validity=Validity.STALE if stale else Validity.UNKNOWN, reasons=[reason])


def _predict(motion: MotionState, obstacle: ObstacleState, now_s: float,
             rules: RuleConfig, closing_speed=None) -> SpatialState:
    """Pure bounded prediction after the caller authenticates registration state."""
    def invalid(reason, stale=False):
        return _unknown(motion, obstacle, now_s, rules, reason, stale)

    if motion.validity != Validity.VALID or motion.xy_m is None or motion.velocity_mps is None:
        return invalid("MOTION_UNAVAILABLE")
    age = now_s - motion.t_s
    if age < 0 or age >= rules.human_max_age_s:
        return invalid("MOTION_STALE_OR_FROM_FUTURE", stale=age >= rules.human_max_age_s)
    if obstacle.validity not in _USABLE or obstacle.status not in ("CONFIRMED", "TEMPORARILY_OCCLUDED"):
        return invalid("OBSTACLE_NOT_CONFIRMED")
    if obstacle.last_confirmed_s is None or obstacle.last_confirmed_s > now_s or now_s >= obstacle.valid_until_s:
        return invalid("OBSTACLE_STALE_OR_FROM_FUTURE", stale=now_s >= obstacle.valid_until_s)
    if len(obstacle.footprint_m) < 3:
        return invalid("FOOTPRINT_MISSING")
    polygon = Polygon(obstacle.footprint_m)
    if not polygon.is_valid or polygon.area <= 0:
        return invalid("FOOTPRINT_INVALID")
    if not np.isfinite([*motion.xy_m, *motion.velocity_mps, motion.uncertainty_m]).all():
        return invalid("MOTION_NONFINITE")
    if rules.horizon_s <= 0:
        return invalid("INVALID_PREDICTION_HORIZON")
    velocity = np.asarray(motion.velocity_mps, dtype=float)
    p = np.asarray(motion.xy_m, dtype=float) + age * velocity
    speed = float(np.linalg.norm(velocity))
    acceleration = float(getattr(rules, "motion_acceleration_noise_mps2", 1.0))
    extrapolation_error = 0.5 * acceleration * age ** 2
    radius = rules.person_radius_m + motion.uncertainty_m + extrapolation_error + rules.obstacle_error_m + rules.safety_margin_m
    if not math.isfinite(radius) or radius < 0:
        return invalid("INVALID_GEOMETRY_MARGIN")
    point = Point(float(p[0]), float(p[1]))
    distance = float(point.distance(polygon))
    safe = polygon.buffer(radius)
    if closing_speed is None:
        # One-sided local derivative is available before enough distance history
        # exists, and is never described as an independently observed trend.
        derivative_dt = min(0.05, rules.horizon_s)
        next_p = p + velocity * derivative_dt
        closing_speed = (distance - Point(float(next_p[0]), float(next_p[1])).distance(polygon)) / derivative_dt
    base = dict(reference_time_s=now_s, computed_at_s=now_s,
                valid_until_s=min(motion.t_s + rules.human_max_age_s, obstacle.valid_until_s),
                horizon_s=rules.horizon_s, calibration_id=motion.calibration_id,
                footprint_version=obstacle.footprint_version, distance_m=distance,
                clearance_m=max(0.0, distance - rules.person_radius_m),
                closing_speed_mps=closing_speed, r_total_m=radius,
                validity=Validity.DEGRADED if obstacle.validity == Validity.DEGRADED else Validity.VALID)
    if safe.covers(point):
        return SpatialState(**base, path_intersects=True, ttc_s=0.0, ttc_status="INSIDE_RISK_REGION")
    if speed < rules.stationary_speed_mps:
        return SpatialState(**base, path_intersects=False, ttc_status="STATIONARY_OUTSIDE")
    endpoint = p + velocity * rules.horizon_s
    path = LineString([p, endpoint])
    intersection = path.intersection(safe)
    if intersection.is_empty:
        return SpatialState(**base, path_intersects=False, ttc_status="NO_INTERSECTION_WITHIN_HORIZON")
    # All intersection points lie on this straight, ordered path; Euclidean
    # distance from its start is the nonnegative arc length to first entry.
    first_distance = float(point.distance(intersection))
    ttc = min(rules.horizon_s, max(0.0, first_distance / speed))
    return SpatialState(**base, path_intersects=True, ttc_s=ttc, ttc_status="PREDICTED_ENTRY")


class SpatialCalculator:
    def __init__(self, site: SiteConfig, rules: RuleConfig):
        self.site = site
        self.rules = rules
        self._previous = None

    def compute(self, motion: MotionState, obstacle: ObstacleState, now_s: float) -> SpatialState:
        if not self.site.calibration_validated or motion.calibration_id != self.site.calibration_id:
            self._previous = None
            return _unknown(motion, obstacle, now_s, self.rules, "CALIBRATION_INVALID_OR_CHANGED")
        if obstacle.footprint_version != self.site.footprint_version or obstacle.obstacle_id != self.site.obstacle_id:
            self._previous = None
            return _unknown(motion, obstacle, now_s, self.rules, "OBSTACLE_REGISTRATION_CHANGED")
        result = _predict(motion, obstacle, now_s, self.rules)
        if result.validity not in _USABLE:
            self._previous = None
            return result
        key = (motion.track_epoch, motion.calibration_id, obstacle.footprint_version)
        closing = result.closing_speed_mps
        if self._previous is not None:
            previous_key, previous_time, previous_distance, previous_closing = self._previous
            dt = now_s - previous_time
            if previous_key == key and 0 < dt <= self.rules.max_gap_s:
                raw_closing = (previous_distance - result.distance_m) / dt
                closing = raw_closing if previous_closing is None else 0.5 * raw_closing + 0.5 * previous_closing
            elif previous_key == key and dt == 0:
                closing = previous_closing
        self._previous = (key, now_s, result.distance_m, closing)
        return result.model_copy(update={"closing_speed_mps": closing})


def recompute_spatial(snapshot: Snapshot, now_s: float, rules: RuleConfig) -> SpatialState:
    """Re-evaluate a trusted snapshot at consumption time without renewing TTL.

    A consumer cannot bootstrap geometry from an invalid original result or
    invent a SiteConfig. Registration/calibration/track versions and quality must
    still agree with the valid source snapshot. This function never mutates it.
    """
    motion, obstacle, original = snapshot.motion, snapshot.obstacle, snapshot.spatial

    def invalid(reason, stale=False):
        return _unknown(motion, obstacle, now_s, rules, reason, stale)

    if original.validity not in _USABLE or original.ttc_status == "UNKNOWN":
        return invalid("SOURCE_SPATIAL_INVALID")
    if now_s < snapshot.t_s or now_s < original.reference_time_s:
        return invalid("SOURCE_SNAPSHOT_FROM_FUTURE")
    if now_s >= original.valid_until_s:
        return invalid("SOURCE_SPATIAL_EXPIRED", stale=True)
    quality = snapshot.quality
    if quality.validity != Validity.VALID or quality.camera_stable is not True:
        return invalid("CAMERA_OR_IMAGE_QUALITY_INVALID")
    if not quality.t_s <= now_s < quality.valid_until_s:
        return invalid("QUALITY_STALE_OR_FROM_FUTURE", stale=now_s >= quality.valid_until_s)
    if (not original.calibration_id or original.calibration_id == "unconfigured"
            or original.calibration_id != snapshot.calibration_id
            or original.calibration_id != motion.calibration_id):
        return invalid("CALIBRATION_INVALID_OR_CHANGED")
    if not motion.track_epoch or motion.track_epoch != snapshot.track_epoch:
        return invalid("TRACK_EPOCH_CHANGED")
    if (not original.footprint_version or original.footprint_version == "unconfigured"
            or original.footprint_version != obstacle.footprint_version):
        return invalid("OBSTACLE_REGISTRATION_CHANGED")
    result = _predict(motion, obstacle, now_s, rules, original.closing_speed_mps)
    if result.validity in _USABLE:
        result = result.model_copy(update={"valid_until_s": min(result.valid_until_s, original.valid_until_s,
                                                                 quality.valid_until_s)})
    return result
