"""Planar geometry in the configured, undistorted image coordinate system."""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from ..config import RuleConfig, SiteConfig
from ..schemas import GroundObservation, HumanObservation, QualityState, Validity


def _points(values, minimum: int, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"{name} requires at least {minimum} two-dimensional points")
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name} must contain finite, distinct points")
    if len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError(f"{name} must not be collinear")
    return points


def fit_calibration(train_img, train_world, val_img, val_world,
                    max_error_m: float) -> tuple[list[list[float]], float]:
    """Fit H with >=6 correspondences; return (H, max independent error in m).

    All image points must already be undistorted. Validation points must not be
    reused training points. Invalid geometry or failed validation raises ValueError.
    """
    image = _points(train_img, 6, "train_img")
    world = _points(train_world, 6, "train_world")
    check_image = _points(val_img, 2, "val_img")
    check_world = _points(val_world, 2, "val_world")
    if len(image) != len(world) or len(check_image) != len(check_world):
        raise ValueError("Image/world correspondence counts differ")
    if not math.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("max_error_m must be finite and positive")
    if np.any(np.linalg.norm(check_image[:, None] - image[None, :], axis=2) < 1e-6):
        raise ValueError("Validation points must be independent of training points")
    h, _ = cv2.findHomography(image, world, method=0)
    if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
        raise ValueError("Cannot fit a nonsingular homography")
    projected_h = np.c_[check_image, np.ones(len(check_image))] @ h.T
    if np.any(np.abs(projected_h[:, 2]) < 1e-10):
        raise ValueError("Validation point lies on the projective horizon")
    projected = projected_h[:, :2] / projected_h[:, 2:]
    error = float(np.linalg.norm(projected - check_world, axis=1).max())
    if not math.isfinite(error) or error > max_error_m:
        raise ValueError(f"Independent calibration error {error:.4f} m exceeds {max_error_m:.4f} m")
    return h.tolist(), error


class Calibration:
    """Fail-closed projection of a deliberately approximate double-ankle proxy.

    The upright gate uses the current pose alone, never a downstream behavior
    label. Its prototype thresholds require validation on the intended view.
    """

    def __init__(self, site: SiteConfig, rules: RuleConfig | None = None):
        self.site = site
        self.rules = rules or RuleConfig()
        self._h = np.asarray(site.homography, dtype=float) if site.homography is not None else None
        self._floor = Polygon(site.floor_polygon_px) if len(site.floor_polygon_px) >= 3 else None
        self._evidence_error = self._check_evidence()

    def _check_evidence(self) -> str | None:
        """The validation flag alone is not sufficient calibration evidence."""
        try:
            limit = float(getattr(self.rules, "calibration_max_error_m", 0.15))
            fit_calibration(self.site.calibration_image_points, self.site.calibration_world_points,
                            self.site.validation_image_points, self.site.validation_world_points, limit)
            if self._h is None or self._h.shape != (3, 3) or not np.isfinite(self._h).all():
                return "INVALID_HOMOGRAPHY"
            image = np.asarray(self.site.validation_image_points, dtype=float)
            world = np.asarray(self.site.validation_world_points, dtype=float)
            projected = np.c_[image, np.ones(len(image))] @ self._h.T
            if np.any(np.abs(projected[:, 2]) < 1e-10):
                return "INVALID_HOMOGRAPHY"
            residual = float(np.linalg.norm(projected[:, :2] / projected[:, 2:] - world, axis=1).max())
            reported = self.site.validation_error_m
            if reported < 0 or not math.isfinite(reported) or residual > limit or residual > reported + 1e-6:
                return "CALIBRATION_EVIDENCE_MISMATCH"
            if self.site.footpoint_error_m < 0:
                return "FOOTPOINT_ERROR_NOT_VALIDATED"
        except (TypeError, ValueError, cv2.error, np.linalg.LinAlgError):
            return "CALIBRATION_EVIDENCE_MISSING_OR_INVALID"
        return None

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.site.camera_matrix is None and self.site.distortion_coefficients is None:
            return frame
        if self.site.camera_matrix is None or self.site.distortion_coefficients is None:
            raise ValueError("Both camera_matrix and distortion_coefficients are required")
        if (frame.shape[1], frame.shape[0]) != self.site.image_size:
            raise ValueError("Cannot undistort an image of a different calibration size")
        matrix = np.asarray(self.site.camera_matrix, dtype=np.float64)
        coefficients = np.asarray(self.site.distortion_coefficients, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(coefficients).all():
            raise ValueError("Invalid camera intrinsics")
        return cv2.undistort(frame, matrix, coefficients, None, matrix)

    def project(self, human: HumanObservation, quality: QualityState,
                image_size: tuple[int, int]) -> GroundObservation:
        base = dict(t_s=human.t_s, calibration_id=self.site.calibration_id,
                    uncertainty_m=max(0.0, self.site.validation_error_m) + max(0.0, self.site.footpoint_error_m))

        def invalid(reason: str) -> GroundObservation:
            return GroundObservation(**base, reasons=[reason])

        if not self.site.calibration_validated or self._h is None:
            return invalid("CALIBRATION_NOT_VALIDATED")
        if self._evidence_error is not None:
            return invalid(self._evidence_error)
        if self._h.shape != (3, 3) or not np.isfinite(self._h).all() or np.linalg.matrix_rank(self._h) < 3:
            return invalid("INVALID_HOMOGRAPHY")
        if tuple(image_size) != self.site.image_size:
            return invalid("IMAGE_SIZE_CHANGED")
        if self._floor is None or not self._floor.is_valid or self._floor.area <= 0:
            return invalid("INVALID_FLOOR_ROI")
        if quality.camera_stable is not True or quality.validity != Validity.VALID:
            return invalid("CAMERA_OR_IMAGE_QUALITY_INVALID")
        if not quality.t_s <= human.t_s < quality.valid_until_s:
            return invalid("QUALITY_STALE_OR_FROM_FUTURE")
        if human.validity != Validity.VALID or human.pose_format != "coco17" or len(human.keypoints) != 17:
            return invalid("POSE_UNAVAILABLE")
        points = np.asarray(human.keypoints, dtype=float)
        required = [5, 6, 11, 12, 13, 14, 15, 16]
        if not np.isfinite(points[required]).all() or np.any(points[required, 2] < self.rules.keypoint_conf):
            return invalid("REQUIRED_KEYPOINTS_UNRELIABLE")
        width, height = image_size
        x1, y1, x2, y2 = human.bbox_px
        if not np.isfinite(human.bbox_px).all() or x2 <= x1 or y2 <= y1:
            return invalid("INVALID_PERSON_BOX")
        ankles = points[[15, 16], :2]
        if y2 >= height - 1 or np.any(ankles[:, 0] <= 1) or np.any(ankles[:, 0] >= width - 1) or np.any(ankles[:, 1] >= height - 1):
            return invalid("LOWER_BODY_CROPPED")
        if np.any(ankles[:, 0] < x1) or np.any(ankles[:, 0] > x2) or np.any(ankles[:, 1] < y1) or np.any(ankles[:, 1] > y2):
            return invalid("ANKLES_OUTSIDE_PERSON_BOX")
        shoulder = points[[5, 6], :2].mean(axis=0)
        hip = points[[11, 12], :2].mean(axis=0)
        ankle = ankles.mean(axis=0)
        trunk = hip - shoulder
        body_height = y2 - y1
        tilt = math.degrees(math.atan2(abs(float(trunk[0])), max(float(trunk[1]), 1e-9)))
        if trunk[1] <= 0 or tilt > 35 or ankle[1] - hip[1] < 0.25 * body_height:
            return invalid("POSE_NOT_IN_VALIDATED_UPRIGHT_RANGE")
        if abs(float(ankles[0, 1] - ankles[1, 1])) > 0.12 * body_height:
            return invalid("ANKLE_HEIGHTS_INCOMPATIBLE")
        for hip_i, knee_i, ankle_i in ((11, 13, 15), (12, 14, 16)):
            a = points[hip_i, :2] - points[knee_i, :2]
            b = points[ankle_i, :2] - points[knee_i, :2]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            knee_angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / denominator, -1, 1)))) if denominator > 1e-9 else 0
            if knee_angle < 135:
                return invalid("LOWER_LIMB_NOT_EXTENDED")
        if not all(self._floor.covers(Point(float(p[0]), float(p[1]))) for p in [*ankles, ankle]):
            return invalid("OUTSIDE_VALIDATED_FLOOR_ROI")
        homogeneous = self._h @ np.array([ankle[0], ankle[1], 1.0])
        if abs(float(homogeneous[2])) < 1e-10:
            return invalid("PROJECTIVE_HORIZON")
        xy = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(xy).all():
            return invalid("NONFINITE_PROJECTION")
        return GroundObservation(**base, xy_m=(float(xy[0]), float(xy[1])),
                                 validity=Validity.VALID, reasons=["VALIDATED_DOUBLE_ANKLE_PROXY"])
