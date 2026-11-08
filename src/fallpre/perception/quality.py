"""Raw-image quality and fail-closed comparison with a saved fixed background."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import RuleConfig, SiteConfig
from ..schemas import QualityState, Validity


def _polygon_mask(shape: tuple[int, int], points) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


class QualityMonitor:
    """Background references and ROI are in the undistorted frame coordinates.

    Missing/textureless references produce UNKNOWN, never an implicit stable
    camera. Confirmed movement is latched until this site is explicitly reset.
    """

    def __init__(self, site: SiteConfig, rules: RuleConfig):
        self.site = site
        self.rules = rules
        self._moved = False
        self._last_t = None
        self._orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=15, fastThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reference = cv2.imread(site.background_reference, cv2.IMREAD_GRAYSCALE) if site.background_reference else None
        self._reference_error = None
        if self._reference is None:
            self._reference_error = "BACKGROUND_REFERENCE_UNAVAILABLE"
        elif (self._reference.shape[1], self._reference.shape[0]) != site.image_size:
            self._reference_error = "BACKGROUND_REFERENCE_SIZE_MISMATCH"
        self._background_mask = _polygon_mask((site.image_size[1], site.image_size[0]), site.background_roi_px)
        self._lighting_mask = _polygon_mask((site.image_size[1], site.image_size[0]), site.floor_polygon_px)

    def update(self, frame: np.ndarray, t_s: float, exclude_boxes=None) -> QualityState:
        if not math.isfinite(t_s):
            raise ValueError("t_s must be finite")
        base = dict(t_s=t_s, valid_until_s=t_s + self.rules.quality_ttl_s)
        if frame is None or frame.size == 0:
            return QualityState(**base, reasons=["EMPTY_FRAME"])
        if (frame.shape[1], frame.shape[0]) != self.site.image_size:
            self._moved = True
            return QualityState(**base, camera_stable=False, reasons=["IMAGE_SIZE_CHANGED_RECALIBRATE_REQUIRED"])
        if self._last_t is not None and t_s < self._last_t:
            return QualityState(**base, reasons=["SOURCE_TIME_REVERSED"])
        self._last_t = t_s
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_mask = self._background_mask.copy()
        lighting_mask = self._lighting_mask.copy()
        for box in exclude_boxes or []:
            if len(box) != 4 or not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            if x2 <= x1 or y2 <= y1:
                continue
            # Apply the same exclusion in reference and current image so moving
            # subjects cannot become matches used to infer camera motion.
            cv2.rectangle(background_mask, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), 0, -1)
            cv2.rectangle(lighting_mask, (x1, y1), (x2, y2), 0, -1)
        values = gray[lighting_mask > 0]
        reasons = []
        light_valid = True
        if len(values) < 64:
            reasons.append("LIGHTING_ROI_UNOBSERVABLE")
            light_valid = False
        else:
            dark = float(np.mean(values < getattr(self.rules, "dark_pixel_threshold", 30)))
            bright = float(np.mean(values > getattr(self.rules, "bright_pixel_threshold", 245)))
            base.update(dark_fraction=dark, bright_fraction=bright)
            if dark > getattr(self.rules, "max_dark_fraction", 0.7):
                reasons.append("LOW_LIGHT")
                light_valid = False
            if bright > getattr(self.rules, "max_bright_fraction", 0.7):
                reasons.append("OVEREXPOSED")
                light_valid = False
        if self._moved:
            return QualityState(**base, camera_stable=False, reasons=reasons + ["CAMERA_MOVED_RECALIBRATE_REQUIRED"])
        if self._reference_error is not None:
            return QualityState(**base, reasons=reasons + [self._reference_error])
        if not light_valid:
            return QualityState(**base, reasons=reasons + ["CAMERA_CHECK_UNRELIABLE"])
        if np.count_nonzero(background_mask) < 64:
            return QualityState(**base, reasons=["BACKGROUND_ROI_UNOBSERVABLE"])
        key_ref, desc_ref = self._orb.detectAndCompute(self._reference, background_mask)
        key_now, desc_now = self._orb.detectAndCompute(gray, background_mask)
        minimum = max(4, int(getattr(self.rules, "camera_min_inliers", 12)))
        if desc_ref is None or desc_now is None or len(desc_ref) < minimum or len(desc_now) < minimum:
            return QualityState(**base, reasons=["INSUFFICIENT_BACKGROUND_FEATURES"])
        pairs = self._matcher.knnMatch(desc_ref, desc_now, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        if len(matches) < minimum:
            return QualityState(**base, reasons=["INSUFFICIENT_BACKGROUND_MATCHES"])
        origin = np.float32([key_ref[match.queryIdx].pt for match in matches])
        current = np.float32([key_now[match.trainIdx].pt for match in matches])
        transform, inliers = cv2.findHomography(origin, current, cv2.RANSAC, 2.0)
        if (transform is None or inliers is None or int(inliers.sum()) < minimum
                or float(inliers.mean()) < 0.6 or not np.isfinite(transform).all()):
            return QualityState(**base, reasons=["BACKGROUND_GEOMETRY_UNRELIABLE"])
        # Inspect displacement across the registered background region, so scale
        # changes or rotations are not hidden by small motion near its centre.
        samples = np.asarray(self.site.background_roi_px, dtype=float)
        samples = np.vstack([samples, samples.mean(axis=0)])
        homogeneous = np.c_[samples, np.ones(len(samples))] @ transform.T
        if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            return QualityState(**base, reasons=["BACKGROUND_GEOMETRY_UNRELIABLE"])
        projected = homogeneous[:, :2] / homogeneous[:, 2:]
        displacement = float(np.linalg.norm(projected - samples, axis=1).max())
        if displacement > getattr(self.rules, "camera_shift_px", 3.0):
            self._moved = True
            return QualityState(**base, camera_stable=False, reasons=["CAMERA_MOVED_RECALIBRATE_REQUIRED"])
        return QualityState(**base, camera_stable=True, validity=Validity.VALID)
