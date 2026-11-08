"""Confirm a manually measured footprint; never infer a footprint from a box."""
from __future__ import annotations

import math

from shapely.geometry import Polygon

from ..config import RuleConfig, SiteConfig
from ..schemas import ObstacleState, Validity


def _box_valid(box) -> bool:
    return (box is not None and len(box) == 4 and all(math.isfinite(float(v)) for v in box)
            and box[2] > box[0] and box[3] > box[1])


def _iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class ObstacleMonitor:
    """Movement/ambiguity is latched for this registered monitor's lifetime.

    To recover, the application must explicitly register the footprint again and
    construct a monitor with the new footprint_version. No detection can unlock it.
    """

    def __init__(self, site: SiteConfig, rules: RuleConfig):
        self.site = site
        self.rules = rules
        self._anchor = site.obstacle_anchor_bbox_px
        self._count = 0
        self._locked = False
        self._last_update = None
        self._state = ObstacleState(obstacle_id=site.obstacle_id,
                                    footprint_m=site.obstacle_footprint_m,
                                    footprint_version=site.footprint_version)
        footprint = Polygon(site.obstacle_footprint_m) if len(site.obstacle_footprint_m) >= 3 else None
        self._configured = (_box_valid(self._anchor) and footprint is not None and footprint.is_valid
                            and footprint.area > 0 and site.footprint_version != "unconfigured")

    def _compatible(self, box) -> bool:
        anchor = self._anchor
        centre_shift = math.hypot((box[0] + box[2] - anchor[0] - anchor[2]) / 2,
                                  (box[1] + box[3] - anchor[1] - anchor[3]) / 2)
        tolerance = max(2.0, 0.1 * min(anchor[2] - anchor[0], anchor[3] - anchor[1]))
        return _iou(box, anchor) >= self.rules.obstacle_iou and centre_shift <= tolerance

    def update(self, detections: list[dict], t_s: float) -> ObstacleState:
        if not math.isfinite(t_s):
            raise ValueError("t_s must be finite")
        if self._last_update is not None and t_s < self._last_update:
            self._locked = True
            self._state = self._state.model_copy(update=dict(status="MOVED_OR_AMBIGUOUS", validity=Validity.UNKNOWN,
                                                             valid_until_s=t_s, reasons=["SOURCE_TIME_REVERSED_REREGISTER_REQUIRED"]))
            return self._state.model_copy(deep=True)
        if self._last_update is not None and t_s == self._last_update:
            # Reprocessing a frame must not satisfy consecutive confirmations.
            return self.age(t_s)
        if self._last_update is not None and t_s - self._last_update >= self.rules.obstacle_ttl_s:
            self._count = 0
        self._last_update = t_s
        if not self._configured:
            self._state = self._state.model_copy(update=dict(t_s=t_s, reasons=["OBSTACLE_NOT_REGISTERED"]))
            return self._state.model_copy(deep=True)
        if self._locked:
            return self.age(t_s)
        candidates = []
        minimum_score = float(getattr(self.rules, "obstacle_min_score", 0.25))
        for detection in detections:
            try:
                score = float(detection.get("score", 0))
                box = detection.get("bbox_px")
                if detection.get("class_name") == "backpack" and math.isfinite(score) and score >= minimum_score and _box_valid(box):
                    candidates.append(tuple(float(v) for v in box))
            except (TypeError, ValueError):
                continue
        if len(candidates) > 1 or (candidates and not self._compatible(candidates[0])):
            self._locked = True
            self._count = 0
            self._state = self._state.model_copy(update=dict(t_s=t_s, status="MOVED_OR_AMBIGUOUS",
                validity=Validity.UNKNOWN, valid_until_s=t_s, bbox_px=candidates[0] if len(candidates) == 1 else None,
                reasons=["OBSTACLE_MOVED_OR_AMBIGUOUS_REREGISTER_REQUIRED"]))
        elif not candidates:
            self._count = 0
            if self._state.last_confirmed_s is not None and t_s < self._state.valid_until_s:
                self._state = self._state.model_copy(update=dict(t_s=t_s, status="TEMPORARILY_OCCLUDED",
                    validity=Validity.DEGRADED, bbox_px=None, reasons=["USING_UNEXPIRED_STATIC_REGISTRATION"]))
            else:
                self._state = self._state.model_copy(update=dict(t_s=t_s, status="STALE", validity=Validity.STALE,
                    bbox_px=None, reasons=["OBSTACLE_NOT_OBSERVED"]))
        else:
            self._count += 1
            already_confirmed = self._state.status == "CONFIRMED" and t_s < self._state.valid_until_s
            if self._count >= max(1, self.rules.obstacle_confirmations) or already_confirmed:
                self._state = self._state.model_copy(update=dict(t_s=t_s, last_confirmed_s=t_s,
                    valid_until_s=t_s + self.rules.obstacle_ttl_s, status="CONFIRMED", validity=Validity.VALID,
                    bbox_px=candidates[0], reasons=[]))
            else:
                unexpired = self._state.last_confirmed_s is not None and t_s < self._state.valid_until_s
                self._state = self._state.model_copy(update=dict(t_s=t_s,
                    status="TEMPORARILY_OCCLUDED" if unexpired else "STALE",
                    validity=Validity.DEGRADED if unexpired else Validity.UNKNOWN,
                    bbox_px=candidates[0], reasons=["AWAITING_CONSECUTIVE_CONFIRMATIONS"]))
        return self.age(t_s)

    def age(self, t_s: float) -> ObstacleState:
        if self._locked:
            return self._state.model_copy(deep=True)
        if self._state.last_confirmed_s is not None and t_s >= self._state.valid_until_s:
            self._state = self._state.model_copy(update=dict(status="STALE", validity=Validity.STALE,
                reasons=["OBSTACLE_CONFIRMATION_EXPIRED"]))
        return self._state.model_copy(deep=True)
