"""YOLO pose observations with source-time, rather than tracker-frame, continuity."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..config import RuleConfig
from ..schemas import HumanObservation, new_id


def _values(value: Any) -> list:
    """Accept a tensor without importing torch in the schema/temporal pipeline."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class HumanPerceptor:
    def __init__(self, weights: str | Path, device: str = "cpu", rules: RuleConfig | None = None):
        self.weights = str(weights)
        self.device = device
        self.rules = rules or RuleConfig()
        self._model: Any = None
        self._last_t: float | None = None
        self._last_frame_id: int | None = None
        self._shape: tuple[int, ...] | None = None
        self._tracks: dict[int, tuple[float, str]] = {}

    def reset(self) -> None:
        """Call on input changes/rebinding even if a tracker might reuse its ID."""
        self._last_t = None
        self._last_frame_id = None
        self._shape = None
        self._tracks.clear()
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def infer(self, frame: Any, t_s: float, frame_id: int) -> list[HumanObservation]:
        if not math.isfinite(t_s):
            raise ValueError("source time must be finite")
        shape = tuple(frame.shape[:2])
        discontinuous = self._last_t is not None and (
            t_s <= self._last_t
            or t_s - self._last_t > self.rules.max_gap_s
            or frame_id <= self._last_frame_id
            or shape != self._shape
        )
        if discontinuous:
            self.reset()
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)

        results = self._model.track(
            frame, persist=True, tracker="bytetrack.yaml", device=self.device,
            imgsz=640, conf=0.1, verbose=False,
        )
        self._last_t, self._last_frame_id, self._shape = t_s, frame_id, shape
        # Tracker retention is measured in processed frames; enforce elapsed source time too.
        self._tracks = {k: v for k, v in self._tracks.items() if t_s - v[0] <= self.rules.max_gap_s}
        observations: list[HumanObservation] = []
        for result in results:
            boxes, points = result.boxes, result.keypoints
            if boxes is None or boxes.id is None or points is None:
                continue
            for track_id, bbox, score, pose in zip(
                _values(boxes.id), _values(boxes.xyxy), _values(boxes.conf), _values(points.data)
            ):
                if len(pose) != 17 or any(len(point) != 3 for point in pose):
                    continue
                flat = [track_id, score, *bbox, *(v for point in pose for v in point)]
                if not all(math.isfinite(float(v)) for v in flat):
                    continue
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                identity = int(track_id)
                previous = self._tracks.get(identity)
                epoch = previous[1] if previous is not None else new_id("track")
                self._tracks[identity] = (t_s, epoch)
                observations.append(HumanObservation(
                    t_s=t_s, frame_id=frame_id, track_id=identity, track_epoch=epoch,
                    bbox_px=tuple(float(v) for v in bbox),
                    keypoints=[tuple(float(v) for v in point) for point in pose],
                    score=float(score), model_version=Path(self.weights).name,
                ))
        return observations
