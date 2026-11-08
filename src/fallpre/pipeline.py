"""Shared observations -> one explicitly bound person -> immutable service snapshots."""
from __future__ import annotations

import hashlib
import queue
from pathlib import Path

from .baseline import PersonalBaseline
from .config import RuleConfig, SiteConfig
from .geometry.calibration import Calibration
from .geometry.motion import MotionEstimator
from .geometry.spatial import SpatialCalculator
from .perception.obstacle import ObstacleMonitor
from .perception.quality import QualityMonitor
from .schemas import GroundObservation, HumanObservation, QualityState, Snapshot, Validity, new_id
from .temporal.behavior import BehaviorEstimator
from .temporal.fall_evidence import FallDetector


class ObservationPipeline:
    def __init__(self, site: SiteConfig, rules: RuleConfig, track_id: int | None, *, source_type="video"):
        self.site, self.rules = site, rules
        self.session_id = new_id("session")
        self.binding_id = f"{site.binding_id}:{self.session_id}"
        self.source_type = source_type
        self.requested_track_id = track_id
        self.bound_epoch: str | None = None
        self.binding_status = "UNBOUND"
        self.last_seen_s: float | None = None
        self.last_t_s: float | None = None
        self.calibration = Calibration(site, rules)
        self.motion = MotionEstimator(rules)
        self.spatial = SpatialCalculator(site, rules)
        self.obstacle = ObstacleMonitor(site, rules)
        self.behavior = BehaviorEstimator(rules)
        self.fall = FallDetector(rules)
        self.baseline = PersonalBaseline(rules)
        self._feedback: queue.SimpleQueue = queue.SimpleQueue()
        self.version = "fallpre-0.1.0:" + hashlib.sha256(rules.model_dump_json().encode()).hexdigest()[:12]

    def _bind(self, humans: list[HumanObservation], t_s: float) -> HumanObservation | None:
        if self.binding_status == "SUSPENDED" or self.requested_track_id is None:
            return None
        candidates = [h for h in humans if h.track_id == self.requested_track_id]
        if len(candidates) != 1:
            if self.last_seen_s is not None and t_s - self.last_seen_s > self.rules.max_gap_s:
                self.binding_status = "SUSPENDED"
            return None
        human = candidates[0]
        if self.bound_epoch is None:
            self.bound_epoch = human.track_epoch
            self.binding_status = "BOUND"
        elif human.track_epoch != self.bound_epoch or (
            self.last_seen_s is not None and t_s - self.last_seen_s > self.rules.max_gap_s
        ):
            self.binding_status = "SUSPENDED"
            return None
        self.last_seen_s = t_s
        return human.model_copy(update={"person_id": self.site.person_id}, deep=True)

    def update(self, humans: list[HumanObservation], detections: list[dict] | None,
               quality: QualityState, t_s: float, frame_id: int, image_size: tuple[int, int]) -> Snapshot:
        if any(abs(h.t_s - t_s) > 1e-6 or h.frame_id != frame_id for h in humans):
            raise ValueError("Human observations must come from the current source frame")
        if self.last_t_s is not None and t_s <= self.last_t_s:
            raise ValueError("New source-time epoch requires a new pipeline/session")
        self.last_t_s = t_s
        if quality.t_s > t_s:
            quality = quality.model_copy(update={"validity": Validity.UNKNOWN, "camera_stable": None,
                                                 "reasons": [*quality.reasons, "quality_from_future"]})
        if quality.camera_stable is False:
            self.binding_status = "SUSPENDED"
        human = self._bind(humans, t_s)
        ground = (self.calibration.project(human, quality, image_size) if human else
                  GroundObservation(t_s=t_s, reasons=["bound_person_not_observable"]))
        motion = self.motion.update(ground, human.track_epoch if human else None)
        obstacle = self.obstacle.update(detections, t_s) if detections is not None else self.obstacle.age(t_s)
        if quality.validity != Validity.VALID or quality.valid_until_s <= t_s or quality.camera_stable is not True:
            obstacle = obstacle.model_copy(update={"validity": Validity.UNKNOWN,
                                                   "reasons": [*obstacle.reasons, "scene_quality_unavailable"]})
        spatial = self.spatial.compute(motion, obstacle, t_s)
        behavior = self.behavior.update(human, motion, t_s)
        stable = quality.camera_stable if quality.validity == Validity.VALID and quality.valid_until_s > t_s else None
        fall = self.fall.update(human, t_s, camera_stable=stable)
        snapshot = Snapshot(t_s=t_s, frame_id=frame_id, person_id=self.site.person_id,
                            binding_id=self.binding_id, binding_status=self.binding_status,
                            session_id=self.session_id, source_type=self.source_type, pipeline_version=self.version,
                            track_epoch=human.track_epoch if human else None,
                            calibration_id=self.site.calibration_id, scene_version=self.site.scene_version,
                            human=human, motion=motion, behavior=behavior, quality=quality,
                            obstacle=obstacle, spatial=spatial, fall=fall)
        intervention_pending = exclusion_pending = False
        while not self._feedback.empty():
            intervention, excluded = self._feedback.get_nowait()
            intervention_pending |= intervention
            exclusion_pending |= excluded
        context = self.baseline.update(snapshot, excluded=exclusion_pending,
                                       intervention=intervention_pending)
        return snapshot.model_copy(update={"personal_context": context}, deep=True)

    def feedback(self, decisions):
        self._feedback.put((any(d.allowed_actions for d in decisions),
                            any(d.outcome in ("WAIT", "REMIND", "ALARM", "REVIEW_REQUIRED") for d in decisions)))


class VideoPipeline:
    def __init__(self, site: SiteConfig, rules: RuleConfig, track_id: int | None,
                 pose_weights: str, detect_weights: str, device="cpu", source_type="video"):
        from .models import BackpackDetector
        from .perception.human import HumanPerceptor
        self.core = ObservationPipeline(site, rules, track_id, source_type=source_type)
        self.human = HumanPerceptor(pose_weights, device=device, rules=rules)
        self.detector = BackpackDetector(detect_weights, device=device)
        fingerprint = hashlib.sha256()
        for weights in (pose_weights, detect_weights):
            fingerprint.update(Path(weights).read_bytes())
        self.core.version += ":weights-" + fingerprint.hexdigest()[:12]
        self.quality = QualityMonitor(site, rules)
        self._last_environment_s: float | None = None
        self._last_quality: QualityState | None = None

    def warmup(self):
        """Do not let first-inference startup latency break a live subject binding."""
        import numpy as np
        width, height = self.core.site.image_size
        frame = np.full((height, width, 3), 127, dtype=np.uint8)
        for i in range(2):
            self.human.infer(frame, i / 10, i)
            self.detector.infer(frame)
        self.human.reset()

    def process(self, frame):
        image = self.core.calibration.undistort(frame.image)
        humans = self.human.infer(image, frame.t_s, frame.frame_id)
        environment_due = self._last_environment_s is None or frame.t_s - self._last_environment_s >= 0.5
        detections = self.detector.infer(image) if environment_due else None
        if environment_due:
            boxes = [h.bbox_px for h in humans] + [d["bbox_px"] for d in detections]
            self._last_quality = self.quality.update(image, frame.t_s, exclude_boxes=boxes)
            self._last_environment_s = frame.t_s
        snapshot = self.core.update(humans, detections, self._last_quality, frame.t_s, frame.frame_id,
                                    (image.shape[1], image.shape[0]))
        return snapshot, image, humans
