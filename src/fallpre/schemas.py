"""Versioned, JSON-safe contracts. All algorithm times are source-clock seconds."""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Validity(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"
    UNSUPPORTED = "UNSUPPORTED"


class Behavior(str, Enum):
    WALKING = "WALKING"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class QualityState(Contract):
    t_s: float = 0
    valid_until_s: float = 0
    validity: Validity = Validity.UNKNOWN
    camera_stable: bool | None = None
    dark_fraction: float | None = None
    bright_fraction: float | None = None
    reasons: list[str] = Field(default_factory=list)


class HumanObservation(Contract):
    t_s: float
    frame_id: int
    track_id: int
    track_epoch: str
    person_id: str | None = None
    bbox_px: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]  # COCO17 x,y,confidence
    score: float
    validity: Validity = Validity.VALID
    pose_format: str = "coco17"
    model_version: str = "yolo11n-pose"


class GroundObservation(Contract):
    t_s: float
    xy_m: tuple[float, float] | None = None
    uncertainty_m: float = 0
    calibration_id: str | None = None
    validity: Validity = Validity.UNKNOWN
    reasons: list[str] = Field(default_factory=list)


class MotionState(Contract):
    t_s: float = 0
    xy_m: tuple[float, float] | None = None
    velocity_mps: tuple[float, float] | None = None
    speed_mps: float | None = None
    uncertainty_m: float = 0
    calibration_id: str | None = None
    track_epoch: str | None = None
    validity: Validity = Validity.UNKNOWN
    reasons: list[str] = Field(default_factory=list)


class ObstacleState(Contract):
    obstacle_id: str = "registered_backpack_01"
    t_s: float = 0
    last_confirmed_s: float | None = None
    valid_until_s: float = 0
    status: str = "STALE"
    validity: Validity = Validity.UNKNOWN
    footprint_m: list[tuple[float, float]] = Field(default_factory=list)
    footprint_source: str = "MANUALLY_MEASURED"
    footprint_version: str = "unconfigured"
    bbox_px: tuple[float, float, float, float] | None = None
    reasons: list[str] = Field(default_factory=list)


class SpatialState(Contract):
    reference_time_s: float = 0
    computed_at_s: float = 0
    valid_until_s: float = 0
    validity: Validity = Validity.UNKNOWN
    distance_m: float | None = None
    clearance_m: float | None = None
    closing_speed_mps: float | None = None
    path_intersects: bool | None = None
    ttc_s: float | None = None
    ttc_status: str = "UNKNOWN"
    horizon_s: float = 3
    r_total_m: float | None = None
    calibration_id: str | None = None
    footprint_version: str | None = None
    reasons: list[str] = Field(default_factory=list)


class FallState(Contract):
    t_s: float = 0
    phase: str = "UNOBSERVABLE"
    candidate_since_s: float | None = None
    vertical_drop_norm: float | None = None
    tilt_deg: float | None = None
    aspect_ratio: float | None = None
    low_pose: bool | None = None
    low_pose_duration_s: float = 0
    had_upright: bool = False
    recovered: bool = False
    validity: Validity = Validity.UNKNOWN
    reasons: list[str] = Field(default_factory=list)


class Snapshot(Contract):
    snapshot_id: str = Field(default_factory=lambda: new_id("snap"))
    t_s: float
    frame_id: int = 0
    person_id: str
    binding_id: str
    binding_status: str = "BOUND"
    session_id: str = "unconfigured"
    source_type: str = "unspecified"
    pipeline_version: str = "fallpre-0.1.0"
    track_epoch: str | None = None
    calibration_id: str | None = None
    scene_version: str = "unconfigured"
    human: HumanObservation | None = None
    motion: MotionState = Field(default_factory=MotionState)
    behavior: Behavior = Behavior.UNKNOWN
    quality: QualityState = Field(default_factory=QualityState)
    obstacle: ObstacleState = Field(default_factory=ObstacleState)
    spatial: SpatialState = Field(default_factory=SpatialState)
    fall: FallState = Field(default_factory=FallState)
    personal_context: dict[str, Any] = Field(default_factory=dict)
    unsupported_capabilities: list[str] = Field(default_factory=lambda: ["wet_floor", "moving_obstacle", "stairs"])


class RiskDecision(Contract):
    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    event_id: str
    event_revision: int
    snapshot_id: str
    person_id: str
    binding_id: str
    branch: str
    outcome: str
    reasons: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    issued_at_s: float
    expires_at_s: float
    rule_version: str = "prototype-0.1"
    calibration_id: str | None = None
    scene_version: str = "unconfigured"
    confirmed_alarm: bool = False
    action_sequence: int = 0


class ExecutionResult(Contract):
    action_id: str = Field(default_factory=lambda: new_id("action"))
    decision_id: str
    event_id: str
    action_type: str
    idempotency_key: str
    status: str = "ACCEPTED"
    channel: str = "dry_run"
    accepted_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    delivery_receipt: dict[str, Any] | None = None
