"""Site configuration is explicit; examples never authorize live execution."""
import json
from pathlib import Path

from pydantic import Field

from .schemas import Contract


class RuleConfig(Contract):
    version: str = "prototype-0.1"
    human_max_age_s: float = 0.5
    quality_ttl_s: float = 1.0
    obstacle_ttl_s: float = 2.0
    max_gap_s: float = 0.5
    horizon_s: float = 3.0
    warning_ttc_s: float = 2.0
    risk_enter_s: float = 0.2
    risk_clear_s: float = 0.8
    clear_distance_m: float = 0.6
    wait_timeout_s: float = 5.0
    reminder_ttl_s: float = 0.5
    walking_enter_mps: float = 0.12
    walking_exit_mps: float = 0.06
    behavior_dwell_s: float = 0.5
    person_radius_m: float = 0.25
    obstacle_error_m: float = 0.05
    safety_margin_m: float = 0.10
    stationary_speed_mps: float = 0.03
    keypoint_conf: float = 0.5
    fall_drop_norm: float = 0.35
    fall_tilt_deg: float = 55.0
    fall_aspect_ratio: float = 1.1
    fall_low_duration_s: float = 1.0
    fall_slow_duration_s: float = 2.0
    recovery_duration_s: float = 1.0
    obstacle_iou: float = 0.5
    obstacle_confirmations: int = 3
    baseline_min_clips: int = 5
    baseline_min_duration_s: float = 30.0
    dark_pixel_threshold: int = 30
    bright_pixel_threshold: int = 245
    max_dark_fraction: float = 0.7
    max_bright_fraction: float = 0.7
    camera_shift_px: float = 3.0
    camera_min_inliers: int = 12
    motion_acceleration_noise_mps2: float = 1.0
    obstacle_min_score: float = 0.25
    upright_tilt_deg: float = 25.0
    upright_reference_s: float = 0.5
    fall_height_ratio: float = 0.7
    calibration_max_error_m: float = 0.15


class SiteConfig(Contract):
    site_id: str = "unconfigured"
    scene_version: str = "unconfigured"
    person_id: str = "test_person_01"
    binding_id: str = "manual_binding_01"
    image_size: tuple[int, int]  # width, height
    calibration_id: str = "unconfigured"
    homography: list[list[float]] | None = None
    floor_polygon_px: list[tuple[float, float]] = Field(default_factory=list)
    validation_error_m: float = 0.0
    calibration_validated: bool = False
    calibration_image_points: list[tuple[float, float]] = Field(default_factory=list)
    calibration_world_points: list[tuple[float, float]] = Field(default_factory=list)
    validation_image_points: list[tuple[float, float]] = Field(default_factory=list)
    validation_world_points: list[tuple[float, float]] = Field(default_factory=list)
    camera_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    footpoint_error_m: float = 0.10
    obstacle_id: str = "registered_backpack_01"
    obstacle_footprint_m: list[tuple[float, float]] = Field(default_factory=list)
    obstacle_anchor_bbox_px: tuple[float, float, float, float] | None = None
    footprint_version: str = "unconfigured"
    background_reference: str | None = None
    background_roi_px: list[tuple[float, float]] = Field(default_factory=list)
    synthetic: bool = False


def load_site(path: str | Path) -> SiteConfig:
    site = SiteConfig.model_validate_json(Path(path).read_text())
    if site.background_reference and not Path(site.background_reference).is_absolute():
        site.background_reference = str((Path(path).resolve().parent / site.background_reference).resolve())
    return site


def load_rules(path: str | Path | None = None) -> RuleConfig:
    return RuleConfig.model_validate(json.loads(Path(path).read_text())) if path else RuleConfig()
