"""Synthetic algorithm integration fixture. These are NOT model accuracy examples."""
from __future__ import annotations

import json
from pathlib import Path

from .config import RuleConfig, SiteConfig
from .pipeline import ObservationPipeline
from .schemas import HumanObservation, QualityState, Validity


def synthetic_site() -> SiteConfig:
    image = [(50, 150), (300, 150), (590, 150), (50, 450), (300, 450), (590, 450)]
    validation = [(150, 250), (450, 350)]
    return SiteConfig(
        site_id="synthetic_only", scene_version="synthetic-v1", image_size=(640, 480),
        calibration_id="synthetic-h01", calibration_validated=True,
        homography=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
        calibration_image_points=image, calibration_world_points=[(x / 100, y / 100) for x, y in image],
        validation_image_points=validation, validation_world_points=[(x / 100, y / 100) for x, y in validation],
        validation_error_m=0.001, floor_polygon_px=[(20, 100), (620, 100), (620, 470), (20, 470)],
        footpoint_error_m=0.05, obstacle_anchor_bbox_px=(400, 370, 450, 450),
        obstacle_footprint_m=[(4, 4.1), (4.5, 4.1), (4.5, 4.5), (4, 4.5)],
        footprint_version="synthetic-footprint-01", synthetic=True,
    )


def synthetic_human(t_s: float, frame_id: int, x: float = 200, *, low: bool = False,
                    epoch: str = "synthetic-track-01") -> HumanObservation:
    points = [(x, 140, 0.99)] * 17
    if low:
        for i, point in {5: (x-55, 370), 6: (x-25, 370), 11: (x+15, 395),
                         12: (x+45, 395), 13: (x+65, 415), 14: (x+85, 415),
                         15: (x+85, 435), 16: (x+105, 435)}.items():
            points[i] = (*point, 0.99)
        box = (x-70, 350, x+115, 445)
    else:
        for i, point in {5: (x-15, 170), 6: (x+15, 170), 11: (x-12, 260),
                         12: (x+12, 260), 13: (x-10, 340), 14: (x+10, 340),
                         15: (x-10, 430), 16: (x+10, 430)}.items():
            points[i] = (*point, 0.99)
        box = (x-30, 130, x+30, 445)
    return HumanObservation(t_s=t_s, frame_id=frame_id, track_id=1, track_epoch=epoch,
                            bbox_px=box, keypoints=points, score=0.99, model_version="SYNTHETIC_NOT_YOLO")


def synthetic_quality(t_s: float) -> QualityState:
    return QualityState(t_s=t_s, valid_until_s=t_s + 1, validity=Validity.VALID,
                        camera_stable=True, dark_fraction=0, bright_fraction=0,
                        reasons=["SYNTHETIC_NOT_CAMERA_VALIDATION"])


def run_demo(output: str | Path) -> dict:
    from .service import MVPService
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        raise FileExistsError("Choose a new output directory for a new replay session")
    site, rules = synthetic_site(), RuleConfig()
    (directory / "site.synthetic.json").write_text(site.model_dump_json(indent=2) + "\n")
    summary = {"data_kind": "synthetic_observations_not_model_accuracy", "dry_run": True, "scenarios": {}}
    for scenario in ("approach_stop_retreat", "fall_after_upright", "initial_low_pose_review"):
        scenario_dir = directory / scenario
        scenario_dir.mkdir(exist_ok=True)
        db = scenario_dir / "events.sqlite3"
        if db.exists():
            raise FileExistsError(db)
        service = MVPService(rules, db_path=str(db), dry_run=True)
        core = ObservationPipeline(site, rules, track_id=1, source_type="synthetic")
        decisions = []
        observed = {"spatial": set(), "fall": set(), "behavior": set()}
        count = 135 if scenario == "approach_stop_retreat" else 85
        with (scenario_dir / "snapshots.jsonl").open("w") as stream:
            for i in range(count):
                t_s = i / 10
                if scenario == "approach_stop_retreat":
                    x = 150 + min(i, 55) * 4 if i < 80 else max(150, 370 - (i-80) * 5)
                    human = synthetic_human(t_s, i, x)
                else:
                    human = synthetic_human(t_s, i, low=scenario == "initial_low_pose_review" or i >= 15)
                snapshot = core.update([human], [{"class_name": "backpack", "score": 0.99,
                                                 "bbox_px": site.obstacle_anchor_bbox_px}],
                                       synthetic_quality(t_s), t_s, i, site.image_size)
                new_decisions = service.ingest_snapshot(snapshot, auto_execute=True)
                core.feedback(new_decisions)
                decisions.extend(new_decisions)
                observed["spatial"].add(snapshot.spatial.ttc_status)
                observed["fall"].add(snapshot.fall.phase)
                observed["behavior"].add(snapshot.behavior.value)
                stream.write(snapshot.model_dump_json() + "\n")
            decisions.extend(service.tick(count / 10 + rules.wait_timeout_s + 0.1))
        with (scenario_dir / "decisions.jsonl").open("w") as stream:
            for decision in {d.decision_id: d for d in decisions}.values():
                stream.write(decision.model_dump_json() + "\n")
        summary["scenarios"][scenario] = {
            "snapshots": count, "outcomes": sorted({d.outcome for d in decisions}),
            "allowed_actions": sorted({a for d in decisions for a in d.allowed_actions}),
            "observed_states": {k: sorted(v) for k, v in observed.items()},
            "database": str(db.resolve()),
        }
        service.close()
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
