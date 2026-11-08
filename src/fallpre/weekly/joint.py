"""Real RGB-D people -> measured floor projection -> visible obstacles within 2 m.

TUM supplies registered sensor depth and camera calibration, never human poses
or semantic labels. The floor is fitted to SegFormer-predicted floor support.
The public sample is oblique, not a ceiling-camera validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .human import LINKS, POSE_SHA256, _local_import_paths, open_video_writer, sha256
from .scene import DEFAULT_MODEL, MODEL_FILES_SHA256, MODEL_ID, MODEL_REVISION, analyze_scene

ROOT = Path(__file__).resolve().parents[3]
SEQUENCE = "rgbd_dataset_freiburg3_walking_static"
DEFAULT_ASSETS = ROOT / "assets/weekly/joint" / SEQUENCE
CALIBRATION = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
               "depth_units_per_m": 5000.0, "width": 640, "height": 480,
               "distortion": [0, 0, 0, 0, 0]}
SOURCES = {"dataset": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
           "calibration": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
           "pose": "https://docs.ultralytics.com/tasks/pose/",
           "semantics": "https://github.com/NVlabs/SegFormer",
           "plane": "https://doi.org/10.1145/358669.358692"}


def read_timestamps(path: Path) -> list[tuple[float, str]]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    result = [(float(row[0]), row[1]) for row in rows]
    if any(not np.isfinite(t) for t, _ in result) or any(b[0] <= a[0] for a, b in zip(result, result[1:])):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if any(Path(p).is_absolute() or ".." in Path(p).parts for _, p in result):
        raise ValueError("Timestamp file contains an unsafe path")
    return result


def associate_rgb_depth(rgb: list[tuple[float, str]], depth: list[tuple[float, str]],
                        max_difference_s: float = .02) -> list[dict]:
    """Globally nearest admissible one-to-one pairs, with source index retained."""
    if not math.isfinite(max_difference_s) or max_difference_s <= 0:
        raise ValueError("Invalid synchronization tolerance")
    candidates = []
    times = np.array([row[0] for row in depth])
    for i, (t, _) in enumerate(rgb):
        lo = np.searchsorted(times, t - max_difference_s, side="left")
        hi = np.searchsorted(times, t + max_difference_s, side="right")
        candidates.extend((abs(t - depth[j][0]), i, j) for j in range(lo, hi))
    used_rgb, used_depth, matches = set(), set(), []
    for delta, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            used_rgb.add(i)
            used_depth.add(j)
            matches.append({"source_frame_id": i, "rgb_timestamp_s": rgb[i][0],
                            "depth_timestamp_s": depth[j][0], "sync_error_s": delta,
                            "rgb_path": rgb[i][1], "depth_path": depth[j][1]})
    return sorted(matches, key=lambda r: r["source_frame_id"])


def depth_to_xyz(depth: np.ndarray, calibration: dict = CALIBRATION) -> np.ndarray:
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Expected original uint16 TUM depth PNG, not an 8-bit preview")
    h, w = depth.shape
    if (w, h) != (calibration["width"], calibration["height"]):
        raise ValueError("Depth dimensions do not match calibration")
    fx, fy, scale = (calibration[k] for k in ("fx", "fy", "depth_units_per_m"))
    if not np.isfinite([fx, fy, scale]).all() or min(fx, fy, scale) <= 0:
        raise ValueError("Invalid metric calibration")
    y, x = np.indices(depth.shape)
    z = depth.astype(np.float64) / scale
    xyz = np.stack(((x - calibration["cx"]) * z / fx, (y - calibration["cy"]) * z / fy, z), axis=-1)
    xyz[depth == 0] = np.nan
    return xyz


def fit_floor(xyz: np.ndarray, labels: np.ndarray, confidence: np.ndarray, *,
              threshold_m: float = .035, min_support: int = 500,
              mounting: str = "upright") -> dict:
    """RANSAC/SVD on learned floor, never the largest arbitrary scene plane.

    Mounting is an explicit calibration prior, not inferred from semantic GT.
    Upright/oblique uses camera -Y; ceiling uses camera -Z and has no lower-image
    requirement. Ground normal is inferred from visible floor, not an IMU.
    """
    if xyz.shape != (*labels.shape, 3) or labels.shape != confidence.shape:
        raise ValueError("Plane support needs aligned labels, confidence and XYZ")
    if not np.isfinite(threshold_m) or threshold_m <= 0 or min_support < 3:
        raise ValueError("Plane residual threshold must be positive with at least three support points")
    if mounting not in {"upright", "ceiling"}:
        raise ValueError("mounting must be upright or ceiling")
    prior = np.array([0., -1., 0.]) if mounting == "upright" else np.array([0., 0., -1.])
    min_height, max_height = (.4, 2.8) if mounting == "upright" else (1.5, 5.)
    valid = np.isfinite(xyz).all(axis=-1)
    support_mask = valid & (labels == 3) & (confidence >= .45)
    yy, _ = np.where(support_mask)
    points = xyz[support_mask]
    if len(points) < min_support:
        raise ValueError("Insufficient semantic floor/depth support")
    if mounting == "upright" and np.mean(yy >= labels.shape[0] * .6) < .5:
        raise ValueError("Floor support is not in the lower image for this upright-camera adapter")
    rng = np.random.default_rng(17)
    sample = points[rng.choice(len(points), min(len(points), 5000), replace=False)]
    best = None
    for _ in range(220):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        d = -float(normal @ a)
        if d < 0:
            normal, d = -normal, -d
        if normal @ prior < .5 or not min_height <= d <= max_height:
            continue
        residual = np.abs(sample @ normal + d)
        inliers = residual <= threshold_m
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), normal, d)
    if best is None:
        raise ValueError("No floor plane compatible with declared camera mounting prior")
    residual = np.abs(points @ best[1] + best[2])
    inliers = residual <= threshold_m
    if inliers.sum() < min_support or inliers.mean() < .65:
        raise ValueError("Predicted floor is not supported by one measured plane")
    selected = points[inliers]
    centroid = selected.mean(axis=0)
    normal = np.linalg.svd(selected - centroid, full_matrices=False)[2][-1]
    d = -float(normal @ centroid)
    if d < 0:
        normal, d = -normal, -d
    residual = np.abs(selected @ normal + d)
    if normal @ prior < .5 or not min_height <= d <= max_height or np.percentile(residual, 95) > threshold_m * 1.5:
        raise ValueError("Refined floor failed orientation/height/residual validation")
    xaxis = np.array([1., 0., 0.]) - normal * normal[0]
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(normal, xaxis)
    basis = np.stack([xaxis, yaxis, normal], axis=1)
    return {"normal_camera": normal.tolist(), "offset_m": d, "basis_columns": basis.tolist(),
            "camera_height_m": d, "downward_pitch_degrees": float(np.degrees(np.arcsin(-normal[2]))),
            "method": "RANSAC_then_SVD_on_model_predicted_floor_registered_sensor_XYZ",
            "mounting": mounting,
            "orientation_prior": f"floor normal dot camera {'-Y' if mounting == 'upright' else '-Z'} >= 0.5",
            "support_pixels": len(points), "inlier_pixels": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()), "median_residual_m": float(np.median(residual)),
            "p95_residual_m": float(np.percentile(residual, 95)),
            "gravity_source": "inferred_floor_normal_not_IMU_or_ground_truth"}


def to_ground(xyz: np.ndarray, plane: dict) -> np.ndarray:
    result = xyz @ np.array(plane["basis_columns"])
    result[..., 2] += plane["offset_m"]
    return result


def person_root(person: dict, xyz: np.ndarray, labels: np.ndarray, plane: dict) -> dict:
    """Pelvis surface proxy from measured depths around detected left/right hips."""
    points = np.asarray(person["kpts"])
    if any(points[i, 2] < .45 for i in (11, 12)):
        raise ValueError("Both hip keypoints must be confident for metric root")
    observed, pixels = [], []
    h, w = labels.shape
    for i in (11, 12):
        x, y = np.rint(points[i, :2]).astype(int)
        if not 0 <= x < w or not 0 <= y < h:
            raise ValueError("Hip keypoint outside aligned image")
        y1, y2, x1, x2 = max(0, y-5), min(h, y+6), max(0, x-5), min(w, x+6)
        patch = xyz[y1:y2, x1:x2]
        valid = np.isfinite(patch).all(axis=-1) & (labels[y1:y2, x1:x2] == 12)
        if valid.sum() < 5:
            raise ValueError("Hip has no valid person-labeled depth support; may be occluded")
        sample = patch[valid]
        median = np.median(sample, axis=0)
        if np.percentile(np.abs(sample[:, 2] - median[2]), 90) > .15:
            raise ValueError("Hip depth crosses a depth discontinuity")
        observed.append(median)
        pixels.append([int(x), int(y)])
    if np.linalg.norm(observed[0] - observed[1]) > .65:
        raise ValueError("Bilateral hip depths are inconsistent")
    camera = np.mean(observed, axis=0)
    ground = to_ground(camera, plane)
    if not .25 <= ground[2] <= 1.5:
        raise ValueError("Pelvis proxy height outside plausible observed range")
    return {"camera_xyz_m": camera.tolist(), "ground_xyz_m": ground.tolist(),
            "center_xy_m": ground[:2].tolist(), "hip_pixels": pixels,
            "source": "detected_bilateral_hip_depth_surface_proxy_projected_onto_inferred_floor",
            "body_center_ground_truth": False}


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-9)


class TargetTracker:
    """Conservative box continuity; reset on missing, time gaps, or ambiguity."""
    def __init__(self):
        self.last = None
        self.last_t = None
        self.epoch = 0

    def update(self, instances, t):
        reason = None
        if self.last_t is not None and (t <= self.last_t or t-self.last_t > .15):
            self.last = None
            reason = "source_time_gap"
        self.last_t = t
        if not instances:
            self.last = None
            return None, "missing_person"
        if self.last is None:
            index = max(range(len(instances)), key=lambda i: instances[i]["det"])
            self.epoch += 1
            reason = reason or "new_target_epoch"
        else:
            matches = sorted(((box_iou(self.last["box"], p["box"]), i) for i, p in enumerate(instances)), reverse=True)
            if matches[0][0] < .2 or (len(matches) > 1 and matches[0][0]-matches[1][0] < .12):
                self.last = None
                return None, "ambiguous_identity_reset"
            index = matches[0][1]
        self.last = instances[index]
        return index, reason


class SemanticModel:
    def __init__(self, directory=DEFAULT_MODEL, device="cpu"):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        for name, expected in MODEL_FILES_SHA256.items():
            if sha256(directory / name) != expected:
                raise ValueError("Pinned SegFormer model checksum mismatch")
        config = json.loads((directory / "preprocessor_config.json").read_text())
        config.pop("feature_extractor_type", None)
        config.pop("reduce_labels", None)
        self.processor = SegformerImageProcessor(**config)
        self.model = SegformerForSemanticSegmentation.from_pretrained(directory, local_files_only=True).to(device).eval()
        self.names = {int(k): v for k, v in self.model.config.id2label.items()}
        self.device = device
        torch.set_num_threads(2)

    def predict(self, rgb):
        import torch
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            confidence, labels = logits.softmax(1).max(1)
        return labels[0].cpu().numpy().astype(np.int16), confidence[0].cpu().numpy()


def subject_mask(person, labels, xyz, root):
    """Conservative mask: learned person pixels inside the selected target box.

    Mask dilation covers uncertain silhouette edges; it may also hide adjacent
    surfaces and therefore those cells remain unknown, never declared free.
    """
    h, w = labels.shape
    x1, y1, x2, y2 = np.rint(person["box"]).astype(int)
    inside = np.zeros(labels.shape, bool)
    inside[max(0,y1):min(h,y2+1), max(0,x1):min(w,x2+1)] = True
    # Do not erase arbitrary same-depth surfaces in a person's rectangle: those
    # may be the very chair or desk the user needs to see as an obstacle.
    mask = inside & (labels == 12)
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def render_panel(bgr, labels, instances, target_index, scene, grid, row, action=None):
    canvas = np.full((680, 1120, 3), (242, 245, 249), np.uint8)
    colors = np.random.default_rng(12).integers(50, 230, (150, 3), dtype=np.uint8)
    colors[3] = [143, 164, 47]
    overlay = cv2.addWeighted(bgr, .76, colors[labels], .24, 0)
    for i, p in enumerate(instances):
        color = (50, 220, 120) if i == target_index else (220, 150, 70)
        x1,y1,x2,y2 = map(int, p["box"])
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        for a,b in LINKS:
            if p["kpts"][a][2] >= .45 and p["kpts"][b][2] >= .45:
                cv2.line(overlay, tuple(map(int,p["kpts"][a][:2])), tuple(map(int,p["kpts"][b][:2])), color, 2)
        cv2.putText(overlay, "TARGET" if i == target_index else "OTHER PERSON", (x1,max(18,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    if scene.get("status") == "available":
        center = np.array(scene["root"]["center_xy_m"])
        angles = np.linspace(0, 2*np.pi, 180)
        circle = np.column_stack([center[0]+scene["radius_m"]*np.cos(angles),
                                  center[1]+scene["radius_m"]*np.sin(angles), np.zeros_like(angles)])
        circle[:,2] -= scene["floor"]["offset_m"]
        camera = circle @ np.array(scene["floor"]["basis_columns"]).T
        projected = np.column_stack([CALIBRATION["fx"]*camera[:,0]/camera[:,2]+CALIBRATION["cx"],
                                     CALIBRATION["fy"]*camera[:,1]/camera[:,2]+CALIBRATION["cy"]])
        for j in range(len(camera)-1):
            if min(camera[j,2],camera[j+1,2]) > .15:
                cv2.line(overlay,tuple(np.rint(projected[j]).astype(int)),tuple(np.rint(projected[j+1]).astype(int)),
                         (230,90,70),2,cv2.LINE_AA)
    canvas[:480,:640] = overlay
    if grid is not None:
        palette = np.array([[220, 218, 213], [146, 171, 63], [73, 164, 229]], np.uint8)
        bev = palette[np.minimum(grid, 2)]
        bev[grid == 255] = [249,245,242]
        bev = cv2.resize(bev[::-1], (460,460), interpolation=cv2.INTER_NEAREST)
        cv2.circle(bev,(230,230),229,(70,60,40),2)
        cv2.drawMarker(bev,(230,230),(70,50,220),cv2.MARKER_CROSS,24,3)
        cv2.putText(bev,f"r = {scene['radius_m']:.2f} m",(155,32),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,60,40),1,cv2.LINE_AA)
        for dynamic in scene["dynamic_people"]:
            if dynamic["status"] == "available" and dynamic["inside_radius"]:
                relative = np.array(dynamic["root"]["center_xy_m"])-scene["root"]["center_xy_m"]
                pixel = (230+int(relative[0]*230/scene["radius_m"]),230-int(relative[1]*230/scene["radius_m"]))
                cv2.circle(bev,pixel,7,(220,110,30),2)
                cv2.putText(bev,f"person {dynamic['horizontal_root_distance_m']:.2f}m",(max(0,pixel[0]-50),max(20,pixel[1]-12)),
                            cv2.FONT_HERSHEY_SIMPLEX,.4,(130,70,20),1,cv2.LINE_AA)
        canvas[10:470,650:1110] = bev
    else:
        cv2.putText(canvas, "METRIC GEOMETRY UNKNOWN", (665,230),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,40,190),2)
    sync = f"{row['sync_error_s']*1000:.2f}ms" if row.get("sync_error_s") is not None else "unmatched"
    radius = scene.get("radius_m", row.get("radius_m", 2.))
    sequence_label = row.get("clip_id", SEQUENCE).replace("rgbd_dataset_freiburg3_", "")
    lines = [f"REAL PERSON + REGISTERED DEPTH | {sequence_label} | t={row['t']:.2f}s | target epoch {row['target_epoch']}",
             f"YOLO11n-Pose + SegFormer-B0 | RGB/depth time error {sync} | radius {radius:.2f}m"]
    if scene.get("status") == "available":
        regions = scene["obstacles"]
        items = [f"{r['class_name']} {r['nearest_visible_surface_xy_distance_m']:.2f}m" for r in regions[:4]]
        lines += ["Visible obstacle regions: " + (" | ".join(items) or "none"),
                  f"Unknown disk area {scene['coverage']['unknown_fraction']:.1%} | ground residual {scene['floor']['median_residual_m']*1000:.1f}mm | other people: {len(scene['dynamic_people'])}"]
    else:
        lines += ["Metric result unavailable: " + scene.get("reason", "unknown")]
    lines += ["Action: " + (str(action.get("label"))+" / "+str(action.get("quality")) if action else "pose stream exported; no action model supplied"),
              "Oblique sensor view, not overhead validation. Gray = unobserved; distances use visible surfaces."]
    for i,line in enumerate(lines):
        cv2.putText(canvas,line[:140],(15,510+i*28),cv2.FONT_HERSHEY_SIMPLEX,.53,(65,50,35),1,cv2.LINE_AA)
    return canvas


def run_joint(output_dir: Path, *, assets_dir: Path = DEFAULT_ASSETS, device="cpu", start_s=0.,
              duration_s=8., scene_stride=3, radius_m=2., action_model: Path | None = None,
              mounting="upright") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if (not np.isfinite([start_s, duration_s, radius_m]).all() or start_s < 0
            or duration_s <= 0 or not isinstance(scene_stride, int) or scene_stride < 1 or radius_m <= 0):
        raise ValueError("Require finite nonnegative start, positive duration/radius and integer stride >= 1")
    if mounting not in {"upright", "ceiling"}:
        raise ValueError("mounting must be upright or ceiling")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    _local_import_paths(ROOT, actions=False)
    from ultralytics import YOLO
    weights = ROOT / "models/yolo11n-pose.pt"
    if sha256(weights) != POSE_SHA256:
        raise ValueError("YOLO checkpoint checksum mismatch")
    pose_model, semantic_model = YOLO(str(weights)), SemanticModel(device=device)
    online = None
    temporal_v2 = False
    if action_model:
        schema = json.loads(Path(action_model).read_text()).get("schema")
        temporal_v2 = schema in {"fallpre.action_tcn.v2", "fallpre.motion_forest.v3", "fallpre.phase_model.v4", "fallpre.completion_model.v5", "fallpre.completion_model.v6", "fallpre.completion_model.v7"}
        if temporal_v2:
            from .action_v2 import OnlineTemporalAction, PersonTracker, TemporalModel
            if schema == "fallpre.completion_model.v7":
                from .action_v7 import OnlineCompletion, CompletionModel
                online = OnlineCompletion(CompletionModel.load(action_model))
            elif schema == "fallpre.completion_model.v6":
                from .action_v6 import OnlineCompletion, CompletionModel
                online = OnlineCompletion(CompletionModel.load(action_model))
            elif schema == "fallpre.completion_model.v5":
                from .action_v5 import OnlineCompletion, CompletionModel
                online = OnlineCompletion(CompletionModel.load(action_model))
            elif schema == "fallpre.phase_model.v4":
                from .action_v4 import OnlinePhase, PhaseModel
                online = OnlinePhase(PhaseModel.load(action_model))
            elif schema == "fallpre.motion_forest.v3":
                from .action_v3 import MotionForest, OnlineMotion
                online = OnlineMotion(MotionForest.load(action_model))
            else:
                online = OnlineTemporalAction(TemporalModel.load(action_model))
        else:
            from .five_actions import FiveActionClassifier, OnlineFiveAction
            online = OnlineFiveAction(FiveActionClassifier.load(action_model))
    rgb_rows, depth_rows = read_timestamps(assets_dir / "rgb.txt"), read_timestamps(assets_dir / "depth.txt")
    pairs = associate_rgb_depth(rgb_rows, depth_rows)
    pairs_by_id = {p["source_frame_id"]: p for p in pairs}
    origin = rgb_rows[0][0]
    selected = [(i,t,p) for i,(t,p) in enumerate(rgb_rows) if start_s <= t-origin < start_s+duration_s]
    if len(selected) < 2:
        raise ValueError("At least two source frames are required in the requested interval")
    output_dir.mkdir(parents=True)
    tracker = PersonTracker() if temporal_v2 else TargetTracker()
    records, scene_records = [], []
    # TUM occasionally drops source frames. A median-interval FPS would shorten
    # playback; preserve the complete requested source span in the CFR preview.
    source_step = (selected[-1][1]-selected[0][1]) / (len(selected)-1)
    source_span = selected[-1][1]-selected[0][1]+source_step
    fps = math.ceil(len(selected)/scene_stride) / source_span
    preview_time_error = max(abs(j/fps-(entry[1]-selected[0][1]))
                             for j,entry in enumerate(selected[::scene_stride]))
    writer, codec = open_video_writer(output_dir / "joint-demo.mp4", float(fps), (1120,680))
    input_hashes = {name:sha256(assets_dir/name) for name in ["rgb.txt","depth.txt"]}
    started = time.perf_counter()
    try:
        with (output_dir / "poses.jsonl").open("w") as poses, (output_dir / "joint-frames.jsonl").open("w") as scenes:
            for sequence_index, (i,t,path) in enumerate(selected):
                bgr = cv2.imread(str(assets_dir / path))
                if bgr is None or bgr.shape[:2] != (480,640):
                    raise ValueError("Missing or invalid source RGB")
                input_hashes[path] = sha256(assets_dir / path)
                prediction = pose_model.predict(bgr, device=device, imgsz=640,
                                                conf=.1 if temporal_v2 else .35,
                                                iou=.45 if temporal_v2 else .7, verbose=False)[0]
                instances = []
                if prediction.keypoints is not None:
                    for box,score,kpts in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy(), prediction.keypoints.data.cpu().numpy()):
                        instances.append({"box":box.tolist(),"det":float(score),"kpts":kpts.tolist()})
                if temporal_v2:
                    row = tracker.update({"t":t-origin,"source_timestamp_s":t,"source_frame_id":i,
                                          "clip_id":assets_dir.name,"instances":instances})
                    target_index, reset_reason = row["target_index"], None
                else:
                    target_index, reset_reason = tracker.update(instances, t)
                    row = {"t":t-origin,"source_timestamp_s":t,"source_frame_id":i,"clip_id":SEQUENCE,
                           "target_epoch":tracker.epoch,"tracking_reset_reason":reset_reason,
                           "all_instances":instances,"target_index":target_index,
                           "instances":[instances[target_index]] if target_index is not None else []}
                action = None
                if online:
                    if reset_reason:
                        online.reset(reset_reason)
                    action = online.update(row)
                    row["action"] = action
                poses.write(json.dumps(row, allow_nan=False)+"\n")
                records.append(row)
                if sequence_index % scene_stride:
                    continue
                pair = pairs_by_id.get(i)
                scene = {"status":"unavailable","reason":"unmatched_depth_timestamp"}
                labels = np.zeros((480,640),np.int16)
                grid = None
                if pair:
                    row.update(pair)
                    input_hashes[path] = sha256(assets_dir / path)
                    input_hashes[pair["depth_path"]] = sha256(assets_dir / pair["depth_path"])
                    labels, confidence = semantic_model.predict(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                    depth = cv2.imread(str(assets_dir / pair["depth_path"]),cv2.IMREAD_UNCHANGED)
                    try:
                        xyz = depth_to_xyz(depth)
                        plane = fit_floor(xyz,labels,confidence,mounting=mounting)
                        if target_index is None:
                            raise ValueError("No identity-continuous detected target")
                        subject = instances[target_index]
                        root = person_root(subject,xyz,labels,plane)
                        mask = subject_mask(subject,labels,xyz,root)
                        scene,grid = analyze_scene(labels,confidence,to_ground(xyz,plane),tuple(root["center_xy_m"]),
                                                  semantic_model.names,floor_z_m=0.,radius_m=radius_m,
                                                  subject_mask=mask,min_region_pixels=100,min_height_m=.08)
                        dynamics = []
                        for other_i,other in enumerate(instances):
                            if other_i == target_index:
                                continue
                            try:
                                other_root = person_root(other,xyz,labels,plane)
                                distance = float(np.linalg.norm(np.array(other_root["center_xy_m"])-root["center_xy_m"]))
                                dynamics.append({"detection_index":other_i,"root":other_root,"horizontal_root_distance_m":distance,
                                                 "inside_radius":distance <= radius_m,"status":"available"})
                            except ValueError as error:
                                dynamics.append({"detection_index":other_i,"status":"unknown","reason":str(error)})
                        scene.update({"status":"available","root":root,"floor":plane,"dynamic_people":dynamics,
                                      "subject_exclusion":"dilated predicted-person pixels inside target bbox; excluded cells unknown"})
                    except ValueError as error:
                        scene = {"status":"unavailable","reason":str(error)}
                row.setdefault("sync_error_s",None)
                row["radius_m"] = radius_m
                scene_record = {"t":row["t"],"source_frame_id":i,"rgb_timestamp_s":t,
                                "depth_timestamp_s":pair["depth_timestamp_s"] if pair else None,
                                "sync_error_s":pair["sync_error_s"] if pair else None,
                                "target_epoch":tracker.epoch,"scene":scene,"action":action}
                scenes.write(json.dumps(scene_record,allow_nan=False)+"\n")
                scene_records.append(scene_record)
                panel = render_panel(bgr,labels,instances,target_index,scene,grid,row,action)
                writer.write(panel)
                if len(scene_records) == 1 or (scene["status"] == "available" and not (output_dir / "joint-panel.png").exists()):
                    cv2.imwrite(str(output_dir / ("first-frame.png" if len(scene_records)==1 else "joint-panel.png")),panel)
                if sequence_index <= len(selected)//2 < sequence_index + scene_stride:
                    cv2.imwrite(str(output_dir / "middle-panel.png"),panel)
                if sequence_index % 30 == 0:
                    print(f"joint frame {sequence_index}/{len(selected)} t={row['t']:.2f}s metric={scene['status']}",flush=True)
    finally:
        writer.release()
    if not (output_dir / "joint-panel.png").exists():
        cv2.imwrite(str(output_dir / "joint-panel.png"),panel)
    available = [r for r in scene_records if r["scene"]["status"] == "available"]
    reasons = {}
    for r in scene_records:
        if r["scene"]["status"] != "available":
            reason = r["scene"]["reason"]
            reasons[reason] = reasons.get(reason,0)+1
    summary = {"schema":"fallpre.weekly.joint.v1","output_dir":str(output_dir.resolve()),
               "dataset":{"name":"TUM RGB-D","sequence":assets_dir.name,"selection":"first requested continuous interval, no accuracy cherry-picking",
                          "source_start_s":start_s,"duration_s":duration_s,"camera_view":"approximately static oblique, not overhead"},
               "calibration":{**CALIBRATION,"declared_mounting":mounting},"registration":"publisher OpenNI 1:1 RGB-depth registration; no extra depth undistortion",
               "synchronization":{"method":"one-to-one nearest timestamp; unmatched frames retain unknown geometry",
                                  "max_allowed_difference_s":.02,"max_observed_difference_s":max((r["sync_error_s"] for r in scene_records if r["sync_error_s"] is not None),default=None)},
               "models":{"pose":"YOLO11n-Pose","pose_sha256":POSE_SHA256,"semantics":MODEL_ID,"semantic_revision":MODEL_REVISION,
                         "action_model":str(action_model) if action_model else None,
                         "tracking":"ByteTrack" if temporal_v2 else "conservative bbox IoU",
                         "action_model_sha256":sha256(action_model) if action_model else None,"inference_executed":True,"device":device},
               "counts":{"rgb_pose_frames":len(records),"scene_frames":len(scene_records),"metric_available":len(available),
                         "metric_unavailable":len(scene_records)-len(available),"failure_reasons":reasons,
                         "target_epochs":tracker.epoch,
                         "actions":{label:sum(r.get("action",{}).get("label")==label for r in records)
                                    for label in sorted({r["action"]["label"] for r in records if r.get("action")})}},
               "video":{"file":"joint-demo.mp4","codec":codec,"fps":float(fps),
                        "duration_s":source_span,"max_preview_timestamp_error_s":preview_time_error,
                        "timing":"CFR preview of subsampled frames, no interpolation; total source span preserved, exact source timestamps in JSONL"},
               "evaluation":{"kind":"public-data integration availability, not accuracy benchmark",
                             "metric_available_fraction":len(available)/len(scene_records),
                             "mean_floor_median_residual_m":float(np.mean([r["scene"]["floor"]["median_residual_m"] for r in available])) if available else None,
                             "human_root_ground_truth_available":False,"semantic_ground_truth_available":False},
               "sources":SOURCES,"input_sha256":input_hashes,"wall_seconds":time.perf_counter()-started,
               "limitations":["Public sequence has no obstacle instance or human metric root GT; availability and plane residual are not detection accuracy.",
                              "Front/oblique RGB-D camera validates the joint pipeline, not ceiling-camera pose domain transfer.",
                              "Root is a detected pelvis surface proxy, not center of mass; depth noise and registration error remain.",
                              "Visible connected semantic regions are not complete physical object footprints; unknown cells are never free space.",
                              "Identity uses conservative box continuity, not biometric identification; ambiguity/missing resets action history."]}
    (output_dir / "joint-result.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=DEFAULT_ASSETS)
    parser.add_argument("--device",default="cpu",choices=["cpu","mps","cuda"])
    parser.add_argument("--start",type=float,default=0.)
    parser.add_argument("--duration",type=float,default=8.)
    parser.add_argument("--scene-stride",type=int,default=3)
    parser.add_argument("--radius",type=float,default=2.)
    parser.add_argument("--action-model",type=Path)
    parser.add_argument("--mounting",choices=["upright","ceiling"],default="upright",
                        help="Declared plane-normal prior; ceiling geometry is unit-tested only. TUM calibration must not be reused for another camera.")
    args=parser.parse_args(argv)
    result=run_joint(args.output,assets_dir=args.assets,device=args.device,start_s=args.start,duration_s=args.duration,
                     scene_stride=args.scene_stride,radius_m=args.radius,action_model=args.action_model,mounting=args.mounting)
    print(json.dumps({k:result[k] for k in ["output_dir","counts","evaluation","video"]},indent=2))


if __name__ == "__main__":
    main()
