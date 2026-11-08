"""Metric scene component: learned semantics, visible XYZ, and a person-centered disk.

The bundled Hypersim demo uses a *virtual* person anchor and ideal geometry.
Ground-truth semantics are read only after inference, for evaluation. This is
neither a monocular metric-depth algorithm nor a person+scene end-to-end test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS = ROOT / "assets/weekly/scene"
DEFAULT_MODEL = ROOT / "models/segformer-b0-ade"
SCENE = "ai_001_001"
FRAME = 35
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "cc2189cbf3173593452db75308756036c990bc44"
MODEL_SHA256 = "0f4df97633cbedd558ecffa3ad228ace5af37e082678390b45a9d22745787c61"
MODEL_FILES_SHA256 = {
    "config.json": "209caa9091e4632f7c8883c11170cd08ad29af68b23c09590aa4a5befb1a2a7f",
    "preprocessor_config.json": "8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c",
    "pytorch_model.bin": MODEL_SHA256,
}
DATA_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip"
SOURCES = {
    "algorithm": "https://github.com/NVlabs/SegFormer",
    "paper": "https://arxiv.org/abs/2105.15203",
    "weights": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
    "dataset": "https://github.com/apple-aiml-research/ml-hypersim",
    "dataset_paper": "https://arxiv.org/abs/2011.02523",
    "tonemap": "https://github.com/apple-aiml-research/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hdf(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path) as handle:
        return handle["dataset"][:]


def tonemap(color: np.ndarray) -> np.ndarray:
    """Hypersim-style 90th-percentile brightness normalization and gamma 2.2."""
    valid = np.all(np.isfinite(color), axis=-1)
    brightness = np.sum(color[valid] * [0.3, 0.59, 0.11], axis=-1)
    percentile = float(np.percentile(brightness, 90))
    scale = 0.8**2.2 / max(percentile, 1e-8)
    mapped = np.power(np.maximum(np.nan_to_num(color), 0) * scale, 1 / 2.2)
    return np.round(np.clip(mapped, 0, 1) * 255).astype(np.uint8)


def infer_semantics(rgb: np.ndarray, model_dir: Path, device: str = "cpu"):
    import torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    for name, expected in MODEL_FILES_SHA256.items():
        if sha256(model_dir / name) != expected:
            raise ValueError(f"SegFormer {name} differs from the pinned, verified NVIDIA model")
    # Use the pinned processor configuration, removing obsolete training-only keys.
    processor_config = json.loads((model_dir / "preprocessor_config.json").read_text())
    processor_config.pop("feature_extractor_type", None)
    processor_config.pop("reduce_labels", None)
    processor = SegformerImageProcessor(**processor_config)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        confidence, labels = logits.softmax(dim=1).max(dim=1)
    return (labels[0].cpu().numpy().astype(np.int16), confidence[0].cpu().numpy(),
            {int(k): v.strip() for k, v in model.config.id2label.items()}, time.perf_counter() - started)


def estimate_floor_height(xyz_m: np.ndarray, labels: np.ndarray, confidence: np.ndarray,
                          threshold: float = 0.35) -> tuple[float, dict]:
    """Robust horizontal plane from learned floor support and known world gravity.

    Dominant 4 cm height bin rejects furniture mislabeled as floor. We do not use
    semantic GT or normals to select the plane. Unsupported geometry fails closed.
    """
    mask = (labels == 3) & (confidence >= threshold) & np.isfinite(xyz_m).all(axis=-1)
    z = xyz_m[..., 2][mask]
    if len(z) < 100:
        raise ValueError("Insufficient model-predicted floor support: need 100 valid pixels")
    bins = np.floor(z / 0.04).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode_height = (values[np.argmax(counts)] + 0.5) * 0.04
    support = z[np.abs(z - mode_height) <= 0.06]
    if len(support) / len(z) < 0.5:
        raise ValueError("Predicted floor does not support one horizontal plane")
    height = float(np.median(support))
    return height, {"method": "dominant_height_on_predicted_floor_known_world_gravity",
                    "candidate_pixels": len(z), "inlier_pixels": len(support),
                    "median_absolute_residual_m": float(np.median(np.abs(support - height)))}


def analyze_scene(labels: np.ndarray, confidence: np.ndarray, xyz_m: np.ndarray,
                  center_xy_m: tuple[float, float], id2label: dict[int, str], *,
                  radius_m: float = 2.0, floor_z_m: float, resolution_m: float = 0.05,
                  confidence_threshold: float = 0.35, min_region_pixels: int = 250,
                  min_height_m: float = 0.06, max_height_m: float = 2.0,
                  subject_mask: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Find visible semantic regions intersecting a metric disk; no GT input.

    Distances use observed surface points, not object centroids. The 2-D grid is
    conservative: unobserved cells stay unknown; there is no hole filling.
    A region is a 2-D connected semantic region, not an instance detector result.
    Real callers may supply a calibrated ground-projected human root as center.
    """
    if labels.shape != confidence.shape or xyz_m.shape != (*labels.shape, 3):
        raise ValueError("labels, confidence and XYZ must share aligned image dimensions")
    if subject_mask is not None and subject_mask.shape != labels.shape:
        raise ValueError("subject_mask must share the aligned image dimensions")
    center = np.asarray(center_xy_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy_m must contain two finite metric coordinates")
    if not (np.isfinite([radius_m, resolution_m, floor_z_m, min_height_m, max_height_m]).all()
            and radius_m > 0 and resolution_m > 0 and 0 <= min_height_m < max_height_m):
        raise ValueError("Require finite positive radius/resolution and an ordered height band")
    if not 0 <= confidence_threshold <= 1 or min_region_pixels < 1:
        raise ValueError("Invalid confidence threshold or minimum region size")
    valid = np.isfinite(xyz_m).all(axis=-1)
    if subject_mask is not None:
        valid &= ~subject_mask.astype(bool)
    distance = np.linalg.norm(xyz_m[..., :2] - center, axis=-1)
    height = xyz_m[..., 2] - floor_z_m
    inside = valid & (distance <= radius_m + 1e-9)
    confident = np.isfinite(confidence) & (confidence >= confidence_threshold)
    elevated = valid & (height >= min_height_m) & (height <= max_height_m)
    # A visible person belongs to the human pipeline; floor/sky/ceiling are not
    # static obstacle classes. Low-confidence elevated geometry remains explicit.
    obstacle_pixels = elevated & ~np.isin(labels, [2, 3, 5, 12]) & confident
    regions = []
    for label in np.unique(labels[obstacle_pixels]):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            ((labels == label) & obstacle_pixels).astype(np.uint8), connectivity=8)
        for component in range(1, count):
            size = int(stats[component, cv2.CC_STAT_AREA])
            if size < min_region_pixels:
                continue
            mask = components == component
            nearest = float(distance[mask].min())
            if nearest > radius_m + 1e-9:
                continue
            xy = xyz_m[..., :2][mask]
            ys, xs = np.where(mask)
            regions.append({
                "region_id": len(regions) + 1, "class_id": int(label),
                "class_name": id2label.get(int(label), f"class_{label}"),
                "representation": "visible_semantic_connected_region_not_instance",
                "nearest_visible_surface_xy_distance_m": nearest,
                "centroid_xy_m": xy.mean(axis=0).tolist(),
                "visible_xy_bounds_m": [xy.min(axis=0).tolist(), xy.max(axis=0).tolist()],
                "height_range_m": [float(height[mask].min()), float(height[mask].max())],
                "mean_class_score": float(confidence[mask].mean()),
                "visible_pixels": size, "visible_pixels_inside_disk": int(np.sum(mask & inside)),
                "bbox_xyxy_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            })
    regions.sort(key=lambda r: r["nearest_visible_surface_xy_distance_m"])
    for index, region in enumerate(regions, 1):
        region["region_id"] = index
    # Symmetric grid: full extent encloses the requested disk even when radius /
    # cell size is not an integer. Fractions count cell centers inside the disk.
    half = int(math.ceil(radius_m / resolution_m))
    axis = (np.arange(2 * half) + 0.5 - half) * resolution_m
    gx, gy = np.meshgrid(axis, axis)
    disk_grid = gx**2 + gy**2 <= radius_m**2
    if not disk_grid.any():
        raise ValueError("radius is too small for the grid resolution; decrease resolution_m")
    grid = np.zeros((2 * half, 2 * half), dtype=np.uint8)  # 0 unknown, 1 floor, 2 occupied
    floor = inside & confident & (labels == 3) & (np.abs(height) <= 0.06)
    occupied = inside & elevated  # even uncertain semantics are geometric obstacles
    for mask, state in [(floor, 1), (occupied, 2)]:
        points = xyz_m[..., :2][mask]
        indices = np.floor((points - center) / resolution_m).astype(int) + half
        indices = indices[np.all((indices >= 0) & (indices < 2 * half), axis=1)]
        grid[indices[:, 1], indices[:, 0]] = state
    total = int(disk_grid.sum())
    coverage = {name: float(np.sum((grid == state) & disk_grid) / total)
                for name, state in [("unknown_fraction", 0), ("visible_floor_fraction", 1),
                                    ("visible_elevated_surface_fraction", 2)]}
    grid[~disk_grid] = 255
    return {
        "center_xy_m": center.tolist(), "radius_m": radius_m, "floor_z_m": floor_z_m,
        "height_band_m": [min_height_m, max_height_m], "obstacles": regions,
        "coverage": {**coverage, "resolution_m": resolution_m, "cells_in_disk": total,
                     "semantics": "visible projected evidence, not guaranteed traversability"},
        "parameters": {"confidence_threshold": confidence_threshold, "min_region_pixels": min_region_pixels,
                       "distance_rule": "minimum horizontal distance to observed surface points <= radius_m",
                       "subject_mask_pixels": int(np.sum(subject_mask)) if subject_mask is not None else 0},
        "unknown_elevated_pixels_inside_disk": int(np.sum(inside & elevated & ~confident)),
        "geometry_valid_fraction": float(valid.mean()),
    }, grid


def binary_iou(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float | None:
    if predicted.shape != target.shape:
        raise ValueError("IoU masks must have equal shape")
    if valid is None:
        valid = np.ones_like(target, dtype=bool)
    union = (predicted | target) & valid
    return float(np.sum(predicted & target & valid) / union.sum()) if union.any() else None


def evaluate_semantics(labels: np.ndarray, target_nyu40: np.ndarray) -> dict:
    """Explicit coarse cross-taxonomy comparison; these are not ADE20K mIoU."""
    valid = target_nyu40 > 0
    # Matched concepts only. ADE chair variants are grouped into NYU chair;
    # coffee/pool tables into table. Unmatched concepts are not averaged.
    mapping = {"floor": ([3], [2]), "wall": ([0], [1]), "chair": ([19, 30, 31, 75], [5]),
               "table": ([15, 56, 64], [7]), "sofa": ([23], [6]), "bed": ([7], [4]),
               "cabinet": ([10], [3]), "ceiling": ([5], [22]), "toilet": ([65], [33]),
               "curtain": ([18], [16]), "blinds": ([63], [13]), "towel": ([81], [27]),
               "door": ([14], [8]), "window": ([8], [9]), "sink": ([47], [34])}
    metrics = {}
    for name, (ade_ids, nyu_ids) in mapping.items():
        truth = np.isin(target_nyu40, nyu_ids)
        prediction = np.isin(labels, ade_ids)
        if truth[valid].any() or prediction[valid].any():
            metrics[name] = {"iou": binary_iou(prediction, truth, valid),
                             "ground_truth_pixels": int(np.sum(truth & valid)),
                             "predicted_pixels": int(np.sum(prediction & valid)),
                             "ade20k_ids": ade_ids, "nyu40_ids": nyu_ids}
    return {"matched_class_pixel_iou": metrics,
            "scope": "one frame, explicit ADE20K-to-NYU40 concept grouping; not dataset mIoU or instance AP",
            "ground_truth_used_for_inference": False}


def render_outputs(rgb: np.ndarray, labels: np.ndarray, result: dict, grid: np.ndarray,
                   center_pixel: tuple[int, int] | None, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    colors = np.random.default_rng(12).integers(50, 230, (150, 3), dtype=np.uint8)
    colors[3] = [47, 164, 143]
    overlay = cv2.addWeighted(rgb, 0.62, colors[labels], 0.38, 0)
    label_rectangles = []
    for region in result["obstacles"]:
        x1, y1, x2, y2 = region["bbox_xyxy_px"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 184, 73), 2)
        text = f"{region['region_id']} {region['class_name']} {region['nearest_visible_surface_xy_distance_m']:.2f}m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = min(x1, rgb.shape[1] - tw - 8)
        ty = min(max(22, y1 + 20), rgb.shape[0] - 6)
        for delta in [0, 24, -24, 48, -48, 72, -72, 96, -96]:
            candidate_y = min(max(22, ty + delta), rgb.shape[0] - 6)
            rectangle = (tx - 2, candidate_y - th - 4, tx + tw + 4, candidate_y + 4)
            if all(rectangle[2] < old[0] or rectangle[0] > old[2]
                   or rectangle[3] < old[1] or rectangle[1] > old[3] for old in label_rectangles):
                ty = candidate_y
                break
        label_rectangles.append((tx - 2, ty - th - 4, tx + tw + 4, ty + 4))
        cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 4), (27, 43, 62), -1)
        cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 240, 210), 1)
    if center_pixel:
        cv2.drawMarker(overlay, center_pixel, (232, 64, 79), cv2.MARKER_CROSS, 28, 3)
    cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "rgb-overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#f4f6f8")
    axes[0].imshow(overlay)
    axes[0].axis("off")
    axes[0].set_title("SegFormer-B0 | real model inference", loc="left", fontweight="bold")
    radius = result["radius_m"]
    extent = grid.shape[0] / 2 * result["coverage"]["resolution_m"]
    shown = np.ma.masked_where(grid == 255, grid)
    axes[1].imshow(shown, origin="lower", extent=[-extent, extent, -extent, extent],
                   cmap=ListedColormap(["#d7dce5", "#3ca992", "#e9a24a"]), vmin=0, vmax=2,
                   interpolation="nearest")
    axes[1].add_patch(Circle((0, 0), radius, fill=False, color="#263c58", linewidth=2))
    axes[1].plot(0, 0, marker="+", color="#d33d52", markersize=18, markeredgewidth=3)
    axes[1].set(xlabel="Ground X relative to anchor (m)", ylabel="Ground Y relative to anchor (m)",
                title=f"Metric disk | radius {radius:g} m")
    axes[1].legend(handles=[Patch(color="#3ca992", label="Visible floor"),
                             Patch(color="#e9a24a", label="Visible elevated surface"),
                             Patch(color="#d7dce5", label="Unknown")], loc="upper right", fontsize=8)
    coverage = result["coverage"]
    fig.suptitle("SCENE COMPONENT DEMO — virtual person anchor", fontsize=17, fontweight="bold")
    caption = (f"Hypersim {SCENE} / cam_00 / frame.{FRAME:04d} | ideal metric XYZ | "
               f"unknown disk area {coverage['unknown_fraction']:.1%}\n"
               "Semantic regions are not object instances. No real human or metric-depth estimator is tested here.")
    fig.text(0.04, 0.025, caption, fontsize=10, color="#43536a")
    fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.93])
    fig.savefig(out / "scene-panel.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(shown, origin="lower", extent=[-extent, extent, -extent, extent],
              cmap=ListedColormap(["#d7dce5", "#3ca992", "#e9a24a"]), vmin=0, vmax=2)
    ax.add_patch(Circle((0, 0), radius, fill=False, color="#263c58", linewidth=2))
    ax.plot(0, 0, "r+", markersize=20)
    ax.set(xlabel="Ground X (m), relative to virtual anchor", ylabel="Ground Y (m)",
           title="Visible metric evidence; gray = unknown")
    fig.savefig(out / "birdseye.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_scene(output_dir: Path | None = None, *, assets_dir: Path = DEFAULT_ASSETS,
              model_dir: Path = DEFAULT_MODEL, center_xy_m: tuple[float, float] | None = None,
              center_pixel: tuple[int, int] | None = None, radius_m: float = 2.0,
              device: str = "cpu") -> dict:
    if center_xy_m is not None and center_pixel is not None:
        raise ValueError("Specify center_xy_m OR center_pixel, not both")
    out = output_dir or ROOT / "output" / ("weekly-" + datetime.now(ZoneInfo("Asia/Shanghai"))
                                          .strftime("%Y%m%d-%H%M%S-%f")) / "scene"
    if out.exists():
        raise FileExistsError(f"Output already exists; choose a new output directory: {out}")
    scene = assets_dir / SCENE
    geometry = scene / "images/scene_cam_00_geometry_hdf5"
    color_path = scene / f"images/scene_cam_00_final_hdf5/frame.{FRAME:04d}.color.hdf5"
    position_path = geometry / f"frame.{FRAME:04d}.position.hdf5"
    semantic_path = geometry / f"frame.{FRAME:04d}.semantic.hdf5"
    scale_path = scene / "_detail/metadata_scene.csv"
    with scale_path.open() as handle:
        scale = float(next(r for r in csv.DictReader(handle) if r["parameter_name"] == "meters_per_asset_unit")
                      ["parameter_value"])
    rgb = tonemap(_hdf(color_path))
    xyz = _hdf(position_path).astype(np.float64) * scale
    labels, confidence, names, inference_seconds = infer_semantics(rgb, model_dir, device)
    floor_z, floor_support = estimate_floor_height(xyz, labels, confidence)
    anchor_source = "operator_supplied_world_xy_virtual_anchor"
    if center_xy_m is None:
        if center_pixel is None:
            from scipy.spatial import cKDTree

            floor = (labels == 3) & (confidence >= 0.35) & np.isfinite(xyz).all(axis=-1)
            floor &= np.abs(xyz[..., 2] - floor_z) <= 0.06
            yy, xx = np.where(floor)
            # A virtual standing person needs plausible clear floor support;
            # prefer floor >= 0.30 m from visible elevated projections. This
            # heuristic is only demo-anchor placement, never a human detector.
            elevated = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] - floor_z >= 0.06)
            elevated &= xyz[..., 2] - floor_z <= 2.0
            if elevated.any():
                tree = cKDTree(xyz[..., :2][elevated][::4])
                clearance, _ = tree.query(xyz[yy, xx, :2])
                candidates = clearance >= 0.30
                if candidates.any():
                    yy, xx = yy[candidates], xx[candidates]
            target = np.array([rgb.shape[1] * 0.5, rgb.shape[0] * 0.72])
            index = np.argmin((xx - target[0])**2 + (yy - target[1])**2)
            center_pixel = (int(xx[index]), int(yy[index]))
            anchor_source = "automatically_selected_visible_floor_virtual_anchor"
        else:
            anchor_source = "operator_selected_floor_pixel_virtual_anchor"
        x, y = center_pixel
        if not 0 <= x < rgb.shape[1] or not 0 <= y < rgb.shape[0]:
            raise ValueError("Center pixel lies outside the image")
        if not np.isfinite(xyz[y, x]).all() or abs(xyz[y, x, 2] - floor_z) > 0.06:
            raise ValueError("Virtual center pixel must lie on the inferred floor plane")
        center_xy_m = tuple(float(v) for v in xyz[y, x, :2])
    result, grid = analyze_scene(labels, confidence, xyz, center_xy_m, names,
                                 floor_z_m=floor_z, radius_m=radius_m)
    orientation = _hdf(scene / "_detail/cam_00/camera_keyframe_orientations.hdf5")[FRAME]
    camera_position = _hdf(scene / "_detail/cam_00/camera_keyframe_positions.hdf5")[FRAME] * scale
    result.update({
        "schema": "fallpre.weekly.scene.v1", "center_source": anchor_source,
        "center_pixel": center_pixel, "person_detection_present": False,
        "geometry_source": "Hypersim ideal ground-truth visible world-space XYZ, converted from asset units",
        "meters_per_asset_unit": scale, "floor_support": floor_support,
        "camera": {"downward_pitch_degrees": float(np.degrees(np.arcsin(np.clip(orientation[2, 2], -1, 1)))),
                   "height_above_inferred_floor_m": float(camera_position[2] - floor_z)},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "checkpoint_sha256": MODEL_SHA256,
                  "device": device, "inference_seconds": inference_seconds},
        "dataset": {"name": "Hypersim", "scene": SCENE, "camera": "cam_00", "frame": FRAME,
                    "split": "train", "scope": "development demonstration; not a held-out benchmark",
                    "license": "CC BY-SA 3.0", "archive_url": DATA_URL},
        "sources": SOURCES,
        "input_hashes": {str(p.relative_to(assets_dir)): sha256(p)
                         for p in [color_path, position_path, semantic_path, scale_path]},
        "limitations": ["Public Hypersim excludes people: anchor is virtual, not a detected human.",
                        "Ideal geometry is supplied by dataset; real depth noise and metric calibration are untested.",
                        "Projected visible surfaces do not reveal occluded footprints or certify a collision-free path.",
                        "Semantic connected regions can merge or split physical objects.",
                        "One development frame is not a generalization benchmark or safety validation."],
    })
    # GT semantic labels enter only evaluation, after all model/geometric inference.
    result["evaluation"] = evaluate_semantics(labels, _hdf(semantic_path))
    out.mkdir(parents=True, exist_ok=False)
    result["output_dir"] = str(out.resolve())
    (out / "scene-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", labels=labels, confidence=confidence, grid=grid)
    render_outputs(rgb, labels, result, grid, center_pixel, out)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    anchors = parser.add_mutually_exclusive_group()
    anchors.add_argument("--center-xy", nargs=2, type=float, metavar=("X_M", "Y_M"))
    anchors.add_argument("--center-pixel", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    result = run_scene(args.output, assets_dir=args.assets, model_dir=args.model,
                       center_xy_m=args.center_xy, center_pixel=args.center_pixel,
                       radius_m=args.radius, device=args.device)
    print(json.dumps({"output_dir": result["output_dir"], "obstacle_regions": len(result["obstacles"]),
                      "coverage": result["coverage"], "evaluation": result["evaluation"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
