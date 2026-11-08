"""Report local prerequisites without downloading data or running models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def inspect_environment() -> dict:
    checks = []
    for name in ("numpy", "scipy", "cv2", "torch", "ultralytics", "transformers", "h5py", "remotezip"):
        found = importlib.util.find_spec(name)
        checks.append({"module": name, "available": found is not None,
                       "origin": found.origin if found else None})
    assets = {
        "pose_weights": "models/yolo11n-pose.pt",
        "action_weights": "runs/meeting-20261014/models/stgcnpp-ntu60-2d-joint.pth",
        "action_source": "third_party/algorithms/mmaction2/mmaction",
        "action_config": "third_party/algorithms/mmaction2/configs/skeleton/stgcnpp/"
                         "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
        "public_fall_rgb": "runs/meeting-20261014/inputs/urfd/fall-01-cam0-rgb.zip",
        "public_fall_timing": "runs/meeting-20261014/inputs/urfd/fall-01-data.csv",
        "public_adl_rgb": "runs/meeting-20261014/inputs/urfd/adl-01-cam0-rgb.zip",
        "existing_action_dependencies": "tmp/group-meeting/action-deps",
        "scene_weights": "models/segformer-b0-ade/pytorch_model.bin",
        "scene_rgb": "assets/weekly/scene/ai_001_001/images/scene_cam_00_final_hdf5/"
                     "frame.0035.color.hdf5",
        "scene_xyz": "assets/weekly/scene/ai_001_001/images/scene_cam_00_geometry_hdf5/"
                     "frame.0035.position.hdf5",
        "five_action_manifest": "configs/five-action-dataset.json",
        "five_action_model": "models/five-actions-v1.json",
        "joint_rgb_index": "assets/weekly/joint/rgbd_dataset_freiburg3_walking_static/rgb.txt",
        "joint_depth_index": "assets/weekly/joint/rgbd_dataset_freiburg3_walking_static/depth.txt",
    }
    files = {name: {"path": str(ROOT / path), "exists": (ROOT / path).exists()}
             for name, path in assets.items()}
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "workspace": str(ROOT),
        "modules": checks,
        "assets": files,
        "notes": [
            "Available means discoverable; the actual demos verify model loading and inference.",
            "The human demo can reuse existing isolated action dependencies.",
            "Scene assets can be restored with scripts/setup_weekly_scene.py.",
            "Install optional packages with: uv pip install --python .venv/bin/python "
            "-r requirements-weekly.txt",
            "Action compatibility packages: uv pip install --python .venv/bin/python "
            "--target tmp/group-meeting/action-deps --no-deps -r requirements-weekly-action.txt",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show weekly demo dependencies and local data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect_environment()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)


if __name__ == "__main__":
    main()
