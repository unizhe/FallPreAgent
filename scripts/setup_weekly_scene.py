"""Download only the pinned public scene sample and official SegFormer weights.

Run with .venv/bin/python scripts/setup_weekly_scene.py after installing
requirements-weekly.txt. Uses HTTP Range ZIP reads (roughly 4 MB of image data),
not the entire 2.25 GB scene archive. Existing verified files are reused; a file
with unexpected contents is never overwritten. No downloaded code is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fallpre.weekly.scene import (  # noqa: E402
    DATA_URL,
    DEFAULT_ASSETS,
    DEFAULT_MODEL,
    MODEL_FILES_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SOURCES,
)

SCENE_FILES = {
    "ai_001_001/_detail/metadata_scene.csv": "60171c3d50c98cfc6ee22f140daf1e2c56ed05604e105bf25368cbecf4cf843a",
    "ai_001_001/_detail/cam_00/camera_keyframe_orientations.hdf5": "c167ceb5bf5e38eaae7307b4aad5414d0910192c14df3e21ea9164dbcc164a6f",
    "ai_001_001/_detail/cam_00/camera_keyframe_positions.hdf5": "2c651665196bf1bab04586b0a07f07b1d6cb78bb2e7a5ec252046911ee5e58ba",
    "ai_001_001/images/scene_cam_00_final_hdf5/frame.0035.color.hdf5": "7d96663928c78538c1c6975284b22abfbd994bdb1e5bb9c6552856b1ee89e0d9",
    "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0035.position.hdf5": "f9494eb8f37a4ec41c1b1f1c37023e5c786a476fbb2b34b2202cfca04c7952ae",
    "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0035.semantic.hdf5": "b48bb05764f2f1ba20bf5c7fb4279aecb9ae32d54d48f6fef7008b5b94ebf225",
}
MODEL_FILES = MODEL_FILES_SHA256
SELECTION = {
    "scene": "ai_001_001", "camera": "cam_00", "frame": 35,
    "included_in_public_release": True, "split": "train",
    "selection_rule": "maximum downward pitch among publicly released frames in this scene; no model metric used",
    "camera_downward_pitch_degrees": 56.90237451584304,
    "excluded_candidate": {"frame": 36, "included_in_public_release": False,
                           "official_exclusion_reason": "CONTENT FLAGGED FOR REMOVAL"},
    "split_source": "https://github.com/apple-aiml-research/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
    "evidence_limit": "development sample, ideal metric geometry, virtual person anchor; not held-out performance",
}


def needs_download(path: Path, expected: str) -> bool:
    if not path.exists():
        return True
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Existing asset has an unexpected SHA-256; preserve and inspect it: {path}")
    return False


def verified_write(path: Path, data: bytes, expected: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Download SHA-256 mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    print(f"Verified {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}", flush=True)


def prepare(assets_dir: Path = DEFAULT_ASSETS, model_dir: Path = DEFAULT_MODEL) -> None:
    missing = [(name, digest) for name, digest in SCENE_FILES.items()
               if needs_download(assets_dir / name, digest)]
    if missing:
        from remotezip import RemoteZip

        with RemoteZip(DATA_URL, timeout=90) as archive:
            for name, digest in missing:
                verified_write(assets_dir / name, archive.read(name), digest)
    for name, digest in MODEL_FILES.items():
        if needs_download(model_dir / name, digest):
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=90) as response:
                verified_write(model_dir / name, response.read(), digest)
    manifest = {"dataset_archive_url": DATA_URL, "scene_files_sha256": SCENE_FILES,
                "model": MODEL_ID, "model_revision": MODEL_REVISION, "model_files_sha256": MODEL_FILES,
                "selection": SELECTION, "sources": SOURCES,
                "licenses": {"Hypersim": "CC BY-SA 3.0", "SegFormer": "NVIDIA Source Code License (research/non-commercial terms)"}}
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_dir / "acquisition-manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All scene assets and model files verified. Ready for scripts/weekly_demo.py scene.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    prepare(args.assets, args.model)


if __name__ == "__main__":
    main()
