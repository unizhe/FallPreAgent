#!/usr/bin/env python3
"""Restore and verify pinned human-demo assets; scene uses setup_weekly_scene.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a5a167dff2399e2d182a60332325f9c0d4663517"
ACTION_URL = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/"
    "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
    "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(path: Path, url: str, expected: str, verify_only: bool) -> None:
    if not path.exists():
        if verify_only:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"Checksum mismatch in download: {temporary}")
        temporary.rename(path)
    if sha256(path) != expected:
        raise ValueError(f"Existing file checksum mismatch; left unchanged: {path}")
    print(f"Verified: {path.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read-only; never download")
    args = parser.parse_args()
    source = ROOT / "third_party/algorithms/mmaction2"
    if not source.exists():
        if args.verify_only:
            raise FileNotFoundError(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/open-mmlab/mmaction2.git", str(source)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "mmaction", "configs/skeleton/stgcnpp",
                        "configs/_base_", "tools/data/skeleton", "requirements"], cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=source, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"MMAction2 has unexpected revision {actual}; left unchanged")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("MMAction2 source has local changes; left unchanged")
    print(f"Verified MMAction2: {actual}", flush=True)
    model = json.loads((ROOT / "models.lock.json").read_text())["yolo11n-pose.pt"]
    fetch(ROOT / "models/yolo11n-pose.pt", model["url"], model["sha256"], args.verify_only)
    fetch(ROOT / "runs/meeting-20261014/models/stgcnpp-ntu60-2d-joint.pth", ACTION_URL,
          "86e1e77a7b27dd53c6ccc1600a4cc3c321ec8e04d17f25a41fbbb7896026945a", args.verify_only)
    data = json.loads((ROOT / "configs/weekly-data.lock.json").read_text())
    for clip in data["clips"].values():
        for asset in clip["assets"].values():
            filename = asset["url"].rsplit("/", 1)[-1]
            fetch(ROOT / "runs/meeting-20261014/inputs/urfd" / filename,
                  asset["url"], asset["sha256"], args.verify_only)
    print("Human assets ready. Scene assets are handled by the scene command. "
          "Dependencies: requirements-weekly.txt")


if __name__ == "__main__":
    main()
