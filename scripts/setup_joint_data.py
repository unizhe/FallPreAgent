"""Acquire the official TUM RGB-D walking_static sequence with resumable ranges.

Downloads public data only; no code from the archive is executed. PNGs are
pre-registered by the publisher. Existing complete files are checksum verified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "rgbd_dataset_freiburg3_walking_static"
URL = f"https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/{SEQUENCE}.tgz"
SIZE = 474074105
EXPECTED_SHA256 = "8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1"
FORMAT_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(output: Path, workers: int = 16):
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{SEQUENCE}.tgz"
    chunks = output / ".download-chunks"
    chunks.mkdir(exist_ok=True)
    if not archive.exists():
        chunk_size = 8 << 20

        def download(index):
            start = index * chunk_size
            end = min(SIZE, start + chunk_size) - 1
            part = chunks / f"{index:03d}.part"
            if part.exists() and part.stat().st_size == end - start + 1:
                return part
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Server did not return the requested byte range")
                with part.open("wb") as f:
                    shutil.copyfileobj(response, f)
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"Incomplete chunk {index}")
            print(f"Downloaded official archive range {index + 1}/{(SIZE + chunk_size - 1) // chunk_size}", flush=True)
            return part

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(download, range((SIZE + chunk_size - 1) // chunk_size)))
        temporary = archive.with_suffix(".assembling")
        with temporary.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target)
        temporary.rename(archive)
    if archive.stat().st_size != SIZE:
        raise ValueError("Archive size does not match official Content-Length")
    checksum = digest(archive)
    if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
        raise ValueError("Official archive SHA256 mismatch")
    manifest_path = output / "acquisition-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["archive_sha256"] != checksum:
            raise ValueError("Archive differs from previous acquisition")
    # Extract only inert expected data files, with traversal/link and size checks.
    with tarfile.open(archive) as tar:
        for member in tar:
            relative = Path(member.name)
            if (relative.is_absolute() or ".." in relative.parts or relative.parts[0] != SEQUENCE
                    or member.issym() or member.islnk()):
                raise ValueError("Unsafe archive path or link")
            if not member.isfile():
                continue
            if relative.suffix not in {".png", ".txt", ".yaml"} or member.size > 20 << 20:
                raise ValueError(f"Unexpected data member: {relative}")
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                data = source.read()
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f"Existing file differs: {path}")
            else:
                path.write_bytes(data)
    manifest = {"dataset": "TUM RGB-D", "sequence": SEQUENCE, "archive_url": URL,
                "archive_size_bytes": SIZE, "archive_sha256": checksum,
                "license": "CC BY 4.0 (dataset website)",
                "source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
                "camera_calibration_source": FORMAT_URL,
                "camera_intrinsics": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
                                      "distortion": [0, 0, 0, 0, 0], "depth_units_per_m": 5000},
                "registration": "Publisher OpenNI registration: RGB and depth pixels correspond 1:1",
                "camera_view": "handheld approximately static front/oblique view, not overhead",
                "selection": "public sequence contains real moving people, furniture, registered measured depth"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/weekly/joint")
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 32:
        parser.error("workers must be 1..32")
    prepare(arguments.output, arguments.workers)
