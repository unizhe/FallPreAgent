#!/usr/bin/env python3
"""Fetch and verify the fixed 22-clip URFD + HADE action pilot.

Only the pinned subset is downloaded (roughly 26 MB), not entire datasets.
The manifest owns the fixed source-clip split and provisional visual labels;
this script never changes labels or invents labels from model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/five-action-dataset.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def local_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Manifest path escapes repository: {value}")
    return path


def fetch_source(asset: dict, verify_only: bool) -> Path:
    path = local_path(asset["path"])
    if not path.exists():
        if verify_only:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".download", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as target:
                request = urllib.request.Request(asset["url"], headers={"User-Agent": "FallPreAgent-public-data/1"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        target.write(block)
            if temporary.stat().st_size != asset["bytes"] or digest(temporary) != asset["sha256"]:
                raise ValueError(f"Downloaded source checksum/size mismatch: {path.name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise ValueError(f"Existing source checksum/size mismatch; left unchanged: {path}")
    return path


def prepare_video(entry: dict, source: Path, verify_only: bool) -> Path:
    import cv2

    video = local_path(entry["video_path"])
    derivation = entry["derivation"]
    if not video.exists():
        if verify_only:
            raise FileNotFoundError(video)
        if derivation["kind"] != "right_half_rgb":
            raise ValueError(f"Original video missing after fetching: {video}")
        if cv2.__version__ != derivation["opencv_version"]:
            raise ValueError(
                f"Exact pinned RGB preview reconstruction requires OpenCV {derivation['opencv_version']}; "
                f"installed {cv2.__version__}. Source files remain verified. "
                "Use the same OpenCV encoder build or explicitly establish a new derived-video lock before inference."
            )
        capture = cv2.VideoCapture(str(source))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (width, height) != (derivation["source_width"], derivation["source_height"]):
            capture.release()
            raise ValueError(f"Unexpected source dimensions: {source}")
        temporary = video.with_name(video.stem + ".preparing.mp4")
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width // 2, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not open mp4v output encoder")
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame[:, width // 2:])
                count += 1
        finally:
            capture.release()
            writer.release()
        if count != entry["frame_count"] or digest(temporary) != entry["video_sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Derived video differs from pinned preview: {entry['clip_id']}. "
                "Encoder builds can vary by platform; do not silently reuse pose caches or overwrite the manifest."
            )
        temporary.replace(video)
    if digest(video) != entry["video_sha256"]:
        raise ValueError(f"Video checksum mismatch; left unchanged: {video}")
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if count != entry["frame_count"] or not math.isclose(fps, entry["fps"], rel_tol=1e-5):
        raise ValueError(f"Unexpected decoded frame count/FPS: {video}")
    return video


def make_review_sheet(entry: dict, source: Path) -> None:
    """Recreate unoverlaid visual evidence; does not verify or revise annotations."""
    import cv2
    from PIL import Image, ImageDraw

    path = local_path(entry["annotation_evidence"])
    path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    overview = path.name.endswith("-overview.jpg")
    urfd = entry["dataset"] == "URFD"
    indices = [round(i * (count - 1) / 11) for i in range(12)] if overview else list(
        range(0, count, 8 if urfd else max(1, round(fps / 5)))
    )
    columns, cell_width, cell_height = (4, 240, 200) if overview else (6, 200, 175 if urfd else 225)
    rows = math.ceil(len(indices) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Cannot decode review frame {index}: {source}")
        if urfd:
            frame = frame[:, frame.shape[1] // 2:]
        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        picture.thumbnail((cell_width, cell_height - 25))
        x, y = (position % columns) * cell_width, (position // columns) * cell_height
        canvas.paste(picture, (x, y))
        draw.text((x, y + cell_height - 25), f"f={index} t={index / fps:.2f}s", fill="black")
    capture.release()
    canvas.save(path)


def validate_manifest(manifest: dict) -> None:
    labels = {"walking", "sitting_down", "standing_up", "bending", "falling", "background", "unknown"}
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    supports: dict[str, dict[str, int]] = {split: {} for split in ("train", "test")}
    for entry in manifest["entries"]:
        split = entry["split"]
        if split not in supports:
            raise ValueError(f"Unexpected split: {split}")
        for mapping, key in ((groups, entry["source_group_id"]), (hashes, entry["source"]["sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Source clip leaks across train/test: {entry['clip_id']}")
            mapping[key] = split
        previous_end = 0.0
        observed = set()
        for segment in entry["segments"]:
            start, end, label = segment["start_s"], segment["end_s"], segment["label"]
            if label not in labels or start < previous_end or end <= start or end > entry["duration_s"]:
                raise ValueError(f"Invalid label interval: {entry['clip_id']} {segment}")
            previous_end = end
            observed.add(label)
        for label in observed:
            supports[split][label] = supports[split].get(label, 0) + 1
    for split in supports:
        for label in labels - {"background", "unknown"}:
            if supports[split].get(label, 0) < 2:
                raise ValueError(f"Need two distinct source clips for {split}/{label}")
    print("Source-clip support:", json.dumps(supports, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true", help="Read-only; never download or recreate files")
    parser.add_argument("--review-sheets", action="store_true", help="Recreate original-frame contact sheets without predictions")
    args = parser.parse_args()
    if args.verify_only and args.review_sheets:
        parser.error("--verify-only cannot be combined with --review-sheets")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    for entry in manifest["entries"]:
        source = fetch_source(entry["source"], args.verify_only)
        prepare_video(entry, source, args.verify_only)
        if args.review_sheets:
            make_review_sheet(entry, source)
        print(f"Verified {entry['split']}: {entry['clip_id']}", flush=True)
    print("All fixed pilot videos verified. Time labels remain provisional agent annotations, not official five-action gold labels.")


if __name__ == "__main__":
    main()
