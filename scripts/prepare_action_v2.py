"""Acquire the predeclared v2 public sources; preserve the v1 split and results."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/action-v2/inputs"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(n):
    name = f"fall-{n:02d}-cam0.mp4"
    path = OUT / name
    url = "https://fenix.ur.edu.pl/~mkepski/ds/data/" + name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        tmp = path.with_suffix(".download")
        tmp.write_bytes(data)
        tmp.replace(path)
    print(name, path.stat().st_size, flush=True)
    return {
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def crop(source, target):
    if target.exists():
        return
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w // 2, h))
    if not cap.isOpened() or not writer.isOpened():
        raise RuntimeError("Cannot decode/crop public RGB preview")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame[:, w // 2 :])
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.tum_only:
        import setup_joint_data as setup

        setup.SEQUENCE = "rgbd_dataset_freiburg3_sitting_static"
        setup.URL = f"https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/{setup.SEQUENCE}.tgz"
        setup.SIZE = 443506009
        lock = OUT / "tum-sitting-lock.json"
        setup.EXPECTED_SHA256 = json.loads(lock.read_text())["archive_sha256"] if lock.exists() else ""
        setup.prepare(OUT / "tum-sitting", workers=4)
        result = json.loads((OUT / "tum-sitting/acquisition-manifest.json").read_text())
        if not lock.exists():
            lock.write_text(json.dumps(result, indent=2) + "\n")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(fetch, range(5, 17)))
    path = OUT / "fall-sources.json"
    if path.exists() and json.loads(path.read_text()) != sources:
        raise ValueError("Downloaded sources differ from previous lock")
    path.write_text(json.dumps(sources, indent=2) + "\n")
    for source in sources:
        p = ROOT / source["path"]
        crop(p, OUT / p.name.replace("-cam0", "-rgb"))
    for n in [9, 10, 17, 19]:
        p = ROOT / f"runs/five-action-20261015/inputs/urfd/adl-{n:02d}-cam0.mp4"
        crop(p, OUT / f"adl-{n:02d}-rgb.mp4")


if __name__ == "__main__":
    main()
