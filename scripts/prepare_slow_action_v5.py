# ruff: noqa: E402
# Standalone entry configures the repository source path before imports.
"""Author-published natural-speed skeletons; no remote code execution or interpolation."""

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.human import LINKS

BASE = ROOT / "runs/action-v5"
SOURCE = "https://github.com/stanfordnmbl/sit2stand-analysis"
SELECT = {
    "TmJDsJCx": "train",
    "I0tA8b18": "train",
    "FWDyTLr2": "train",
    "VOHFfenU": "dev",
    "kP1305wr": "test",
    "jsr1WRji": "test",
}
MAP = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (ROOT / "runs/action-v5/training/freeze.json").exists():
        raise ValueError("V5 artifacts are frozen; create a new version for data changes")
    rows = {r["subjectid"]: r for r in csv.DictReader(open(BASE / "inputs/stanford-dataMovement.csv"))}
    fpsmeta = json.loads((BASE / "inputs/stanford-framerates.json").read_text())
    z = zipfile.ZipFile(BASE / "inputs/stanford-np.zip")
    entries = []
    protocol = json.loads((ROOT / "configs/action-v5-protocol.json").read_text())
    protocol["real_slow_source"] = {
        "source": SOURCE,
        "subjects": SELECT,
        "selection": "Three available recordings with author-metric mean ascent>=2s, plus three available 1.77-1.92s recordings. YcE3Tlrl, zN0O4jLu, Ydhtzejw are absent from author archive and are excluded for availability only before model predictions. Natural timestamps; no speed augmentation. Explicit subject IDs from author code.",
        "labels": "Manual review of skeleton contact sheets; not clinical slow diagnosis. Frames retain OpenPose confidence.",
        "REMAP": "Public files still return empty response; replaced by accessible Stanford source.",
    }
    (ROOT / "configs/action-v5-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for sid, split in SELECT.items():
        a = np.load(io.BytesIO(z.read("np/" + sid + ".npy")), allow_pickle=False).reshape(-1, 25, 3)
        fps = float(fpsmeta[sid]["framerate"])
        assert sid not in rows or abs(fps - float(rows[sid]["framerate"])) < 0.01
        cid = "stanford-" + sid
        dest = BASE / "inputs/derived" / f"{cid}.mp4"
        cache = BASE / "poses" / cid
        cache.mkdir(parents=True, exist_ok=True)
        pts = a[:, MAP, :].copy()
        good = pts[:, :, 2] > 0.2
        xy = pts[:, :, :2][good]
        extent = np.quantile(xy, [0.005, 0.995], axis=0)
        scale = min(600 / max(extent[1, 0] - extent[0, 0], 1), 430 / max(extent[1, 1] - extent[0, 1], 1))
        pts[:, :, :2] = (pts[:, :, :2] - extent[0]) * scale + np.array([20, 20])
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
        obs = []
        tiles = []
        for i, k in enumerate(pts):
            canvas = np.full((480, 640, 3), 235, np.uint8)
            valid = k[:, 2] > 0.2
            for x, y in LINKS:
                if valid[x] and valid[y]:
                    cv2.line(
                        canvas, tuple(k[x, :2].astype(int)), tuple(k[y, :2].astype(int)), (60, 120, 40), 3
                    )
            cv2.putText(
                canvas,
                f"{sid} | natural t={i / fps:.2f}s",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
            )
            writer.write(canvas)
            if i % max(1, round(fps * 0.5)) == 0:
                tiles.append(cv2.resize(canvas, (200, 150)))
            if valid.sum() >= 6:
                v = k[valid, :2]
                box = [
                    float(v[:, 0].min() - 10),
                    float(v[:, 1].min() - 10),
                    float(v[:, 0].max() + 10),
                    float(v[:, 1].max() + 10),
                ]
                instances = [dict(kpts=k.tolist(), box=box, det=float(k[valid, 2].mean()))]
            else:
                instances = []
            obs.append(dict(t=i / fps, source_frame_id=i + 1, instances=instances))
        writer.release()
        panel = np.full((int(np.ceil(len(tiles) / 5)) * 150, 1000, 3), 235, np.uint8)
        for i, tile in enumerate(tiles):
            y, x = divmod(i, 5)
            panel[y * 150 : (y + 1) * 150, x * 200 : (x + 1) * 200] = tile
        cv2.imwrite(str(BASE / "review" / f"{cid}.jpg"), panel)
        (cache / "observations.jsonl").write_text("".join(json.dumps(o) + "\n" for o in obs))
        e = dict(
            clip_id=cid,
            subject_id=cid,
            source_group_id=cid,
            parent_source_id=cid,
            dataset="Stanford-Sit2Stand",
            split=split,
            video_path=str(dest.relative_to(ROOT)),
            video_sha256=sha(dest),
            fps=fps,
            frame_count=len(a),
            duration_s=len(a) / fps,
            source=dict(
                url=SOURCE,
                archive_sha256=sha(BASE / "inputs/stanford-np.zip"),
                member="np/" + sid + ".npy",
                member_sha256=hashlib.sha256(z.read("np/" + sid + ".npy")).hexdigest(),
                author_mean_ascent_s=float(rows[sid]["time_sit2stand"]) / float(rows[sid]["n_sit2stand"])
                if sid in rows
                else None,
            ),
            input_mode="public_openpose_skeleton",
            segments=[],
            phase_segments=[],
            state_segments=[],
        )
        entries.append(e)
        print(cid, split, len(a), fps, e["source"]["author_mean_ascent_s"], flush=True)
        (cache / "meta.json").write_text(
            json.dumps(
                dict(
                    fps=fps,
                    frame_count=len(a),
                    video_sha256=e["video_sha256"],
                    input_mode="public_openpose_skeleton",
                )
            )
        )
    (BASE / "slow-entries-unannotated.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
