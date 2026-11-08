# ruff: noqa: E402
# Standalone script: configure the local source path before importing project modules.
"""Prepare fixed 20-second clips/contact sheets; no model predictions are accessed."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_action_v3 import sha

OUT = ROOT / "runs/action-v4"


def contact(video, dest, times):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(5)
    canvas = np.full((int(np.ceil(len(times) / 5)) * 145, 1000, 3), 240, np.uint8)
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (200, 120))
        y, x = divmod(i, 5)
        canvas[y * 145 : y * 145 + 120, x * 200 : x * 200 + 200] = img
        cv2.putText(
            canvas, f"{t:.2f}s", (x * 200 + 3, y * 145 + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1
        )
    cap.release()
    cv2.imwrite(str(dest), canvas)


def main():
    rows = json.loads((OUT / "inputs/acquisition.json").read_text())
    entries = []
    for r in rows:
        cid = f"telkom-{r['subject']}-{r['when']}"
        target = OUT / "inputs/derived" / f"{cid}.mp4"
        target.parent.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / r["path"]))
        fps = cap.get(5)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
        n = 0
        while n < round(20 * fps):
            ok, img = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(img, (640, 360)))
            n += 1
        cap.release()
        writer.release()
        entries.append(
            dict(
                clip_id=cid,
                split="train" if r["subject"] in [11, 12] else "dev" if r["subject"] == 13 else "test",
                subject_id=f"telkom-{r['subject']}",
                parent_source_id=cid,
                source_group_id=cid,
                dataset="Telkom-STS",
                video_path=str(target.relative_to(ROOT)),
                video_sha256=sha(target),
                fps=fps,
                frame_count=n,
                duration_s=n / fps,
                source=r,
                segments=[],
                state_segments=[],
                phase_segments=[],
            )
        )
        contact(target, OUT / "review" / f"{cid}.jpg", np.arange(0, min(20, n / fps), 0.5))
    path = OUT / "new-entries-unannotated.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print([(e["clip_id"], e["frame_count"]) for e in entries])


if __name__ == "__main__":
    main()
