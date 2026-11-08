"""V2/V3 full-video comparison: source-time overlays, explicit live/replay modes."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .action_v2 import OnlineTemporalAction, PersonTracker
from .action_v3 import V3_CACHE, MotionForest, OnlineMotion, extract, observations
from .five_actions import label_at
from .human import LINKS, open_video_writer, sha256


def render_comparison(manifest, model_path, clip_id, output, pose_mode="infer"):
    e = next(e for e in manifest["entries"] if e["clip_id"] == clip_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if pose_mode not in {"infer", "replay"}:
        raise ValueError("Unknown pose mode")
    cache = V3_CACHE
    if pose_mode == "infer":
        cache = output / "fresh-poses"
        extract(dict(manifest, entries=[e]), cache, splits=[e["split"]])
    meta, rows = observations(e, cache)
    model = MotionForest.load(model_path)
    online = OnlineMotion(model)
    old = OnlineTemporalAction(model.frozen)
    tracker = PersonTracker(meta["fps"])
    cap = cv2.VideoCapture(e["video_path"])
    writer, codec = open_video_writer(output / "comparison.mp4", e["fps"], (1120, 680))
    counts = {"v2": Counter(), "v3": Counter()}
    fall_identical = True
    try:
        with (output / "actions.jsonl").open("x") as f:
            for row in rows:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Video/pose alignment failed")
                tr = tracker.update(row)
                a = old.update(tr)
                b = online.update(tr)
                fall_identical &= (a["label"] == "falling") == (b["label"] == "falling")
                counts["v2"][a["label"]] += 1
                counts["v3"][b["label"]] += 1
                record = dict(
                    source_frame_id=row["source_frame_id"],
                    source_time_s=row["t"],
                    reference=label_at(e, row["t"]),
                    v2=a,
                    v3=b,
                    target_id=tr["target_id"],
                )
                f.write(json.dumps(record, allow_nan=False) + "\n")
                for person in tr["instances"]:
                    box = list(map(int, person["box"]))
                    pts = person["kpts"]
                    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (90, 220, 150), 2)
                    for i, j in LINKS:
                        if min(pts[i][2], pts[j][2]) >= 0.2:
                            cv2.line(
                                frame,
                                tuple(map(int, pts[i][:2])),
                                tuple(map(int, pts[j][:2])),
                                (90, 220, 150),
                                2,
                            )
                h, w = frame.shape[:2]
                scale = min(640 / w, 480 / h)
                res = cv2.resize(frame, (round(w * scale), round(h * scale)))
                canvas = np.full((680, 1120, 3), (29, 26, 24), np.uint8)
                canvas[65 : 65 + res.shape[0], : res.shape[1]] = res
                cv2.putText(canvas, clip_id, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
                lines = [
                    f"POSE: {pose_mode} | split: {e['split']}",
                    f"Source time: {row['t']:.2f}s",
                    f"ByteTrack ID: {tr['target_id']}",
                    "",
                    f"V2: {a['label']}",
                    f"V3: {b['label']}",
                    f"Raw classifier: {b.get('raw_top1', 'unknown')}",
                    f"Upward evidence: {b.get('upward_torso_units', 0):.3f}",
                    f"Branch: {b.get('branch', 'unknown')}",
                    "Reason: " + (b.get("reason") or "available"),
                    "",
                    "Reference: " + record["reference"],
                    "Reference is never a model input.",
                    "",
                    "V3: multi-scale past-motion features",
                    "ExtraTrees + frozen causal TCN fall branch",
                    "0.2 / 0.5 / 1 / 2 second windows",
                    "Missing observations remain unknown.",
                    "",
                    "Public staged data; failures retained.",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line[:60],
                        (652, 45 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (90, 220, 150) if i == 5 else (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    canvas,
                    "Original source clock in JSONL; CFR video is a preview.",
                    (18, 620),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 225, 225),
                    1,
                )
                writer.write(canvas)
                if row["source_frame_id"] == max(1, len(rows) // 2):
                    cv2.imwrite(str(output / "preview.jpg"), canvas)
    finally:
        cap.release()
        writer.release()
    if not fall_identical:
        raise AssertionError("V2 falling branch changed in online demo")
    result = dict(
        clip_id=clip_id,
        frames=len(rows),
        pose_mode=pose_mode,
        fresh_pose_inference=pose_mode == "infer",
        counts={k: dict(v) for k, v in counts.items()},
        falling_decisions_identical=fall_identical,
        model_sha256=sha256(model_path),
        video_codec=codec,
        output=str(output.resolve()),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
