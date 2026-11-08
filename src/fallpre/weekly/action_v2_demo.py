"""Display every frame of a selected recording, including errors and abstentions."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .action_v2 import OnlineTemporalAction, PersonTracker, TemporalModel, extract, read_cache
from .five_actions import FiveActionClassifier, OnlineFiveAction, label_at
from .human import LINKS, ROOT, open_video_writer, sha256
from .joint import TargetTracker


def render_comparison(manifest, cache, model_path, clip_id, output, pose_mode="infer"):
    entry = next(e for e in manifest["entries"] if e["clip_id"] == clip_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if pose_mode not in {"infer", "replay"}:
        raise ValueError("Unknown pose mode")
    if pose_mode == "infer":
        cache = output / "fresh-poses"
        extract(dict(manifest, entries=[entry]), cache)
    meta, rows = read_cache(entry, cache)
    model = TemporalModel.load(model_path)
    online = OnlineTemporalAction(model)
    tracker = PersonTracker(meta["fps"])
    old = OnlineFiveAction(FiveActionClassifier.load(ROOT / "models/five-actions-v1.json"))
    old_tracker = TargetTracker()
    cap = cv2.VideoCapture(entry["video_path"])
    writer, codec = open_video_writer(output / "comparison.mp4", meta["fps"], (1120, 660))
    counts = {"v1": Counter(), "v2": Counter()}
    try:
        with (output / "actions.jsonl").open("x") as handle:
            for row in rows:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Video/pose alignment failed")
                tracked = tracker.update(row)
                prediction = online.update(tracked)
                legacy = dict(row, instances=[p for p in row["instances"] if p["det"] >= 0.35])
                if clip_id.startswith("tum-"):
                    idx, reason = old_tracker.update(legacy["instances"], row["t"])
                    if reason:
                        old.reset(reason)
                    legacy["instances"] = [] if idx is None else [legacy["instances"][idx]]
                baseline = old.update(legacy)
                counts["v1"][baseline["label"]] += 1
                counts["v2"][prediction["label"]] += 1
                record = dict(
                    source_frame_id=row["source_frame_id"],
                    source_time_s=row["t"],
                    v1=baseline,
                    v2=prediction,
                    target_id=tracked["target_id"],
                    reference=label_at(entry, row["t"]),
                )
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                h, w = frame.shape[:2]
                for person in tracked["instances"]:
                    points = person["kpts"]
                    for a, b in LINKS:
                        if min(points[a][2], points[b][2]) >= 0.2:
                            cv2.line(
                                frame,
                                tuple(map(int, points[a][:2])),
                                tuple(map(int, points[b][:2])),
                                (90, 220, 150),
                                2,
                            )
                    box = list(map(int, person["box"]))
                    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (90, 220, 150), 2)
                canvas = np.full((660, 1120, 3), (29, 26, 24), np.uint8)
                scale = min(640 / w, 480 / h)
                resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
                canvas[70 : 70 + resized.shape[0], : resized.shape[1]] = resized
                cv2.putText(canvas, clip_id, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
                lines = [
                    f"POSE: {pose_mode} | time {row['t']:.2f}s",
                    f"ByteTrack target: {tracked['target_id']}",
                    "",
                    f"V1: {baseline['label']}",
                    f"V2: {prediction['label']}",
                    f"Raw TCN: {prediction.get('raw_top1', 'unavailable')}",
                    f"Upward evidence: {prediction.get('upward_torso_units', 0):.3f}",
                    "Reason: " + (prediction.get("reason") or "available"),
                    "",
                    "Reference: " + record["reference"],
                    "Reference is never a model input.",
                    "",
                    "V2: ByteTrack + causal TCN",
                    "Static pose does not prove rising.",
                    "Missing observations remain unknown.",
                    "",
                    "Independent test: 16 recordings.",
                    "Fall events detected: 6/6.",
                    "Standing-up test recall: 0/11.",
                    "Walking / bending still regress.",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line[:58],
                        (655, 50 + 27 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (90, 220, 150) if i == 4 else (232, 235, 238),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    canvas,
                    "Full recording; provisional action labels need human review.",
                    (20, 615),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(canvas)
                if row["source_frame_id"] == max(1, len(rows) // 2):
                    cv2.imwrite(str(output / "preview.jpg"), canvas)
    finally:
        cap.release()
        writer.release()
    result = dict(
        clip_id=clip_id,
        frames=len(rows),
        counts={k: dict(v) for k, v in counts.items()},
        pose_mode=pose_mode,
        fresh_pose_inference=pose_mode == "infer",
        video_codec=codec,
        model_sha256=sha256(model_path),
        output=str(output.resolve()),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
