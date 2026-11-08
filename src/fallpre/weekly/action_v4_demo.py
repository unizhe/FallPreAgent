"""Render fresh-pose V3/V4 comparison and causal stage trace."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .action_v3 import extract
from .action_v4 import CACHE, PROTECTED, PhaseModel, iter_predictions
from .five_actions import label_at
from .human import LINKS, open_video_writer, sha256


def render(manifest, model_path, clip_id, output, pose_mode="infer"):
    e = next(e for e in manifest["entries"] if e["clip_id"] == clip_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = CACHE
    if pose_mode == "infer":
        cache = output / "fresh-poses"
        extract(dict(manifest, entries=[e]), cache, splits=[e["split"]])
    elif pose_mode != "replay":
        raise ValueError("Unknown pose mode")
    model = PhaseModel.load(model_path)
    cap = cv2.VideoCapture(e["video_path"])
    writer, codec = open_video_writer(output / "comparison.mp4", e["fps"], (1280, 720))
    counts = {"v3": Counter(), "v4": Counter()}
    phases = Counter()
    n = 0
    identical = True
    try:
        with (output / "actions.jsonl").open("x") as f:
            for row, tr, p in iter_predictions(e, model, cache):
                ok, img = cap.read()
                if not ok:
                    raise ValueError("Video/pose alignment failed")
                n += 1
                counts["v3"][p["v3_label"]] += 1
                counts["v4"][p["label"]] += 1
                phases[p["phase"]] += 1
                for protected in PROTECTED:
                    identical &= (p["v3_label"] == protected) == (p["label"] == protected)
                ref = label_at(e, row["t"])
                phase_ref = next(
                    (
                        s["label"]
                        for s in e.get("phase_segments", [])
                        if s["start_s"] <= row["t"] < s["end_s"]
                    ),
                    "unannotated",
                )
                f.write(
                    json.dumps(
                        dict(
                            p,
                            source_time_s=row["t"],
                            source_frame_id=row["source_frame_id"],
                            reference=ref,
                            reference_phase=phase_ref,
                            track_id=tr["target_id"],
                        )
                    )
                    + "\n"
                )
                for person in tr["instances"]:
                    pts = person["kpts"]
                    box = list(map(int, person["box"]))
                    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (90, 220, 150), 2)
                    for a, b in LINKS:
                        if min(pts[a][2], pts[b][2]) >= 0.2:
                            cv2.line(
                                img,
                                tuple(map(int, pts[a][:2])),
                                tuple(map(int, pts[b][:2])),
                                (90, 220, 150),
                                2,
                            )
                h, w = img.shape[:2]
                scale = min(680 / w, 480 / h)
                frame = cv2.resize(img, (round(w * scale), round(h * scale)))
                canvas = np.full((720, 1280, 3), (29, 26, 24), np.uint8)
                canvas[65 : 65 + frame.shape[0], : frame.shape[1]] = frame
                cv2.putText(canvas, clip_id, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
                lines = [
                    f"POSE {pose_mode} / {e['split']}",
                    f"Source clock: {row['t']:.2f}s | ID {tr['target_id']}",
                    "",
                    f"V3: {p['v3_label']}",
                    f"V4: {p['label']}",
                    f"Phase: {p['phase']}",
                    f"Raw phase: {p.get('raw_phase', 'unknown')}",
                    f"Seat-relative rise: {p.get('phase_up', 0):.3f} torso",
                    f"Knee opening: {p.get('phase_opening', 0):.3f}",
                    f"Branch: {p['branch']}",
                    "",
                    f"Reference action: {ref}",
                    f"Reference phase: {phase_ref}",
                    "Reference never enters model input.",
                    "Phases: provisional visual proxies.",
                    "Protected V3 walk / bend / fall retained.",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line,
                        (710, 50 + i * 29),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.51,
                        (90, 220, 150) if i in [4, 5] else (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                stage_names = ["seated", "forward_lean", "rising", "standing"]
                for i, name in enumerate(stage_names):
                    x = 20 + i * 305
                    color = (80, 170, 90) if name == p["phase"] else (65, 60, 55)
                    cv2.rectangle(canvas, (x, 575), (x + 285, 625), color, -1)
                    cv2.putText(
                        canvas, name, (x + 12, 607), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1
                    )
                cv2.putText(
                    canvas,
                    "Causal phases | leaning alone is not a confirmed rise | missing evidence remains unknown",
                    (20, 670),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    1,
                )
                writer.write(canvas)
                if n == e["frame_count"] // 2:
                    cv2.imwrite(str(output / "preview.jpg"), canvas)
    finally:
        cap.release()
        writer.release()
    assert identical and n == e["frame_count"]
    result = dict(
        clip_id=clip_id,
        frames=n,
        pose_mode=pose_mode,
        fresh_pose_inference=pose_mode == "infer",
        counts={k: dict(v) for k, v in counts.items()},
        phase_counts=dict(phases),
        protected_decisions_identical=identical,
        model_sha256=sha256(model_path),
        video_codec=codec,
        output=str(output.resolve()),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
