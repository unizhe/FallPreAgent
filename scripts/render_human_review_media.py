"""Prepare raw-RGB human-review media, without model predictions or labels.

Only observations already present in the frozen input manifest are used.
Missing UTK frames are never invented. Slower videos hold observed images.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from render_abc_v7_media import ROOT, source_frames

from fallpre.weekly.human import open_video_writer, sha256

WIDTH, HEIGHT, FPS = 1400, 680, 30
CASES = [
    dict(id="contact", clip_id="urfd-fall-01-original", title="URFD fall01：首次非足部接地复核",
         context_frame_range=[91, 115], candidate_frame_range=[101, 107],
         roi=[145, 180, 445, 480]),
    dict(id="stable", clip_id="urfd-fall-01-original", title="URFD fall01：稳定倒地起点复核",
         context_frame_range=[120, 160], candidate_frame_range=[132, 138],
         roi=[180, 280, 480, 480]),
    dict(id="utk-s01-seat-off", clip_id="utk-s01_e01-transition", title="UTK s01：离座边界复核",
         context_frame_range=[680, 782], candidate_frame_range=[704, 740],
         roi=[90, 215, 320, 455]),
    dict(id="utk-s10-seat-off", clip_id="utk-s10_e01-transition", title="UTK s10：离座边界复核",
         context_frame_range=[730, 832], candidate_frame_range=[754, 810],
         roi=[110, 200, 350, 440]),
]


def put(canvas, text, position, scale=.66, color=(236, 238, 243), thick=1):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def frame_canvas(image, case, fid, time_s, gap_s):
    canvas = np.full((HEIGHT, WIDTH, 3), (25, 22, 20), np.uint8)
    put(canvas, f"{case['id']} | source frame {fid} | source timeline {time_s:.3f} s",
        (20, 33), .89, thick=2)
    source_note = ("URFD official CSV clock" if case["clip_id"].startswith("urfd")
                   else "UTK sparse observations; relative timeline = source frame delta / 30")
    gap = "first context observation" if gap_s is None else f"gap from previous observed frame: {gap_s:.3f} s"
    put(canvas, f"{source_note} | {gap}", (20, 66), .60, (148, 202, 238))
    put(canvas, "Original RGB - no model overlays", (20, 101), .65)
    put(canvas, f"Fixed crop x:{case['roi'][0]}-{case['roi'][2]} y:{case['roi'][1]}-{case['roi'][3]}",
        (740, 101), .65)
    assert image.shape == (480, 640, 3), image.shape
    canvas[120:600, 20:660] = image
    x1, y1, x2, y2 = case["roi"]
    roi = image[y1:y2, x1:x2]
    scale = min(640 / roi.shape[1], 480 / roi.shape[0])
    enlarged = cv2.resize(roi, (round(roi.shape[1] * scale), round(roi.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)
    x = 740 + (640 - enlarged.shape[1]) // 2
    y = 120 + (480 - enlarged.shape[0]) // 2
    canvas[y:y + enlarged.shape[0], x:x + enlarged.shape[1]] = enlarged
    put(canvas, "Human review material only. Candidate ranges are search aids, not confirmed event boundaries.",
        (20, 631), .62)
    put(canvas, "No temporal interpolation, pose inference, model labels, or automatic boundary decisions.",
        (20, 660), .60, (171, 185, 200))
    return canvas


def save_contact_sheet(images, selected, path, case):
    """Keep every available candidate observation; make pages if necessary."""
    cols, cell_w, cell_h = 3, 640, 530
    cells = []
    for frame in selected:
        im = images[frame["frame_id"]]
        cell = np.full((cell_h, cell_w, 3), (25, 22, 20), np.uint8)
        put(cell, f"source frame {frame['frame_id']} | {frame['source_time_s']:.3f} s", (12, 32), .76, thick=2)
        cell[50:530] = im
        cells.append(cell)
    rows = math.ceil(len(cells) / cols)
    sheet = np.full((rows * cell_h + 72, cols * cell_w, 3), (25, 22, 20), np.uint8)
    put(sheet, f"{case['id']} | candidate search observations only - NOT confirmed boundaries",
        (18, 32), .80, thick=2)
    put(sheet, "Original RGB, ordered by source frame. Sparse UTK observations remain sparse.", (18, 59), .67)
    for i, cell in enumerate(cells):
        y, x = 72 + (i // cols) * cell_h, (i % cols) * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = cell
    assert cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return dict(width=sheet.shape[1], height=sheet.shape[0], source_observations=len(cells))


def render_video(path, rows, canvases, speed):
    writer, codec = open_video_writer(path, FPS, (WIDTH, HEIGHT))
    goals = [round((f["source_time_s"] - rows[0]["source_time_s"]) / speed * FPS) for f in rows]
    assert all(b > a for a, b in zip(goals, goals[1:]))
    output_count = 0
    mapping = []
    for i, row in enumerate(rows):
        first = goals[i]
        end_exclusive = goals[i + 1] if i + 1 < len(goals) else first + 1
        canvas = canvases[row["frame_id"]].copy()
        canvas[640:680] = (25, 22, 20)
        put(canvas, f"Baked playback {speed:g}x | repeated observed images only | displayed observation stays frame {row['frame_id']}",
            (20, 666), .62, (148, 202, 238))
        assert output_count == first
        for _ in range(first, end_exclusive):
            writer.write(canvas)
            output_count += 1
        mapping.append(dict(frame_id=row["frame_id"], source_time_s=row["source_time_s"],
                            output_frame_start_0based=first, output_frame_end_0based=end_exclusive - 1,
                            output_start_s=first / FPS, output_end_exclusive_s=end_exclusive / FPS,
                            repeated_display_frames=end_exclusive - first))
    writer.release()
    return dict(speed=speed, fps=FPS, frames=output_count, width=WIDTH, height=HEIGHT, codec=codec,
                sha256=sha256(path), duration_s=output_count / FPS,
                rule="Each observed image is held until the next observed source time, scaled by speed; no synthetic/interpolated images.",
                final_frame_padding_s=1 / FPS,
                final_frame_note="Final observed image has exactly one display frame, not evidence of continued event state.",
                timeline_mapping=mapping)


def validate_video(path, expected):
    cap = cv2.VideoCapture(str(path))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count, shapes = 0, set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        shapes.add(tuple(frame.shape[:2]))
    cap.release()
    assert count == declared == expected, (path, count, declared, expected)
    assert shapes == {(HEIGHT, WIDTH)}
    assert abs(fps - FPS) < .001
    return dict(file=str(path), complete_decode=True, decoded_frames=count, declared_frames=declared,
                expected_frames=expected, fps=fps, width=WIDTH, height=HEIGHT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/abc-v7-20261020/human-review/media")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "runs/public-samples-20261020/manifest.json"
    inputs = json.loads(manifest_path.read_text())
    result = dict(schema="human-review-raw-rgb-v1", confirmed_annotations=False,
                  model_inference=False, model_outputs_included=False,
                  input_manifest=str(manifest_path), input_manifest_sha256=sha256(manifest_path),
                  purpose="Independent human review material. Candidate intervals are retained as search aids, never confirmed automatically.",
                  source_gap_policy="Only available decoded observations. No fabricated intermediate frames or joint overlays.",
                  cases=[])
    validations = []
    caches = {}
    for case in CASES:
        entry = next(e for e in inputs["entries"] if e["clip_id"] == case["clip_id"])
        source_path = Path(entry.get("zip_path", entry.get("video_path")))
        assert source_path.exists() and sha256(source_path) == entry["input_sha256"]
        ids = entry.get("source_original_frame_ids", list(range(1, entry["frame_count"] + 1)))
        times = entry["source_times_s"]
        if case["clip_id"] not in caches:
            decoded = list(source_frames(entry))
            assert len(decoded) == len(ids) == len(times)
            caches[case["clip_id"]] = dict(zip(ids, decoded))
        images = caches[case["clip_id"]]
        lo, hi = case["context_frame_range"]
        selected = [dict(frame_id=fid, source_time_s=t) for fid, t in zip(ids, times) if lo <= fid <= hi]
        dest = out / case["id"]
        frames_dir = dest / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        canvases, gaps = {}, []
        for i, frame in enumerate(selected):
            fid, t = frame["frame_id"], frame["source_time_s"]
            delta = None if i == 0 else t - selected[i - 1]["source_time_s"]
            canvas = frame_canvas(images[fid], case, fid, t, delta)
            canvases[fid] = canvas
            path = frames_dir / f"{fid:06d}.jpg"
            assert cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 97])
            frame.update(file=str(path.relative_to(out.parent)), sha256=sha256(path),
                         previous_observation_gap_s=delta,
                         available_original_rgb=True)
            if i:
                previous_id = selected[i - 1]["frame_id"]
                gaps.append(dict(previous_frame_id=previous_id, next_frame_id=fid,
                                 source_frame_id_step=fid - previous_id,
                                 unavailable_between=max(0, fid - previous_id - 1),
                                 timeline_gap_s=delta))
        core_lo, core_hi = case["candidate_frame_range"]
        candidates = [r for r in selected if core_lo <= r["frame_id"] <= core_hi]
        sheet_path = dest / "candidate-contact-sheet.jpg"
        sheet_meta = save_contact_sheet(images, candidates, sheet_path, case)
        videos = {}
        for name, speed in (("normal", 1.0), ("slow", .25)):
            path = dest / f"{name}.mp4"
            videos[name] = render_video(path, selected, canvases, speed)
            videos[name]["file"] = str(path.relative_to(out.parent))
            check = validate_video(path, videos[name]["frames"])
            validations.append(check)
        source_note = (
            "URFD original RGB ZIP with official source CSV timestamps; consecutive source frames."
            if case["clip_id"].startswith("urfd") else
            "UTK input contains sparse original RGB observations. Frame-ID-derived relative timeline (source frame delta / 30); no missing frames reconstructed and no calibrated camera-clock claim.")
        tail = selected[-1]["source_time_s"] - max(r["source_time_s"] for r in candidates)
        record = dict(case, frames=selected, candidate_available_frame_ids=[r["frame_id"] for r in candidates],
                      contact_sheet=str(sheet_path.relative_to(out.parent)), contact_sheet_dimensions=sheet_meta,
                      video_normal=videos["normal"]["file"], video_slow=videos["slow"]["file"],
                      clock_note=source_note, source_file=str(source_path),
                      source_sha256=entry["input_sha256"], sampling_gaps=gaps,
                      max_sampling_gap_s=max(g["timeline_gap_s"] for g in gaps),
                      available_observations=len(selected),
                      missing_integer_frame_ids=[i for i in range(lo, hi + 1) if i not in images],
                      source_time_range_s=[selected[0]["source_time_s"], selected[-1]["source_time_s"]],
                      context_after_candidate_s=tail,
                      videos=videos, status="pending_human_review")
        if case["id"] == "stable":
            assert tail >= .5
        result["cases"].append(record)
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(case["id"], len(selected), "observations", len(candidates), "candidate observations", flush=True)
    (out / "validation.json").write_text(json.dumps(dict(
        all_passed=True, videos=len(validations), checks=validations,
        jpeg_count=sum(len(c["frames"]) for c in result["cases"]),
        annotation_confirmation_performed=False), ensure_ascii=False, indent=2) + "\n")
    print(out / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
