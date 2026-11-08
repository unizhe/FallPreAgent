"""Render completed unknown-reduction trials on their measured wall-clock axis, without inference.

Default selection is C repeat 2, fixed before viewing its result. Each panel
holds the raw source image associated with its latest COMPLETED V7 output.
Holds and dropped frames are explicit; no source image is shown as a prediction
before its recorded processing completion. Run only after all timing trials end.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "runs/abc-v7-20261020/unknown-reduction-20261021"
WIDTH, HEIGHT, FPS = 1440, 980, 30
BG, PANEL = (23, 21, 19), (36, 32, 28)
WHITE, MUTED = (244, 242, 237), (187, 181, 167)
AMBER, GREEN, RED = (105, 196, 248), (143, 223, 137), (139, 137, 252)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def put(image, text, x, y, scale=.58, color=WHITE, thick=1, width=None):
    if width is not None:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        if size > width:
            scale *= width / size
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def load_trial(experiment, mode, policy, repeat):
    directory = experiment / "trials" / f"{policy}-r{repeat}"
    rows_path = directory / "frames.jsonl"
    summary_path = directory / "summary.json"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = read_json(summary_path)
    assert rows and len(rows) == summary["processed_frames"]
    outputs = [row["output_wall_s"] for row in rows]
    selections = [row["selection_wall_s"] for row in rows]
    assert all(b > a for a, b in zip(outputs, outputs[1:]))
    assert all(b > a for a, b in zip(selections, selections[1:]))
    assert all(row["source_time_s"] <= row["selection_wall_s"] <= row["output_wall_s"] for row in rows)
    cumulative_dropped, total = [], 0
    for row in rows:
        total += len(row["dropped_before_this_frame"])
        cumulative_dropped.append(total)
    assert total == summary["dropped_frames"]
    return dict(policy=policy, rows=rows, outputs=outputs, selections=selections,
                cumulative_dropped=cumulative_dropped, summary=summary,
                rows_path=rows_path, summary_path=summary_path)


def panel(canvas, trial, images, wall_s, x, total_sources):
    cv2.rectangle(canvas, (x, 150), (x + 690, 922), PANEL, -1)
    policy = trial["policy"]
    title = "C LATEST | 200 ms elapsed-only budget" if policy == "budget200ms" else "C LATEST | 200 ms + call-cost admission"
    accent = AMBER if policy == "budget200ms" else GREEN
    put(canvas, title, x + 20, 181, .76, accent, 2, 650)
    index = bisect_right(trial["outputs"], wall_s) - 1
    selected_index = bisect_right(trial["selections"], wall_s) - 1
    dropped = trial["cumulative_dropped"][selected_index] if selected_index >= 0 else 0
    completed = index + 1
    put(canvas, f"Completed: {completed}/{total_sources}  |  discarded at selection: {dropped}",
        x + 20, 211, .56, MUTED, width=650)
    if index < 0:
        cv2.rectangle(canvas, (x + 25, 256), (x + 665, 736), (10, 10, 10), -1)
        put(canvas, "NO COMPLETED OUTPUT YET", x + 84, 470, .83, AMBER, 2, 530)
        put(canvas, "The current camera image is not a model result.", x + 59, 514, .58, MUTED, width=575)
        put(canvas, "Source frame / output / age: unavailable", x + 20, 775, .64, MUTED, width=650)
        return dict(index=None, source_frame_id=None, displayed_output_wall_s=None,
                    selection_discarded_count=dropped, displayed_prediction=None)

    row = trial["rows"][index]
    held_s = max(0.0, wall_s - row["output_wall_s"])
    is_new = held_s < 1 / FPS
    state = "NEW COMPLETED OUTPUT" if is_new else "HOLDING LAST COMPLETED OUTPUT"
    put(canvas, state, x + 20, 242, .67, GREEN if is_new else AMBER, 2, 650)
    image = images[row["source_frame_id"]]
    assert image.shape[:2] == (480, 640)
    canvas[256:736, x + 25:x + 665] = image
    # The persistent HOLD badge prevents repeated display from implying new inference.
    badge = canvas[256:292, x + 25:x + 665]
    badge[:] = (badge.astype(np.float32) * .25).astype(np.uint8)
    put(canvas, f"{'NEW' if is_new else 'HOLD'} | raw source frame {row['source_frame_id']} | held {held_s:.3f}s",
        x + 38, 281, .62, GREEN if is_new else AMBER, 2, 615)
    label = row["prediction"]["label"]
    put(canvas, f"V7: {label}  |  phase: {row['prediction']['phase']}  |  target: {row['target_id']}",
        x + 20, 769, .71, RED if label == "falling" else WHITE, 2, 650)
    put(canvas, f"Source: frame {row['source_frame_id']} / {row['source_time_s']:.3f}s  |  output: {row['output_wall_s']:.3f}s",
        x + 20, 799, .60, WHITE, width=650)
    put(canvas, f"Age at completion: {row['output_age_s']:.3f}s  |  display age now: {wall_s-row['source_time_s']:.3f}s",
        x + 20, 828, .57, MUTED, width=650)
    put(canvas, f"Recovery: {row['stages_s']['recovery']:.3f}s | recovered: {row['recovery_selected']} | observed: {row['observed']}",
        x + 20, 856, .59, MUTED, width=650)
    first = trial["summary"]["first_falling"]
    if first and wall_s >= first["output_wall_s"]:
        margin = first["contact_margin_s"]
        note = f"First falling at wall {first['output_wall_s']:.3f}s; contact margin [{margin[0]:+.3f}, {margin[1]:+.3f}]s"
        put(canvas, note, x + 20, 898, .55, GREEN if margin[0] > 0 else RED, 2, 650)
    else:
        put(canvas, "First falling: no completed falling output yet", x + 20, 898, .57, MUTED, width=650)
    return dict(index=index, source_frame_id=row["source_frame_id"],
                displayed_output_wall_s=row["output_wall_s"],
                selection_discarded_count=dropped, displayed_prediction=label)


def render_canvas(trials, images, wall_s, end_s, mode, repeat, bounds, total_sources):
    canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    put(canvas, f"URFD fall01 | fixed V7 | {mode} repeat {repeat} | measured output availability",
        20, 34, .86, WHITE, 2, WIDTH - 40)
    put(canvas, f"WALL CLOCK {wall_s:06.3f} s  |  playback 1x  |  images appear only AFTER their outputs complete",
        20, 68, .69, AMBER, 2, WIDTH - 40)
    if wall_s < bounds[0]:
        contact = "Before earliest confirmed contact candidate"
    elif wall_s <= bounds[1]:
        contact = "WITHIN CONTACT UNCERTAINTY INTERVAL"
    else:
        contact = "Past latest confirmed contact candidate"
    put(canvas, f"Human contact interval: {bounds[0]:.3f}-{bounds[1]:.3f}s | exact instant unknown | {contact}",
        20, 98, .59, RED if bounds[0] <= wall_s <= bounds[1] else MUTED, width=WIDTH - 40)
    left, right, y = 20, WIDTH - 20, 122
    cv2.rectangle(canvas, (left, y - 4), (right, y + 4), (78, 70, 59), -1)
    px = lambda value: left + round((right - left) * min(value, end_s) / end_s)
    cv2.rectangle(canvas, (px(bounds[0]), y - 10), (max(px(bounds[0]) + 2, px(bounds[1])), y + 10), RED, -1)
    cv2.circle(canvas, (px(wall_s), y), 7, WHITE, -1)
    shown = [panel(canvas, trial, images, wall_s, 20 + i * 710, total_sources)
             for i, trial in enumerate(trials)]
    put(canvas, "HOLD = old output retained on screen, not a fresh prediction. Dropped frames have no interpolated labels or poses.",
        20, 946, .57, AMBER, width=WIDTH - 40)
    put(canvas, "Local scheduled file replay only; no sensor/network/notification delivery timing. No training or model changes.",
        20, 971, .56, MUTED, width=WIDTH - 40)
    return canvas, shown


def validate_video(path, expected_frames):
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened(), path
    declared = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    fourcc = int(reader.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
    decoded = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        assert frame.shape == (HEIGHT, WIDTH, 3)
        decoded += 1
    reader.release()
    assert declared == decoded == expected_frames
    assert abs(fps - FPS) < .001
    assert codec.lower() in {"avc1", "h264"}, codec
    return dict(complete_decode=True, expected_frames=expected_frames,
                decoded_frames=decoded, declared_frames=declared, fps=fps, codec=codec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT)
    parser.add_argument("--mode", choices=["C"], default="C")
    parser.add_argument("--repeat", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    results_path = experiment / "results.json"
    assert results_path.exists() and read_json(results_path).get("complete"), "Wait until all timing trials complete"
    protocol_path = experiment / "protocol.json"
    protocol = read_json(protocol_path)
    assert protocol["training"] is False
    assert read_json(results_path)["all_frozen_hashes_unchanged"]
    model_path = ROOT / "runs/action-v7/training/model.json"
    assert sha256(model_path) == protocol["model_sha256"]
    baseline_path = experiment.parent / "evaluation-protocol.json"
    source = read_json(baseline_path)["source_inputs"][protocol["clip_id"]]
    source_path = Path(source["file"])
    assert sha256(source_path) == source["sha256"]
    trials = [load_trial(experiment, args.mode, policy, args.repeat) for policy in ["budget200ms", "costaware200ms"]]
    assert all(trial["summary"]["immutable_hashes_match"] for trial in trials)
    out = experiment / "media"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"C-r{args.repeat}-budget-vs-costaware"
    path, temp = out / f"{stem}.mp4", out / f"{stem}.partial.mp4"
    metadata_path = out / "metadata.json"
    assert not path.exists() and not temp.exists(), "Refuse to overwrite a rendered comparison"
    requested_ids = {row["source_frame_id"] for trial in trials for row in trial["rows"]}
    images = {}
    with zipfile.ZipFile(source_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".png"))
        assert len(names) == protocol["source_frames"]
        for fid in sorted(requested_ids):
            image = cv2.imdecode(np.frombuffer(archive.read(names[fid - 1]), np.uint8), cv2.IMREAD_COLOR)
            assert image is not None
            images[fid] = image
    end_s = max(trial["outputs"][-1] for trial in trials)
    frame_count = math.ceil(end_s * FPS) + 1
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "H.264 video writer unavailable"
    mapping = []
    contact_frame = math.ceil(protocol["contact_interval_s"][0] * FPS)
    poster_path = out / f"{stem}-contact.jpg"
    try:
        for frame in range(frame_count):
            wall_s = frame / FPS
            canvas, shown = render_canvas(trials, images, wall_s, end_s, args.mode, args.repeat,
                                           protocol["contact_interval_s"], protocol["source_frames"])
            writer.write(canvas)
            for item in shown:
                assert item["displayed_output_wall_s"] is None or item["displayed_output_wall_s"] <= wall_s
            mapping.append(dict(video_frame_0based=frame, wall_s=wall_s,
                                budget200ms=shown[0], costaware200ms=shown[1]))
            if frame == contact_frame:
                assert cv2.imwrite(str(poster_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    finally:
        writer.release()
    validation = validate_video(temp, frame_count)
    temp.rename(path)
    mapping_path = out / f"{stem}.display-map.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = read_json(metadata_path) if metadata_path.exists() else dict(
        schema="fallpre.unknown_reduction_wall_clock_media.v1", videos=[])
    artifact = dict(file=str(path.relative_to(experiment)), poster=str(poster_path.relative_to(experiment)),
        created_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode, repeat=args.repeat,
        selection="C repeat 2 was chosen before its timing results were available; repeat is not selected for best outcome.",
        wall_clock_playback=True, playback_speed=1, model_inference=False, training=False,
        frozen_model_sha256=protocol["model_sha256"], frozen_models_verified_by_trial=True,
        inputs={str(p): sha256(p) for p in [results_path, protocol_path, baseline_path, source_path,
               *(trial["rows_path"] for trial in trials), *(trial["summary_path"] for trial in trials)]},
        contact_interval_s=protocol["contact_interval_s"], exact_contact_unknown=True,
        time_axis="Video time equals measured elapsed wall time from scheduled local arrival origin, at 1x.",
        display_rule="Latest completed output at or before each display tick; raw associated source image held explicitly. No future frame or interpolated model output.",
        timing_quantization="Display rounds completed outputs up to the next 30 Hz display tick (<33.334 ms); displayed numeric output times retain measurements.",
        drop_count_rule="Cumulative drops whose recorded selection_wall_s <= displayed wall time; discarded only at worker selection as in the measured runner.",
        age_rule="Age at completion = output wall minus source time; display age now = current video wall minus held source time.",
        measurement_limit="Controlled local replay, excluding capture, network and notification delivery; not physical warning delivery.",
        last_measured_output_wall_s=end_s, encoded_duration_s=frame_count / FPS,
        tail_padding_s=frame_count / FPS - end_s, source_frames=protocol["source_frames"],
        width=WIDTH, height=HEIGHT, fps=FPS, frames=frame_count, sha256=sha256(path),
        display_map=str(mapping_path.relative_to(experiment)), display_map_sha256=sha256(mapping_path),
        validation=validation)
    metadata["videos"].append(artifact)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(video=str(path), poster=str(poster_path), metadata=str(metadata_path),
                         duration_s=artifact["encoded_duration_s"], validation=validation), ensure_ascii=False))


if __name__ == "__main__":
    main()
