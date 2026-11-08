"""URFD RGB -> COCO17 -> past-window ST-GCN++ action recognition.

No future frames, labels, or prerecorded action scores enter the classifier.
Pose replay is explicit and still runs a fresh action model. Single-person
continuity is bounded by source IDs/time and box motion, not asserted identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
SOURCES = {
    "yolo_pose": "https://docs.ultralytics.com/tasks/pose/",
    "stgcnpp_paper": "https://arxiv.org/abs/2205.09443",
    "stgcnpp_implementation": "https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/stgcnpp",
    "stgcnpp_model_zoo": "https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html#stgcn",
    "public_dataset": SOURCE_URL,
}
LABEL_SUPPORT = {
    "sitting_down": {"status": "exact_ntu60_class", "label": "sitting down"},
    "standing_up": {"status": "exact_ntu60_class", "label": "standing up (from sitting position)"},
    "falling": {"status": "exact_ntu60_class", "label": "falling"},
    "walking": {"status": "unsupported_single_person_class", "label": None},
    "bending": {"status": "no_exact_class; projected_geometry_candidate_only", "label": None},
}
LINKS = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4))
CONFIG_NAME = "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
POSE_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
ACTION_SHA256 = "86e1e77a7b27dd53c6ccc1600a4cc3c321ec8e04d17f25a41fbbb7896026945a"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_urfd_frames(rgb_zip: Path, sync_csv: Path, clip: str) -> list[dict]:
    """Exact ZIP/CSV ID join; the CSV's third column is not an action label."""
    stamps = {}
    previous_id, previous_ms = 0, -1
    with sync_csv.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            frame_id, ms = int(row[0]), int(row[1])
            if frame_id <= previous_id or ms <= previous_ms:
                raise ValueError("Source frame IDs and timestamps must strictly increase")
            stamps[frame_id] = ms
            previous_id, previous_ms = frame_id, ms
    members = {}
    pattern = re.compile(re.escape(clip) + r"-cam0-rgb-(\d+)\.png$")
    with zipfile.ZipFile(rgb_zip) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = pattern.fullmatch(Path(member.filename).name)
            if not match or member.file_size > 20 * (1 << 20):
                raise ValueError("Unexpected or oversized URFD ZIP member")
            frame_id = int(match[1])
            if frame_id in members:
                raise ValueError("Duplicate RGB source frame ID")
            members[frame_id] = member.filename
    if not stamps or set(stamps) != set(members):
        raise ValueError("RGB ZIP and synchronization CSV frame IDs differ")
    return [{"source_frame_id": i, "frame_index": i - 1, "source_timestamp_ms": stamps[i],
             "t": stamps[i] / 1000.0, "zip_member": members[i]} for i in sorted(stamps)]


def validate_instances(instances: list[dict]) -> None:
    for instance in instances:
        box, points, confidence = instance["box"], instance["kpts"], instance["det"]
        if len(box) != 4 or len(points) != 17 or any(len(p) != 3 for p in points):
            raise ValueError("Expected xyxy box and COCO17 (x,y,confidence)")
        if not all(math.isfinite(float(x)) for x in [*box, confidence, *(x for p in points for x in p)]):
            raise ValueError("Nonfinite pose observation")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty detection box")
        if not 0 <= confidence <= 1 or any(not 0 <= p[2] <= 1 for p in points):
            raise ValueError("Confidence outside [0,1]")


class PastWindow:
    """Never stitch missing/multiple outputs, frame gaps, or large box jumps."""

    def __init__(self, frames: int = 100, stride: int = 10, max_gap_s: float = .75):
        if frames < 100 or stride < 1 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Use >=100 frames, positive stride and finite positive time gap")
        self.size, self.stride, self.max_gap_s = frames, stride, max_gap_s
        self.history: deque[dict] = deque(maxlen=frames)
        self.run_id = 0
        self.count = 0
        self.reason = "warming_up"

    def reset(self, reason: str) -> None:
        self.history.clear()
        self.count = 0
        self.run_id += 1
        self.reason = reason

    def add(self, row: dict) -> list[dict] | None:
        if not math.isfinite(row["t"]):
            raise ValueError("Nonfinite source timestamp")
        validate_instances(row["instances"])
        if len(row["instances"]) != 1:
            self.reset("missing_person" if not row["instances"] else "multiple_people")
            return None
        if self.history:
            previous = self.history[-1]
            gap = row["t"] - previous["t"]
            if gap <= 0:
                self.reset("nonincreasing_source_time")
                return None
            a, b = previous["instances"][0]["box"], row["instances"][0]["box"]
            movement = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
            if row["source_frame_id"] != previous["source_frame_id"] + 1:
                self.reset("source_frame_gap")
            elif gap > self.max_gap_s:
                self.reset("source_time_gap")
            elif movement / (a[3] - a[1]) > 1.2:
                self.reset("box_continuity_break")
        self.history.append(row)
        self.count += 1
        if len(self.history) < self.size:
            self.reason = "warming_up"
            return None
        self.reason = "available"
        return list(self.history) if (self.count - self.size) % self.stride == 0 else None


def projected_signals(instance: dict, threshold: float = .6) -> dict:
    """View-dependent image measurements; no metric speed or semantic truth."""
    points = instance["kpts"]
    if any(points[i][2] < threshold for i in (5, 6, 11, 12, 13, 14, 15, 16)):
        return {"quality": "unavailable", "reason": "insufficient_joint_confidence"}
    shoulder = [(points[5][i] + points[6][i]) / 2 for i in (0, 1)]
    hip = [(points[11][i] + points[12][i]) / 2 for i in (0, 1)]
    ankle = [(points[15][i] + points[16][i]) / 2 for i in (0, 1)]
    torso_length = math.dist(shoulder, hip)
    if torso_length < 1e-6:
        return {"quality": "unavailable", "reason": "degenerate_torso"}
    angles = []
    for h, k, a in ((11, 13, 15), (12, 14, 16)):
        hip_p, knee, foot = [points[i][:2] for i in (h, k, a)]
        norm = math.dist(hip_p, knee) * math.dist(foot, knee)
        if norm < 1e-6:
            return {"quality": "unavailable", "reason": "degenerate_knee"}
        cosine = sum((hip_p[i] - knee[i]) * (foot[i] - knee[i]) for i in (0, 1)) / norm
        angles.append(math.degrees(math.acos(max(-1., min(1., cosine)))))
    return {"quality": "available", "torso_tilt_deg": math.degrees(math.atan2(
        abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1]))),
        "knee_mean_deg": statistics.mean(angles), "leg_vertical_over_torso": (ankle[1] - hip[1]) / torso_length,
        "hip_y_px": hip[1], "torso_length_px": torso_length}


class ProjectedTransitions:
    """Transparent exploratory candidates for current projected motion only."""

    def __init__(self):
        self.history: deque[tuple[float, dict]] = deque()
        self.run_id: int | None = None

    def update(self, row: dict, run_id: int) -> dict:
        signals = projected_signals(row["instances"][0]) if len(row["instances"]) == 1 else {
            "quality": "unavailable", "reason": "not_one_person"}
        if self.run_id != run_id or signals["quality"] != "available":
            self.history.clear()
        self.run_id = run_id
        result = {"kind": "heuristic_projected_motion", "candidate": "unknown", "signals": signals,
                  "source_time_s": row["t"], "evidence_window_s": [row["t"], row["t"]],
                  "limitation": "view-dependent candidate; no semantic ground truth or intention prediction"}
        if signals["quality"] != "available":
            return result
        self.history.append((row["t"], signals))
        while len(self.history) > 1 and self.history[0][0] < row["t"] - .6:
            self.history.popleft()
        start, old = self.history[0]
        result["evidence_window_s"] = [start, row["t"]]
        ratio_delta = signals["leg_vertical_over_torso"] - old["leg_vertical_over_torso"]
        knee_delta = signals["knee_mean_deg"] - old["knee_mean_deg"]
        signals.update(leg_ratio_delta=ratio_delta, knee_delta_deg=knee_delta)
        if row["t"] - start >= .15:
            if ratio_delta >= .18 and knee_delta >= 8:
                result["candidate"] = "rising_observed_candidate"
            elif ratio_delta <= -.18 and knee_delta <= -8:
                result["candidate"] = "sitting_down_observed_candidate"
        if result["candidate"] == "unknown" and signals["torso_tilt_deg"] >= 40 and signals["knee_mean_deg"] >= 155:
            result["candidate"] = "bent_torso_extended_knees_candidate"
        return result


def _local_import_paths(root: Path, *, actions: bool) -> None:
    paths = [root / "third_party/ultralytics"]
    if actions:
        paths += [root / "tmp/group-meeting/action-deps", root / "third_party/algorithms/mmaction2"]
    for path in reversed(paths):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class SkeletonClassifier:
    def __init__(self, root: Path, device: str = "cpu"):
        _local_import_paths(root, actions=True)
        import torch
        from mmaction.apis import inference_skeleton, init_recognizer

        base = root / "third_party/algorithms/mmaction2"
        config = base / "configs/skeleton/stgcnpp" / CONFIG_NAME
        weights = root / "runs/meeting-20261014/models/stgcnpp-ntu60-2d-joint.pth"
        labels = base / "tools/data/skeleton/label_map_ntu60.txt"
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(
                "Missing MMAction2 config/checkpoint. Install requirements-weekly-action.txt into the isolated action dependency directory; "
                "obtain the official config from SOURCES[stgcnpp_implementation] and checkpoint "
                "from the official model zoo, and place them at the paths reported here: "
                f"{config}; {weights}"
            )
        if sha256(weights) != ACTION_SHA256:
            raise ValueError("ST-GCN++ checkpoint checksum mismatch")
        torch.set_num_threads(2)
        self.model = init_recognizer(str(config), str(weights), device=device)
        self.inference = inference_skeleton
        self.labels = labels.read_text().splitlines()
        self.provenance = {"algorithm": "ST-GCN++", "training_dataset": "NTU60 2D / cross-subject",
                           "weights": str(weights), "weights_sha256": sha256(weights),
                           "config": str(config), "config_sha256": sha256(config),
                           "labels_sha256": sha256(labels), "device": device,
                           "test_pipeline": self.model.cfg.test_pipeline,
                           "versions": {name: importlib.metadata.version(name) for name in
                                        ("torch", "mmengine", "mmcv-lite", "numpy",
                                         "transformers", "tokenizers", "timm")},
                           "official_code_commit": "a5a167dff2399e2d182a60332325f9c0d4663517"}

    def classify(self, rows: list[dict], image_size: tuple[int, int]) -> dict:
        import numpy as np

        poses = []
        for row in rows:
            points = np.asarray(row["instances"][0]["kpts"], dtype=np.float32)
            poses.append({"keypoints": points[None, :, :2], "keypoint_scores": points[None, :, 2]})
        start = time.perf_counter()
        result = self.inference(self.model, poses, (image_size[1], image_size[0]))
        seconds = time.perf_counter() - start
        scores = result.pred_score.cpu().numpy()
        if scores.shape != (60,) or not np.isfinite(scores).all():
            raise ValueError("Invalid ST-GCN++ result")
        order = np.argsort(-scores)
        return {"kind": "past_window_action_classification", "model_inference_executed": True,
                "window_start_s": rows[0]["t"], "window_end_s": rows[-1]["t"],
                "source_frame_ids": [r["source_frame_id"] for r in rows],
                "source_times_s": [r["t"] for r in rows], "frames": len(rows),
                "top5": [{"label": self.labels[i], "label_index": int(i), "score": float(scores[i])}
                         for i in order[:5]],
                "all_scores": dict(zip(self.labels, map(float, scores))),
                "inference_wall_s": seconds, "short_sequence_wraparound": False,
                "interpretation": "closed-set past-window scores, not current-frame truth or intention"}


def _overlay(image, row: dict, latest: dict | None, transition: dict, mode: str, buffered: int):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    canvas = np.full((max(480, h), w + 460, 3), (29, 26, 24), dtype=np.uint8)
    canvas[:h, :w] = image
    for person in row["instances"]:
        x1, y1, x2, y2 = map(int, person["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (104, 224, 160), 2)
        for a, b in LINKS:
            if person["kpts"][a][2] >= .6 and person["kpts"][b][2] >= .6:
                cv2.line(canvas, tuple(map(int, person["kpts"][a][:2])),
                         tuple(map(int, person["kpts"][b][:2])), (104, 224, 160), 2)
        for x, y, confidence in person["kpts"]:
            if confidence >= .6:
                cv2.circle(canvas, (int(x), int(y)), 3, (64, 204, 255), -1)
    lines = [f"URFD / {row['clip_id']} / pose: {mode}",
             f"Frame {row['source_frame_id']} | CSV {row['t']:.3f}s",
             f"People {len(row['instances'])} | COCO17 skeleton", "",
             "ST-GCN++ / real model / NTU60 classes"]
    if latest:
        lines += [f"Past window: {latest['window_start_s']:.3f} - {latest['window_end_s']:.3f}s",
                  f"Result age: {row['t'] - latest['window_end_s']:.3f}s"]
        for item in latest["top5"][:3]:
            lines.append(f"{item['score']:.3f}  {item['label']}")
    else:
        lines += [f"Warming up / continuous poses: {buffered}", "No available action window"]
    lines += ["", "Projected motion (experimental):", transition["candidate"], "",
              "Walking: no generic single-person NTU class",
              "Bending: geometry candidate only", "Scores classify a PAST window.",
              "No subjective intention or future prediction."]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line[:62], (w + 14, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, (232, 235, 238) if i else (104, 224, 160), 1, cv2.LINE_AA)
    return canvas



def open_video_writer(path: Path, fps: float, size: tuple[int, int]):
    """Prefer H.264 for VSCode/browser playback, with a portable fallback."""
    import cv2

    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("Neither H.264 nor MPEG-4 video encoder is available")


def run_human_demo(output: str | Path, *, clip: str = "fall-01", pose_mode: str = "infer",
                   pose_backend: str = "yolo", device: str = "cpu", max_frames: int | None = None,
                   action_stride: int = 10, window_frames: int = 100, root: str | Path = ROOT) -> dict:
    """Create a new run directory. Never overwrite experiments or download silently."""
    import cv2
    import numpy as np

    root, output = Path(root).resolve(), Path(output).resolve()
    if pose_mode not in {"infer", "replay"} or pose_backend not in {"yolo", "rtmo"}:
        raise ValueError("pose_mode must be infer/replay and pose_backend yolo/rtmo")
    if pose_mode == "infer" and pose_backend != "yolo":
        raise ValueError("Fresh pose inference currently supports yolo; rtmo supports explicit replay")
    if clip not in {"fall-01", "adl-01"}:
        raise ValueError("This pinned public demo supports fall-01 and adl-01")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    window = PastWindow(window_frames, action_stride)
    observer = ProjectedTransitions()
    inputs = root / "runs/meeting-20261014/inputs"
    manifest = json.loads((root / "configs/weekly-data.lock.json").read_text())
    assets = manifest["clips"][clip]["assets"]
    rgb_zip, sync_csv = inputs / "urfd" / f"{clip}-cam0-rgb.zip", inputs / "urfd" / f"{clip}-data.csv"
    for path, kind in ((rgb_zip, "rgb"), (sync_csv, "sync")):
        if sha256(path) != assets[kind]["sha256"]:
            raise ValueError(f"URFD {kind} checksum mismatch")
    all_frames = load_urfd_frames(rgb_zip, sync_csv, clip)
    frames = all_frames[:max_frames] if max_frames else all_frames
    replay_header, replay_rows, model = None, {}, None
    pose_weights = root / "models/yolo11n-pose.pt"
    if pose_mode == "replay":
        replay_path = root / f"runs/meeting-20261014/{pose_backend}/{clip}/observations.jsonl"
        with replay_path.open() as handle:
            replay_header = json.loads(next(handle))
            for line in handle:
                row = json.loads(line)
                if row["source_frame_id"] in replay_rows:
                    raise ValueError("Duplicate replay frame")
                replay_rows[row["source_frame_id"]] = row
        for name, expected in (("source_sha256", assets["rgb"]["sha256"]),
                               ("sync_sha256", assets["sync"]["sha256"])):
            if replay_header.get(name) != expected:
                raise ValueError(f"Replay {name} differs from current input")
        pose_provenance = {"mode": "replay", "path": str(replay_path), "sha256": sha256(replay_path),
                           "original_model": replay_header["model"], "original_metadata": replay_header,
                           "model_inference_executed_this_run": False}
    else:
        _local_import_paths(root, actions=False)
        from ultralytics import YOLO
        if sha256(pose_weights) != POSE_SHA256:
            raise ValueError("YOLO pose checkpoint checksum mismatch")
        model = YOLO(str(pose_weights))
        pose_provenance = {"mode": "infer", "algorithm": "YOLO11n-pose", "device": device,
                           "weights_sha256": sha256(pose_weights), "model_inference_executed_this_run": True}
    classifier = SkeletonClassifier(root, device)
    output.mkdir(parents=True, exist_ok=False)
    counts: Counter = Counter()
    actions, pose_times, latest, writer, image_size = [], [], None, None, None
    video_codec = None
    fps = 1 / statistics.median([b["t"] - a["t"] for a, b in zip(all_frames, all_frames[1:])])
    metadata = {"kind": "meta", "schema_version": 1, "clip_id": clip, "synthetic": False,
                "source_url": SOURCE_URL, "camera_view": "cam0", "source_sha256": assets["rgb"]["sha256"],
                "sync_sha256": assets["sync"]["sha256"], "license": manifest["license"],
                "clock": "official synchronization CSV milliseconds / 1000",
                "pose": pose_provenance, "action": classifier.provenance,
                "sources": SOURCES, "requested_actions_support": LABEL_SUPPORT,
                "warmup": f"Requires {window_frames} uninterrupted source frames before first classification; "
                          "100 frames spans about 3.3s at 30Hz, not a current-frame label",
                "labels_used_by_models": False, "window_frames": window_frames, "action_stride": action_stride,
                "continuity": "single output, adjacent source IDs, <=0.75s gap, <=1.2 body-height box center jump",
                "interpretation": "past observed actions and projected candidates; no intention/forecast"}
    start = time.perf_counter()
    try:
        with zipfile.ZipFile(rgb_zip) as archive, (output / "observations.jsonl").open("x") as poses_out, \
                (output / "actions.jsonl").open("x") as actions_out, \
                (output / "transitions.jsonl").open("x") as transitions_out:
            _json_line(poses_out, metadata)
            for source in frames:
                frame = cv2.imdecode(np.frombuffer(archive.read(source["zip_member"]), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Could not decode public RGB frame")
                size = (frame.shape[1], frame.shape[0])
                if image_size is not None and size != image_size:
                    raise ValueError("Image dimensions changed")
                image_size = size
                if replay_header and list(size) != replay_header["image_size"]:
                    raise ValueError("Replay and image dimensions differ")
                if model is not None:
                    before = time.perf_counter()
                    result = model.predict(frame, imgsz=640, conf=.35, iou=.45, device=device,
                                           verbose=False, augment=False, half=False)[0]
                    latency = time.perf_counter() - before
                    boxes, scores, points = (result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                             result.keypoints.data.cpu().tolist())
                    if not len(boxes) == len(scores) == len(points):
                        raise ValueError("Pose instance alignment mismatch")
                    instances = [{"box": b, "det": s, "kpts": p} for b, s, p in zip(boxes, scores, points)]
                    pose_times.append(latency)
                else:
                    replay = replay_rows[source["source_frame_id"]]
                    if replay["t"] != source["t"] or replay["frame_index"] != source["frame_index"]:
                        raise ValueError("Replay source timestamp/frame mismatch")
                    instances, latency = replay["instances"], None
                row = {k: v for k, v in source.items() if k != "zip_member"}
                row.update(clip_id=clip, instances=instances, inference_wall_s=latency)
                selected = window.add(row)
                if not window.history or len(window.history) == 1:
                    latest = None  # Never display a cached action across an identity/gap boundary.
                transition = observer.update(row, window.run_id)
                if selected is not None:
                    latest = classifier.classify(selected, image_size)
                    latest["continuity_run_id"] = window.run_id
                    actions.append(latest)
                    _json_line(actions_out, latest)
                _json_line(poses_out, row)
                _json_line(transitions_out, transition)
                counts["zero_people" if not instances else "one_person" if len(instances) == 1 else "multiple_people"] += 1
                if len(instances) == 1 and all(instances[0]["kpts"][i][2] >= .6 for i in (5, 6, 11, 12, 15, 16)):
                    counts["single_six_joints_ge_0_6"] += 1
                counts["transition_" + transition["candidate"]] += 1
                canvas = _overlay(frame, row, latest, transition, pose_mode, len(window.history))
                if writer is None:
                    writer, video_codec = open_video_writer(
                        output / "overlay.mp4", fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
                if selected is not None:
                    cv2.imwrite(str(output / "preview.jpg"), canvas)
                    cv2.imwrite(str(output / f"action-window-{source['source_frame_id']:04d}.jpg"), canvas)
                if source == frames[-1]:
                    cv2.imwrite(str(output / "final.jpg"), canvas)
                    if not actions:
                        cv2.imwrite(str(output / "preview.jpg"), canvas)
    finally:
        if writer is not None:
            writer.release()
    count = len(frames)
    metrics = {"clip_id": clip, "frame_count": count, "total_source_frames": len(all_frames),
               "source_start_s": frames[0]["t"], "source_end_s": frames[-1]["t"],
               "counts": dict(counts), "single_person_coverage": counts["one_person"] / count,
               "six_joint_coverage": counts["single_six_joints_ge_0_6"] / count,
               "action_windows": len(actions), "action_top1_histogram": dict(Counter(a["top5"][0]["label"] for a in actions)),
               "pose_inference_mean_ms": 1000 * statistics.mean(pose_times) if pose_times else None,
               "action_inference_mean_ms": 1000 * statistics.mean(a["inference_wall_s"] for a in actions) if actions else None,
               "processing_wall_s": time.perf_counter() - start,
               "accuracy": None, "accuracy_reason": "No frame-level 60-class ground truth supplied; coverage is not accuracy",
               "video_codec": video_codec,
               "video_timing": "CFR display at median source FPS; JSON retains exact irregular source times",
               "metadata": metadata, "output_dir": str(output)}
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/weekly-human-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    parser.add_argument("--clip", choices=["fall-01", "adl-01"], default="fall-01")
    parser.add_argument("--pose-mode", choices=["infer", "replay"], default="infer")
    parser.add_argument("--pose-backend", choices=["yolo", "rtmo"], default="yolo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--action-stride", type=int, default=10)
    parser.add_argument("--window-frames", type=int, default=100)
    args = parser.parse_args(argv)
    result = run_human_demo(**vars(args))
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
