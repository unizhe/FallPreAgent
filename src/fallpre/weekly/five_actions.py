"""Frozen, supervised six-class skeleton baseline for five actions + background.

COCO17 -> interpretable past-window features -> class-balanced multinomial
logistic regression (L2, scipy L-BFGS). This is a small-data research baseline,
not the unrelated NTU60 label space and not subjective intention prediction.
All normalization and weights are learned from train clips only. Evaluation
retains missing poses, warmup and uncertain classifications as ``unknown``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .human import (
    LINKS,
    POSE_SHA256,
    ROOT,
    _local_import_paths,
    open_video_writer,
    sha256,
    validate_instances,
)

LABELS = ("walking", "sitting_down", "standing_up", "bending", "falling", "background")
ALL_LABELS = LABELS + ("unknown",)
ALGORITHM_SOURCES = {
    "multinomial_logistic_regression": "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
    "optimizer": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html",
    "pose": "https://docs.ultralytics.com/tasks/pose/",
}


@dataclass(frozen=True)
class FeatureConfig:
    window_s: float = 1.0
    min_span_s: float = .8
    min_frames: int = 6
    max_gap_s: float = .25
    joint_confidence: float = .25
    min_valid_fraction: float = .7
    confidence_threshold: float = .4
    l2: float = .03
    sample_stride_s: float = .1

    def __post_init__(self):
        if not (0 < self.min_span_s <= self.window_s and self.min_frames >= 3 and self.max_gap_s > 0
                and 0 < self.min_valid_fraction <= 1 and 0 <= self.joint_confidence <= 1
                and 0 <= self.confidence_threshold <= 1 and self.l2 > 0 and self.sample_stride_s > 0):
            raise ValueError("Invalid feature or classifier configuration")


SIGNALS = ("torso_tilt", "left_knee", "right_knee", "left_hip", "right_hip",
           "hip_above_ankle", "shoulder_above_ankle", "foot_width", "foot_height_difference",
           "knee_width", "body_width", "body_height", "box_aspect", "torso_length",
           "hip_x", "hip_y")
FEATURE_NAMES = tuple(f"{name}_{stat}" for stat in ("mean", "std", "last", "delta", "slope") for name in SIGNALS)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    left, right = a - b, c - b
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.arccos(np.clip(np.dot(left, right) / max(norm, 1e-8), -1, 1)) / np.pi)


def _signals(instance: dict, config: FeatureConfig) -> tuple[np.ndarray, float] | None:
    """Normalize by torso, never by image scale or a test-set statistic.

    One reliable body side suffices. A low-confidence counterpart is mirrored
    to the reliable side; quality and side coverage are recorded separately.
    Hip image translation is centered over the window by extract_features.
    """
    p = np.asarray(instance["kpts"], dtype=float)
    left, right = (5, 11, 13, 15), (6, 12, 14, 16)
    good_left = np.min(p[list(left), 2]) >= config.joint_confidence
    good_right = np.min(p[list(right), 2]) >= config.joint_confidence
    if not good_left and not good_right:
        return None
    if not good_left:
        p[list(left)] = p[list(right)]
    if not good_right:
        p[list(right)] = p[list(left)]
    shoulder, hip, ankle = [p[list(indices), :2].mean(axis=0) for indices in ((5, 6), (11, 12), (15, 16))]
    torso = float(np.linalg.norm(hip - shoulder))
    if torso < 5:
        return None
    x1, y1, x2, y2 = instance["box"]
    values = [
        float(np.arctan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1])) / (np.pi / 2)),
        _angle(p[11, :2], p[13, :2], p[15, :2]), _angle(p[12, :2], p[14, :2], p[16, :2]),
        _angle(p[5, :2], p[11, :2], p[13, :2]), _angle(p[6, :2], p[12, :2], p[14, :2]),
        (ankle[1] - hip[1]) / torso, (ankle[1] - shoulder[1]) / torso,
        abs(p[15, 0] - p[16, 0]) / torso, abs(p[15, 1] - p[16, 1]) / torso,
        abs(p[13, 0] - p[14, 0]) / torso, (x2 - x1) / torso, (y2 - y1) / torso,
        (x2 - x1) / (y2 - y1), torso, hip[0], hip[1],
    ]
    return np.asarray(values), (int(good_left) + int(good_right)) / 2


def extract_features(rows: list[dict], config: FeatureConfig | None = None) -> dict:
    """Pure past-window feature extraction; labels are never read here."""
    config = config or FeatureConfig()
    result = {"quality": "unavailable", "reason": "warming_up", "features": None,
              "evidence_window_s": [rows[0]["t"], rows[-1]["t"]] if rows else None,
              "support_frames": len(rows), "valid_frames": 0}
    if len(rows) < config.min_frames:
        return result
    times = np.asarray([r["t"] for r in rows], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Window timestamps must be finite and strictly increasing")
    span = times[-1] - times[0]
    if span < config.min_span_s:
        return result
    if span > config.window_s + 1e-6:
        raise ValueError("Feature window exceeds frozen past-window duration")
    if np.max(np.diff(times)) > config.max_gap_s:
        return dict(result, reason="source_time_gap")
    values, valid_times, sides = [], [], []
    for row in rows:
        validate_instances(row["instances"])
        if len(row["instances"]) != 1:
            return dict(result, reason="not_one_target_person")
        item = _signals(row["instances"][0], config)
        if item is not None:
            values.append(item[0])
            valid_times.append(row["t"])
            sides.append(item[1])
    result.update(valid_frames=len(values), valid_fraction=len(values) / len(rows))
    if (len(values) < config.min_frames or len(values) / len(rows) < config.min_valid_fraction
            or _signals(rows[-1]["instances"][0], config) is None):
        return dict(result, reason="insufficient_joint_confidence")
    a = np.asarray(values)
    scale = float(np.median(a[:, 13]))
    a[:, 13] /= scale
    a[:, 14:16] = (a[:, 14:16] - a[0, 14:16]) / scale
    # Irregular source clock is retained. No synthetic wraparound or future padding.
    t = np.asarray(valid_times) - np.mean(valid_times)
    slope = t @ a / max(float(t @ t), 1e-8)
    delta = a[-1] - a[0]
    features = np.concatenate((a.mean(axis=0), a.std(axis=0), a[-1], delta, slope))
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite temporal features")
    return dict(result, quality="available", reason=None, features=features,
                reliable_body_side_fraction=float(np.mean(sides)))


class FiveActionClassifier:
    """Portable frozen JSON model; scores are uncalibrated softmax outputs."""

    def __init__(self, artifact: dict):
        if artifact.get("schema") != "fallpre.five_actions.model.v1" or tuple(artifact["labels"]) != LABELS:
            raise ValueError("Unsupported model schema or label ordering")
        if tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Model feature specification mismatch")
        self.artifact = artifact
        self.config = FeatureConfig(**artifact["config"])
        self.mean, self.scale = (np.asarray(artifact[k], dtype=float) for k in ("mean", "scale"))
        self.weights = np.asarray(artifact["weights"], dtype=float)
        self.bias = np.asarray(artifact["bias"], dtype=float)
        n = len(FEATURE_NAMES)
        if (self.mean.shape != (n,) or self.scale.shape != (n,) or self.weights.shape != (n, len(LABELS))
                or self.bias.shape != (len(LABELS),) or np.any(self.scale <= 0)
                or not all(np.isfinite(x).all() for x in (self.mean, self.scale, self.weights, self.bias))):
            raise ValueError("Invalid model parameter shapes or numbers")

    @classmethod
    def load(cls, path: str | Path) -> "FiveActionClassifier":
        return cls(json.loads(Path(path).read_text()))

    def predict(self, rows: list[dict]) -> dict:
        evidence = extract_features(rows, self.config)
        features = evidence.pop("features")
        evidence.update(label="unknown", scores={}, model_inference_executed=False)
        if features is None:
            return evidence
        logits = ((features - self.mean) / self.scale) @ self.weights + self.bias
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        index = int(np.argmax(scores))
        evidence.update(scores=dict(zip(LABELS, map(float, scores))), model_inference_executed=True,
                        confidence=float(scores[index]), top1=LABELS[index])
        if scores[index] < self.config.confidence_threshold:
            evidence.update(quality="uncertain", reason="low_classifier_confidence")
        else:
            evidence["label"] = LABELS[index]
        return evidence


class OnlineFiveAction:
    """One explicitly selected person, current/past rows only, reset on breaks."""

    def __init__(self, model: FiveActionClassifier, *, expected_frame_stride: int = 1):
        if not isinstance(expected_frame_stride, int) or expected_frame_stride < 1:
            raise ValueError("Expected source frame stride must be a positive integer")
        self.model = model
        self.expected_frame_stride = expected_frame_stride
        self.history: deque[dict] = deque()
        self.last_source = None
        self.run_id = 0
        self.reset_reason = "warming_up"

    def reset(self, reason: str = "identity_change") -> None:
        self.history.clear()
        self.last_source = None
        self.run_id += 1
        self.reset_reason = reason

    def update(self, row: dict) -> dict:
        if not math.isfinite(row["t"]):
            raise ValueError("Nonfinite source timestamp")
        validate_instances(row["instances"])
        reason = None
        if len(row["instances"]) != 1:
            reason = "missing_person" if not row["instances"] else "ambiguous_target_person"
        elif self.last_source is not None:
            previous = self.last_source
            gap = row["t"] - previous["t"]
            a, b = previous["instances"][0]["box"], row["instances"][0]["box"]
            jump = math.hypot(a[0] + a[2] - b[0] - b[2], a[1] + a[3] - b[1] - b[3]) / 2
            if gap <= 0:
                reason = "nonincreasing_source_time"
            elif gap > self.model.config.max_gap_s:
                reason = "source_time_gap"
            elif ("source_frame_id" in previous and "source_frame_id" in row
                  and row["source_frame_id"] != previous["source_frame_id"] + self.expected_frame_stride):
                reason = "source_frame_gap"
            elif jump > 1.2 * (a[3] - a[1]):
                reason = "box_continuity_break"
        if reason:
            self.reset(reason)
        if len(row["instances"]) == 1 and reason != "nonincreasing_source_time":
            self.history.append(row)
            self.last_source = row
            while self.history and self.history[0]["t"] < row["t"] - self.model.config.window_s - 1e-8:
                self.history.popleft()
        result = self.model.predict(list(self.history))
        if reason:
            result["reason"] = reason
        result.update(source_time_s=row["t"], source_frame_id=row.get("source_frame_id"), continuity_run_id=self.run_id)
        return result


def load_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    entries = manifest["entries"]
    seen, assets, source_groups, original_hashes = set(), {}, set(), set()
    for entry in entries:
        clip, split = entry["clip_id"], entry["split"]
        if clip in seen or split not in {"train", "dev", "test"}:
            raise ValueError("Duplicate clip IDs or invalid split; clips cannot cross splits")
        seen.add(clip)
        group = entry.get("source_group_id", clip)
        original = entry.get("source", {}).get("sha256")
        if group in source_groups or (original is not None and original in original_hashes):
            raise ValueError("Duplicate underlying source group across manifest entries")
        source_groups.add(group)
        if original is not None:
            original_hashes.add(original)
        raw_path = Path(entry["video_path"])
        video = raw_path if raw_path.is_absolute() else ROOT / raw_path
        entry["video_path"] = str(video.resolve())
        # Also catch renamed duplicate source videos, not only duplicate clip names.
        digest = entry.get("video_sha256") or (sha256(video) if video.exists() else None)
        if digest is not None:
            if video.exists() and sha256(video) != digest:
                raise ValueError(f"Video checksum mismatch: {clip}")
            if digest in assets:
                raise ValueError("Duplicate source video across manifest entries")
            assets[digest] = split
            entry["video_sha256"] = digest
        previous_end = -math.inf
        for segment in entry["segments"]:
            start, end = segment["start_s"], segment["end_s"]
            if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start
                    or start < previous_end - 1e-8 or segment["label"] not in ALL_LABELS):
                raise ValueError("Invalid/overlapping action annotation intervals")
            if "duration_s" in entry and end > entry["duration_s"] + 1e-6:
                raise ValueError("Action annotation extends beyond the video duration")
            previous_end = end
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256(path)
    return manifest


def label_at(entry: dict, time_s: float) -> str:
    """Half-open visual annotation intervals; gaps are unknown, never background."""
    for segment in entry["segments"]:
        if segment["start_s"] <= time_s < segment["end_s"]:
            return segment["label"]
    return "unknown"


def pose_cache_path(entry: dict, cache_dir: Path) -> Path:
    return cache_dir / entry["clip_id"] / "observations.jsonl"


def read_pose_cache(entry: dict, cache_dir: Path) -> tuple[dict, list[dict]]:
    path = pose_cache_path(entry, cache_dir)
    with path.open() as handle:
        header = json.loads(next(handle))
        rows = [json.loads(line) for line in handle if line.strip()]
    if (header.get("kind") != "meta" or header.get("clip_id") != entry["clip_id"]
            or header.get("source_sha256") != entry["video_sha256"]
            or header.get("weights_sha256") != POSE_SHA256):
        raise ValueError("Pose cache source/model provenance mismatch")
    if len(rows) != header["frame_count"]:
        raise ValueError("Incomplete pose cache")
    for index, row in enumerate(rows):
        if row["source_frame_id"] != index + 1 or row["clip_id"] != entry["clip_id"]:
            raise ValueError("Pose cache frame/clip alignment mismatch")
        expected_t = index / header["fps"]
        if not math.isclose(row["t"], expected_t, abs_tol=1e-8):
            raise ValueError("Pose cache clock differs from source video clock")
        validate_instances(row["instances"])
    return header, rows


def extract_poses(manifest: dict, cache_dir: str | Path, *, device: str = "cpu", clips: list[str] | None = None) -> dict:
    """Fresh pretrained pose inference on public RGB videos; no action labels read."""
    import cv2

    _local_import_paths(ROOT, actions=False)
    from ultralytics import YOLO

    weights = ROOT / "models/yolo11n-pose.pt"
    if sha256(weights) != POSE_SHA256:
        raise ValueError("YOLO checkpoint checksum mismatch")
    model = None
    cache_dir = Path(cache_dir)
    summaries = []
    for entry in manifest["entries"]:
        if clips and entry["clip_id"] not in clips:
            continue
        path = pose_cache_path(entry, cache_dir)
        if path.exists():
            header, rows = read_pose_cache(entry, cache_dir)
            summaries.append({"clip_id": entry["clip_id"], "frame_count": len(rows), "mode": "verified_cache"})
            continue
        if model is None:
            model = YOLO(str(weights))
        cap = cv2.VideoCapture(entry["video_path"])
        if not cap.isOpened():
            raise ValueError(f"Cannot open {entry['video_path']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError("Video has no valid FPS")
        rows, index, size = [], 0, None
        try:
            while True:
                okay, frame = cap.read()
                if not okay:
                    break
                size = [frame.shape[1], frame.shape[0]]
                pred = model.predict(frame, imgsz=640, conf=.35, iou=.45, device=device,
                                     verbose=False, augment=False, half=False)[0]
                boxes, scores, points = (pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(),
                                         pred.keypoints.data.cpu().tolist())
                if not len(boxes) == len(scores) == len(points):
                    raise ValueError("YOLO pose instance alignment mismatch")
                rows.append({"clip_id": entry["clip_id"], "source_frame_id": index + 1, "frame_index": index,
                             "t": index / fps,
                             "instances": [{"box": b, "det": s, "kpts": p} for b, s, p in zip(boxes, scores, points)]})
                index += 1
        finally:
            cap.release()
        if not rows or len(rows) != expected:
            raise ValueError("Video decoded frame count differs from container frame count")
        header = {"kind": "meta", "schema": "fallpre.five_actions.pose.v1", "clip_id": entry["clip_id"],
                  "source_sha256": entry["video_sha256"], "weights_sha256": POSE_SHA256,
                  "algorithm": "YOLO11n-pose", "device": device, "frame_count": len(rows), "fps": fps,
                  "image_size": size, "clock": "decoded public derived RGB video frame_index / container FPS",
                  "action_labels_used_by_pose": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            for row in [header, *rows]:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        summaries.append({"clip_id": entry["clip_id"], "frame_count": len(rows), "mode": "infer"})
        print(json.dumps(summaries[-1]), flush=True)
    return {"clips": summaries}


def _windows(rows: list[dict], config: FeatureConfig):
    """Same online history and reset policy for training, evaluation and demo."""
    # A shape-valid inert model lets the exact streaming window implementation
    # serve training too, without any annotation or trained parameters in it.
    empty = {"schema": "fallpre.five_actions.model.v1", "labels": LABELS, "feature_names": FEATURE_NAMES,
             "config": asdict(config), "mean": np.zeros(len(FEATURE_NAMES)).tolist(),
             "scale": np.ones(len(FEATURE_NAMES)).tolist(),
             "weights": np.zeros((len(FEATURE_NAMES), len(LABELS))).tolist(), "bias": [0.] * len(LABELS)}
    stream = OnlineFiveAction(FiveActionClassifier(empty))
    next_t = -math.inf
    for row in rows:
        stream.update(row)
        if row["t"] + 1e-8 >= next_t:
            yield row, list(stream.history)
            next_t = row["t"] + config.sample_stride_s


def fit_model(features: np.ndarray, targets: list[str], *, config: FeatureConfig | None = None,
              provenance: dict | None = None) -> FiveActionClassifier:
    """Actual supervised optimization; call with train-only rows."""
    import scipy
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    config = config or FeatureConfig()
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(targets) or not np.isfinite(x).all():
        raise ValueError("Invalid training matrix")
    counts = Counter(targets)
    if set(counts) != set(LABELS):
        raise ValueError(f"Train split needs supported windows for all six classes; got {dict(counts)}")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < .05, .05, scale)
    x = (x - mean) / scale
    y = np.asarray([LABELS.index(label) for label in targets])
    sample_weight = np.asarray([1 / (len(LABELS) * counts[label]) for label in targets])
    n, k = len(FEATURE_NAMES), len(LABELS)

    def objective(flat):
        w, b = flat[:n*k].reshape(n, k), flat[n*k:]
        logits = x @ w + b
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -float(sample_weight @ log_prob[np.arange(len(y)), y]) + config.l2 * float((w*w).sum()) / 2
        error = np.exp(log_prob)
        error[np.arange(len(y)), y] -= 1
        error *= sample_weight[:, None]
        grad = np.concatenate(((x.T @ error + config.l2 * w).ravel(), error.sum(axis=0)))
        return loss, grad

    optimized = minimize(objective, np.zeros(n*k+k), jac=True, method="L-BFGS-B",
                         options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"Supervised optimizer did not converge: {optimized.message}")
    artifact = {"schema": "fallpre.five_actions.model.v1", "algorithm": "class-balanced multinomial logistic regression",
                "labels": LABELS, "feature_names": FEATURE_NAMES, "config": asdict(config),
                "mean": mean.tolist(), "scale": scale.tolist(),
                "weights": optimized.x[:n*k].reshape(n, k).tolist(), "bias": optimized.x[n*k:].tolist(),
                "training": {"window_counts": dict(counts), "optimizer": "scipy L-BFGS-B",
                             "iterations": int(optimized.nit), "objective": float(optimized.fun),
                             "converged": bool(optimized.success), "provenance": provenance or {}},
                "implementation_sha256": sha256(Path(__file__)),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "sources": ALGORITHM_SOURCES,
                "limitations": ["Small clip-disjoint public-data baseline; not subject-disjoint unless separately documented.",
                                "Scores are not calibrated probabilities; no future intention prediction.",
                                "2D features are view dependent; top-down transfer needs separate evaluation."]}
    return FiveActionClassifier(artifact)


def train(manifest: dict, cache_dir: str | Path, model_path: str | Path,
          *, config: FeatureConfig | None = None) -> dict:
    config = config or FeatureConfig()
    x, y, sources, skipped = [], [], [], Counter()
    for entry in manifest["entries"]:
        if entry["split"] != "train":
            continue
        _, rows = read_pose_cache(entry, Path(cache_dir))
        sources.append({"clip_id": entry["clip_id"], "video_sha256": entry["video_sha256"],
                        "pose_sha256": sha256(pose_cache_path(entry, Path(cache_dir))),
                        "segments": entry["segments"]})
        for row, window in _windows(rows, config):
            target = label_at(entry, row["t"])
            evidence = extract_features(window, config)
            if target == "unknown":
                skipped["unannotated"] += 1
            elif evidence["features"] is None:
                skipped[evidence["reason"]] += 1
            else:
                x.append(evidence["features"])
                y.append(target)
    model = fit_model(np.asarray(x), y, config=config,
                      provenance={"manifest_sha256": manifest["manifest_sha256"], "train_clips": sources,
                                  "skipped_train_windows": dict(skipped),
                                  "annotation_status": manifest.get("annotation_status", "unspecified"),
                                  "independent_human_annotators": manifest.get("independent_human_annotators"),
                                  "test_data_used_for_fit_or_hyperparameters": False})
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(model.artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return dict(model.artifact["training"], model_path=str(output.resolve()), model_sha256=sha256(output))


def classification_metrics(records: list[dict]) -> dict:
    matrix = np.zeros((len(ALL_LABELS), len(ALL_LABELS)), dtype=int)
    for record in records:
        matrix[ALL_LABELS.index(record["target"]), ALL_LABELS.index(record["label"])] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp, support = int(matrix[i, i]), int(matrix[i].sum())
        # Unannotated targets are excluded from precision's denominator.
        predicted = int(matrix[:len(LABELS), i].sum())
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else None
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1 if support else None,
                            "support": support, "predicted": predicted,
                            "unknown_predictions": int(matrix[i, -1]),
                            "support_clips": sorted({r["clip_id"] for r in records if r["target"] == label})}
    known = int(matrix[:len(LABELS)].sum())
    correct = int(sum(matrix[i, i] for i in range(len(LABELS))))
    classified = known - int(matrix[:len(LABELS), -1].sum())
    f1s = [per_class[label]["f1"] for label in LABELS[:5] if per_class[label]["support"]]
    return {"labels": ALL_LABELS, "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "per_class": per_class, "annotated_windows": known, "unannotated_windows": int(matrix[-1].sum()),
            "accuracy_including_unknown_as_error": correct / known if known else None,
            "classified_coverage": classified / known if known else None,
            "selective_accuracy": correct / classified if classified else None,
            "five_action_macro_f1_supported_classes": float(np.mean(f1s)) if f1s else None,
            "all_five_actions_have_test_support": all(per_class[label]["support"] > 0 for label in LABELS[:5]),
            "five_action_macro_f1": float(np.mean(f1s)) if len(f1s) == 5 else None}


def evaluate(manifest: dict, cache_dir: str | Path, model_path: str | Path, output: str | Path,
             *, split: str = "test") -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation requires an explicit dev or test split")
    model = FiveActionClassifier.load(model_path)
    fitted = model.artifact["training"]["provenance"].get("train_clips", [])
    if model.artifact["training"]["provenance"].get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed after fitting; evaluation requires the frozen annotation/split manifest")
    fitted_ids = {entry["clip_id"] for entry in fitted}
    fitted_hashes = {entry["video_sha256"] for entry in fitted}
    records, clips = [], []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        if entry["clip_id"] in fitted_ids or entry["video_sha256"] in fitted_hashes:
            raise ValueError("Evaluation source overlaps fitted training clips")
        _, rows = read_pose_cache(entry, Path(cache_dir))
        clip_records = []
        stream = OnlineFiveAction(model)
        next_t = -math.inf
        for row in rows:
            # Use the actual online result so missing-person, ambiguity and
            # continuity-reset reasons survive. Re-predicting an empty history
            # would preserve the unknown label but incorrectly call it warmup.
            prediction = stream.update(row)
            if row["t"] + 1e-8 < next_t:
                continue
            next_t = row["t"] + model.config.sample_stride_s
            prediction.update(clip_id=entry["clip_id"], source_frame_id=row["source_frame_id"],
                              source_time_s=row["t"], target=label_at(entry, row["t"]))
            records.append(prediction)
            clip_records.append(prediction)
        clips.append({"clip_id": entry["clip_id"], "video_sha256": entry["video_sha256"],
                      "pose_sha256": sha256(pose_cache_path(entry, Path(cache_dir))),
                      "metrics": classification_metrics(clip_records)})
    if not clips:
        raise ValueError(f"No {split} clips in manifest")
    result = {"schema": "fallpre.five_actions.evaluation.v1", "split": split,
              "model_sha256": sha256(model_path), "manifest_sha256": manifest["manifest_sha256"],
              "dataset": manifest.get("dataset"), "annotation_status": manifest.get("annotation_status", "unspecified"),
              "independent_human_annotators": manifest.get("independent_human_annotators"),
              "metric_interpretation": "Agreement against the manifest's provisional visual labels; "
                                       "not an official or independently human-validated five-action benchmark.",
              "model_config": asdict(model.config), "metrics": classification_metrics(records), "clips": clips,
              "unknown_reasons": dict(Counter(r["reason"] for r in records if r["label"] == "unknown")),
              "protocol": "Whole clips disjoint. Train-only scaling and fitting. Frozen config. "
                          "Causal past 1s window predicts annotation at its endpoint. "
                          "Unannotated intervals remain unknown. Warmup/pose failures count as abstentions, not dropped tests.",
              "limitations": ["Overlapping windows are correlated; window counts are not independent sample counts.",
                              "Provisional agent-visual annotations require human review; not official five-action ground truth.",
                              "If classes come from different datasets, source/class confounding limits interpretation.",
                              "Clip-disjoint split does not establish subject-disjoint or overhead-camera generalization."]}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (out / "predictions.jsonl").open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    _plot_evaluation(result, out / "confusion-matrix.png")
    return dict(result, output_dir=str(out.resolve()))


def _plot_evaluation(result: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(result["metrics"]["confusion_matrix_rows_true_columns_predicted"])
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(matrix, cmap="Blues")
    for (r, c), value in np.ndenumerate(matrix):
        axis.text(c, r, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    axis.set(xticks=range(7), yticks=range(7), xticklabels=ALL_LABELS, yticklabels=ALL_LABELS,
             xlabel="Predicted (unknown = abstain)", ylabel="Provisional visual endpoint label",
             title="Public RGB subset | clip-disjoint frozen baseline")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    fig.text(.5, .015, "Overlapping windows are correlated. Counts are windows, not independent subjects.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_demo(manifest: dict, cache_dir: str | Path, model_path: str | Path, clip_id: str,
                output: str | Path, *, pose_mode: str = "infer", device: str = "cpu") -> dict:
    """Fresh model inference per frame, optional explicit cached pose replay."""
    import cv2

    if pose_mode not in {"infer", "replay"}:
        raise ValueError("pose_mode must be infer or replay")
    entry = next((e for e in manifest["entries"] if e["clip_id"] == clip_id), None)
    if entry is None:
        raise ValueError(f"Unknown clip {clip_id}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    if pose_mode == "infer":
        # A fresh directory guarantees infer really executes the pose network.
        local_cache = out / "pose-inference"
        extract_poses(manifest, local_cache, device=device, clips=[clip_id])
    else:
        local_cache = Path(cache_dir)
    header, rows = read_pose_cache(entry, local_cache)
    model = FiveActionClassifier.load(model_path)
    stream = OnlineFiveAction(model)
    cap = cv2.VideoCapture(entry["video_path"])
    writer, codec, counts = None, None, Counter()
    try:
        with (out / "actions.jsonl").open("x") as handle:
            for row in rows:
                okay, frame = cap.read()
                if not okay:
                    raise ValueError("Video/pose alignment ended early")
                prediction = stream.update(row)
                # Provisional visual labels are used only for display AFTER inference.
                record = dict(prediction, target=label_at(entry, row["t"]), clip_id=clip_id)
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                counts[prediction["label"]] += 1
                h, w = frame.shape[:2]
                canvas = np.full((max(h, 480), w + 460, 3), (29, 26, 24), dtype=np.uint8)
                canvas[:h, :w] = frame
                for person in row["instances"]:
                    box, points = person["box"], person["kpts"]
                    cv2.rectangle(canvas, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (104, 224, 160), 2)
                    for a, b in LINKS:
                        if min(points[a][2], points[b][2]) >= model.config.joint_confidence:
                            cv2.line(canvas, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])), (104, 224, 160), 2)
                lines = [f"Public video {clip_id} | split: {entry['split']}", f"Pose: {pose_mode} / YOLO11n COCO17",
                         f"Time {row['t']:.2f}s | frame {row['source_frame_id']}", "", "Five actions + background / trained baseline",
                         f"PREDICTED: {prediction['label']}", f"Provisional label: {record['target']}",
                         f"Quality: {prediction['quality']}", f"Reason: {prediction['reason'] or 'available'}",
                         f"Past support: {prediction['support_frames']} frames"]
                if prediction["evidence_window_s"]:
                    lines.append("Window: %.2f - %.2fs" % tuple(prediction["evidence_window_s"]))
                lines += [""] + [f"{score:.3f}  {label}" for label, score in sorted(prediction["scores"].items(), key=lambda p: -p[1])[:3]]
                lines += ["", "Reference labels need human review.", "View-dependent research baseline.", "No subjective intention or future prediction."]
                for i, line in enumerate(lines):
                    cv2.putText(canvas, line[:62], (w + 12, 25 + i*22), cv2.FONT_HERSHEY_SIMPLEX, .46,
                                (104, 224, 160) if i in (0, 5) else (232, 235, 238), 1, cv2.LINE_AA)
                if writer is None:
                    writer, codec = open_video_writer(out / "overlay.mp4", header["fps"], (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
                if row is rows[-1] or row["source_frame_id"] == max(1, len(rows)//2):
                    cv2.imwrite(str(out / "preview.jpg"), canvas)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    result = {"clip_id": clip_id, "split": entry["split"], "pose_mode": pose_mode,
              "pose_model_inference_executed_this_run": pose_mode == "infer", "action_model_inference_executed": True,
              "model_sha256": sha256(model_path), "frames": len(rows), "prediction_counts": dict(counts),
              "video_codec": codec, "output_dir": str(out.resolve())}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "train", "evaluate", "demo"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/five-action-dataset.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/five-actions/poses")
    parser.add_argument("--model", type=Path, default=ROOT / "models/five-actions-v1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--pose-mode", choices=["infer", "replay"], default="infer")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "extract":
        result = extract_poses(manifest, args.cache, device=args.device, clips=[args.clip] if args.clip else None)
    elif args.command == "train":
        result = train(manifest, args.cache, args.model)
    elif args.command == "evaluate":
        if args.output is None:
            parser.error("evaluate requires --output (a new directory)")
        result = evaluate(manifest, args.cache, args.model, args.output, split=args.split)
    else:
        if args.output is None or args.clip is None:
            parser.error("demo requires --output (a new directory) and --clip")
        result = render_demo(manifest, args.cache, args.model, args.clip, args.output,
                             pose_mode=args.pose_mode, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
