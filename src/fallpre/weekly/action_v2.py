"""ByteTrack + masked causal temporal convolution + observable transition guards.

The v1 test set is development data. New test sources are predeclared separately.
This implementation does not interpolate a missing observation into a detected person.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .five_actions import (
    LABELS,
    FiveActionClassifier,
    OnlineFiveAction,
    classification_metrics,
    label_at,
    load_manifest,
)
from .human import POSE_SHA256, ROOT, _local_import_paths, sha256

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))
torch.set_num_threads(2)


@dataclass(frozen=True)
class Config:
    samples: int = 21
    step_s: float = 0.1
    min_history_s: float = 0.3
    retain_missing_s: float = 0.35
    sample_max_age_s: float = 0.12
    pose_conf: float = 0.1
    joint_conf: float = 0.2
    decision_conf: float = 0.35
    rise_min_up: float = 0.12
    channels: int = 32


class PersonTracker:
    """Official ByteTrack with original detection indices preserved across both passes."""

    def __init__(self, fps=30.0):
        _local_import_paths(ROOT, actions=False)
        from ultralytics.trackers.byte_tracker import BYTETracker

        class IndexedTracker(BYTETracker):
            def init_track(self, results, img=None):
                return [
                    self.track_class(np.r_[b, idx], score, 0)
                    for b, idx, score in zip(results.xywh, results.indices, results.conf)
                ]

        self.tracker = IndexedTracker(
            SimpleNamespace(
                track_high_thresh=0.35,
                track_low_thresh=0.1,
                new_track_thresh=0.45,
                track_buffer=max(2, round(fps * 0.5)),
                match_thresh=0.85,
                fuse_score=False,
            )
        )
        self.target = None
        self.last_seen = None
        self.last_t = None
        self.epoch = 0

    def update(self, row):
        instances, t = row["instances"], float(row["t"])
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError("Tracker timestamps must strictly increase")
        if self.last_t is not None and t - self.last_t > 0.5:
            self.tracker.reset()
            self.target = None
        self.last_t = t

        class Detections:
            def __init__(self, data, indices=None):
                self.data = data
                self.indices = np.arange(len(data)) if indices is None else indices

            @property
            def conf(self):
                return self.data[:, 4]

            @property
            def xywh(self):
                boxes = self.data[:, :4].copy()
                boxes[:, :2] = (self.data[:, :2] + self.data[:, 2:4]) / 2
                boxes[:, 2:] = self.data[:, 2:4] - self.data[:, :2]
                return boxes

            def __len__(self):
                return len(self.data)

            def __getitem__(self, i):
                return Detections(self.data[i], self.indices[i])

        data = np.array([p["box"] + [p["det"]] for p in instances], dtype=np.float32).reshape(-1, 5)
        tracked = self.tracker.update(Detections(data))
        candidates = {int(p[4]): int(p[-1]) for p in tracked}
        reason = None
        if self.target is not None and self.target not in candidates:
            if t - self.last_seen > 0.5:
                self.target = None
                reason = "target_expired"
            else:
                reason = "temporarily_unobserved"
        if self.target is None and candidates:
            # Only the current image is used. No future track length or labels.
            def rank(tid):
                p = instances[candidates[tid]]
                return p["det"] * float(np.mean(np.array(p["kpts"])[5:, 2]))

            self.target = max(candidates, key=rank)
            self.epoch += 1
            reason = "new_target"
        idx = candidates.get(self.target)
        if idx is not None:
            self.last_seen = t
        return dict(
            row,
            all_instances=instances,
            instances=[] if idx is None else [instances[idx]],
            target_index=idx,
            target_id=self.target,
            target_epoch=self.epoch,
            tracking_reason=reason or ("associated" if idx is not None else "missing_person"),
        )


def angle(a, b, c):
    u, v = a - b, c - b
    return float(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-8), -1, 1)) / np.pi)


def body_signal(person, config):
    """Partial-body features plus explicit reliability; ankles are not mandatory."""
    p = np.asarray(person["kpts"], dtype=float)
    box = np.asarray(person["box"], dtype=float)
    reliable = p[:, 2] >= config.joint_conf
    shoulder_ok, hip_ok = reliable[5:7].any(), reliable[11:13].any()

    def mid(ids, fallback):
        ids = [i for i in ids if reliable[i]]
        return p[ids, :2].mean(0) if ids else np.asarray(fallback)

    center = (box[:2] + box[2:]) / 2
    shoulder = mid([5, 6], [center[0], box[1] + 0.25 * (box[3] - box[1])])
    hip = mid([11, 12], center)
    torso = np.linalg.norm(hip - shoulder) if shoulder_ok and hip_ok else 0.3 * (box[3] - box[1])
    torso = max(torso, 5.0)
    tilt = math.atan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1])) / (np.pi / 2)
    knees, hips = [], []
    for a, b, c, d in [(5, 11, 13, 15), (6, 12, 14, 16)]:
        if all(reliable[[b, c, d]]):
            knees.append(angle(p[b, :2], p[c, :2], p[d, :2]))
        if all(reliable[[a, b, c]]):
            hips.append(angle(p[a, :2], p[b, :2], p[c, :2]))
    raw = np.array(
        [
            hip[0],
            hip[1],
            shoulder[1],
            center[0],
            center[1],
            box[2] - box[0],
            box[3] - box[1],
            tilt,
            np.mean(knees) if knees else 0.0,
            np.mean(hips) if hips else 0.0,
            (box[2] - box[0]) / max(box[3] - box[1], 1),
            float(hip_ok),
            float(shoulder_ok),
            float(bool(knees)),
            float(bool(hips)),
            float(reliable[5:].mean()),
        ],
        dtype=float,
    )
    return raw, torso


def sequence_features(history, config=Config()):
    """Previous-sample resampling only. Missing inputs are zero with a validity mask."""
    if not history:
        return None, {"reason": "warming_up"}
    now = history[-1]["t"]
    good = [r for r in history if len(r["instances"]) == 1]
    if not good or not history[-1]["instances"]:
        return None, {"reason": "missing_observation"}
    if now - good[0]["t"] < config.min_history_s:
        return None, {"reason": "warming_up"}
    if now - good[-1]["t"] > config.retain_missing_s:
        return None, {"reason": "stale_track"}
    signals = {r["t"]: body_signal(r["instances"][0], config) for r in good}
    scale = float(np.median([v[1] for v in signals.values()]))
    first = next(iter(signals.values()))[0]
    reference = np.array([first[0], first[1], first[1], first[0], first[1], 0.0, 0.0])
    times = np.array([r["t"] for r in history])
    data = np.zeros((config.samples, 17), dtype=np.float32)
    sampled = []
    for i, t in enumerate(now - np.arange(config.samples - 1, -1, -1) * config.step_s):
        j = np.searchsorted(times, t + 1e-7, side="right") - 1
        if j < 0 or t - times[j] > config.sample_max_age_s or not history[j]["instances"]:
            continue
        raw = signals[times[j]][0].copy()
        raw[:7] = (raw[:7] - reference) / scale
        data[i, :16], data[i, 16] = raw, 1.0
        sampled.append((times[j], raw))
    if data[-1, 16] == 0 or data[:, 16].sum() < 3:
        return None, {"reason": "insufficient_recent_support"}
    velocity = np.zeros((config.samples, 11), dtype=np.float32)
    pairs = data[1:, 16] * data[:-1, 16]
    velocity[1:] = (data[1:, :11] - data[:-1, :11]) / config.step_s * pairs[:, None]
    # Short and long upward motion evidence, using measured hip points only.
    recent = [(t, v) for t, v in sampled if now - t <= 0.8 and v[11] and v[12]]
    upward = 0.0
    if len(recent) >= 4 and recent[-1][0] - recent[0][0] >= 0.25:
        k = max(1, len(recent) // 3)
        upward = float(np.median([v[1] for _, v in recent[:k]]) - np.median([v[1] for _, v in recent[-k:]]))
    return np.concatenate([data, velocity], axis=1).T, {
        "reason": None,
        "observed_samples": int(data[:, 16].sum()),
        "upward_torso_units": upward,
        "evidence_window_s": [good[0]["t"], now],
        "support_frames": len(good),
    }


class CausalTCN(nn.Module):
    """Three left-padded dilated Conv1d blocks; output sees current and past only."""

    def __init__(self, channels=32):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Conv1d(28 if i == 0 else channels, channels, 3, dilation=2**i) for i in range(3)]
        )
        self.head = nn.Linear(channels, len(LABELS))

    def forward(self, x):
        for i, conv in enumerate(self.blocks):
            x = torch.relu(conv(torch.nn.functional.pad(x, (2 * 2**i, 0))))
        return self.head(x[:, :, -1])


class TemporalModel:
    def __init__(self, artifact):
        if artifact["schema"] != "fallpre.action_tcn.v2":
            raise ValueError("Unsupported v2 model")
        self.artifact = artifact
        self.config = Config(**artifact["config"])
        self.mean = np.asarray(artifact["mean"], dtype=np.float32)[:, None]
        self.scale = np.asarray(artifact["scale"], dtype=np.float32)[:, None]
        self.network = CausalTCN(self.config.channels).eval()
        self.network.load_state_dict({k: torch.tensor(v) for k, v in artifact["state_dict"].items()})

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, history, *, guard=True):
        x, info = sequence_features(history, self.config)
        result = dict(info, label="unknown", scores={}, quality="unavailable", model_inference_executed=False)
        if x is None:
            return result
        with torch.inference_mode():
            scores = self.network(torch.from_numpy((x - self.mean) / self.scale)[None]).softmax(1)[0].numpy()
        raw_top = LABELS[int(scores.argmax())]
        result.update(
            scores=dict(zip(LABELS, map(float, scores))),
            raw_top1=raw_top,
            model_inference_executed=True,
            quality="available",
        )
        if guard and info["upward_torso_units"] < self.config.rise_min_up:
            # A static pose cannot substantiate a transition. Keep raw score for audit.
            scores = scores.copy()
            scores[LABELS.index("background")] += scores[LABELS.index("standing_up")]
            scores[LABELS.index("standing_up")] = 0
            if raw_top == "standing_up":
                result["reason"] = "rise_rejected_no_upward_motion"
        top = int(scores.argmax())
        result.update(confidence=float(scores[top]), label=LABELS[top])
        if scores[top] < self.config.decision_conf:
            result.update(label="unknown", quality="uncertain", reason="low_classifier_confidence")
        return result


class OnlineTemporalAction:
    def __init__(self, model):
        self.model = model
        self.history = deque()
        self.epoch = None
        self.last_observed = None

    def reset(self, reason="identity_change"):
        self.history.clear()
        self.last_observed = None

    def update(self, row):
        t = row["t"]
        if self.history and t <= self.history[-1]["t"]:
            raise ValueError("Non-increasing source time")
        epoch = row.get("target_epoch", 0)
        if epoch != self.epoch or (self.history and t - self.history[-1]["t"] > 0.35):
            self.reset()
        self.epoch = epoch
        if row["instances"]:
            if self.last_observed is not None and t - self.last_observed > self.model.config.retain_missing_s:
                self.reset()
            self.last_observed = t
        self.history.append(row)
        while self.history and self.history[0]["t"] < t - 2.05:
            self.history.popleft()
        result = self.model.predict(list(self.history))
        result.update(source_time_s=t, source_frame_id=row.get("source_frame_id"), target_epoch=epoch)
        return result


def cache_path(entry, cache):
    return Path(cache) / entry["clip_id"] / "observations.jsonl"


def read_cache(entry, cache):
    lines = [json.loads(line) for line in cache_path(entry, cache).read_text().splitlines()]
    meta, rows = lines[0], lines[1:]
    if meta["source_sha256"] != entry["video_sha256"] or meta["weights_sha256"] != POSE_SHA256:
        raise ValueError("Pose provenance mismatch")
    if len(rows) != meta["frame_count"] or meta["detector_confidence"] != 0.1:
        raise ValueError("Wrong or incomplete v2 detection cache")
    return meta, rows


def extract(manifest, cache, split=None):
    import cv2

    _local_import_paths(ROOT, actions=False)
    from ultralytics import YOLO

    weights = ROOT / "models/yolo11n-pose.pt"
    if sha256(weights) != POSE_SHA256:
        raise ValueError("Wrong pose model")
    model = None
    for entry in manifest["entries"]:
        if split and entry["split"] not in split:
            continue
        path = cache_path(entry, cache)
        if path.exists():
            read_cache(entry, cache)
            continue
        model = model or YOLO(str(weights))
        cap = cv2.VideoCapture(entry["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pred = model.predict(frame, imgsz=640, conf=0.1, iou=0.45, device="cpu", verbose=False)[0]
            rows.append(
                {
                    "clip_id": entry["clip_id"],
                    "source_frame_id": len(rows) + 1,
                    "t": len(rows) / fps,
                    "instances": [
                        {"box": b, "det": s, "kpts": p}
                        for b, s, p in zip(
                            pred.boxes.xyxy.cpu().tolist(),
                            pred.boxes.conf.cpu().tolist(),
                            pred.keypoints.data.cpu().tolist(),
                        )
                    ],
                }
            )
        cap.release()
        if not rows or len(rows) != expected:
            raise ValueError("Incomplete source decode")
        meta = {
            "kind": "meta",
            "source_sha256": entry["video_sha256"],
            "weights_sha256": POSE_SHA256,
            "detector_confidence": 0.1,
            "fps": fps,
            "frame_count": len(rows),
            "inference_executed": True,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in [meta, *rows]) + "\n")
        print(entry["clip_id"], len(rows), flush=True)


def tracked_rows(entry, cache):
    meta, rows = read_cache(entry, cache)
    tracker = PersonTracker(meta["fps"])
    return [tracker.update(r) for r in rows]


def windows(rows, config=Config()):
    history = deque()
    epoch = None
    last_observed = None
    next_t = 0.0
    for row in rows:
        if row.get("target_epoch", 0) != epoch:
            history.clear()
            last_observed = None
        epoch = row.get("target_epoch", 0)
        if row["instances"]:
            if last_observed is not None and row["t"] - last_observed > config.retain_missing_s:
                history.clear()
            last_observed = row["t"]
        history.append(row)
        while history and history[0]["t"] < row["t"] - 2.05:
            history.popleft()
        if row["t"] + 1e-8 >= next_t:
            yield row, list(history)
            next_t = row["t"] + config.step_s


def train(manifest, cache, output):
    """Train on train only; checkpoint selection uses only the existing dev partition."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = Config()
    sets = {s: [] for s in ["train", "dev"]}
    for e in manifest["entries"]:
        if e["split"] not in sets:
            continue
        for row, history in windows(tracked_rows(e, cache), config):
            x, info = sequence_features(history, config)
            target = label_at(e, row["t"])
            sets[e["split"]].append((x, target, e["clip_id"], info, history))
    usable = [v for v in sets["train"] if v[0] is not None and v[1] in LABELS]
    x = np.stack([v[0] for v in usable])
    y = np.array([LABELS.index(v[1]) for v in usable])
    mean = x.mean((0, 2))
    scale = np.maximum(x.std((0, 2)), 0.1)
    inputs = torch.tensor((x - mean[None, :, None]) / scale[None, :, None])
    targets = torch.tensor(y)
    counts = np.bincount(y, minlength=len(LABELS))
    if not counts.all():
        raise ValueError("Training lacks a class")
    weights = torch.tensor(len(y) / len(LABELS) / counts, dtype=torch.float32)
    torch.manual_seed(20261015)
    np.random.seed(20261015)
    network = CausalTCN(config.channels)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=weights)
    log = []
    best = None
    for epoch in range(1, 121):
        network.train()
        order = torch.randperm(len(inputs))
        for idx in order.split(128):
            batch = inputs[idx].clone()
            # Small measurement jitter, never labels or held-out observations.
            batch += torch.randn_like(batch) * 0.02
            optimizer.zero_grad()
            loss = criterion(network(batch), targets[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            optimizer.step()
        if epoch % 10:
            continue
        artifact = {
            "schema": "fallpre.action_tcn.v2",
            "config": asdict(config),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "state_dict": {k: v.detach().tolist() for k, v in network.state_dict().items()},
        }
        candidate = TemporalModel(artifact)
        records = []
        for _, target, clip, _, history in sets["dev"]:
            prediction = candidate.predict(history)
            records.append(dict(prediction, target=target, clip_id=clip))
        metrics = classification_metrics(records)
        score = metrics["five_action_macro_f1"] or 0.0
        log.append({"epoch": epoch, "loss": float(loss.detach()), "dev_macro_f1": score})
        print(log[-1], flush=True)
        if best is None or score > best[0]:
            best = (score, copy.deepcopy(artifact), epoch)
    artifact = best[1]
    artifact.update(
        training={
            "seed": 20261015,
            "selected_epoch": best[2],
            "selection": "maximum dev macro F1, fixed 120-epoch budget",
            "train_clips": [e["clip_id"] for e in manifest["entries"] if e["split"] == "train"],
            "dev_clips": [e["clip_id"] for e in manifest["entries"] if e["split"] == "dev"],
            "window_counts": dict(zip(LABELS, map(int, counts))),
            "test_predictions_accessed": False,
        },
        manifest_sha256=manifest["manifest_sha256"],
        protocol_sha256=sha256(ROOT / "configs/action-v2-protocol.json"),
        sources={"tracking": "https://arxiv.org/abs/2110.06864", "TCN": "https://arxiv.org/abs/1803.01271"},
    )
    (output / "model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(log, indent=2) + "\n")
    (output / "freeze.json").write_text(
        json.dumps(
            {
                "model_sha256": sha256(output / "model.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "test_not_evaluated": True,
            },
            indent=2,
        )
        + "\n"
    )
    return output / "model.json"


def evaluate(manifest, cache, model_path, output, split="test"):
    """Frozen, matched-input comparison. All failures stay in the denominator."""
    model = TemporalModel.load(model_path)
    if model.artifact["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed after model freeze")
    if split == "test":
        frozen = json.loads((Path(model_path).parent / "freeze.json").read_text())
        if (
            frozen["model_sha256"] != sha256(model_path)
            or frozen["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise ValueError("Model or dataset differs from the frozen independent-test configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old_model = FiveActionClassifier.load(ROOT / "models/five-actions-v1.json")
    variants = {name: [] for name in ["v1", "v1_bytetrack", "v2_without_rise_guard", "v2"]}
    per_clip = []
    for entry in manifest["entries"]:
        if entry["split"] != split:
            continue
        if (
            entry["clip_id"]
            in model.artifact["training"]["train_clips"] + model.artifact["training"]["dev_clips"]
            and split == "test"
        ):
            raise ValueError("Test source was used for model development")
        meta, raw = read_cache(entry, cache)
        tracked = tracked_rows(entry, cache)
        old, old_tracked = OnlineFiveAction(old_model), OnlineFiveAction(old_model)
        from .joint import TargetTracker

        old_target = TargetTracker()
        legacy = []
        ablation = []
        last_epoch = None
        for r, tr in zip(raw, tracked):
            filtered = [p for p in r["instances"] if p["det"] >= 0.35]
            legacy_row = dict(r, instances=filtered)
            if entry["clip_id"].startswith("tum-"):
                index, reason = old_target.update(filtered, r["t"])
                if reason:
                    old.reset(reason)
                legacy_row["instances"] = [] if index is None else [filtered[index]]
            legacy.append(old.update(legacy_row))
            if tr["target_epoch"] != last_epoch:
                old_tracked.reset()
            last_epoch = tr["target_epoch"]
            ablation.append(old_tracked.update(tr))
        clip_records = {name: [] for name in variants}
        for row, history in windows(tracked, model.config):
            idx = row["source_frame_id"] - 1
            target = label_at(entry, row["t"])
            seated = any(
                s["state"] == "seated" and s["start_s"] <= row["t"] < s["end_s"]
                for s in entry.get("state_segments", [])
            )
            predictions = {
                "v1": legacy[idx],
                "v1_bytetrack": ablation[idx],
                "v2_without_rise_guard": model.predict(history, guard=False),
                "v2": model.predict(history),
            }
            official = entry.get("official_frame_states", {}).get(str(row["source_frame_id"]))
            for name, pred in predictions.items():
                record = dict(
                    pred,
                    clip_id=entry["clip_id"],
                    source_frame_id=row["source_frame_id"],
                    source_time_s=row["t"],
                    target=target,
                    seated=seated,
                    official_fall_state=official,
                    split=split,
                )
                variants[name].append(record)
                clip_records[name].append(record)
        per_clip.append(
            {
                "clip_id": entry["clip_id"],
                "video_sha256": entry["video_sha256"],
                "metrics": {name: classification_metrics(rows) for name, rows in clip_records.items()},
                "tracking": {
                    "raw_multi_person_frames": sum(len(r["instances"]) > 1 for r in raw),
                    "observed_target_frames": sum(bool(r["instances"]) for r in tracked),
                    "frames": len(raw),
                    "epochs": max(r["target_epoch"] for r in tracked),
                },
            }
        )
    summaries = {}
    for name, records in variants.items():
        summary = classification_metrics(records)
        seated = [r for r in records if r["seated"]]
        summary["seated_false_rise"] = {
            "support_windows": len(seated),
            "false_positive_windows": sum(r["label"] == "standing_up" for r in seated),
            "unknown_windows": sum(r["label"] == "unknown" for r in seated),
            "false_positive_fraction": sum(r["label"] == "standing_up" for r in seated) / len(seated)
            if seated
            else None,
        }
        official = [r for r in records if r["official_fall_state"] is not None]
        positives = [r for r in official if r["official_fall_state"] == 0]
        events = []
        for entry in manifest["entries"]:
            if entry["split"] != split or not entry.get("official_frame_states"):
                continue
            rr = [r for r in records if r["clip_id"] == entry["clip_id"] and r["official_fall_state"] == 0]
            detected = [r for r in rr if r["label"] == "falling"]
            onset = (min(int(f) for f, s in entry["official_frame_states"].items() if s == 0) - 1) / entry[
                "fps"
            ]
            events.append(
                {
                    "clip_id": entry["clip_id"],
                    "detected_during_fall": bool(detected),
                    "first_detection_delay_s": detected[0]["source_time_s"] - onset if detected else None,
                }
            )
        summary["official_falling"] = {
            "positive_windows": len(positives),
            "recall_including_unknown": sum(r["label"] == "falling" for r in positives) / len(positives)
            if positives
            else None,
            "negative_windows": sum(r["official_fall_state"] != 0 for r in official),
            "false_positive_windows": sum(
                r["official_fall_state"] != 0 and r["label"] == "falling" for r in official
            ),
            "events": events,
            "events_detected": sum(e["detected_during_fall"] for e in events),
            "event_count": len(events),
        }
        summary["unknown_reasons"] = dict(
            Counter(r.get("reason") for r in records if r["label"] == "unknown")
        )
        summaries[name] = summary
        (output / f"{name}-predictions.jsonl").write_text(
            "\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n"
        )
    result = {
        "schema": "fallpre.action_v2.evaluation.v1",
        "split": split,
        "model_sha256": sha256(model_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "comparison": summaries,
        "clips": per_clip,
        "test_policy": "new source-video split; no parameter fitting or checkpoint selection on these predictions",
        "limitations": [
            "Video-disjoint, not verified subject-disjoint.",
            "Mixed official fall-state labels and provisional unreviewed action/state intervals.",
            "Overlapping windows are correlated. Report independent event counts as well.",
            "Unknown is an error for annotated classes; not silently discarded.",
            "TUM v1 comparator uses its original target tracker; standalone v1 rejects multiple detections as before.",
        ],
    }
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "train", "evaluate"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/action-v2-dataset.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/action-v2/poses")
    parser.add_argument("--split", nargs="+", choices=["train", "dev", "test"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/action-v2/training")
    parser.add_argument("--model", type=Path, default=ROOT / "runs/action-v2/training/model.json")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "extract":
        extract(manifest, args.cache, args.split)
    elif args.command == "train":
        print(train(manifest, args.cache, args.output))
    else:
        result = evaluate(manifest, args.cache, args.model, args.output, (args.split or ["test"])[0])
        print(
            json.dumps(
                {
                    k: {
                        "macro_f1": v["five_action_macro_f1"],
                        "fall": v["official_falling"],
                        "seated": v["seated_false_rise"],
                    }
                    for k, v in result["comparison"].items()
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
