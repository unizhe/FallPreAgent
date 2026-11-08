"""Causal multi-scale motion forest with a frozen V2 falling branch.

Only current/past observations are inputs. Labels, subject names and dataset IDs
are never classifier features. Small-data temporal statistics complement the
frozen causal TCN; this is an explicitly routed ensemble, not a new pretrained net.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .action_v2 import Config, PersonTracker, TemporalModel, body_signal, read_cache, windows
from .five_actions import LABELS, classification_metrics, label_at
from .five_actions import load_manifest as load_base_manifest
from .human import POSE_SHA256, ROOT, _local_import_paths, sha256

V2_MODEL = ROOT / "runs/action-v2/training/model.json"
V2_CACHE = ROOT / "runs/action-v2/poses"
V3_CACHE = ROOT / "runs/action-v3/poses"
ORDINARY = tuple(x for x in LABELS if x != "falling")


def load_manifest(path):
    manifest = load_base_manifest(path)
    membership = {}
    for entry in manifest["entries"]:
        if entry.get("legacy_v2") and entry["split"] == "test":
            raise ValueError("Historical source cannot be an independent test")
        for field in ["subject_id", "parent_source_id"]:
            identity = entry.get(field)
            if identity is None:
                continue
            key = (field, identity)
            previous = membership.setdefault(key, entry["split"])
            if previous != entry["split"]:
                raise ValueError(f"{field} crosses train/dev/test: {identity}")
    return manifest


def observations(entry, cache=V3_CACHE):
    location = Path(cache) / entry["clip_id"] / "observations.jsonl"
    if not location.exists():
        return read_cache(entry, V2_CACHE)
    lines = [json.loads(line) for line in location.read_text().splitlines()]
    meta, rows = lines[0], lines[1:]
    if meta["source_sha256"] != entry["video_sha256"] or meta["weights_sha256"] != POSE_SHA256:
        raise ValueError("Pose cache checksum mismatch")
    expected = entry.get("source_times_s", [i / entry["fps"] for i in range(len(rows))])
    if len(rows) != entry["frame_count"] or len(rows) != len(expected):
        raise ValueError("Incomplete pose cache")
    for i, (r, t) in enumerate(zip(rows, expected)):
        if r["source_frame_id"] != i + 1 or abs(r["t"] - t) > 1e-7:
            raise ValueError("Pose source-clock mismatch")
    return meta, rows


def tracked(entry, cache=V3_CACHE):
    meta, rows = observations(entry, cache)
    tracker = PersonTracker(meta["fps"])
    return [tracker.update(r) for r in rows]


def extract(manifest, cache=V3_CACHE, splits=("train", "dev"), clips=None):
    import cv2

    _local_import_paths(ROOT, actions=False)
    from ultralytics import YOLO

    weights = ROOT / "models/yolo11n-pose.pt"
    if sha256(weights) != POSE_SHA256:
        raise ValueError("Pose weights changed")
    model = None
    for entry in manifest["entries"]:
        if entry["split"] not in splits or (clips and entry["clip_id"] not in clips):
            continue
        target = Path(cache) / entry["clip_id"] / "observations.jsonl"
        if target.exists():
            observations(entry, cache)
            continue
        if entry.get("legacy_v2") and Path(cache) == V3_CACHE:
            observations(entry, cache)
            continue
        model = model or YOLO(str(weights))
        cap = cv2.VideoCapture(entry["video_path"])
        rows = []
        times = entry.get("source_times_s")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i = len(rows)
            t = times[i] if times is not None else i / entry["fps"]
            pred = model.predict(frame, imgsz=640, conf=0.1, iou=0.45, device="cpu", verbose=False)[0]
            rows.append(
                dict(
                    clip_id=entry["clip_id"],
                    source_frame_id=i + 1,
                    t=t,
                    instances=[
                        dict(box=b, det=s, kpts=p)
                        for b, s, p in zip(
                            pred.boxes.xyxy.cpu().tolist(),
                            pred.boxes.conf.cpu().tolist(),
                            pred.keypoints.data.cpu().tolist(),
                        )
                    ],
                )
            )
        cap.release()
        if len(rows) != entry["frame_count"]:
            raise ValueError("Incomplete video decoding")
        meta = dict(
            kind="meta",
            source_sha256=entry["video_sha256"],
            weights_sha256=POSE_SHA256,
            frame_count=len(rows),
            fps=entry["fps"],
            detector_confidence=0.1,
            source_clock="manifest source_times_s or native video fps",
            fresh_inference=True,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(r) for r in [meta, *rows]) + "\n")
        print(entry["clip_id"], len(rows), flush=True)


def signals(person):
    raw, scale = body_signal(person, Config())
    p = np.array(person["kpts"])
    valid = p[:, 2] >= 0.2
    hip = raw[:2]
    shoulder_y = raw[2]
    # Shape ratios and reliability masks; no absolute image position or pixel size.
    shape = list(raw[7:]) + [raw[5] / scale, raw[6] / scale, (hip[1] - shoulder_y) / scale]
    for ids in [(13, 14), (15, 16)]:
        good = [i for i in ids if valid[i]]
        q = p[good, :2].mean(0) if good else hip
        shape.extend(
            [
                (q[1] - hip[1]) / scale,
                abs(p[ids[0], 0] - p[ids[1], 0]) / scale if all(valid[list(ids)]) else 0.0,
                float(bool(good)),
            ]
        )
    # Root and shoulders are in pixel coordinates only until window normalization.
    return np.array(shape, dtype=float), np.array([hip[0], hip[1], shoulder_y, raw[3], raw[4]]), scale


def motion_features(history):
    if not history or not history[-1]["instances"]:
        return None, dict(reason="missing_observation")
    now = history[-1]["t"]
    good = [r for r in history if r["instances"] and now - r["t"] <= 2.0]
    if len(good) < 3 or now - good[0]["t"] < 0.16:
        return None, dict(reason="warming_up")
    if any(b["t"] - a["t"] > 0.35 for a, b in zip(good, good[1:])):
        return None, dict(reason="source_gap")
    vals = [signals(r["instances"][0]) for r in good]
    shape = np.stack([v[0] for v in vals])
    root = np.stack([v[1] for v in vals])
    scale = float(np.median([v[2] for v in vals]))
    root = (root - root[-1]) / scale
    a = np.column_stack((shape, root))
    tt = np.array([r["t"] for r in good])
    features = []
    up = []
    for span in [0.2, 0.5, 1.0, 2.0]:
        mask = tt >= now - span
        v = a[mask]
        t = tt[mask]
        center = t - t.mean()
        den = max(float(center @ center), 1e-6)
        features.extend(np.r_[v[-1], v.mean(0), v.std(0), v[-1] - v[0], center @ v / den])
        features.extend([len(v) / max(1, len(good)), t[-1] - t[0]])
        reliable = (shape[mask, 4] > 0.5) & (shape[mask, 5] > 0.5)
        vr = v[reliable]
        tr = t[reliable]
        if len(vr) >= 3 and tr[-1] - tr[0] >= 0.12:
            # Root dimensions are the final five columns. Hip and shoulder both rise.
            up.append(float(min(vr[0, -4] - vr[-1, -4], vr[0, -3] - vr[-1, -3])))
    info = dict(
        reason=None,
        upward_torso_units=max(up[:3], default=0.0),
        support_frames=len(good),
        evidence_window_s=[good[0]["t"], now],
        observed_samples=len(good),
    )
    return np.nan_to_num(np.clip(features, -30, 30)).astype(np.float32), info


class MotionForest:
    """Portable JSON inference for sklearn ExtraTrees. No pickle execution."""

    def __init__(self, artifact):
        self.artifact = artifact
        self.frozen = TemporalModel(artifact["frozen_v2"])
        self.config = self.frozen.config
        self.classes = artifact["classes"]
        self.trees = artifact["trees"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def forest_scores(self, x):
        scores = np.zeros(len(self.classes))
        for t in self.trees:
            i = 0
            while t["left"][i] != -1:
                i = t["left"][i] if x[t["feature"][i]] <= t["threshold"][i] else t["right"][i]
            scores += t["value"][i]
        return scores / len(self.trees)

    def predict(self, history):
        frozen = self.frozen.predict(history)
        if frozen["label"] == "falling":
            return dict(frozen, branch="frozen_v2_falling", raw_top1="falling")
        x, info = motion_features(history)
        result = dict(
            info,
            label="unknown",
            scores={},
            quality="unavailable",
            model_inference_executed=False,
            branch="multiscale_motion_forest",
        )
        if x is None:
            return result
        scores = self.forest_scores(x)
        raw = self.classes[int(scores.argmax())]
        result.update(
            scores=dict(zip(self.classes, map(float, scores))),
            raw_top1=raw,
            quality="available",
            model_inference_executed=True,
        )
        scores = scores.copy()
        if info["upward_torso_units"] < self.artifact["rise_min_up"]:
            scores[self.classes.index("background")] += scores[self.classes.index("standing_up")]
            scores[self.classes.index("standing_up")] = 0
            if raw == "standing_up":
                result["reason"] = "rise_rejected_no_coherent_upward_motion"
        k = int(scores.argmax())
        result.update(label=self.classes[k], confidence=float(scores[k]))
        if scores[k] < self.artifact["confidence"]:
            result.update(label="unknown", quality="uncertain", reason="low_classifier_confidence")
        return result


class OnlineMotion:
    def __init__(self, model):
        self.model = model
        self.history = deque()
        self.epoch = None
        self.last_seen = None

    def reset(self, reason=None):
        self.history.clear()
        self.last_seen = None

    def update(self, row):
        t = row["t"]
        if self.history and t <= self.history[-1]["t"]:
            raise ValueError("Non-increasing source time")
        if row.get("target_epoch") != self.epoch or (self.history and t - self.history[-1]["t"] > 0.35):
            self.reset()
        self.epoch = row.get("target_epoch")
        if row["instances"]:
            if self.last_seen is not None and t - self.last_seen > 0.35:
                self.reset()
            self.last_seen = t
        self.history.append(row)
        while self.history and self.history[0]["t"] < t - 2.05:
            self.history.popleft()
        return dict(
            self.model.predict(list(self.history)),
            source_time_s=t,
            source_frame_id=row.get("source_frame_id"),
            target_epoch=self.epoch,
        )


def export_forest(clf):
    trees = []
    for estimator in clf.estimators_:
        t = estimator.tree_
        values = t.value[:, 0, :]
        values = values / np.maximum(values.sum(1, keepdims=True), 1e-10)
        trees.append(
            dict(
                left=t.children_left.tolist(),
                right=t.children_right.tolist(),
                feature=t.feature.tolist(),
                threshold=t.threshold.tolist(),
                value=values.tolist(),
            )
        )
    return trees


def records(manifest, split, cache=V3_CACHE):
    for e in manifest["entries"]:
        if e["split"] != split:
            continue
        for row, hist in windows(tracked(e, cache)):
            yield e, row, hist


def train(manifest, output, cache=V3_CACHE):
    from sklearn.ensemble import ExtraTreesClassifier

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = []
    dev = []
    for e, row, h in records(manifest, "train", cache):
        x, info = motion_features(h)
        label = label_at(e, row["t"])
        if x is not None and label in ORDINARY:
            data.append((x, label, e["clip_id"]))
    for e, row, h in records(manifest, "dev", cache):
        x, info = motion_features(h)
        dev.append((e, row, h, x, info))
    x = np.stack([d[0] for d in data])
    y = np.array([d[1] for d in data])
    counts = Counter(y)
    # Each clip/class contributes comparable total weight, then class balance.
    group_counts = Counter((d[2], d[1]) for d in data)
    weight = np.array([1 / np.sqrt(group_counts[(d[2], d[1])]) for d in data])
    weight /= weight.mean()
    v2 = json.loads(V2_MODEL.read_text())
    best = None
    logs = []
    frozen = TemporalModel(v2)
    baseline = [
        dict(frozen.predict(h), target=label_at(e, r["t"]), clip_id=e["clip_id"]) for e, r, h, _, _ in dev
    ]
    for depth, leaf in [(10, 2), (16, 2), (None, 3), (16, 6)]:
        clf = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=0.7,
            class_weight="balanced",
            random_state=20261018,
            n_jobs=2,
        )
        clf.fit(x, y, sample_weight=weight)
        valid = [i for i, d in enumerate(dev) if d[3] is not None]
        probs = clf.predict_proba(np.stack([dev[i][3] for i in valid]))
        probmap = dict(zip(valid, probs))
        for rise in [0.08, 0.12, 0.18]:
            predicted = []
            seated_fp = 0
            seated_n = 0
            for i, (e, row, h, features, info) in enumerate(dev):
                result = baseline[i].copy()
                if result["label"] != "falling":
                    if features is None:
                        result.update(label="unknown")
                    else:
                        s = probmap[i].copy()
                        if info["upward_torso_units"] < rise:
                            s[list(clf.classes_).index("background")] += s[
                                list(clf.classes_).index("standing_up")
                            ]
                            s[list(clf.classes_).index("standing_up")] = 0
                        k = int(s.argmax())
                        result.update(label=str(clf.classes_[k]) if s[k] >= 0.35 else "unknown")
                predicted.append(result)
                seated = any(
                    seg["state"] == "seated" and seg["start_s"] <= row["t"] < seg["end_s"]
                    for seg in e.get("state_segments", [])
                )
                if seated:
                    seated_n += 1
                    seated_fp += result["label"] == "standing_up"
            metrics = classification_metrics(predicted)
            score = metrics["five_action_macro_f1"]
            # Explicit cost for seated regression; fall decisions are identical by construction.
            objective = score - 2 * seated_fp / max(1, seated_n)
            log = dict(
                depth=depth,
                leaf=leaf,
                rise_min_up=rise,
                macro_f1=score,
                seated_fp=seated_fp,
                seated_support=seated_n,
                objective=objective,
                per_class=metrics["per_class"],
            )
            logs.append(log)
            print({k: v for k, v in log.items() if k != "per_class"}, flush=True)
            if best is None or objective > best[0]:
                best = (objective, copy.deepcopy(clf), rise, metrics)
    clf = best[1]
    artifact = dict(
        schema="fallpre.motion_forest.v3",
        classes=list(clf.classes_),
        trees=export_forest(clf),
        frozen_v2=v2,
        rise_min_up=best[2],
        confidence=0.35,
        manifest_sha256=manifest["manifest_sha256"],
        protocol_sha256=sha256(ROOT / "configs/action-v3-protocol.json"),
        training=dict(
            train_clips=sorted(set(d[2] for d in data)),
            dev_clips=[e["clip_id"] for e in manifest["entries"] if e["split"] == "dev"],
            window_counts=dict(counts),
            selected_dev_metrics=best[3],
            selection="max dev macro F1 minus 2*seated false-positive fraction; 12 fixed candidates; falling exactly protected",
            seed=20261018,
        ),
        sources=dict(
            extra_trees="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
            frozen_tcn="https://arxiv.org/abs/1803.01271",
        ),
    )
    model = MotionForest(artifact)
    for vector, expected in zip(x[:20], clf.predict_proba(x[:20])):
        if not np.allclose(model.forest_scores(vector), expected, atol=1e-12):
            raise ValueError("JSON export changed tree probabilities")
    (output / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    (output / "selection.json").write_text(json.dumps(logs, indent=2) + "\n")
    (output / "baseline-dev.json").write_text(json.dumps(classification_metrics(baseline), indent=2) + "\n")
    return output / "model.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["extract", "train"])
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/action-v3-dataset.json")
    p.add_argument("--output", type=Path, default=ROOT / "runs/action-v3/training")
    p.add_argument("--split", nargs="+", default=["train", "dev"])
    args = p.parse_args()
    m = load_manifest(args.manifest)
    if args.command == "extract":
        extract(m, splits=args.split)
    else:
        print(train(m, args.output))


def evaluate(manifest, model_path, output, *, split="test", legacy_only=False, cache=V3_CACHE):
    """Matched observations and original source clocks; unknown stays in denominators."""
    model = MotionForest.load(model_path)
    if manifest["manifest_sha256"] != model.artifact["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if split == "test":
        frozen = json.loads((Path(model_path).parent / "freeze.json").read_text())
        if (
            frozen["model_sha256"] != sha256(model_path)
            or frozen["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise ValueError("Freeze mismatch")
        for path, digest in frozen["source_sha256"].items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"Inference source changed after freeze: {path}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    variants = {"v2": [], "v3": []}
    clip_results = []
    development = set(model.artifact["training"]["train_clips"] + model.artifact["training"]["dev_clips"])
    for e in manifest["entries"]:
        if e["split"] != split or (legacy_only and not e.get("legacy_v2")):
            continue
        if split == "test" and (e.get("legacy_v2") or e["clip_id"] in development):
            raise ValueError("Test source used in development")
        clip = {"v2": [], "v3": []}
        rows = tracked(e, cache)
        for row, hist in windows(rows):
            reference = label_at(e, row["t"])
            seated = any(
                s["state"] == "seated" and s["start_s"] <= row["t"] < s["end_s"]
                for s in e.get("state_segments", [])
            )
            for name, pred in [("v2", model.frozen.predict(hist)), ("v3", model.predict(hist))]:
                record = dict(
                    pred,
                    clip_id=e["clip_id"],
                    dataset=e.get("dataset", "legacy"),
                    subject_id=e.get("subject_id"),
                    source_frame_id=row["source_frame_id"],
                    source_time_s=row["t"],
                    target=reference,
                    seated=seated,
                    official_fall_state=e.get("official_frame_states", {}).get(str(row["source_frame_id"])),
                )
                variants[name].append(record)
                clip[name].append(record)
        clip_results.append(
            dict(
                clip_id=e["clip_id"],
                subject_id=e.get("subject_id"),
                dataset=e.get("dataset", "legacy"),
                metrics={name: classification_metrics(rr) for name, rr in clip.items()},
                tracked_frames=sum(bool(r["instances"]) for r in rows),
                frames=len(rows),
                epochs=max(r["target_epoch"] for r in rows),
            )
        )
    summary = {}
    for name, rr in variants.items():
        metrics = classification_metrics(rr)
        seated = [r for r in rr if r["seated"]]
        metrics["seated_false_rise"] = dict(
            support=len(seated),
            false_rise=sum(r["label"] == "standing_up" for r in seated),
            unknown=sum(r["label"] == "unknown" for r in seated),
        )
        events = []
        for cid in sorted(set(r["clip_id"] for r in rr if r["official_fall_state"] == 0)):
            selected = [r for r in rr if r["clip_id"] == cid and r["official_fall_state"] == 0]
            hits = [r for r in selected if r["label"] == "falling"]
            e = next(e for e in manifest["entries"] if e["clip_id"] == cid)
            onset = (min(int(k) for k, v in e["official_frame_states"].items() if v == 0) - 1) / e["fps"]
            events.append(
                dict(
                    clip_id=cid,
                    detected_during_fall=bool(hits),
                    delay_s=hits[0]["source_time_s"] - onset if hits else None,
                )
            )
        official = [r for r in rr if r["official_fall_state"] is not None]
        metrics["official_falling"] = dict(
            positive_windows=sum(r["official_fall_state"] == 0 for r in official),
            true_positive_windows=sum(
                r["official_fall_state"] == 0 and r["label"] == "falling" for r in official
            ),
            negative_windows=sum(r["official_fall_state"] != 0 for r in official),
            false_positive_windows=sum(
                r["official_fall_state"] != 0 and r["label"] == "falling" for r in official
            ),
            events=events,
            events_detected=sum(e["detected_during_fall"] for e in events),
        )
        metrics["by_dataset"] = {
            d: classification_metrics([r for r in rr if r["dataset"] == d])
            for d in sorted(set(r["dataset"] for r in rr))
        }
        summary[name] = metrics
        (output / f"{name}-predictions.jsonl").write_text(
            "\n".join(json.dumps(r, allow_nan=False) for r in rr) + "\n"
        )
    # The protected falling decision is audited on every sampled observation, not just positives.
    same_falls = all(
        (a["label"] == "falling") == (b["label"] == "falling") for a, b in zip(variants["v2"], variants["v3"])
    )
    if not same_falls:
        raise AssertionError("Protected falling branch changed")
    result = dict(
        schema="fallpre.action_v3.evaluation.v1",
        split=split,
        legacy_only=legacy_only,
        model_sha256=sha256(model_path),
        manifest_sha256=manifest["manifest_sha256"],
        comparison=summary,
        clips=clip_results,
        falling_decisions_identical=same_falls,
        limitations=[
            "UTKinect action interval labels official; Weizmann clip labels are weak temporal labels.",
            "New UTKinect/standard Weizmann actors held out within dataset; rooms shared; URFD actor independence unknown.",
            "Seated intervals are provisional annotations inferred from official sequence order and inspected raw contacts; no human adjudication.",
            "Overlapping windows are correlated; unknown is included as error.",
        ],
    )
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    main()
