# ruff: noqa: E402
# Standalone script: configure the local source path before importing project modules.
"""Train phase emissions on train only, choose gates on development only."""

import copy
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]
from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v3 import MotionForest, OnlineMotion, export_forest, load_manifest, signals
from fallpre.weekly.action_v4 import OnlinePhase, PhaseModel, phase_features, read_observations
from fallpre.weekly.five_actions import classification_metrics, label_at
from fallpre.weekly.human import sha256

BASE = ROOT / "runs/action-v4"


def phase_target(e, row, all_rows):
    t = row["t"]
    for s in e.get("phase_segments", []):
        if s["start_s"] <= t < s["end_s"]:
            return s["label"]
    if e.get("phase_segments"):
        return "unknown"
    for s in e.get("state_segments", []):
        if s["state"] == "seated" and s["start_s"] <= t < s["end_s"]:
            return "seated"
    target = label_at(e, t)
    if target in ["walking", "bending", "falling"]:
        return "other"
    if target == "sitting_down" and row["instances"]:
        segment = next(s for s in e["segments"] if s["label"] == target and s["start_s"] <= t < s["end_s"])
        ys = [
            signals(r["instances"][0])[1][1]
            for r in all_rows
            if r["instances"] and segment["start_s"] <= r["t"] < segment["end_s"]
        ]
        if len(ys) > 3 and max(ys) - min(ys) > 8:
            progress = (signals(row["instances"][0])[1][1] - min(ys)) / (max(ys) - min(ys))
            if progress > 0.88:
                return "seated"
        return "descending"
    if target == "standing_up" and row["instances"]:
        # Weak TRAIN labels, based on full annotated transition geometry only.
        # These labels are never used to evaluate new video phases.
        segment = next(s for s in e["segments"] if s["label"] == target and s["start_s"] <= t < s["end_s"])
        ys = [
            signals(r["instances"][0])[1][1]
            for r in all_rows
            if r["instances"] and segment["start_s"] <= r["t"] < segment["end_s"]
        ]
        if len(ys) > 3:
            span = max(ys) - min(ys)
            if span > 8:
                progress = (max(ys) - signals(row["instances"][0])[1][1]) / span
                return "forward_lean" if progress < 0.12 else "rising" if progress < 0.88 else "standing"
    return "unknown"


def tracked(e):
    meta, rows = read_observations(e)
    tracker = PersonTracker(meta["fps"])
    return [tracker.update(r) for r in rows]


def features(rows, with_info=False):
    h = deque()
    epoch = None
    last = None
    for r in rows:
        if (
            r["target_epoch"] != epoch
            or (h and r["t"] - h[-1]["t"] > 0.35)
            or (r["instances"] and last is not None and r["t"] - last > 0.35)
        ):
            h.clear()
        if r["instances"]:
            last = r["t"]
        epoch = r["target_epoch"]
        h.append(r)
        while h and h[0]["t"] < r["t"] - 4.05:
            h.popleft()
        value = phase_features(list(h))
        yield value if with_info else value[0]


def main():
    m = load_manifest(ROOT / "configs/action-v4-dataset.json")
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "training"
    output.mkdir(parents=True, exist_ok=False)
    xx = []
    yy = []
    groups = []
    train_ids = []
    development = []
    base = MotionForest.load(ROOT / "runs/action-v3/training/model.json")
    for e in m["entries"]:
        if e["split"] == "test":
            continue
        print("prepare", e["clip_id"], flush=True)
        rows = tracked(e)
        if e["split"] == "train":
            train_ids.append(e["clip_id"])
            targets = [phase_target(e, r, rows) for r in rows]
            for dilation in [1.0, 1.5, 2.0]:
                augmented = [dict(r, t=r["t"] * dilation) for r in rows]
                for i, (x, target) in enumerate(zip(features(augmented), targets)):
                    if x is not None and target != "unknown" and i % 3 == 0:
                        xx.append(x)
                        yy.append(target)
                        groups.append((e["clip_id"], target))
        else:
            stream = OnlineMotion(base)
            baseline = [stream.update(r) for r in rows]
            development.append((e, rows, list(features(rows, with_info=True)), baseline))
    x = np.stack(xx)
    y = np.array(yy)
    count = Counter(groups)
    weight = np.array([1 / np.sqrt(count[g]) for g in groups])
    print("train", x.shape, Counter(yy), flush=True)
    classifier = ExtraTreesClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=20261019,
        n_jobs=2,
    )
    classifier.fit(x, y, sample_weight=weight)
    config = dict(
        min_up=0.1, min_opening=0.04, confirm_s=0.1, seat_probability=0.45, stable_speed=0.15, stable_s=0.2
    )
    artifact = dict(
        schema="fallpre.phase_model.v4",
        frozen_v3=base.artifact,
        classes=classifier.classes_.tolist(),
        trees=export_forest(classifier),
        phase_config=config,
        manifest_sha256=m["manifest_sha256"],
        training=dict(
            train_clips=train_ids,
            dev_clips=[e["clip_id"] for e, *_ in development],
            phase_counts=dict(Counter(yy)),
            augmentation="train-only 1x/1.5x/2x timestamp dilation; no fabricated frames",
            weak_legacy_labels="past/future geometry within official TRAIN action interval used for training phase targets, never for inference or new evaluation",
        ),
    )
    model = PhaseModel(artifact)
    # Match fast sklearn batch emissions to the exported JSON implementation.
    for z, expected in zip(x[:10], classifier.predict_proba(x[:10])):
        assert np.allclose(list(model.scores(z).values()), expected)
    cache = {}
    for _, _, vectors, _ in development:
        good = [v[0] for v in vectors if v[0] is not None]
        probabilities = classifier.predict_proba(np.stack(good))
        for v, p in zip(good, probabilities):
            cache[v.tobytes()] = dict(zip(classifier.classes_, map(float, p)))
    model.scores = lambda v: cache[v.tobytes()]
    logs = []
    best = None
    for up in [0.06, 0.10, 0.16]:
        for confirm in [0.0, 0.10, 0.20]:
            config.update(min_up=up, confirm_s=confirm)
            records = []
            seated = 0
            fp = 0
            changed = 0
            for e, rows, vectors, baseline in development:
                stream = OnlinePhase(model)
                stream.feature_provider = lambda h, vectors=vectors: vectors[h[-1]["source_frame_id"] - 1]
                for r, old in zip(rows, baseline):
                    # Same causal V3 result, precomputed once; never a target label.
                    stream.baseline.update = lambda unused, old=old: old
                    new = stream.update(r)
                    assert (
                        old["label"] not in {"walking", "bending", "falling"} or new["label"] == old["label"]
                    )
                    if r["source_frame_id"] % 3 != 1:
                        continue
                    records.append(dict(new, target=label_at(e, r["t"]), clip_id=e["clip_id"]))
                    is_seated = any(
                        s["state"] == "seated" and s["start_s"] <= r["t"] < s["end_s"]
                        for s in e.get("state_segments", [])
                    )
                    seated += is_seated
                    fp += is_seated and new["label"] == "standing_up"
                    changed += new["label"] != old["label"]
            metrics = classification_metrics(records)
            rise = metrics["per_class"]["standing_up"]["f1"]
            score = rise - 3 * fp / max(1, seated)
            log = dict(
                config=dict(config),
                rise_f1=rise,
                seated_fp=fp,
                seated_support=seated,
                macro_f1=metrics["five_action_macro_f1"],
                objective=score,
                changed=changed,
            )
            logs.append(log)
            print(log, flush=True)
            if fp == 0 and (best is None or score > best[0]):
                best = (score, copy.deepcopy(config))
    if best is None:
        raise ValueError("No development candidate satisfies zero seated false rises")
    artifact["phase_config"] = best[1]
    artifact["selection_rule"] = (
        "Among development candidates with zero seated false rises, choose maximum rising F1; no new test predictions accessed."
    )
    (output / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    (output / "selection.json").write_text(json.dumps(logs, indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                v3_sha256=sha256(ROOT / "runs/action-v3/training/model.json"),
                new_test_predictions_accessed=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
