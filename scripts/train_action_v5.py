# ruff: noqa: E402
# Standalone entry configures the repository source path before imports.
"""Train terminal completion on explicit TRAIN phase labels, select only on DEV."""

import copy
import json
import pickle
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v3 import export_forest
from fallpre.weekly.action_v4 import OnlinePhase, PhaseModel
from fallpre.weekly.action_v5 import (
    CompletionModel,
    OnlineCompletion,
    load_manifest,
    read_observations,
    terminal_features,
)
from fallpre.weekly.five_actions import classification_metrics, label_at

BASE = ROOT / "runs/action-v5"


def phase_at(e, t):
    return next(
        (s["label"] for s in e.get("phase_segments", []) if s["start_s"] <= t < s["end_s"]), "unknown"
    )


def build(e, model):
    meta, raw = read_observations(e)
    tr = PersonTracker(meta["fps"])
    rows = [tr.update(r) for r in raw]
    stream = OnlinePhase(model)
    pred = [stream.update(r) for r in rows]
    return e, rows, None, pred


def completion_vectors(rows, pred):
    h = deque()
    start = None
    epoch = None
    last = None
    result = []
    for r, p in zip(rows, pred):
        t = r["t"]
        if epoch != r["target_epoch"] or last is not None and t - last > 0.35:
            h.clear()
            start = None
        epoch = r["target_epoch"]
        last = t
        h.append(r)
        while h and h[0]["t"] < t - 0.7:
            h.popleft()
        if p["phase"] != "rising":
            start = None
            result.append(None)
            continue
        if start is None:
            start = t
        result.append(terminal_features(list(h), p, start))
    return result


def main():
    m = load_manifest(ROOT / "configs/action-v5-dataset.json")
    out = BASE / "training"
    out.mkdir(exist_ok=False)
    v4 = PhaseModel.load(ROOT / "runs/action-v4/training/model.json")
    datafile = BASE / "completion-cache.pkl"
    if datafile.exists():
        data = pickle.load(datafile.open("rb"))
        assert data["manifest_sha256"] == m["manifest_sha256"]
        training = data["training"]
        development = data["development"]
    else:
        old = pickle.load((BASE / "dev-cache.pkl").open("rb"))
        entries = {e["clip_id"]: e for e in m["entries"]}
        development = []
        training = []
        for e, rows, _, pred in old:
            development.append((entries[e["clip_id"]], rows, pred, completion_vectors(rows, pred)))
        for e in m["entries"]:
            if (
                not e.get("phase_segments")
                or e["split"] == "test"
                or e["split"] == "dev"
                and not e.get("input_mode")
            ):
                continue
            print("prepare", e["clip_id"], flush=True)
            _, rows, _, pred = build(e, v4)
            batch = (e, rows, pred, completion_vectors(rows, pred))
            (training if e["split"] == "train" else development).append(batch)
        pickle.dump(
            dict(manifest_sha256=m["manifest_sha256"], training=training, development=development),
            datafile.open("wb"),
        )
    xx = []
    yy = []
    groups = []
    for e, rows, pred, vectors in training:
        count = 0
        for i, (r, x) in enumerate(zip(rows, vectors)):
            phase = phase_at(e, r["t"])
            if x is not None and phase in ["standing", "rising"] and i % 2 == 0:
                xx.append(x)
                yy.append(phase)
                groups.append(e["clip_id"])
                count += 1
        print("train samples", e["clip_id"], count, flush=True)
    if len(set(yy)) < 2:
        raise ValueError("No usable completion training")
    x = np.stack(xx)
    y = np.array(yy)
    counts = Counter(groups)
    weight = np.array([1 / np.sqrt(counts[g]) for g in groups])
    clf = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=20261019,
        n_jobs=2,
    )
    clf.fit(x, y, sample_weight=weight)
    artifact = dict(
        schema="fallpre.completion_model.v5",
        frozen_v4=v4.artifact,
        classes=clf.classes_.tolist(),
        trees=export_forest(clf),
        completion_config={},
        manifest_sha256=m["manifest_sha256"],
        training=dict(
            clips=[e["clip_id"] for e, *_ in training],
            phase_counts=dict(Counter(yy)),
            per_clip_samples=dict(counts),
            feature_dim=x.shape[1],
            natural_speed=True,
            no_new_test_predictions=True,
        ),
    )
    model = CompletionModel(artifact)
    probabilities = {}
    for e, rows, pred, vectors in development:
        good = [v for v in vectors if v is not None]
        if good:
            for v, p in zip(good, clf.predict_proba(np.stack(good))[:, list(clf.classes_).index("standing")]):
                probabilities[v.tobytes()] = float(p)
    for v in x[:10]:
        assert (
            abs(model.probability(v) - clf.predict_proba(v[None])[0, list(clf.classes_).index("standing")])
            < 1e-8
        )
    model.probability = lambda x: probabilities[x.tobytes()]
    baseline = []
    base_events = set()
    for e, rows, pred, _ in development:
        next_t = -1
        for r, p in zip(rows, pred):
            if r["t"] + 1e-8 < next_t:
                continue
            next_t = r["t"] + 0.1
            baseline.append(dict(p, target=label_at(e, r["t"]), clip_id=e["clip_id"]))
            for j, s in enumerate(e["segments"]):
                if (
                    s["label"] == "standing_up"
                    and s["start_s"] <= r["t"] < s["end_s"]
                    and p["label"] == "standing_up"
                ):
                    base_events.add((e["clip_id"], j))
    base_metrics = classification_metrics(baseline)
    base_recall = base_metrics["per_class"]["standing_up"]["recall"]
    logs = []
    best = None
    for threshold in [0.35, 0.45, 0.55, 0.65, 0.75]:
        for window in [0.10, 0.20, 0.30]:
            cfg = dict(threshold=threshold, window_s=window, vote_fraction=0.65, minimum_rise_s=0.15)
            artifact["completion_config"] = cfg
            model.config = cfg
            records = []
            events = set()
            standing_fp = 0
            for e, rows, pred, vectors in development:
                stream = OnlineCompletion(model)
                next_t = -1
                for i, (r, p) in enumerate(zip(rows, pred)):
                    stream.baseline.update = lambda unused, p=p: p
                    stream.feature_provider = lambda h, old, start, i=i, vectors=vectors: vectors[i]
                    new = stream.update(r)
                    if r["t"] + 1e-8 < next_t:
                        continue
                    next_t = r["t"] + 0.1
                    records.append(dict(new, target=label_at(e, r["t"]), clip_id=e["clip_id"]))
                    standing_fp += new["label"] == "standing_up" and phase_at(e, r["t"]) == "standing"
                    for j, s in enumerate(e["segments"]):
                        if (
                            s["label"] == "standing_up"
                            and s["start_s"] <= r["t"] < s["end_s"]
                            and new["label"] == "standing_up"
                        ):
                            events.add((e["clip_id"], j))
            metrics = classification_metrics(records)
            rise = metrics["per_class"]["standing_up"]
            eligible = events >= base_events and rise["recall"] >= base_recall * 0.95
            log = dict(
                config=cfg,
                rise={k: rise[k] for k in ["precision", "recall", "f1"]},
                events=len(events),
                old_events=len(base_events),
                standing_fp=standing_fp,
                eligible=eligible,
            )
            logs.append(log)
            print(log, flush=True)
            if eligible and (best is None or rise["f1"] > best[0]):
                best = (rise["f1"], copy.deepcopy(cfg))
    if best is None:
        raise ValueError("No candidate preserves dev event recall and >=95% window recall")
    artifact["completion_config"] = best[1]
    artifact["selection"] = (
        "Maximum dev F1 subject to retaining every V4 dev detected event and >=95% V4 rising window recall; no test used."
    )
    (out / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    (out / "selection.json").write_text(
        json.dumps(dict(baseline=base_metrics, candidates=logs), indent=2) + "\n"
    )
    print("selected", best, flush=True)


if __name__ == "__main__":
    main()
