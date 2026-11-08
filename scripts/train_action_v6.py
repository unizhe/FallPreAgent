# ruff: noqa: E402
# Standalone script configures repository imports.
"""Fit short-window completion on TRAIN; select only on former-test development."""

import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_action_v5 import phase_at

from fallpre.weekly.action_v3 import export_forest
from fallpre.weekly.action_v6 import CompletionModel, OnlineCompletion
from fallpre.weekly.five_actions import classification_metrics, label_at


def score(development, model=None, probabilities=None):
    records = []
    events = set()
    groups = Counter()
    standing = 0
    late = 0
    for e, rows, pred, vectors in development:
        stream = OnlineCompletion(model) if model else None
        next_t = -1
        for i, (r, p, x) in enumerate(zip(rows, pred, vectors)):
            if stream:
                stream.baseline.update = lambda unused, p=p: p
                stream.feature_provider = lambda h, p, s, x=x: x
                new = stream.update(r)
            else:
                new = p
            if r["t"] + 1e-8 < next_t:
                continue
            next_t = r["t"] + 0.1
            target = label_at(e, r["t"])
            phase = phase_at(e, r["t"])
            records.append(dict(new, target=target, clip_id=e["clip_id"]))
            if new["label"] == "standing_up":
                if target == "standing_up":
                    groups[e["clip_id"]] += 1
                if phase == "standing":
                    standing += 1
                    start = next(
                        s["start_s"] for s in e["phase_segments"] if s["start_s"] <= r["t"] < s["end_s"]
                    )
                    late += r["t"] - start > 0.25 + 1e-8
                for j, s in enumerate(e["segments"]):
                    if s["label"] == "standing_up" and s["start_s"] <= r["t"] < s["end_s"]:
                        events.add((e["clip_id"], j))
    return dict(
        rise=classification_metrics(records)["per_class"]["standing_up"],
        groups=dict(groups),
        events=sorted(events),
        standing_fp=standing,
        standing_fp_beyond025=late,
    )


def main():
    base = ROOT / "runs/action-v6"
    d = pickle.load(open(base / "completion-cache.pkl", "rb"))
    xx = []
    yy = []
    ids = []
    for e, rows, pred, vec in d["training"]:
        for i, (r, x) in enumerate(zip(rows, vec)):
            phase = phase_at(e, r["t"])
            if x is not None and phase in ["standing", "rising"] and i % 2 == 0:
                xx.append(x)
                yy.append(phase)
                ids.append(e["clip_id"])
    x = np.stack(xx)
    y = np.array(yy)
    cnt = Counter(ids)
    baseline = score(d["development"])
    base_events = set(map(tuple, baseline["events"]))
    best = None
    logs = []
    v5 = json.load(open(ROOT / "runs/action-v5/training/model.json"))
    for indices in [list(range(15)) + [45, 46], list(range(30)) + [45, 46]]:
        clf = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=20261019,
            n_jobs=2,
        )
        clf.fit(x[:, indices], y, sample_weight=[1 / np.sqrt(cnt[i]) for i in ids])
        artifact = dict(
            schema="fallpre.completion_model.v6",
            frozen_v5=v5,
            classes=clf.classes_.tolist(),
            trees=export_forest(clf),
            feature_indices=indices,
            completion_config={},
        )
        model = CompletionModel(artifact)
        probs = {}
        for e, rows, pred, vec in d["development"]:
            good = [v for v in vec if v is not None]
            if good:
                for v, p in zip(
                    good,
                    clf.predict_proba(np.stack(good)[:, indices])[:, list(clf.classes_).index("standing")],
                ):
                    probs[v.tobytes()] = float(p)
        model.probability = lambda x: probs[x.tobytes()]
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for window in [0, 0.067, 0.1]:
                cfg = dict(threshold=threshold, window_s=window, vote_fraction=0.65, minimum_rise_s=0.15)
                model.config = cfg
                s = score(d["development"], model)
                # Per-clip TP retention is stronger than the preregistered overall constraint.
                eligible = set(map(tuple, s["events"])) >= base_events and all(
                    s["groups"].get(k, 0) >= 0.99 * v for k, v in baseline["groups"].items()
                )
                log = dict(features=len(indices), config=cfg, **s, eligible=eligible)
                logs.append(log)
                print({k: v for k, v in log.items() if k not in ["events", "groups"]}, flush=True)
                if eligible and (best is None or s["rise"]["precision"] > best[0]):
                    artifact["completion_config"] = cfg
                    best = (s["rise"]["precision"], copy.deepcopy(artifact), log)
    if best is None:
        raise ValueError("No eligible model")
    out = base / "training"
    out.mkdir(exist_ok=False)
    artifact = best[1]
    artifact["training"] = dict(
        clips=dict(cnt),
        counts=dict(Counter(yy)),
        selection="Max precision with >=99% V5 TP windows per clip and every V5 event retained; other labels immutable",
    )
    (out / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    (out / "selection.json").write_text(
        json.dumps(dict(baseline=baseline, candidates=logs, selected=best[2]), indent=2) + "\n"
    )
    print("SELECTED", best[2], flush=True)


if __name__ == "__main__":
    main()
