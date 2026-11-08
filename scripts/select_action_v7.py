# ruff: noqa: E402
"""Calibrate completion on development only; retain every V5 true-positive window."""

import copy
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.five_actions import classification_metrics, label_at


def main():
    base = ROOT / "runs/action-v7"
    data = pickle.load(open(base / "completion-cache.pkl", "rb"))["development"]
    required = set()
    baseline = []
    old_seated = 0
    cases = []
    for e, rows, pred, vec in data:
        indices = {}
        next_t = -1
        for i, (r, p) in enumerate(zip(rows, pred)):
            if r["t"] + 1e-8 < next_t:
                continue
            next_t = r["t"] + 0.1
            target = label_at(e, r["t"])
            phase = next(
                (s["label"] for s in e.get("phase_segments", []) if s["start_s"] <= r["t"] < s["end_s"]),
                "unknown",
            )
            seated = any(
                s["state"] == "seated" and s["start_s"] <= r["t"] < s["end_s"]
                for s in e.get("state_segments", [])
            )
            indices[i] = (target, phase, seated)
            baseline.append(dict(p, target=target, clip_id=e["clip_id"]))
            if target == "standing_up" and p["v5_label"] == "standing_up":
                required.add((e["clip_id"], i))
            old_seated += seated and p["label"] == "standing_up"
        cases.append((e, rows, pred, vec, indices))
    baseline_metrics = classification_metrics(baseline)["per_class"]["standing_up"]
    artifact = dict(
        schema="fallpre.completion_model.v7",
        frozen_v6=json.load(open(ROOT / "runs/action-v6/training/model.json")),
        completion_config={},
    )
    model = CompletionModel(artifact)
    best = None
    logs = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for speed in [0.1, 0.2, 0.3, 0.4, 0.6]:
            for knee in [0.9, 0.94, 0.96]:
                for window in [0, 0.067]:
                    cfg = dict(threshold=threshold, max_up_speed=speed, minimum_knee=knee, window_s=window)
                    model.config = cfg
                    rr = []
                    kept = set()
                    fp = 0
                    seated_fp = 0
                    old62 = 0
                    for e, rows, pred, vec, indices in cases:
                        stream = OnlineCompletion(model)
                        for i, (r, p, x) in enumerate(zip(rows, pred, vec)):
                            stream.baseline.update = lambda unused, p=p: p
                            stream.feature_provider = lambda h, p, s, x=x: x
                            new = stream.update(r)
                            if i not in indices:
                                continue
                            target, phase, seated = indices[i]
                            label = new["label"]
                            rr.append(dict(label=label, target=target, clip_id=e["clip_id"]))
                            if label == "standing_up":
                                if target == "standing_up":
                                    kept.add((e["clip_id"], i))
                                fp += phase == "standing"
                                seated_fp += seated
                                old62 += phase == "standing" and e["clip_id"] in {
                                    f"telkom-{j}-{w}" for j in [1, 2, 3] for w in ["before", "after"]
                                }
                    rise = classification_metrics(rr)["per_class"]["standing_up"]
                    eligible = kept >= required and seated_fp <= old_seated
                    log = dict(
                        config=cfg,
                        rise={k: rise[k] for k in ["precision", "recall", "f1"]},
                        missed_required=len(required - kept),
                        standing_fp=fp,
                        seated_fp=seated_fp,
                        old62=old62,
                        eligible=eligible,
                    )
                    logs.append(log)
                    if eligible and (best is None or rise["precision"] > best[0]):
                        best = (rise["precision"], copy.deepcopy(cfg), log)
        print("threshold", threshold, "best", best, flush=True)
    if best is None:
        raise ValueError("No recall-preserving candidate")
    out = base / "training"
    out.mkdir(exist_ok=False)
    artifact["completion_config"] = best[1]
    artifact["selection"] = dict(
        method="Development-only calibration, no forest retraining",
        required_v5_tp_windows=len(required),
        selected=best[2],
    )
    (out / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    (out / "selection.json").write_text(
        json.dumps(dict(baseline=baseline_metrics, candidates=logs, selected=best[2]), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
