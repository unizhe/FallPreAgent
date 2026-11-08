# ruff: noqa: E402
# Standalone script configures repository imports.
"""Audit every former-test standing FP; keeps V5 labels and predictions intact."""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.action_v6 import CompletionModel, OnlineCompletion

BASE = ROOT / "runs/action-v6"


def main():
    old = json.load(open(BASE / "review/remaining70.json"))
    wanted = {(r["clip_id"], r["source_frame_id"]): r for r in old["rows"]}
    model = CompletionModel.load(BASE / "training/model.json")
    cache = pickle.load(open(BASE / "completion-cache.pkl", "rb"))
    audited = []
    for e, rows, pred, vec in cache["development"]:
        if not any(cid == e["clip_id"] for cid, _ in wanted):
            continue
        s = OnlineCompletion(model)
        for r, p, x in zip(rows, pred, vec):
            s.baseline.update = lambda unused, p=p: p
            s.feature_provider = lambda h, p, start, x=x: x
            new = s.update(r)
            key = (e["clip_id"], r["source_frame_id"])
            if key in wanted:
                record = dict(
                    wanted[key],
                    v6_label=new["label"],
                    fast_probability=new["fast_completion_probability"],
                    fast_latched=new["fast_completion_latched"],
                    short_hip_speed=float(x[10]) if x is not None else None,
                    long_hip_speed=float(x[40]) if x is not None else None,
                )
                audited.append(record)
    assert len(audited) == 70
    result = dict(
        total=70,
        within025=sum(r["after_stand_s"] <= 0.25 + 1e-8 for r in audited),
        v6_remaining=sum(r["v6_label"] == "standing_up" for r in audited),
        v5_beyond025=sum(r["after_stand_s"] > 0.25 + 1e-8 for r in audited),
        v6_beyond025=sum(
            r["after_stand_s"] > 0.25 + 1e-8 and r["v6_label"] == "standing_up" for r in audited
        ),
        by_clip={
            cid: dict(
                v5=sum(r["clip_id"] == cid for r in audited),
                v6=sum(r["clip_id"] == cid and r["v6_label"] == "standing_up" for r in audited),
            )
            for cid in sorted({r["clip_id"] for r in audited})
        },
        causes={
            "all_in_upstream_rising": all(r["v4_phase"] == "rising" for r in audited),
            "below_completion_threshold": sum(r["completion_probability"] < 0.65 for r in audited),
            "above_threshold_waiting_votes": sum(r["completion_probability"] >= 0.65 for r in audited),
        },
        rows=audited,
    )
    (BASE / "review/remaining70-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
