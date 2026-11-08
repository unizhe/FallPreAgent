# ruff: noqa: E402
"""Build local development cache from frozen V5/V6 and source-clock observations."""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_action_v5 import completion_vectors

from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v5 import CompletionModel as V5Model
from fallpre.weekly.action_v5 import OnlineCompletion as V5Stream
from fallpre.weekly.action_v6 import CompletionModel, OnlineCompletion, read_observations


def main():
    base = ROOT / "runs/action-v7"
    d = pickle.load(open(ROOT / "runs/action-v6/completion-cache.pkl", "rb"))
    m = json.load(open(ROOT / "configs/action-v6-dataset.json"))
    v5 = V5Model.load(ROOT / "runs/action-v5/training/model.json")
    for e in m["entries"]:
        if e["split"] != "test":
            continue
        e = dict(e, split="dev", legacy_v6=True)
        print("prepare", e["clip_id"], flush=True)
        meta, raw = read_observations(e)
        tracker = PersonTracker(meta["fps"])
        rows = [tracker.update(r) for r in raw]
        online = V5Stream(v5)
        pred = [online.update(r) for r in rows]
        # Completion vectors use the upstream V4 phase, not the V5 terminated phase.
        p4 = [dict(p, phase=p["v4_phase"]) for p in pred]
        d["development"].append((e, rows, pred, completion_vectors(rows, p4)))
    model = CompletionModel.load(ROOT / "runs/action-v6/training/model.json")
    for part in ["training", "development"]:
        out = []
        for e, rows, pred, vectors in d[part]:
            online = OnlineCompletion(model)
            new = []
            for r, p, x in zip(rows, pred, vectors):
                online.baseline.update = lambda unused, p=p: p
                online.feature_provider = lambda h, p, start, x=x: x
                new.append(online.update(r))
            out.append((e, rows, new, vectors))
        d[part] = out
    pickle.dump(d, open(base / "completion-cache.pkl", "wb"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
