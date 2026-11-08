# ruff: noqa: E402
# Standalone script configures repository imports.
import json
import pickle
import sys
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
from train_action_v5 import build, completion_vectors

from fallpre.weekly.action_v4 import PhaseModel
from fallpre.weekly.action_v5 import CompletionModel, OnlineCompletion

b = Path("runs/action-v6")
d = pickle.load(open("runs/action-v5/completion-cache.pkl", "rb"))
m = json.load(open("configs/action-v5-dataset.json"))
base = PhaseModel.load("runs/action-v4/training/model.json")
for e in m["entries"]:
    if e["split"] == "test":
        e = dict(e, split="dev", legacy_v5=True)
        print("prepare", e["clip_id"], flush=True)
        _, rows, _, p = build(e, base)
        d["development"].append((e, rows, p, completion_vectors(rows, p)))
model = CompletionModel.load("runs/action-v5/training/model.json")
for partition in ["training", "development"]:
    out = []
    for e, rows, p, v in d[partition]:
        online = OnlineCompletion(model)
        new = []
        prob = []
        for i, (r, old) in enumerate(zip(rows, p)):
            online.baseline.update = lambda unused, old=old: old
            online.feature_provider = lambda h, o, start, i=i: v[i]
            pred = online.update(r)
            new.append(pred)
            prob.append(pred.get("completion_probability"))
        out.append((e, rows, new, v))
    d[partition] = out
pickle.dump(d, open(b / "completion-cache.pkl", "wb"))
print("done", flush=True)
