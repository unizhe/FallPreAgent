# ruff: noqa: E402
# Standalone entry configures the repository source path before imports.
"""Development-only terminal-phase diagnostics and reusable causal cache."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_action_v4 import features

from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v4 import OnlinePhase, PhaseModel, read_observations


def main():
    m = json.loads((ROOT / "configs/action-v4-dataset.json").read_text())
    model = PhaseModel.load(ROOT / "runs/action-v4/training/model.json")
    out = ROOT / "runs/action-v5"
    out.mkdir(exist_ok=True)
    cache = []
    for e in m["entries"]:
        if e["split"] == "train":
            continue
        meta, raw = read_observations(e)
        tracker = PersonTracker(meta["fps"])
        rows = [tracker.update(r) for r in raw]
        vectors = list(features(rows, True))
        stream = OnlinePhase(model)
        stream.feature_provider = lambda h: vectors[h[-1]["source_frame_id"] - 1]
        pred = [stream.update(r) for r in rows]
        cache.append((e, rows, vectors, pred))
        if e["clip_id"].startswith("telkom"):
            for label in ["rising", "standing"]:
                ids = [
                    i
                    for i, r in enumerate(rows)
                    if any(
                        s["label"] == label and s["start_s"] <= r["t"] < s["end_s"]
                        for s in e["phase_segments"]
                    )
                    and vectors[i][1]
                    and pred[i]["phase"] == "rising"
                ]
                print(
                    e["clip_id"],
                    label,
                    len(ids),
                    {
                        k: np.round(np.quantile([vectors[i][1][k] for i in ids], [0.1, 0.5, 0.9]), 3).tolist()
                        for k in ["hip_speed", "knee", "hip_angle", "tilt"]
                    }
                    if ids
                    else {},
                    flush=True,
                )
        else:
            print(e["clip_id"], flush=True)
    with (out / "dev-cache.pkl").open("wb") as f:
        pickle.dump(cache, f)
    print("done", len(cache), flush=True)


if __name__ == "__main__":
    main()
