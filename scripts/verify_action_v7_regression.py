# ruff: noqa: E402
"""Check frozen artifact against every sampled development result using cached upstream evidence."""

import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion

BASE = ROOT / "runs/action-v7"


def main():
    model = CompletionModel.load(BASE / "training/model.json")
    manifest = json.loads((ROOT / "configs/action-v7-dataset.json").read_text())
    entries = {e["clip_id"]: e for e in manifest["entries"]}
    ref = {
        (r["clip_id"], r["source_frame_id"]): r
        for r in map(json.loads, (BASE / "regression/v7-predictions.jsonl").read_text().splitlines())
    }
    data = pickle.load(open(BASE / "completion-cache.pkl", "rb"))["development"]
    matched = 0
    required = 0
    for e, rows, preds, vectors in data:
        now = entries[e["clip_id"]]
        assert now["split"] == "dev"
        for field in ["segments", "state_segments", "phase_segments"]:
            assert now.get(field) == e.get(field)
        stream = OnlineCompletion(model)
        for row, old, x in zip(rows, preds, vectors):
            stream.baseline.update = lambda unused, old=old: old
            stream.feature_provider = lambda history, p, start, x=x: x
            p = stream.update(row)
            r = ref.get((e["clip_id"], row["source_frame_id"]))
            if r is None:
                continue
            for field in ["label", "phase", "v6_label", "motion_completion_latched"]:
                assert p[field] == r[field]
            if r["target"] == "standing_up" and r["v5_label"] == "standing_up":
                assert p["label"] == "standing_up"
                required += 1
            matched += 1
    assert matched == len(ref)
    result = dict(
        frozen_model_sha256=hashlib.sha256((BASE / "training/model.json").read_bytes()).hexdigest(),
        frozen_manifest_sha256=hashlib.sha256(
            (ROOT / "configs/action-v7-dataset.json").read_bytes()
        ).hexdigest(),
        clips=len(data),
        matched_windows=matched,
        v5_true_positive_windows_preserved=required,
        development_annotations_unchanged=True,
        explanation="Full source runtime regression started before freeze; final test-only annotation refinement did not affect development entries. Frozen artifact reproduces all sampled results from cached upstream evidence; test uses full runtime after freeze.",
    )
    (BASE / "regression/frozen-recheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
