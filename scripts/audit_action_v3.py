"""Read-only supplemental audits of frozen V3 evaluation; never trains a model."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]


def main():
    from fallpre.weekly.five_actions import classification_metrics
    from fallpre.weekly.human import sha256

    base = ROOT / "runs/action-v3"
    manifest = json.loads((ROOT / "configs/action-v3-dataset.json").read_text())
    old = json.loads((ROOT / "configs/action-v2-dataset.json").read_text())
    old_test = {e["clip_id"] for e in old["entries"] if e["split"] == "test"}
    result = {
        "analysis": "Post-hoc descriptive audit only; no model changes or selection using these results.",
        "legacy_v2_test_regression": {},
        "independent_rising_events": {},
        "checks": {},
    }
    for name in ["v2", "v3"]:
        historical = [
            json.loads(line)
            for line in (base / f"regression/{name}-predictions.jsonl").read_text().splitlines()
        ]
        rr = [r for r in historical if r["clip_id"] in old_test]
        result["legacy_v2_test_regression"][name] = classification_metrics(rr)
        test = [
            json.loads(line)
            for line in (base / f"independent-test/{name}-predictions.jsonl").read_text().splitlines()
        ]
        events = []
        for e in manifest["entries"]:
            if e["split"] != "test":
                continue
            for s in e["segments"]:
                if s["label"] != "standing_up":
                    continue
                hit = [
                    r
                    for r in test
                    if r["clip_id"] == e["clip_id"]
                    and s["start_s"] <= r["source_time_s"] < s["end_s"]
                    and r["label"] == "standing_up"
                ]
                events.append(
                    dict(
                        clip_id=e["clip_id"],
                        detected=bool(hit),
                        first_hit_s=hit[0]["source_time_s"] if hit else None,
                    )
                )
        result["independent_rising_events"][name] = events
        if name == "v3":
            result["rise_miss_reasons"] = dict(
                Counter(
                    str((r["label"], r.get("reason")))
                    for r in test
                    if r["target"] == "standing_up" and r["label"] != "standing_up"
                )
            )
    freeze = json.loads((base / "training/freeze.json").read_text())
    assert sha256(ROOT / "configs/action-v3-dataset.json") == freeze["manifest_sha256"]
    assert sha256(base / "training/model.json") == freeze["model_sha256"]
    for p, digest in freeze["source_sha256"].items():
        assert sha256(ROOT / p) == digest
    result["checks"]["freeze_intact"] = True
    result["checks"]["new_test_has_no_legacy_ids"] = not any(
        e["clip_id"] in {r["clip_id"] for r in old["entries"]}
        for e in manifest["entries"]
        if e["split"] == "test"
    )
    assert result["checks"]["new_test_has_no_legacy_ids"]
    (base / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
