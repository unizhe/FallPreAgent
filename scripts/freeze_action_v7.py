"""Bind the trained artifact to its manifest, then freeze before new-test inference."""

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    target = ROOT / "runs/action-v7/training/freeze.json"
    if target.exists():
        raise ValueError("Already frozen; create a new version")
    if (ROOT / "runs/action-v7/independent-test").exists():
        raise ValueError("New test already used")
    model_path = ROOT / "runs/action-v7/training/model.json"
    artifact = json.loads(model_path.read_text())
    artifact["manifest_sha256"] = sha("configs/action-v7-dataset.json")
    model_path.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    sources = [
        "src/fallpre/weekly/" + f
        for f in [
            "action_v7.py",
            "action_v6.py",
            "action_v5.py",
            "action_v4.py",
            "action_v3.py",
            "action_v2.py",
            "five_actions.py",
            "action_v7_demo.py",
        ]
    ] + ["scripts/" + f for f in ["evaluate_action_v7.py", "select_action_v7.py", "annotate_action_v7.py"]]
    freeze = dict(
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_sha256=sha(model_path),
        manifest_sha256=sha("configs/action-v7-dataset.json"),
        protocol_sha256=sha("configs/action-v7-protocol.json"),
        source_sha256={s: sha(s) for s in sources},
        new_test_predictions_before_freeze=False,
    )
    target.write_text(json.dumps(freeze, indent=2) + "\n")


if __name__ == "__main__":
    main()
