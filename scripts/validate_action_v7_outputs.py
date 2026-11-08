"""Integrity, sampled fresh/replayed parity and full video decode checks."""

import hashlib
import json
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v7"
sys.path.insert(0, str(ROOT / "src"))


def read(p):
    return json.loads(p.read_text())


def main():
    f = read(BASE / "training/freeze.json")
    checks = dict(f["source_sha256"])
    checks.update(
        {
            "runs/action-v7/training/model.json": f["model_sha256"],
            "configs/action-v7-dataset.json": f["manifest_sha256"],
            "configs/action-v7-protocol.json": f["protocol_sha256"],
        }
    )
    for p, h in checks.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
    ref = {}
    for split in ["regression", "independent-test"]:
        for line in (BASE / split / "v7-predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            ref[(r["clip_id"], r["source_frame_id"])] = r
    result = dict(freeze_intact=True, online_parity=[], videos=[])
    videos = []
    for p in sorted(BASE.glob("demo-*")) + sorted(ROOT.glob("runs/weekly-stage8-demo-*")):
        m = read(p / "result.json")
        reference_kind = "frozen_evaluation"
        if not any(cid == m["clip_id"] for cid, frame_id in ref):
            from fallpre.weekly.action_v7 import CompletionModel, iter_predictions, load_manifest

            manifest = load_manifest(ROOT / "configs/action-v7-dataset.json")
            entry = next(e for e in manifest["entries"] if e["clip_id"] == m["clip_id"])
            assert entry["split"] == "train", "Missing held-out evaluation evidence"
            reference_kind = "direct_source_replay_training_only"
            model = CompletionModel.load(BASE / "training/model.json")
            for row, tracked, prediction in iter_predictions(entry, model):
                ref[(entry["clip_id"], row["source_frame_id"])] = prediction
        assert m["model_sha256"] == f["model_sha256"]
        assert m["protected_decisions_identical"]
        assert m["fresh_pose_inference"] or m.get("input_mode") == "public_openpose_skeleton"
        matched = 0
        for line in (p / "actions.jsonl").read_text().splitlines():
            r = json.loads(line)
            a = ref.get((m["clip_id"], r["source_frame_id"]))
            if a is None:
                continue
            for key in ["label", "phase", "v6_label", "motion_completion_latched"]:
                assert a[key] == r[key], (p, key, r["source_frame_id"])
            if a["fast_completion_probability"] is not None:
                assert abs(a["fast_completion_probability"] - r["fast_completion_probability"]) < 1e-7
            matched += 1
        assert matched
        result["online_parity"].append(
            dict(
                folder=str(p.relative_to(ROOT)),
                windows=matched,
                mode=m.get("input_mode", "rgb"),
                reference=reference_kind,
            )
        )
        videos.append((p / "comparison.mp4", m["frames"]))
    for p in BASE.glob("joint-*"):
        videos.append((p / "joint-demo.mp4", read(p / "joint-result.json")["counts"]["scene_frames"]))
    for p, expected in videos:
        cap = cv2.VideoCapture(str(p))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.size
            n += 1
        cap.release()
        assert n == expected, (p, n, expected)
        result["videos"].append(dict(path=str(p.relative_to(ROOT)), frames=n))
    ui = BASE / "ui-verification.json"
    if ui.exists():
        result["vscode_ui"] = read(ui)
    log = ROOT / "tmp/action-v7-tests.log"
    matched = re.search(r"(\d+) passed", log.read_text())
    assert matched, "No completed successful test log"
    result["automated_tests"] = {"passed": int(matched.group(1)), "log": str(log.relative_to(ROOT))}
    (BASE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
