"""Verify frozen artifact integrity, online/evaluation parity, and playable outputs."""

import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v4"


def read(p):
    return json.loads(p.read_text())


def main():
    frozen = read(BASE / "training/freeze.json")
    checks = dict(frozen["source_sha256"])
    checks.update(
        {
            "runs/action-v4/training/model.json": frozen["model_sha256"],
            "configs/action-v4-dataset.json": frozen["manifest_sha256"],
            "configs/action-v4-protocol.json": frozen["protocol_sha256"],
        }
    )
    for p, d in checks.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == d, p
    reference = {}
    for split in ["regression", "independent-test"]:
        for line in (BASE / split / "v4-predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            reference[(r["clip_id"], r["source_frame_id"])] = r
    result = dict(
        freeze_intact=True,
        online_parity=[],
        videos=[],
        tests_log="tmp/action-v4-tests.log",
        lint_log="tmp/action-v4-ruff.log",
    )
    folders = sorted(BASE.glob("demo-*")) + sorted(ROOT.glob("runs/weekly-stage5-demo-*"))
    videos = []
    for p in folders:
        meta = read(p / "result.json")
        assert meta["fresh_pose_inference"] and meta["protected_decisions_identical"]
        matched = 0
        for line in (p / "actions.jsonl").read_text().splitlines():
            r = json.loads(line)
            ref = reference.get((meta["clip_id"], r["source_frame_id"]))
            if ref is None:
                continue
            assert abs(r["source_time_s"] - ref["source_time_s"]) < 1e-7
            for k in ["label", "phase", "v3_label"]:
                assert r[k] == ref[k], (p, k, r["source_frame_id"])
            for k, v in ref["phase_scores"].items():
                assert abs(r["phase_scores"][k] - v) < 1e-6
            matched += 1
        assert matched
        result["online_parity"].append(dict(folder=str(p.relative_to(ROOT)), matched_windows=matched))
        videos.append((p / "comparison.mp4", meta["frames"]))
    for p in sorted(BASE.glob("joint-*")):
        videos.append((p / "joint-demo.mp4", read(p / "joint-result.json")["counts"]["scene_frames"]))
    for p, expected in videos:
        cap = cv2.VideoCapture(str(p))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.size
            n += 1
        cap.release()
        assert n == expected, (p, n, expected)
        result["videos"].append(
            dict(
                path=str(p.relative_to(ROOT)),
                frames=n,
                codec="".join(chr((code >> 8 * i) & 255) for i in range(4)),
            )
        )
    (BASE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
