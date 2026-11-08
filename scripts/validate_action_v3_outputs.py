"""Audit frozen hashes, complete video decoding, and fresh-demo/evaluation parity."""

import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v3"


def read(path):
    return json.loads(path.read_text())


def main():
    freeze = read(BASE / "training/freeze.json")
    paths = dict(freeze["source_sha256"])
    paths.update(
        {
            "runs/action-v3/training/model.json": freeze["model_sha256"],
            "configs/action-v3-dataset.json": freeze["manifest_sha256"],
            "configs/action-v3-protocol.json": freeze["protocol_sha256"],
        }
    )
    for path, sha in paths.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    predictions = {}
    for split in ["independent-test", "regression"]:
        for version in ["v2", "v3"]:
            for line in (BASE / split / f"{version}-predictions.jsonl").read_text().splitlines():
                r = json.loads(line)
                predictions[(version, r["clip_id"], r["source_frame_id"])] = r
    result = dict(
        freeze_intact=True,
        videos=[],
        fresh_online_parity=[],
        tests_log="tmp/action-v3-tests.log",
        lint_log="tmp/action-v3-ruff.log",
    )
    folders = sorted(BASE.glob("demo-*")) + sorted(ROOT.glob("runs/weekly-stage4-demo-*"))
    for folder in folders:
        m = read(folder / "result.json")
        assert m["fresh_pose_inference"] and m["falling_decisions_identical"]
        checked = 0
        for line in (folder / "actions.jsonl").read_text().splitlines():
            row = json.loads(line)
            for version in ["v2", "v3"]:
                ref = predictions.get((version, m["clip_id"], row["source_frame_id"]))
                if ref is None:
                    continue
                assert abs(ref["source_time_s"] - row["source_time_s"]) < 1e-7
                assert ref["label"] == row[version]["label"], (folder, row["source_frame_id"], version)
                for name, score in ref["scores"].items():
                    assert abs(score - row[version]["scores"][name]) < 1e-6
                checked += 1
        assert checked
        result["fresh_online_parity"].append(
            dict(folder=str(folder.relative_to(ROOT)), checked_variant_windows=checked)
        )
    videos = [(p / "comparison.mp4", read(p / "result.json")["frames"]) for p in folders]
    videos += [
        (p / "joint-demo.mp4", read(p / "joint-result.json")["counts"]["scene_frames"])
        for p in sorted(BASE.glob("joint-*"))
    ]
    for path, expected in videos:
        cap = cv2.VideoCapture(str(path))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.size
            frames += 1
        cap.release()
        assert frames == expected, (path, frames, expected)
        result["videos"].append(
            dict(
                path=str(path.relative_to(ROOT)),
                decoded_frames=frames,
                codec="".join(chr((codec >> 8 * i) & 255) for i in range(4)),
            )
        )
    (BASE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
