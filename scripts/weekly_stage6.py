"""V5 causal phases, independent test, and RGB-D demonstration."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]


def main():
    from evaluate_action_v5 import evaluate

    from fallpre.weekly.action_v5 import load_manifest
    from fallpre.weekly.action_v5_demo import render
    from fallpre.weekly.joint import DEFAULT_ASSETS, run_joint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["demo", "evaluate", "regression", "joint", "joint-seated"])
    p.add_argument("--clip", default="telkom-14-before")
    p.add_argument("--pose-mode", choices=["infer", "replay"], default="infer")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = args.output or ROOT / "runs" / (
        "weekly-stage6-"
        + args.command
        + "-"
        + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    )
    model = ROOT / "runs/action-v5/training/model.json"
    manifest = load_manifest(ROOT / "configs/action-v5-dataset.json")
    print("OUTPUT:", out.resolve(), flush=True)
    if args.command == "demo":
        r = render(manifest, model, args.clip, out, args.pose_mode)
    elif args.command in ["evaluate", "regression"]:
        result = evaluate(manifest, model, out, "test" if args.command == "evaluate" else "dev")
        r = {
            k: {key: m[key] for key in ["per_class", "seated_false_rise"]}
            for k, m in result["comparison"].items()
        }
    else:
        assets = (
            DEFAULT_ASSETS
            if args.command == "joint"
            else ROOT / "runs/action-v2/inputs/tum-sitting/rgbd_dataset_freiburg3_sitting_static"
        )
        r = run_joint(out, assets_dir=assets, action_model=model)
        r = {"counts": r["counts"], "video": r["video"]}
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
