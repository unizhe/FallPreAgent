"""V3: subject-separated evaluation and live-inference demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]


def main():
    from fallpre.weekly.action_v3 import evaluate, load_manifest, train
    from fallpre.weekly.action_v3_demo import render_comparison
    from fallpre.weekly.joint import DEFAULT_ASSETS, run_joint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["evaluate", "regression", "demo", "joint", "joint-seated", "train"])
    p.add_argument("--output", type=Path)
    p.add_argument("--clip", default="utk-s08_e01-transition")
    p.add_argument("--pose-mode", choices=["infer", "replay"], default="infer")
    args = p.parse_args()
    output = args.output or ROOT / "runs" / (
        "weekly-stage4-"
        + args.command
        + "-"
        + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    )
    manifest = load_manifest(ROOT / "configs/action-v3-dataset.json")
    model = ROOT / "runs/action-v3/training/model.json"
    print(f"OUTPUT: {output.resolve()}", flush=True)
    if args.command in {"evaluate", "regression"}:
        result = evaluate(
            manifest,
            model,
            output,
            split="test" if args.command == "evaluate" else "dev",
            legacy_only=args.command == "regression",
        )
        result = {
            k: {key: v[key] for key in ["five_action_macro_f1", "seated_false_rise", "official_falling"]}
            for k, v in result["comparison"].items()
        }
    elif args.command == "train":
        result = {"model": str(train(manifest, output))}
    elif args.command == "demo":
        result = render_comparison(manifest, model, args.clip, output, args.pose_mode)
    else:
        assets = (
            DEFAULT_ASSETS
            if args.command == "joint"
            else ROOT / "runs/action-v2/inputs/tum-sitting/rgbd_dataset_freiburg3_sitting_static"
        )
        r = run_joint(output, assets_dir=assets, action_model=model)
        result = {k: r[k] for k in ["counts", "evaluation", "video"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
