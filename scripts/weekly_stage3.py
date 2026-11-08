"""Frozen v2 comparison, fresh pose inference and RGB-D demos for VSCode."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]


def main():
    from fallpre.weekly.action_v2 import evaluate, load_manifest
    from fallpre.weekly.action_v2_demo import render_comparison
    from fallpre.weekly.joint import DEFAULT_ASSETS, run_joint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "demo", "joint", "joint-seated"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip", default="urfd-fall-11")
    parser.add_argument("--pose-mode", choices=["infer", "replay"], default="infer")
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / (
        "weekly-stage3-" + args.command + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    model = ROOT / "runs/action-v2/training/model.json"
    cache = ROOT / "runs/action-v2/poses"
    manifest = load_manifest(ROOT / "configs/action-v2-dataset.json")
    print(f"OUTPUT: {output.resolve()}", flush=True)
    if args.command == "evaluate":
        result = evaluate(manifest, cache, model, output)
        result = {
            k: {m: v[m] for m in ["five_action_macro_f1", "seated_false_rise", "official_falling"]}
            for k, v in result["comparison"].items()
        }
    elif args.command == "demo":
        result = render_comparison(manifest, cache, model, args.clip, output, args.pose_mode)
    else:
        assets = (
            DEFAULT_ASSETS
            if args.command == "joint"
            else ROOT / "runs/action-v2/inputs/tum-sitting/rgbd_dataset_freiburg3_sitting_static"
        )
        result = run_joint(output, assets_dir=assets, action_model=model)
        result = {k: result[k] for k in ["counts", "evaluation", "video"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
