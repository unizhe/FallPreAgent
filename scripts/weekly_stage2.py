"""Repeatable VSCode entry points with separate, timestamped output directories."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "tmp/ultralytics"))


def main(argv=None):
    from fallpre.weekly.five_actions import LABELS, evaluate, extract_poses, load_manifest, render_demo, train

    parser = argparse.ArgumentParser(description="五动作公开数据验证 / 真人2米邻域联合演示")
    parser.add_argument("command", choices=["train-evaluate", "evaluate", "action-demo", "joint-demo"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/five-action-dataset.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/five-actions-v1.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/five-actions/poses")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action", choices=["all", *LABELS[:5]], default="all")
    parser.add_argument("--clip", help="Optional exact clip ID instead of selecting by action")
    parser.add_argument("--duration", type=float, default=8.)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "runs" / (
        "weekly-stage2-" + args.command + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    print(f"OUTPUT: {output.resolve()}", flush=True)
    if args.command == "joint-demo":
        from fallpre.weekly.joint import run_joint

        result = run_joint(output, device=args.device, duration_s=args.duration, action_model=args.model)
        print(json.dumps({k: result[k] for k in ("output_dir", "counts", "evaluation", "video")}, indent=2))
        return
    manifest = load_manifest(args.manifest)
    if args.command == "evaluate":
        result = evaluate(manifest, args.cache, args.model, output)
        print(json.dumps(result["metrics"], indent=2))
        return
    output.mkdir(parents=True)
    if args.command == "train-evaluate":
        extracted = extract_poses(manifest, args.cache, device=args.device)
        model_path = output / "five-actions.json"
        training = train(manifest, args.cache, model_path)
        result = evaluate(manifest, args.cache, model_path, output / "evaluation")
        summary = {"output_dir": str(output.resolve()), "extraction": extracted, "training": training,
                   "evaluation": result["metrics"], "evaluation_path": str(output / "evaluation/evaluation.json")}
    else:
        if args.clip:
            clips = [args.clip]
        else:
            actions = LABELS[:5] if args.action == "all" else [args.action]
            clips = []
            for action in actions:
                entry = next((e for e in manifest["entries"] if e["split"] == "test"
                              and any(s["label"] == action for s in e["segments"])), None)
                if entry is None:
                    raise ValueError(f"No held-out video annotated with {action}")
                if entry["clip_id"] not in clips:
                    clips.append(entry["clip_id"])
        demos = [render_demo(manifest, args.cache, args.model, clip, output / clip,
                             pose_mode="infer", device=args.device) for clip in clips]
        summary = {"output_dir": str(output.resolve()), "requested_action": None if args.clip else args.action,
                   "explicit_clip": args.clip, "demos": demos,
                   "selection": ("Explicit clip ID. " if args.clip else
                                 "First manifest test clip per requested annotated action, deduplicated. ") +
                                "These visual examples do not replace evaluation over the entire test split."}
    (output / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
