# ruff: noqa: E402
# Standalone script: configure the local source path before importing project modules.
"""Evaluate frozen V7 using the same causal path as the video demo."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]
from fallpre.weekly.action_v4 import PHASES, PROTECTED
from fallpre.weekly.action_v7 import CompletionModel, iter_predictions, load_manifest
from fallpre.weekly.five_actions import classification_metrics, label_at
from fallpre.weekly.human import sha256


def evaluate(manifest, model_path, output, split="test"):
    model = CompletionModel.load(model_path)
    if model.artifact["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if split == "test":
        frozen = json.loads((model_path.parent / "freeze.json").read_text())
        assert sha256(model_path) == frozen["model_sha256"]
        assert manifest["manifest_sha256"] == frozen["manifest_sha256"]
        for p, d in frozen["source_sha256"].items():
            assert sha256(ROOT / p) == d, p
    output.mkdir(parents=True, exist_ok=False)
    variants = {"v6": [], "v7": []}
    matrix = [[0] * len(PHASES) for _ in PHASES]
    clips = []
    events = []
    for e in manifest["entries"]:
        if e["split"] != split:
            continue
        if split == "test" and e.get("legacy_v6"):
            raise ValueError("Legacy test leakage")
        records = []
        next_t = -1
        for row, tr, new in iter_predictions(e, model):
            if row["t"] + 1e-8 < next_t:
                continue
            next_t = row["t"] + 0.1
            phase = next(
                (s["label"] for s in e.get("phase_segments", []) if s["start_s"] <= row["t"] < s["end_s"]),
                "unknown",
            )
            state_seated = any(
                s["state"] == "seated" and s["start_s"] <= row["t"] < s["end_s"]
                for s in e.get("state_segments", [])
            )
            record = dict(
                new,
                clip_id=e["clip_id"],
                source_time_s=row["t"],
                source_frame_id=row["source_frame_id"],
                target=label_at(e, row["t"]),
                reference_phase=phase,
                seated=state_seated,
                subject_id=e.get("subject_id"),
                input_mode=e.get("input_mode", "rgb"),
                target_epoch=tr["target_epoch"],
            )
            assert record["v6_label"] not in PROTECTED or record["label"] == record["v6_label"]
            assert record["label"] not in PROTECTED or record["label"] == record["v6_label"]
            records.append(record)
            variants["v7"].append(record)
            variants["v6"].append(dict(record, label=record["v6_label"]))
            if phase != "unknown":
                matrix[PHASES.index(phase)][PHASES.index(record["phase"])] += 1
        clips.append(
            dict(
                clip_id=e["clip_id"],
                subject_id=e.get("subject_id"),
                input_mode=e.get("input_mode", "rgb"),
                comparison={
                    v: classification_metrics(
                        [dict(r, label=r["v6_label"]) if v == "v6" else r for r in records]
                    )
                    for v in variants
                },
            )
        )
        for s in e["segments"]:
            if s["label"] != "standing_up":
                continue
            event = dict(clip_id=e["clip_id"], start_s=s["start_s"], end_s=s["end_s"])
            for v in variants:
                hits = [
                    r
                    for r in records
                    if s["start_s"] <= r["source_time_s"] < s["end_s"]
                    and (r["label"] if v == "v7" else r["v6_label"]) == "standing_up"
                ]
                event[v] = dict(
                    detected=bool(hits), delay_s=hits[0]["source_time_s"] - s["start_s"] if hits else None
                )
            events.append(event)
        print(e["clip_id"], flush=True)
    summary = {}
    for v, rr in variants.items():
        metrics = classification_metrics(rr)
        for name, select in [
            ("seated", lambda r: r["seated"]),
            ("forward_lean", lambda r: r["reference_phase"] == "forward_lean"),
            ("standing", lambda r: r["reference_phase"] == "standing"),
        ]:
            selected = [r for r in rr if select(r)]
            metrics[name + "_false_rise"] = dict(
                support=len(selected),
                false_rise=sum(r["label"] == "standing_up" for r in selected),
                unknown=sum(r["label"] == "unknown" for r in selected),
            )
        summary[v] = metrics
        (output / f"{v}-predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
    result = dict(
        schema="fallpre.completion_evaluation.v7",
        split=split,
        model_sha256=sha256(model_path),
        manifest_sha256=manifest["manifest_sha256"],
        comparison=summary,
        phase_labels=PHASES,
        phase_confusion_rows_true_columns_pred=matrix,
        clips=clips,
        rising_events=events,
        protected_decisions_identical=True,
        phase_counts=dict(Counter(r["phase"] for r in variants["v7"])),
        subsets={
            mode: {
                v: classification_metrics([r for r in rr if r["input_mode"] == mode])
                for v, rr in variants.items()
            }
            for mode in ["rgb", "public_openpose_skeleton"]
        },
        annotation_limit="New phases are agent visual provisional labels, ~0.25s boundary uncertainty; three held-out Telkom people 7/8/9. Legacy official interval metrics reported separately; no forced relabeling.",
    )
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "runs/action-v7/training/model.json")
    args = p.parse_args()
    r = evaluate(load_manifest(ROOT / "configs/action-v7-dataset.json"), args.model, args.output, args.split)
    print({v: (m["per_class"]["standing_up"], m["seated_false_rise"]) for v, m in r["comparison"].items()})


if __name__ == "__main__":
    main()
