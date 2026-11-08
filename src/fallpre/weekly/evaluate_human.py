"""Post-inference URFD state annotation; no labels enter either neural model.

The official CSV labels describe cam0 depth-frame posture, not NTU60 actions.
This module aligns by official sequence/frame number and never interprets
lying=1 as the NTU action 'falling'. It does not calculate five-action accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "runs/weekly-20261015/labels"
OFFICIAL = {
    "falls": {"filename": "urfall-cam0-falls.csv",
              "sha256": "4ed29af7040098ba0150d1030eeadd97bbfde5c3a4fb9b07df786d673e22a65c"},
    "adls": {"filename": "urfall-cam0-adls.csv",
             "sha256": "e19b2d8379538c16892241aa4cdc9e26a53b296473229c568331c23d7b3cdad1"},
}
STATE_NAMES = {-1: "not_lying", 0: "fall_transition", 1: "lying_on_ground", None: "unknown"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_states(data: bytes) -> dict[tuple[str, int], int]:
    """Read only sequence/frame/state, deliberately ignoring author features."""
    result = {}
    for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
        if not row:
            continue
        if len(row) < 3:
            raise ValueError("URFD state row needs sequence, frame and state")
        clip, frame, state = row[0], int(row[1]), int(row[2])
        if frame < 1 or state not in {-1, 0, 1}:
            raise ValueError("Invalid URFD frame/state")
        key = (clip, frame)
        if key in result:
            raise ValueError("Duplicate official state frame")
        result[key] = state
    if not result:
        raise ValueError("Empty state file")
    return result


def load_official_states(labels_dir: Path, *, download: bool = True) -> tuple[dict, list[dict]]:
    states, sources = {}, []
    for asset in OFFICIAL.values():
        path = labels_dir / asset["filename"]
        url = "https://fenix.ur.edu.pl/~mkepski/ds/data/" + asset["filename"]
        if path.exists():
            data = path.read_bytes()
        elif download:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(1_000_001)
            if len(data) > 1_000_000:
                raise ValueError("Official state download exceeds expected size bound")
        else:
            raise FileNotFoundError(f"Missing official state CSV: {path}")
        if _sha(data) != asset["sha256"]:
            raise ValueError(f"Official state checksum mismatch: {path.name}")
        parsed = parse_states(data)
        if states.keys() & parsed.keys():
            raise ValueError("Duplicate state key across official files")
        states.update(parsed)
        if not path.exists():
            labels_dir.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        sources.append({"url": url, "path": str(path.resolve()), "sha256": _sha(data),
                        "bytes": len(data), "rows": len(parsed)})
    return states, sources


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_states(clip: str, frames: list[int], states: dict[tuple[str, int], int]) -> dict:
    counts = Counter(STATE_NAMES[states.get((clip, frame))] for frame in frames)
    return {"frame_count": len(frames),
            "counts": {name: counts[name] for name in STATE_NAMES.values()},
            "fractions": {name: counts[name] / len(frames) if frames else None for name in STATE_NAMES.values()},
            "missing_frame_ids": [frame for frame in frames if (clip, frame) not in states]}


def evaluate_records(metadata: dict, observations: list[dict], actions: list[dict], metrics: dict,
                     states: dict[tuple[str, int], int]) -> dict:
    """Validate the exact source join before summarizing GT support per window."""
    clip = metrics["clip_id"]
    if metadata.get("clip_id") != clip or not (clip.startswith("fall-") or clip.startswith("adl-")):
        raise ValueError("Run metadata and clip differ or are not a URFD clip")
    if metadata.get("labels_used_by_models") is not False:
        raise ValueError("Run must explicitly declare that models did not use labels")
    if len(observations) != metrics["frame_count"] or len(actions) != metrics["action_windows"]:
        raise ValueError("Run records differ from declared metrics counts")
    by_frame, previous_id, previous_time = {}, 0, -math.inf
    for row in observations:
        frame, time = row["source_frame_id"], row["t"]
        if (not isinstance(frame, int) or frame <= previous_id or not math.isfinite(time)
                or time <= previous_time or row.get("clip_id") != clip):
            raise ValueError("Observation frames/times must strictly increase within the same clip")
        by_frame[frame] = row
        previous_id, previous_time = frame, time
    if not by_frame:
        raise ValueError("Cannot evaluate an empty run")
    frames = list(by_frame)
    official_transition = {frame for frame in frames if states.get((clip, frame)) == 0}
    covered, covered_transition, windows = set(), set(), []
    for action in actions:
        ids, times = action["source_frame_ids"], action["source_times_s"]
        if (not ids or len(ids) != len(times) or len(ids) != action["frames"]
                or any(b != a + 1 for a, b in zip(ids, ids[1:]))):
            raise ValueError("Action window must contain adjacent distinct source frames")
        if any(frame not in by_frame for frame in ids):
            raise ValueError("Action references a frame absent from this run")
        if any(not math.isclose(t, by_frame[f]["t"], abs_tol=1e-8, rel_tol=0) for f, t in zip(ids, times)):
            raise ValueError("Action window source timestamps differ from observations")
        if (not math.isclose(action["window_start_s"], times[0], abs_tol=1e-8, rel_tol=0)
                or not math.isclose(action["window_end_s"], times[-1], abs_tol=1e-8, rel_tol=0)):
            raise ValueError("Action window endpoints differ from source timestamps")
        composition = summarize_states(clip, ids, states)
        intersection = set(ids) & official_transition
        covered.update(ids)
        covered_transition.update(intersection)
        windows.append({"window_start_s": times[0], "window_end_s": times[-1],
                        "source_frame_ids": ids, "top1": action["top5"][0],
                        "official_state_composition": composition,
                        "official_fall_transition_overlap_frames": len(intersection),
                        "fraction_of_run_transition_frames_covered": len(intersection) / len(official_transition)
                        if official_transition else None,
                        "interpretation": "window composition, not a target label for the last frame"})
    transition_no_pose = sum(len(by_frame[f]["instances"]) != 1 for f in official_transition)
    return {
        "schema": "fallpre.weekly.human_evaluation.v1", "clip_id": clip,
        "evaluation_phase": "post_inference_only; no model called by this evaluator",
        "alignment": "official cam0 sequence/frame-number join; labels describe depth frames; "
                     "RGB-depth registration precision is not independently evaluated",
        "official_state_meaning": {str(k): v for k, v in STATE_NAMES.items() if k is not None},
        "run_state_composition": summarize_states(clip, frames, states),
        "action_windows": windows,
        "coverage": {"run_frames": len(frames), "frames_in_any_classified_window": len(covered),
                     "classified_window_source_frame_coverage": len(covered) / len(frames),
                     "official_transition_frames_in_run": len(official_transition),
                     "transition_frames_with_no_unique_pose": transition_no_pose,
                     "transition_frames_in_any_classified_window": len(covered_transition),
                     "transition_window_coverage": len(covered_transition) / len(official_transition)
                     if official_transition else None},
        "clip_level_descriptive_probe": {
            "official_clip_group": "fall_sequence" if clip.startswith("fall-") else "adl_sequence",
            "window_count": len(windows),
            "top1_label_counts": dict(Counter(window["top1"]["label"] for window in windows)),
            "any_window_top1_is_falling": any(window["top1"]["label"] == "falling" for window in windows)
            if windows else None,
            "interpretation": "descriptive top1 occurrence, not a calibrated event detector or frame accuracy"},
        "five_action_accuracy": None,
        "accuracy_reason": "URFD posture states are not five-action or NTU60 labels; lying is not falling",
        "limitations": ["Unknown official labels remain unknown; no negative labels are filled in.",
                        "Overlapping windows are correlated and are not independent test samples.",
                        "Previously inspected demonstration clips do not establish held-out performance.",
                        "Transition label boundaries are annotations, not independently verified impact times."],
    }


def render_timeline(result: dict, observations: list[dict], states: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {-1: "#43a592", 0: "#e1a34b", 1: "#d46569", None: "#b5bdc9"}
    clip = result["clip_id"]
    times = [row["t"] for row in observations]
    interval = statistics.median(b-a for a, b in zip(times, times[1:])) if len(times) > 1 else .033
    windows = result["action_windows"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1, max(2, len(windows) * .55)]})
    for index, row in enumerate(observations):
        left, right = times[index], times[index + 1] if index + 1 < len(times) else times[index] + interval
        axes[0].axvspan(left, right, color=colors[states.get((clip, row["source_frame_id"]))], linewidth=0)
        axes[1].axvspan(left, right, color="#497eae" if len(row["instances"]) == 1 else "#b5bdc9", linewidth=0)
    axes[0].set(title="Official cam0 posture states (not action labels)", yticks=[])
    figure.legend(handles=[Patch(color=c, label=STATE_NAMES[s]) for s, c in colors.items()],
                  loc="upper left", ncol=4, bbox_to_anchor=(.047, .915), fontsize=9)
    axes[1].set(title="Pose observation: blue = one person; gray = no unique pose", yticks=[])
    for index, window in enumerate(windows):
        y = len(windows) - 1 - index
        left, right = window["window_start_s"], window["window_end_s"]
        axes[2].barh(y, right-left, left=left, height=.54, color="#c7d9ea")
        count = window["official_state_composition"]["counts"]
        text = (f"{window['top1']['label']} | GT counts nonlying/transition/lying/unknown: "
                f"{count['not_lying']}/{count['fall_transition']}/{count['lying_on_ground']}/{count['unknown']}")
        axes[2].text(left + .02, y, text, va="center", fontsize=8)
    if not windows:
        axes[2].text(.5, .5, "No available action window — unknown", transform=axes[2].transAxes, ha="center")
    axes[2].set(yticks=[], xlabel="Official source time (s)", title="Past-window ST-GCN++ top1, with GT state composition")
    axes[2].set_xlim(times[0], times[-1] + interval)
    for axis in axes:
        axis.spines[["right", "top", "left"]].set_visible(False)
    figure.suptitle(f"URFD {clip} | post-inference evidence alignment", fontweight="bold", fontsize=15)
    figure.text(.06, .025, "No five-action accuracy. Lying is not the falling action. Missing labels remain unknown. "
                "Overlapping windows are not independent samples.", fontsize=9)
    figure.tight_layout(rect=(.03, .065, .99, .86), h_pad=1.5)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_evaluation(run_dir: Path, output_dir: Path | None = None, *, labels_dir: Path = DEFAULT_LABELS,
                   download: bool = True) -> dict:
    run_dir = run_dir.resolve()
    out = output_dir.resolve() if output_dir else run_dir / "evaluation"
    if out.exists():
        raise FileExistsError(f"Evaluation output already exists: {out}")
    observations_path, actions_path, metrics_path = [run_dir / name for name in
                                                    ("observations.jsonl", "actions.jsonl", "metrics.json")]
    rows = _read_jsonl(observations_path)
    if not rows or rows[0].get("kind") != "meta":
        raise ValueError("Expected human observation metadata first")
    actions = _read_jsonl(actions_path)
    metrics = json.loads(metrics_path.read_text())
    states, sources = load_official_states(labels_dir, download=download)
    result = evaluate_records(rows[0], rows[1:], actions, metrics, states)
    result.update({"source_files": sources, "run_dir": str(run_dir), "output_dir": str(out),
                   "run_input_hashes": {path.name: _sha(path.read_bytes()) for path in
                                         (observations_path, actions_path, metrics_path)}})
    out.mkdir(parents=True, exist_ok=False)
    render_timeline(result, rows[1:], states, out / "timeline.png")
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--offline", action="store_true", help="Require already downloaded, checksum-verified labels")
    args = parser.parse_args(argv)
    result = run_evaluation(args.run, args.output, labels_dir=args.labels_dir, download=not args.offline)
    print(json.dumps({key: result[key] for key in ("clip_id", "output_dir", "coverage", "clip_level_descriptive_probe")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
