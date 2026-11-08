"""Audit frozen V7 A/B/C logs with explicit provisional-reference accounting.

No training or visual inference. Source-time weighted durations are left-held
sample estimates, capped at 0.35 s; the last sample has zero duration. MultiCam
has no physical clock and is excluded from action/time metrics.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/abc-v7-20261020"
SOURCES = {
    "A": ROOT / "runs/public-samples-20261020",
    "B": ROOT / "runs/verified-tracking-20261020-v2",
    "C": ROOT / "runs/evidence-recovery-20261020",
}
LABELS = ["walking", "sitting_down", "standing_up", "bending", "falling", "background", "unknown"]
ACTIVE = LABELS[:5]
MAX_HOLD_S = 0.35


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def sample_weights(times, max_hold=MAX_HOLD_S):
    """Return per-sample held duration and unobserved gaps; never invent an end."""
    if not times or all(t is None for t in times):
        return [None] * len(times), None
    if any(t is None or not math.isfinite(t) for t in times):
        raise ValueError("Mixed or invalid source clock")
    differences = [b - a for a, b in zip(times, times[1:])]
    if any(d <= 0 for d in differences):
        raise ValueError("Source clock must strictly increase")
    return [min(d, max_hold) for d in differences] + [0.0], sum(
        max(0, d - max_hold) for d in differences
    )


def segment_value(segment, kind):
    return segment.get(kind, segment.get("label", segment.get("state", segment.get("visibility"))))


def contains(segment, frame):
    return segment["start_frame"] <= frame <= segment["end_frame"]


def reference_at(segments, frame, kind="label"):
    matches = [segment_value(s, kind) for s in segments if contains(s, frame)]
    if len(matches) > 1:
        raise ValueError(f"Overlapping {kind} labels at frame {frame}: {matches}")
    return matches[0] if matches else None


def normalize(mode, row):
    if mode == "A":
        indices = [int(k) for k, v in row["detection_track_ids"].items()
                   if v == row["target_id"]] if row["target_observed"] else []
        index = indices[0] if len(indices) == 1 else None
        state = dict(observed=row["target_observed"], target_index=index,
                     target_id=row["target_id"],
                     target_epoch=row["prediction"].get("target_epoch"),
                     reason=row["tracking_reason"], prediction=row["prediction"])
        poses = row["instances"]
    else:
        state = row["after"]
        poses = row["pose_instances"]
        index = state["target_index"]
    person = poses[index] if index is not None else None
    joints = sum(float(k[2]) >= 0.2 for k in person["kpts"]) if person else 0
    return dict(
        source_frame_id=row["original_frame_id"],
        source_time_s=row["source_time_s"],
        observed=bool(state["observed"]),
        target_id=state["target_id"],
        epoch=state.get("target_epoch", state["prediction"].get("target_epoch")),
        target_index=index,
        selected_box=person["box"] if person else None,
        selected_keypoints=person["kpts"] if person else None,
        confidence_supported_joints=joints,
        presence_only=bool(person and person.get("action_eligible") is False),
        prediction=state["prediction"],
        reason=state.get("reason"),
        recovery_wall_s=row.get("recovery", {}).get("wall_s") if row.get("recovery") else None,
    )


def runs_of_true(mask, rows, weights):
    """Consecutive observation episodes; no bridging unknown or large gaps."""
    result = []
    current = None
    for i, flag in enumerate(mask):
        gap = (i > 0 and rows[i]["source_time_s"] is not None and
               rows[i]["source_time_s"] - rows[i - 1]["source_time_s"] > MAX_HOLD_S)
        if gap or not flag:
            if current is not None:
                result.append(current)
                current = None
        if flag:
            if current is None:
                current = dict(start_frame=rows[i]["source_frame_id"],
                               start_s=rows[i]["source_time_s"], samples=0, duration_s=0.0)
            current.update(end_frame=rows[i]["source_frame_id"], end_s=rows[i]["source_time_s"])
            current["samples"] += 1
            if weights[i] is None:
                current["duration_s"] = None
            else:
                current["duration_s"] += weights[i]
    if current is not None:
        result.append(current)
    return result


def event_hits(rows, segment):
    label = segment_value(segment, "label")
    return [r for r in rows if contains(segment, r["source_frame_id"])
            and r["prediction"]["label"] == label]


def classification(confusion):
    result = {}
    for label in LABELS[:-1]:
        tp = confusion.get(label, {}).get(label, 0.0)
        support = sum(confusion.get(label, {}).values())
        predicted = sum(row.get(label, 0.0) for row in confusion.values())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support and support + predicted else None
        result[label] = dict(support_s=support, predicted_s=predicted, correct_s=tp,
                             precision=precision, recall=recall, f1=f1)
    return result


def evaluate_clip(mode, rows, annotation):
    cid = annotation["clip_id"]
    times = [r["source_time_s"] for r in rows]
    weights, gaps = sample_weights(times)
    clocked = weights[0] is not None
    actions = annotation["action_segments"]
    states = annotation["state_segments"]
    visibility = annotation["visibility_segments"]
    refs = [reference_at(actions, r["source_frame_id"]) for r in rows]
    state_refs = [reference_at(states, r["source_frame_id"], "state") for r in rows]
    vis_refs = [reference_at(visibility, r["source_frame_id"], "visibility") for r in rows]
    preds = [r["prediction"]["label"] for r in rows]
    phases = [r["prediction"]["phase"] for r in rows]
    confusion = defaultdict(lambda: defaultdict(float))
    if clocked:
        for ref, pred, weight in zip(refs, preds, weights):
            if ref in LABELS[:-1]:
                confusion[ref][pred] += weight
    confusion = {k: dict(v) for k, v in confusion.items()}
    unknown = runs_of_true([p == "unknown" for p in preds], rows, weights)
    false_runs = {}
    for label in ACTIVE:
        false_runs[label] = runs_of_true(
            [ref in LABELS[:-1] and ref != label and pred == label
             for ref, pred in zip(refs, preds)], rows, weights
        ) if clocked else []
    events = []
    if clocked:
        for i, seg in enumerate(actions):
            label = segment_value(seg, "label")
            if label not in ACTIVE:
                continue
            hits = event_hits(rows, seg)
            source_start = seg.get("start_s")
            if source_start is None:
                available = [r["source_time_s"] for r in rows if contains(seg, r["source_frame_id"])]
                source_start = available[0] if available else None
            events.append(dict(
                mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
                event_id=f'{annotation["event_group_id"]}:{label}:{i}',
                label=label, start_frame=seg["start_frame"], end_frame=seg["end_frame"],
                start_s=source_start, detected=bool(hits),
                first_hit_frame=hits[0]["source_frame_id"] if hits else None,
                first_hit_s=hits[0]["source_time_s"] if hits else None,
                first_hit_delay_s=hits[0]["source_time_s"] - source_start
                if hits and source_start is not None else None,
                primary=annotation.get("primary_event_representation", cid != "urfd-fall-01"),
                human_reviewed=seg.get("human_reviewed", False),
                reference_status=seg.get("status"), reference_source=seg.get("source"),
                rule="one or more current V7 label hits INSIDE annotated interval; diagnostic, no debounce",
            ))
    def duration(mask):
        return sum(w for w, ok in zip(weights, mask) if ok) if clocked else None

    metrics = dict(
        mode=mode, clip_id=cid, event_group_id=annotation["event_group_id"],
        primary=annotation.get("primary_event_representation", cid != "urfd-fall-01"),
        frame_only=not clocked, samples=len(rows),
        source_span_s=times[-1] - times[0] if clocked else None,
        held_observation_s=sum(weights) if clocked else None, unobserved_clock_gap_s=gaps,
        annotated_action_s=sum(sum(v.values()) for v in confusion.values()) if clocked else None,
        unscored_action_s=duration([ref not in LABELS[:-1] for ref in refs]),
        unscored_falling_samples=sum(ref not in LABELS[:-1] and pred == "falling"
                                    for ref, pred in zip(refs, preds)) if clocked else None,
        unscored_falling_s=duration([ref not in LABELS[:-1] and pred == "falling"
                                    for ref, pred in zip(refs, preds)]),
        unknown_samples=preds.count("unknown") if clocked else None,
        unknown_s=duration([p == "unknown" for p in preds]),
        longest_unknown_s=max((r["duration_s"] for r in unknown), default=0) if clocked else None,
        target_unavailable_samples=sum(not r["observed"] for r in rows),
        target_unavailable_s=duration([not r["observed"] for r in rows]),
        visible_reference_samples=vis_refs.count("visible"),
        visible_target_unavailable_samples=sum(v == "visible" and not r["observed"]
                                               for r, v in zip(rows, vis_refs)),
        empty_reference_samples=vis_refs.count("empty"),
        empty_false_target_samples=sum(v == "empty" and r["observed"] for r, v in zip(rows, vis_refs)),
        presence_only_samples=sum(r["presence_only"] for r in rows),
        logical_epochs=len({r["epoch"] for r in rows if r["observed"] and r["epoch"] is not None}),
        target_ids=len({r["target_id"] for r in rows if r["observed"]}),
        seated_reference_samples=state_refs.count("seated"),
        seated_forward_lean_samples=sum(s == "seated" and p == "forward_lean"
                                        for s, p in zip(state_refs, phases)),
        seated_false_rise_samples=sum(s == "seated" and p == "standing_up"
                                      for s, p in zip(state_refs, preds)),
        seated_false_rise_s=duration([s == "seated" and p == "standing_up"
                                      for s, p in zip(state_refs, preds)]),
        seated_phase_unknown_samples=sum(s == "seated" and p == "unknown"
                                         for s, p in zip(state_refs, phases)),
        action_event_hits=sum(e["detected"] for e in events) if clocked else None,
        action_event_total=len(events) if clocked else None,
        action_counts=dict(Counter(preds)), phase_counts=dict(Counter(phases)),
        false_action_episodes={label: len(episodes) for label, episodes in false_runs.items()} if clocked else None,
        false_action_s={label: sum(r["duration_s"] for r in episodes)
                        for label, episodes in false_runs.items()} if clocked else None,
        confusion_duration_s=confusion, per_class=classification(confusion),
        human_reviewed=False,
    )
    details = dict(unknown_runs=unknown, false_action_runs=false_runs)
    return metrics, events, details


def aggregate(metrics, events):
    result = {}
    for mode in SOURCES:
        selected = [m for m in metrics if m["mode"] == mode and m["primary"] and not m["frame_only"]]
        expected_groups = {m["event_group_id"] for m in metrics
                           if m["mode"] == mode and not m["frame_only"]}
        primary_counts = Counter(m["event_group_id"] for m in selected)
        if set(primary_counts) != expected_groups or any(n != 1 for n in primary_counts.values()):
            raise ValueError(f"{mode}: each clocked event group needs exactly one primary representation")
        ev = [e for e in events if e["mode"] == mode and e["primary"]]
        confusion = defaultdict(lambda: defaultdict(float))
        for m in selected:
            for ref, values in m["confusion_duration_s"].items():
                for label, value in values.items():
                    confusion[ref][label] += value
        conf = {k: dict(v) for k, v in confusion.items()}
        by_action = {}
        for label in ACTIVE:
            chosen = [e for e in ev if e["label"] == label]
            by_action[label] = dict(detected=sum(e["detected"] for e in chosen), events=len(chosen))
        result[mode] = dict(
            unique_clocked_groups=len(selected),
            observations=sum(m["samples"] for m in selected),
            observed_span_s=sum(m["source_span_s"] for m in selected),
            annotated_action_s=sum(m["annotated_action_s"] for m in selected),
            unscored_action_s=sum(m["unscored_action_s"] for m in selected),
            unscored_falling_samples=sum(m["unscored_falling_samples"] for m in selected),
            unscored_falling_s=sum(m["unscored_falling_s"] for m in selected),
            unknown_s=sum(m["unknown_s"] for m in selected),
            target_unavailable_s=sum(m["target_unavailable_s"] for m in selected),
            seated_false_rise_samples=sum(m["seated_false_rise_samples"] for m in selected),
            seated_forward_lean_samples=sum(m["seated_forward_lean_samples"] for m in selected),
            seated_reference_samples=sum(m["seated_reference_samples"] for m in selected),
            action_event_hits=sum(e["detected"] for e in ev), action_event_total=len(ev),
            events_by_action=by_action,
            false_action_episodes={label: sum(m["false_action_episodes"][label] for m in selected)
                                   for label in ACTIVE},
            false_action_s={label: sum(m["false_action_s"][label] for m in selected) for label in ACTIVE},
            confusion_duration_s=conf, per_class=classification(conf),
        )
    return result


def build_attribution(streams, annotations):
    issues = []
    for cid, modes in streams.items():
        for pair in [("A", "B"), ("B", "C"), ("A", "C")]:
            before, after = modes[pair[0]], modes[pair[1]]
            for a, b in zip(before, after):
                label_change = a["prediction"]["label"] != b["prediction"]["label"]
                presence_change = a["observed"] != b["observed"]
                if not label_change and not presence_change:
                    continue
                ref = reference_at(annotations[cid]["action_segments"], b["source_frame_id"])
                if presence_change:
                    cause = "current_target_recovered" if b["observed"] else "target_withheld_or_lost"
                elif b["presence_only"]:
                    cause = "partial_presence_action_withheld"
                else:
                    cause = "temporal_output_changed_with_frontend"
                if ref not in LABELS[:-1]:
                    effect = "unscored_reference_gap"
                elif b["prediction"]["label"] == ref and a["prediction"]["label"] != ref:
                    effect = "provisional_reference_match_improved"
                elif a["prediction"]["label"] == ref and b["prediction"]["label"] != ref:
                    effect = "provisional_reference_match_regressed"
                else:
                    effect = "no_reference_match_gain"
                issues.append(dict(
                    clip_id=cid, pair="→".join(pair), source_frame_id=b["source_frame_id"],
                    source_time_s=b["source_time_s"], before_observed=a["observed"],
                    after_observed=b["observed"], before_label=a["prediction"]["label"],
                    after_label=b["prediction"]["label"], reference=ref,
                    cause=cause, provisional_effect=effect, after_reason=b["reason"],
                    human_reviewed=False,
                ))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = read_json(SOURCES["A"] / "manifest.json")
    annotation_path = out / "annotations/annotations.json"
    annotations = {e["clip_id"]: e for e in read_json(annotation_path)["entries"]}
    model = ROOT / "runs/action-v7/training/model.json"
    expected_hash = manifest["frozen_models"][str(model)]
    assert digest(model) == expected_hash
    protocol = dict(
        schema="fallpre.abc_v7.evaluation.v1", model_sha256=expected_hash,
        groups={"A": "legacy/off", "B": "verified/off", "C": "verified/evidence"},
        training=False, sources={k: str(v) for k, v in SOURCES.items()},
        annotations_sha256=digest(annotation_path),
        metric_rules={
            "sample_duration": "left-held until next source timestamp, cap 0.35s; last sample zero",
            "gap_policy": "excess gap excluded from held duration and separately reported",
            "action_reference": "inclusive source frame intervals; gaps unscored, unknown prediction retained",
            "events": "any label hit inside interval; no future smoothing or tuned debounce",
            "false_episodes": "consecutive samples of a class outside its annotated interval; gap/unknown breaks episode",
            "deduplication": "URFD fall01 original primary; low-res same event secondary",
            "multicam": "frame-only target observations; no action, seconds, speed or lead-time",
            "status": "development regression; all project annotation intervals await independent human review",
            "latency": "historical recovery_wall_s covers recovery only, not end-to-end latency",
            "epochs": "tracker continuity diagnostic, not ground-truth identity switch count",
        },
        source_logs={}, source_inputs={},
    )
    streams, metrics, events, all_details = {}, [], [], {}
    for entry in manifest["entries"]:
        cid = entry["clip_id"]
        assert cid in annotations
        src = Path(entry.get("zip_path", entry.get("video_path")))
        assert digest(src) == entry["input_sha256"]
        protocol["source_inputs"][cid] = dict(file=str(src), sha256=entry["input_sha256"])
        streams[cid] = {}
        for mode, folder in SOURCES.items():
            path = folder / cid / "frames.jsonl"
            raw = read_rows(path)
            assert len(raw) == entry["frame_count"]
            rows = [normalize(mode, r) for r in raw]
            streams[cid][mode] = rows
            protocol["source_logs"][f"{mode}/{cid}"] = dict(file=str(path), sha256=digest(path))
            metric, ev, details = evaluate_clip(mode, rows, annotations[cid])
            metrics.append(metric)
            events.extend(ev)
            all_details[f"{mode}/{cid}"] = details
        signatures = [[(r["source_frame_id"], r["source_time_s"]) for r in streams[cid][mode]]
                      for mode in SOURCES]
        assert signatures[0] == signatures[1] == signatures[2]
    result = dict(
        protocol=protocol, clips=metrics, aggregate=aggregate(metrics, events),
        unique_event_groups=len({e["event_group_id"] for e in annotations.values()}),
        source_streams=len(streams),
        input_observations=sum(len(v["A"]) for v in streams.values()),
    )
    issues = build_attribution(streams, annotations)
    write_json(out / "evaluation-protocol.json", protocol)
    write_json(out / "metrics.json", result)
    write_json(out / "events.json", events)
    write_json(out / "episodes.json", all_details)
    write_json(out / "attribution.json", issues)
    scalar_metrics = [{k: v for k, v in m.items() if not isinstance(v, (dict, list))} for m in metrics]
    write_csv(out / "metrics.csv", scalar_metrics)
    write_csv(out / "events.csv", events)
    write_csv(out / "attribution.csv", issues)
    action_metrics = []
    for m in metrics:
        if m["frame_only"]:
            continue
        for label in ACTIVE:
            action_metrics.append(dict(
                mode=m["mode"], clip_id=m["clip_id"], primary=m["primary"], label=label,
                **m["per_class"][label],
                false_episodes=m["false_action_episodes"][label],
                false_duration_s=m["false_action_s"][label],
                human_reviewed=False,
            ))
    write_csv(out / "action-metrics.csv", action_metrics)
    assert digest(model) == expected_hash
    assert digest(annotation_path) == protocol["annotations_sha256"]
    for record in protocol["source_logs"].values():
        assert digest(record["file"]) == record["sha256"]
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
