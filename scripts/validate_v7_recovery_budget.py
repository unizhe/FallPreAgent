"""Independent, stdlib-only audit of fixed-V7 recovery-budget logs.

This reads recorded results; it never imports the runner, scheduling helper,
model code, torch or a neural runtime. Source-gap flags are reset conditions,
not an observed count of all actual model/tracker resets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/abc-v7-20261020"
DEFAULT = BASE / "recovery-budget-20261021"
CLIP = "urfd-fall-01-original"
KNOWN_LABELS = {"walking", "sitting_down", "standing_up", "bending", "falling", "background"}
TOLERANCE_S = 1e-6


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, location="value"):
    """Compare recomputed aggregate floats with a fixed sub-millisecond tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{location}: expected object")
        for key, value in expected.items():
            require(key in actual, f"{location}: missing {key}")
            equal(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                f"{location}: unequal sequence lengths")
        for i, value in enumerate(expected):
            equal(actual[i], value, f"{location}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=TOLERANCE_S),
                f"{location}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{location}: {actual!r} != {expected!r}")


def distribution(values):
    ordered = sorted(values)
    p = .95 * (len(ordered) - 1)
    lower = math.floor(p)
    upper = math.ceil(p)
    return {"n": len(ordered), "mean_s": statistics.mean(ordered),
            "median_s": statistics.median(ordered), "max_s": ordered[-1],
            "p95_s": ordered[lower] + (ordered[upper] - ordered[lower]) * (p - lower)}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, operation):
        started = time.perf_counter()
        try:
            detail = operation()
            result = {"name": name, "passed": True, "evidence": detail}
        except Exception as error:
            result = {"name": name, "passed": False,
                      "error": f"{type(error).__name__}: {error}"}
        result["elapsed_s"] = time.perf_counter() - started
        self.checks.append(result)


def validate_trial(audit, trial, entry, cached, aggregate):
    data = rows(trial / "frames.jsonl")
    summary = read(trial / "summary.json")
    times = entry["clock"]["source_times_s"]
    bounds = entry["event_landmarks"]["contact"]["time_interval_s"]
    n = len(times)
    strategy, repeat_text = trial.name.rsplit("-", 1)
    repeat = int(repeat_text.removeprefix("r"))
    mode, policy = "C", "latest"
    budget = None if strategy == "baseline" else .200
    ids = [r["source_frame_id"] for r in data]
    dropped = [fid for r in data for fid in r["dropped_before_this_frame"]]
    by_frame = {r["source_frame_id"]: r for r in data}
    refs = {}
    for fid in range(1, n + 1):
        hits = [segment["label"] for segment in entry["action_segments"]
                if segment["start_frame"] <= fid <= segment["end_frame"]]
        require(len(hits) <= 1, f"Overlapping action reference at {fid}")
        refs[fid] = hits[0] if hits and hits[0] in KNOWN_LABELS else None
    weights = [min(b - a, .35) for a, b in zip(times, times[1:])] + [0.0]

    def partition():
        require(ids == sorted(set(ids)), "Processed IDs not unique/ascending")
        require(dropped == sorted(set(dropped)), "Dropped IDs not unique/ascending")
        require(not set(ids) & set(dropped), "Processed and dropped overlap")
        equal(sorted(ids + dropped), list(range(1, n + 1)), "source partition")
        require(ids[-1] == n, "Final source frame was not processed")
        if policy == "fifo":
            equal(ids, list(range(1, n + 1)), "FIFO preserves every source observation")
        equal(summary, {"source_observations": n, "processed_frames": len(ids),
                        "dropped_frames": len(dropped), "dropped_source_frame_ids": dropped,
                        "processed_fraction": len(ids) / n})
        return {"source": n, "processed": len(ids), "dropped": len(dropped)}
    audit.check(f"{trial.name}/160_frame_partition", partition)

    def schedule():
        cursor = 0
        previous_output = 0.0
        for r in data:
            selection = r["selection_wall_s"]
            fid = r["source_frame_id"]
            arrived = sum(t <= selection for t in times)
            require(arrived > cursor, f"Frame {fid}: selected before any pending arrival")
            expected_index = cursor if policy == "fifo" else arrived - 1
            equal(fid, expected_index + 1, f"frame selection at {selection}")
            equal(r["dropped_before_this_frame"], list(range(cursor + 1, expected_index + 1)),
                  f"Frame {fid} superseded IDs")
            equal(r["arrived_unprocessed_before_selection"], arrived - cursor,
                  f"Frame {fid} arrived pending count")
            require(selection >= previous_output, f"Frame {fid}: worker overlap")
            require(r["source_time_s"] <= selection, f"Frame {fid}: future input")
            cursor = fid
            previous_output = r["output_wall_s"]
        return {"checked_selections": len(data), "policy": policy,
                "future_inputs": 0, "overlapping_worker_calls": 0}
    audit.check(f"{trial.name}/causal_selection", schedule)

    def clock():
        previous_source = None
        for r in data:
            fid, t = r["source_frame_id"], r["source_time_s"]
            equal(t, times[fid - 1], f"Frame {fid} original CSV time")
            equal(r["prediction"]["source_frame_id"], fid, f"Frame {fid} V7 source ID")
            equal(r["prediction"]["source_time_s"], t, f"Frame {fid} V7 source time")
            equal(r["mode"], mode)
            equal(r["policy"], policy)
            equal(r["repeat"], repeat)
            equal(r["strategy"], strategy)
            equal(r["budget_s"], budget)
            require(r["selection_wall_s"] <= r["service_start_wall_s"] <= r["output_wall_s"],
                    f"Frame {fid}: invalid clock ordering")
            equal(r["queue_wait_s"], r["service_start_wall_s"] - t, f"Frame {fid} queue")
            equal(r["service_s"], r["output_wall_s"] - r["service_start_wall_s"], f"Frame {fid} service")
            equal(r["output_age_s"], r["output_wall_s"] - t, f"Frame {fid} output age")
            equal(r["output_age_s"], r["queue_wait_s"] + r["service_s"], f"Frame {fid} age decomposition")
            require(all(math.isfinite(v) and v >= 0 for v in r["stages_s"].values()),
                    f"Frame {fid}: non-finite/negative stage timing")
            equal(sum(r["stages_s"].values()), r["service_s"], f"Frame {fid} stage sum")
            gap = None if previous_source is None else t - previous_source
            equal(r["source_gap_s"], gap, f"Frame {fid} source gap")
            equal(r["gap_exceeds_action_reset"], gap is not None and gap > .35)
            equal(r["gap_exceeds_tracker_reset"], gap is not None and gap > .5)
            previous_source = t
        return {"checked_rows": len(data), "tolerance_s": TOLERANCE_S,
                "source_gap_flags": "Threshold conditions only, not all actual resets"}
    audit.check(f"{trial.name}/source_clock_and_wall_timing", clock)

    def pending():
        for r in data:
            waiting = sum(t <= r["output_wall_s"] for t in times) - r["source_frame_id"]
            equal(r["arrived_not_yet_accounted_at_output"], max(0, waiting))
            equal(r["pending_slots_at_output"], min(1, waiting) if policy == "latest" else waiting)
        maximum = max(r["pending_slots_at_output"] for r in data)
        equal(summary["max_pending_slots"], maximum)
        if policy == "latest":
            require(maximum <= 1, "Latest policy retained >1 pending slot")
        return {"max_pending_slots": maximum,
                "semantics": "Logical scheduling slots, not independently sampled camera-driver queue"}
    audit.check(f"{trial.name}/pending_accounting", pending)

    def prediction_comparison():
        label_changes, phase_changes, exact_count = [], [], 0
        for r in data:
            old_row = cached[r["source_frame_id"] - 1]
            old = old_row["prediction"] if mode == "A" else old_row["after"]["prediction"]
            pred = r["prediction"]
            same_label, same_phase, same_all = pred["label"] == old["label"], pred["phase"] == old["phase"], pred == old
            equal(r["same_source_full_fifo_label_equal"], same_label)
            equal(r["same_source_full_fifo_phase_equal"], same_phase)
            equal(r["same_source_full_fifo_prediction_equal"], same_all)
            if not same_label:
                label_changes.append(r["source_frame_id"])
            if not same_phase:
                phase_changes.append(r["source_frame_id"])
            exact_count += same_all
        equal(summary, {"same_source_fifo_label_changes": label_changes,
                        "same_source_fifo_phase_changes": phase_changes,
                        "full_prediction_equality_count": exact_count})
        return {"exact_cached_predictions": exact_count, "label_changes": label_changes,
                "phase_changes": phase_changes, "latest_differences_allowed": policy == "latest"}
    audit.check(f"{trial.name}/full_fifo_prediction_comparison", prediction_comparison)

    def detection():
        first = next((r for r in data if r["prediction"]["label"] == "falling"), None)
        if first is None:
            expected, category = None, "miss_no_falling_output"
        else:
            expected = {"frame": first["source_frame_id"], "source_time_s": first["source_time_s"],
                        "output_wall_s": first["output_wall_s"],
                        "source_margin_s": [b - first["source_time_s"] for b in bounds],
                        "contact_margin_s": [b - first["output_wall_s"] for b in bounds],
                        **{k: first[k] for k in ("queue_wait_s", "service_s", "output_age_s")}}
            category = ("before_contact_certain" if first["output_wall_s"] < bounds[0]
                        else "within_contact_uncertainty" if first["output_wall_s"] <= bounds[1]
                        else "after_contact_certain")
        equal(summary["first_falling"], expected)
        equal(summary["contact_result"], category)
        segment = next(s for s in entry["action_segments"] if s["label"] == "falling")
        event = set(range(segment["start_frame"], segment["end_frame"] + 1))
        hits = [r["source_frame_id"] for r in data if r["source_frame_id"] in event and r["prediction"]["label"] == "falling"]
        equal(summary["official_fall_event"], {"start_frame": segment["start_frame"],
              "end_frame": segment["end_frame"], "total_reference_frames": len(event),
              "processed_reference_frames": len(event & set(ids)),
              "dropped_reference_frames": len(event & set(dropped)),
              "falling_output_frames": hits, "hit": bool(hits),
              "reference_status": segment.get("status"),
              "human_reviewed": segment.get("human_reviewed", False)})
        false = [r["source_frame_id"] for r in data if refs[r["source_frame_id"]] not in (None, "falling") and r["prediction"]["label"] == "falling"]
        unscored = [r["source_frame_id"] for r in data if refs[r["source_frame_id"]] is None and r["prediction"]["label"] == "falling"]
        equal(summary["provisional_labeled_false_falling_frames"], false)
        equal(summary["unscored_falling_frames"], unscored)
        return {"first_falling": expected, "contact_result": category,
                "official_event_hit": bool(hits), "provisional_false_falling": false,
                "unscored_falling": unscored}
    audit.check(f"{trial.name}/first_trigger_contact_and_official_event", detection)

    def coverage():
        known = [r for r in data if refs[r["source_frame_id"]] is not None]
        matched = [r for r in known if refs[r["source_frame_id"]] == r["prediction"]["label"]]
        expected = {"total_source_reference_samples": n,
                    "processed_labeled_samples": len(known),
                    "matched_processed_labeled_samples": len(matched),
                    "all_labeled_source_samples": sum(v is not None for v in refs.values()),
                    "dropped_labeled_samples": sum(refs[i] is not None for i in dropped),
                    "matched_source_weight_s": sum(weights[r["source_frame_id"] - 1] for r in matched),
                    "processed_source_weight_s": sum(weights[i - 1] for i in ids),
                    "dropped_source_weight_s": sum(weights[i - 1] for i in dropped)}
        equal(summary["coverage"], expected)
        equal(expected["processed_source_weight_s"] + expected["dropped_source_weight_s"], sum(weights))
        require(expected["processed_labeled_samples"] + expected["dropped_labeled_samples"] == expected["all_labeled_source_samples"], "Labeled denominator lost skipped samples")
        require(all(refs[i] is None for i in range(113, 138)), "Rolling interval must remain unscored")
        return expected
    audit.check(f"{trial.name}/unchanged_denominators_and_unscored_interval", coverage)

    def aggregates():
        epochs = [r["target_epoch"] for r in data if r["observed"]]
        expected = {"last_output_wall_s": data[-1]["output_wall_s"],
                    "source_span_s": times[-1] - times[0],
                    "service": distribution([r["service_s"] for r in data]),
                    "queue_wait": distribution([r["queue_wait_s"] for r in data]),
                    "output_age": distribution([r["output_age_s"] for r in data]),
                    "stages": {key: distribution([r["stages_s"][key] for r in data]) for key in data[0]["stages_s"]},
                    "max_source_gap_s": max(r["source_gap_s"] or 0 for r in data),
                    "source_gaps_gt_0_35_s": sum(r["gap_exceeds_action_reset"] for r in data),
                    "source_gaps_gt_0_5_s": sum(r["gap_exceeds_tracker_reset"] for r in data),
                    "outputs_age_gt_0_35_s": sum(r["output_age_s"] > .35 for r in data),
                    "outputs_age_gt_0_5_s": sum(r["output_age_s"] > .5 for r in data),
                    "unknown_processed_frames": sum(r["prediction"]["label"] == "unknown" for r in data),
                    "unobserved_processed_frames": sum(not r["observed"] for r in data),
                    "epoch_transitions_among_observed": sum(a != b for a, b in zip(epochs, epochs[1:])),
                    "processed_label_counts": dict(Counter(r["prediction"]["label"] for r in data)),
                    "recovery_executed_frames": [r["source_frame_id"] for r in data if r["recovery"] and r["recovery"]["executed"]]}
        equal(summary, expected)
        equal(summary["runtime"], {"torch_threads": 8, "opencv_threads": 1})
        equal(summary["immutable_hashes_match"], True)
        equal(aggregate, summary)
        return {"all_summary_fields_recomputed": True, "runtime": summary["runtime"]}
    audit.check(f"{trial.name}/aggregate_metrics", aggregates)
    recovery_checks(audit, trial.name, data, summary, budget)
    return {"trial": trial.name, "processed_frames": len(ids), "dropped_frames": len(dropped),
            "frames_sha256": digest(trial / "frames.jsonl"),
            "summary_sha256": digest(trial / "summary.json")}


def recovery_checks(audit, name, data, summary, budget):
    """Recompute selection, state continuity and budget semantics from raw rows."""
    def selected_observation():
        for row in data:
            fid = row["source_frame_id"]
            index = row["target_index"]
            selected = row["selected_instance"]
            require((index is None) == (selected is None), f"{fid}: selection/index mismatch")
            if selected is not None:
                require(isinstance(index, int) and 0 <= index < len(row["pose_instances"]),
                        f"{fid}: selected index out of range")
                equal(selected, row["pose_instances"][index], f"{fid}: selected pose")
                require(row["target_id"] is not None, f"{fid}: observed without a target")
            equal(row["observed"], selected is not None)
            eligible = bool(selected is not None and selected.get("action_eligible", True))
            equal(row["action_eligible"], eligible)
            recovery = row["recovery"]
            selected_recovery = bool(recovery["accepted"] and selected is not None
                                     and index == recovery.get("appended_index")
                                     and selected.get("recovered"))
            equal(row["recovery_selected"], selected_recovery)
            if recovery["accepted"]:
                appended = recovery["appended_index"]
                equal(appended, len(row["native_pose_instances"]))
                require(len(row["pose_instances"]) == appended + 1, f"{fid}: malformed recovery append")
                require(row["pose_instances"][appended].get("recovered") is True,
                        f"{fid}: accepted recovery has no recovered flag")
            equal(row["pose_instances"][:len(row["native_pose_instances"])],
                  row["native_pose_instances"], f"{fid}: native pose preservation")
            if selected is not None and not eligible:
                equal(row["prediction"]["label"], "unknown")
                equal(row["prediction"]["phase"], "unknown")
                equal(row["prediction"].get("pose_input_withheld"), True)
            if index is not None:
                evidence = next(item for item in row["admission"] if item["index"] == index)
                equal(evidence["accepted"], True)
        return {"rows": len(data), "actual_selected_recoveries": sum(r["recovery_selected"] for r in data),
                "accepted_alone_not_counted_as_selected": True}
    audit.check(f"{name}/selected_recovery_and_action_eligibility", selected_observation)

    def tracker_state():
        previous = None
        for row in data:
            before, after = row["tracker_before"], row["tracker_after"]
            fid, t = row["source_frame_id"], row["source_time_s"]
            if previous is not None:
                equal(before, previous, f"{fid}: state chaining")
            else:
                equal(before, {"step": 0, "last_t": None, "last_box": None,
                               "confirmed_anchor": None, "last_seen": None,
                               "last_step": None, "target": None, "epoch": 0})
            equal(after["step"], before["step"] + 1)
            equal(after["last_t"], t)
            equal(after["target"], row["target_id"])
            equal(after["epoch"], row["target_epoch"])
            new_epoch = row["tracking_reason"] == "new_confirmed_epoch"
            equal(after["epoch"], before["epoch"] + int(new_epoch))
            if row["observed"]:
                equal(after["last_seen"], t)
                equal(after["last_step"], after["step"])
                equal(after["last_box"], row["selected_instance"]["box"])
            else:
                equal(after["last_seen"], before["last_seen"])
                equal(after["last_step"], before["last_step"])
            if before["target"] is not None and t - before["last_seen"] > .5 + 1e-9:
                require(new_epoch or row["target_id"] is None,
                        f"{fid}: expired target incorrectly retained")
                if not new_epoch:
                    equal(row["tracking_reason"], "expired_requires_new_confirmation")
                    equal(after["last_box"], None)
                    equal(after["confirmed_anchor"], None)
            if before["target"] is not None and row["target_id"] is None:
                require(t - before["last_seen"] > .5 + 1e-9, f"{fid}: unexplained target expiry")
            previous = after
        return {"rows": len(data), "state_chains_consistent": True,
                "identity_truth": "Algorithmic target continuity only; not human identity validation"}
    audit.check(f"{name}/tracker_state_and_actual_expiry", tracker_state)

    def soft_budget():
        exhausted_frames, call_counts = [], Counter()
        rtmo_views = [f"{prefix}_{turn}" for prefix in ("full", "crop") for turn in range(4)]
        for row in data:
            fid = row["source_frame_id"]
            recovery = row["recovery"]
            detail = recovery["budget_audit"]
            calls = detail["calls"]
            equal(detail["budget_s"], budget)
            equal(detail["enforcement"], "soft_non_preemptive")
            elapsed = detail["elapsed_s"]
            require(math.isfinite(elapsed) and elapsed >= 0, f"{fid}: invalid budget elapsed")
            require(elapsed <= row["stages_s"]["recovery"] + TOLERANCE_S,
                    f"{fid}: wrapper time exceeds recovery stage")
            counts = Counter()
            previous_end, seen_refiner = 0.0, False
            for i, call in enumerate(calls):
                kind = call["kind"]
                require(kind in ("rtmo", "rtmpose"), f"{fid}: unknown backend")
                counts[kind] += 1
                equal(call["index"], counts[kind])
                require(previous_end <= call["start_s"] <= call["end_s"] <= elapsed,
                        f"{fid}: invalid/overlapping call clock")
                equal(call["duration_s"], call["end_s"] - call["start_s"])
                if kind == "rtmo":
                    require(not seen_refiner and counts[kind] <= 8, f"{fid}: changed RTMO order")
                    equal(call["view"], rtmo_views[counts[kind] - 1])
                    equal(call["rotation"], (counts[kind] - 1) % 4)
                    equal(call["expansion"], None)
                else:
                    seen_refiner = True
                    require(counts[kind] <= 8, f"{fid}: extra RTMPose call")
                    equal(call["rotation"], (counts[kind] - 1) // 2)
                    equal(call["expansion"], (1.0, 1.15)[(counts[kind] - 1) % 2])
                if budget is not None:
                    require(call["start_s"] < budget, f"{fid}: new call started after budget")
                    if call["end_s"] >= budget:
                        require(i == len(calls) - 1 and detail["exhausted"],
                                f"{fid}: did not stop after non-preemptible overrun")
                        equal(detail["exhaustion_stage"], "after_call")
                previous_end = call["end_s"]
            call_counts.update(counts)
            equal(detail["rtmo_calls"], counts["rtmo"])
            equal(detail["rtmpose_calls"], counts["rtmpose"])
            equal(detail["total_calls"], len(calls))
            equal(detail["max_call_s"], max((c["duration_s"] for c in calls), default=0.0))
            equal(detail["overrun_s"], max(0.0, elapsed - budget) if budget is not None else 0.0)
            equal(recovery["executed"], bool(calls))
            equal(recovery["inference_passes"], counts["rtmo"])
            equal(recovery["refinement"]["inference_passes"], counts["rtmpose"])
            equal(detail["discarded_incomplete_recovery"], detail["exhausted"])
            if detail["exhausted"]:
                require(budget is not None, f"{fid}: unlimited baseline exhausted")
                exhausted_frames.append(fid)
                require(detail["exhaustion_elapsed_s"] >= budget, f"{fid}: premature exhaustion")
                require(detail["exhaustion_elapsed_s"] <= elapsed, f"{fid}: exhaustion after return")
                stage = detail["exhaustion_stage"]
                require(stage in ("before_call", "after_call", "after_augment"), f"{fid}: unknown stop boundary")
                if stage == "after_call":
                    require(bool(calls), f"{fid}: after_call without a call")
                    equal(detail["exhaustion_elapsed_s"], calls[-1]["end_s"])
                elif stage == "after_augment":
                    require(bool(calls) and all(c["end_s"] < budget for c in calls),
                            f"{fid}: invalid postprocessing timeout")
                    equal(detail["blocked_call"], None)
                else:
                    blocked = detail["blocked_call"]
                    require(isinstance(blocked, dict), f"{fid}: missing blocked call")
                    equal(blocked["index"], counts[blocked["kind"]] + 1)
                    require(all(c["end_s"] < budget for c in calls), f"{fid}: before-call stop after earlier timeout")
                equal(recovery["reason"], "recovery_budget_exhausted")
                equal(recovery["accepted"], False)
                equal(recovery["refinement"]["accepted"], False)
                equal(recovery["candidates"], [])
                equal(recovery["incomplete_recovery_discarded"], True)
                equal(row["pose_instances"], row["native_pose_instances"])
                equal(row["recovery_selected"], False)
            else:
                equal(detail["exhaustion_stage"], None)
                equal(detail["exhaustion_elapsed_s"], None)
                equal(detail["blocked_call"], None)
                require(recovery.get("reason") != "recovery_budget_exhausted", f"{fid}: inconsistent timeout reason")
        return {"exhausted_frames": exhausted_frames, "backend_call_counts": dict(call_counts),
                "budget_s": budget, "hard_deadline_claimed": False,
                "unfinished_candidates_submitted": 0}
    audit.check(f"{name}/soft_budget_call_boundaries_and_discard", soft_budget)

    def recovery_aggregates():
        attempts = [r for r in data if r["recovery"]["executed"]]
        selected = [r for r in attempts if r["recovery_selected"]]
        following = []
        for i, row in enumerate(data):
            if not row["recovery"]["executed"]:
                continue
            nxt = data[i + 1] if i + 1 < len(data) else None
            following.append({"frame": row["source_frame_id"], "accepted": row["recovery"]["accepted"],
                              "selected": row["recovery_selected"], "action_eligible": row["action_eligible"],
                              "next_frame": nxt["source_frame_id"] if nxt else None,
                              "next_source_gap_s": nxt["source_gap_s"] if nxt else None,
                              "next_observed": nxt["observed"] if nxt else None,
                              "next_target_id": nxt["target_id"] if nxt else None,
                              "next_tracking_reason": nxt["tracking_reason"] if nxt else None})
        expected = {"attempt_frames": [r["source_frame_id"] for r in attempts],
                    "accepted_frames": [r["source_frame_id"] for r in attempts if r["recovery"]["accepted"]],
                    "selected_frames": [r["source_frame_id"] for r in selected],
                    "selected_action_eligible_frames": [r["source_frame_id"] for r in selected if r["action_eligible"]],
                    "selected_presence_only_frames": [r["source_frame_id"] for r in selected if not r["action_eligible"]],
                    "budget_aborted_frames": [r["source_frame_id"] for r in attempts if r["recovery"]["reason"] == "recovery_budget_exhausted"],
                    "attempt_service": distribution([r["stages_s"]["recovery"] for r in attempts]) if attempts else None,
                    "following_observations": following}
        equal(summary["recovery_statistics"], expected)
        gaps = [data[0]["output_wall_s"]] + [b["output_wall_s"] - a["output_wall_s"] for a, b in zip(data, data[1:])]
        equal(summary["output_gaps"], distribution(gaps))
        tracking = {"observed_frames": [r["source_frame_id"] for r in data if r["observed"]],
                    "target_id_null_frames": [r["source_frame_id"] for r in data if r["target_id"] is None],
                    "target_expired_frames": [r["source_frame_id"] for r in data if r["tracker_before"]["target"] is not None and r["target_id"] is None],
                    "new_epoch_frames": [r["source_frame_id"] for r in data if r["tracking_reason"] == "new_confirmed_epoch"],
                    "last_frame_observed": data[-1]["observed"], "last_frame_target_id": data[-1]["target_id"]}
        equal(summary["tracking_statistics"], tracking)
        for region, lo, hi in (("official_fall", 83, 112), ("post_contact_unscored", 113, 137), ("tail_provisional", 138, 160)):
            region_rows = [r for r in data if lo <= r["source_frame_id"] <= hi]
            equal(summary["regions"][region], {"range": [lo, hi], "reference_samples": hi - lo + 1,
                  "processed": len(region_rows), "dropped": hi - lo + 1 - len(region_rows),
                  "observed": sum(r["observed"] for r in region_rows),
                  "unknown": sum(r["prediction"]["label"] == "unknown" for r in region_rows),
                  "target_id_null": sum(r["target_id"] is None for r in region_rows),
                  "recovery_selected": sum(r["recovery_selected"] for r in region_rows)})
        return {"output_gap_max_including_initial_s": max(gaps), "recovery": expected,
                "tracking": tracking, "region_denominators_unchanged": True}
    audit.check(f"{name}/longest_output_gap_recovery_followup_and_region_totals", recovery_aggregates)

    def native_reappearance():
        evidence = []
        for i, row in enumerate(data):
            if not row["recovery"]["executed"]:
                continue
            observed_after = next((r for r in data[i + 1:] if r["observed"]), None)
            before = row["tracker_before"]
            gap = (observed_after["source_time_s"] - before["last_seen"]
                   if observed_after is not None and before["last_seen"] is not None else None)
            same_target = bool(observed_after is not None and before["target"] is not None
                               and observed_after["target_id"] == before["target"]
                               and observed_after["target_epoch"] == before["epoch"])
            native = bool(observed_after is not None and not observed_after["selected_instance"].get("recovered"))
            if observed_after is not None and native:
                require(observed_after["target_index"] < len(observed_after["native_pose_instances"]),
                        "Native reappearance must select a current native observation")
            evidence.append({"attempt_frame": row["source_frame_id"],
                             "budget_aborted": row["recovery"]["budget_audit"]["exhausted"],
                             "attempt_selected_recovery": row["recovery_selected"],
                             "first_observed_after_frame": observed_after["source_frame_id"] if observed_after else None,
                             "first_observed_after_uses_native_pose": native,
                             "source_gap_from_last_seen_before_attempt_s": gap,
                             "same_logical_target": same_target,
                             "within_unchanged_half_second_continuity_window": gap is not None and gap <= .5 + 1e-9})
        return {"reappearance_after_attempts": evidence,
                "not_recovery_success": "Native reappearance is separate from an accepted and selected recovery pose"}
    audit.check(f"{name}/native_reappearance_after_recovery_attempt", native_reappearance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "results.json").exists():
        raise SystemExit("Completed results.json is absent; no partial validation file was written.")
    protocol = read(output / "protocol.json")
    results = read(output / "results.json")
    audit = Audit()
    annotation_path = output / "annotations.reviewed.snapshot.json"
    annotation = read(annotation_path)
    entry = next(e for e in annotation["entries"] if e["clip_id"] == CLIP)
    frozen = read(BASE / "evaluation-protocol.json")
    expected_names = [f"{item['strategy']}-r{item['repeat']}" for item in protocol["order"]]

    def completeness():
        expected_order = [{"strategy": strategy, "repeat": repeat}
                          for repeat in (1, 2, 3)
                          for strategy in (("baseline", "budget200ms") if repeat % 2
                                           else ("budget200ms", "baseline"))]
        equal(protocol["order"], expected_order)
        require(len(expected_names) == 6, "Protocol does not contain six paired trials")
        equal(protocol["soft_budget_s"], .200)
        equal(protocol["repeats_per_cell"], 3)
        equal(protocol["representative_video_repeat"], 2)
        require(results["complete"] is True, "Results not marked complete")
        equal([s["trial"] for s in results["trials"]], expected_names)
        equal(sorted(p.name for p in (output / "trials").iterdir() if p.is_dir()), sorted(expected_names))
        equal(protocol["training"], False)
        equal(protocol["source_frames"], 160)
        equal(protocol["contact_interval_s"], [3.336, 3.537])
        equal(protocol["model_sha256"], frozen["model_sha256"])
        return {"trials": 6, "group": "C", "policy": "latest", "repeats_per_cell": 3}
    audit.check("complete_six_trial_design", completeness)

    def immutable():
        mismatches = [{"file": name, "expected": expected, "actual": digest(name)}
                      for name, expected in protocol["frozen_hashes"].items()
                      if digest(name) != expected]
        require(not mismatches, f"Frozen hash mismatches: {mismatches}")
        equal(results["all_frozen_hashes_unchanged"], True)
        live_annotation = BASE / "annotations/annotations.reviewed.json"
        equal(digest(annotation_path), protocol["frozen_hashes"][str(live_annotation)])
        return {"verified_file_count": len(protocol["frozen_hashes"]),
                "annotation_snapshot_sha256": digest(annotation_path), "mismatches": []}
    audit.check("frozen_input_model_code_hashes", immutable)

    def historical_separation():
        equal(results["historical_benchmark"], "../latency-optimization-20261021/results.json")
        historic = BASE / "latency-optimization-20261021"
        old_rows = []
        for repeat in (1, 2, 3):
            path = historic / "trials" / f"C-latest-r{repeat}" / "frames.jsonl"
            equal(digest(path), protocol["frozen_hashes"][str(path)])
            data = rows(path)
            gaps = [data[0]["output_wall_s"]] + [b["output_wall_s"] - a["output_wall_s"] for a, b in zip(data, data[1:])]
            old_rows.append({"trial": path.parent.name, "role": "historical_diagnosis_only",
                             "max_output_gap_s": max(gaps),
                             "max_output_age_s": max(r["output_age_s"] for r in data),
                             "attempt_frames": [r["source_frame_id"] for r in data if r["recovery"]["executed"]]})
        created = datetime.fromisoformat(protocol["created_utc"])
        for trial in results["trials"]:
            require(datetime.fromisoformat(trial["started_utc"]) >= created,
                    "A primary trial started before the new protocol was frozen")
            require(trial["trial"] in expected_names, "Historical trial was pooled into the primary comparison")
        return {"fresh_primary_trials": expected_names, "historical_trials": old_rows,
                "historical_trials_pooled_into_primary_comparison": False}
    audit.check("fresh_primary_comparison_historical_diagnosis_separation", historical_separation)

    cached = {mode: rows(frozen["source_logs"][f"{mode}/{CLIP}"]["file"]) for mode in "C"}
    evidence = []
    by_trial = {s["trial"]: s for s in results["trials"]}
    for name in expected_names:
        try:
            evidence.append(validate_trial(audit, output / "trials" / name, entry,
                                           cached["C"], by_trial[name]))
        except Exception as error:
            audit.check(f"{name}/readability", lambda error=error: (_ for _ in ()).throw(error))
    failures = [check for check in audit.checks if not check["passed"]]
    payload = {"schema": "fallpre.v7.recovery_budget.independent_validation.v1",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "validator_file": str(Path(__file__).resolve()), "validator_sha256": digest(__file__),
               "protocol_sha256": digest(output / "protocol.json"),
               "results_sha256": digest(output / "results.json"),
               "passed": not failures, "checks_passed": len(audit.checks) - len(failures),
               "checks_total": len(audit.checks), "inference_executed": False,
               "independence": "Stdlib recomputation; no runner/helper/model imports or inference",
               "limitations": ["Source-gap flags are reset threshold conditions, not counts of every actual reset.",
                               "Single-slot occupancy is the logical replay policy, not measured camera capture.",
                               "Log validation cannot establish sensor-to-alert performance or generalization.",
                               "A 0.200s soft budget cannot preempt a backend call already running.",
                               "Target/epoch continuity is algorithmic state, not human-verified identity truth."],
               "trials": evidence, "checks": audit.checks}
    (output / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    suite = ET.Element("testsuite", name="v7_recovery_budget_independent_audit",
                       tests=str(len(audit.checks)), failures=str(len(failures)), errors="0")
    for check in audit.checks:
        case = ET.SubElement(suite, "testcase", name=check["name"], time=f"{check['elapsed_s']:.6f}")
        if not check["passed"]:
            ET.SubElement(case, "failure", message=check["error"]).text = check["error"]
    ET.ElementTree(suite).write(output / "tests.xml", encoding="utf-8", xml_declaration=True)
    print(json.dumps({"passed": not failures, "checks_passed": len(audit.checks) - len(failures),
                      "checks_total": len(audit.checks), "failures": failures,
                      "validation": str(output / "validation.json")}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
