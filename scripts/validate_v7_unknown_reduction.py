"""Independent, stdlib-only audit of fixed-V7 unknown-reduction logs.

This reads recorded results; it never imports the runner, scheduling helper,
model code, torch or a neural runtime. Source-gap flags are reset conditions,
not an observed count of all actual model/tracker resets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
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
DEFAULT = BASE / "unknown-reduction-20261021"
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
    budget = .200
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
        cost_aware = name.startswith("costaware200ms-")
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
            completed_same_kind = {"rtmo": [], "rtmpose": []}
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
                    if cost_aware and completed_same_kind[kind]:
                        require(call["start_s"] + max(completed_same_kind[kind]) < budget,
                                f"{fid}: admitted {kind} call despite predicted overrun")
                    if call["end_s"] >= budget:
                        require(i == len(calls) - 1 and detail["exhausted"],
                                f"{fid}: did not stop after non-preemptible overrun")
                        equal(detail["exhaustion_stage"], "after_call")
                completed_same_kind[kind].append(call["duration_s"])
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
                require(budget is not None, f"{fid}: exhausted without budget")
                exhausted_frames.append(fid)
                require(detail["exhaustion_elapsed_s"] <= elapsed, f"{fid}: exhaustion after return")
                stage = detail["exhaustion_stage"]
                require(stage in ("before_call", "after_call", "after_augment", "predicted_call_overrun"),
                        f"{fid}: unknown stop boundary")
                if stage == "predicted_call_overrun":
                    require(cost_aware, f"{fid}: estimate gate appeared in elapsed-only baseline")
                    require(detail["exhaustion_elapsed_s"] < budget,
                            f"{fid}: actual elapsed check did not take precedence")
                    blocked = detail["blocked_call"]
                    require(isinstance(blocked, dict), f"{fid}: no blocked call")
                    kind = blocked["kind"]
                    require(kind in completed_same_kind and completed_same_kind[kind],
                            f"{fid}: estimator has no same-kind current-attempt evidence")
                    estimate = max(completed_same_kind[kind])
                    remaining = budget - detail["exhaustion_elapsed_s"]
                    require(estimate >= remaining, f"{fid}: rejected a predicted fitting call")
                    equal(blocked["index"], counts[kind] + 1)
                    if kind == "rtmo":
                        equal(blocked["view"], rtmo_views[counts[kind]])
                        equal(blocked["rotation"], counts[kind] % 4)
                    else:
                        equal(blocked["rotation"], counts[kind] // 2)
                        equal(blocked["expansion"], (1.0, 1.15)[counts[kind] % 2])
                    equal(blocked["estimated_duration_s"], estimate)
                    equal(blocked["remaining_budget_s"], remaining)
                    equal(detail["estimated_next_call_s"], estimate)
                    equal(detail["remaining_budget_s"], remaining)
                    equal(detail["exhaustion_cause"], "predicted_call_overrun")
                    require(all(c["end_s"] < budget for c in calls),
                            f"{fid}: prediction used after actual call timeout")
                elif stage == "after_call":
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
                if stage != "predicted_call_overrun":
                    require(detail["exhaustion_elapsed_s"] >= budget, f"{fid}: premature actual exhaustion")
                    if cost_aware:
                        equal(detail["exhaustion_cause"], "elapsed_budget_exhausted")
                        equal(detail["estimated_next_call_s"], None)
                        equal(detail["remaining_budget_s"], None)
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
                if cost_aware:
                    equal(detail["exhaustion_cause"], None)
                    equal(detail["estimated_next_call_s"], None)
                    equal(detail["remaining_budget_s"], None)
        return {"exhausted_frames": exhausted_frames, "backend_call_counts": dict(call_counts),
                "budget_s": budget, "hard_deadline_claimed": False,
                "unfinished_candidates_submitted": 0,
                "cost_admission": "same-kind completed calls in each row only" if cost_aware else "disabled",
                "per_attempt_estimator_recomputed": cost_aware}
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


def validate_action_audit(audit, output, results):
    """Check the saved frozen-V7 replay's accounting without importing that replay."""
    path = output / "action-audit.json"
    if not path.exists():
        audit.check("action_audit_present", lambda: require(False, "action-audit.json is absent"))
        return
    replay = read(path)
    by_name = {trial["trial"]: trial for trial in replay["trials"]}
    summaries = {trial["trial"]: trial for trial in results["trials"]}

    def provenance():
        equal(sorted(by_name), sorted(summaries))
        require(replay["all_predictions_exactly_equal"], "Frozen V7 replay differs from primary logs")
        require(replay["history_reset_predicates_verified"], "Replay history reset mismatch")
        require(replay["observed_v3_v4_reset_calls_match_predicates"],
                "Observed V3/V4 reset calls differ from predicates")
        require(replay["frozen_hashes_unchanged"], "Replay inputs changed")
        for name, value in replay["frozen_hashes"].items():
            equal(digest(name), value, f"action replay provenance: {name}")
        equal(replay["processed_prediction_count"], sum(s["processed_frames"] for s in results["trials"]))
        return {"saved_prediction_replay_matches": replay["processed_prediction_count"],
                "validator_performed_neural_inference": False}
    audit.check("frozen_action_replay_provenance", provenance)

    for name in summaries:
        def histories(name=name):
            logged = rows(output / "trials" / name / "frames.jsonl")
            trial = by_name[name]
            equal(trial["input_sha256"], digest(output / "trials" / name / "frames.jsonl"))
            equal(trial["processed_frames"], [r["source_frame_id"] for r in logged])
            require(len(trial["frames"]) == len(logged), "Action audit omitted a processed row")
            equal(trial["prediction_mismatches"], [])
            equal(trial["history_reset_mismatches"], [])
            equal(trial["full_prediction_matches"], len(logged))
            snapshots = {}
            history_rows = {"v3": [], "v4": []}
            last_valid, initial_valid, ever_known, origin = None, False, False, None
            categories = Counter()
            reset_totals = {"v3": Counter(), "v4": Counter()}
            for row, frame in zip(logged, trial["frames"]):
                fid, t = row["source_frame_id"], row["source_time_s"]
                valid = row["action_eligible"]
                equal(frame, {"source_frame_id": fid, "source_time_s": t,
                              "observed": row["observed"], "action_observed": valid,
                              "target_id": row["target_id"], "target_epoch": row["target_epoch"],
                              "label": row["prediction"]["label"],
                              "reason": row["prediction"].get("reason"),
                              "previous_valid": last_valid,
                              "gap_since_last_valid_s": None if last_valid is None else t-last_valid["time_s"],
                              "source_gap_s": row["source_gap_s"],
                              "dropped_before_this_frame": row["dropped_before_this_frame"],
                              "prediction_exactly_equal": True})
                for version, window in (("v3", 2.05), ("v4", 4.05)):
                    detail = frame["histories"][version]
                    before, after = detail["before"], detail["after"]
                    if version in snapshots:
                        equal(before, snapshots[version], f"{fid}/{version}: action state chain")
                    else:
                        equal(before, {"epoch": None, "last_t": None, "last_seen": None,
                                       "history_frames": [], "history_times": [],
                                       "valid_history_frames": [], "valid_history_times": []})
                    reasons = []
                    if row["target_epoch"] != before["epoch"]:
                        reasons.append("target_epoch_changed")
                    previous = (before["history_times"][-1] if before["history_times"] else None) if version == "v3" else before["last_t"]
                    if previous is not None and t-previous > .35:
                        reasons.append("adjacent_processed_gap_gt_0_35")
                    if ((version == "v4" or not reasons) and valid
                            and before["last_seen"] is not None and t-before["last_seen"] > .35):
                        reasons.append("last_valid_observation_gap_gt_0_35")
                    reset = bool(reasons)
                    equal(detail["reasons"], reasons)
                    equal(detail["clear_called"], reset)
                    equal(detail["audit_method"], "wrapped_reset_method")
                    observed_calls = detail["observed_reset_calls"]
                    equal(len(observed_calls), int(reset))
                    for actual_call in observed_calls:
                        equal(actual_call["caller_file"], str(ROOT / f"src/fallpre/weekly/action_{version}.py"))
                        equal(actual_call["caller_function"], "update")
                        code_line = Path(actual_call["caller_file"]).read_text().splitlines()[actual_call["caller_line"]-1]
                        require("self.reset(" in code_line, f"{fid}: reset observer points outside a reset call")
                        equal(actual_call["inferred_reasons"], reasons)
                        equal(actual_call["before"], before)
                        equal(actual_call["after"], dict(before, last_seen=None,
                                                        history_frames=[], history_times=[],
                                                        valid_history_frames=[], valid_history_times=[]))
                    equal(detail["cleared_nonempty_history"], reset and bool(before["history_frames"]))
                    equal(detail["cleared_valid_history"], reset and bool(before["valid_history_frames"]))
                    if reset:
                        history_rows[version] = []
                        reset_totals[version]["all_clear_calls"] += 1
                        reset_totals[version]["nonempty_history_clear_calls"] += bool(before["history_frames"])
                        reset_totals[version]["valid_history_clear_calls"] += bool(before["valid_history_frames"])
                        reset_totals[version].update("reason:" + reason for reason in reasons)
                    history_rows[version].append((fid, t, valid))
                    history_rows[version] = [r for r in history_rows[version] if r[1] >= t-window]
                    kept = history_rows[version]
                    last_seen = t if valid else None if reset else before["last_seen"]
                    equal(after, {"epoch": row["target_epoch"],
                                  "last_t": None if version == "v3" else t,
                                  "last_seen": last_seen,
                                  "history_frames": [r[0] for r in kept],
                                  "history_times": [r[1] for r in kept],
                                  "valid_history_frames": [r[0] for r in kept if r[2]],
                                  "valid_history_times": [r[1] for r in kept if r[2]]})
                    snapshots[version] = after
                v3_reasons = frame["histories"]["v3"]["reasons"]
                if v3_reasons:
                    origin = {"source_frame_id": fid, "time_s": t, "reasons": v3_reasons,
                              "previous_last_valid": last_valid,
                              "previous_valid_gap_s": None if last_valid is None else t-last_valid["time_s"],
                              "initial": not initial_valid}
                equal(frame["v3_warmup_origin"], origin)
                category = None
                pred = row["prediction"]
                abandoned = row["recovery"]["budget_audit"]["exhausted"]
                equal(frame["recovery_abandoned"], abandoned)
                if pred["label"] == "unknown":
                    reason = pred.get("reason")
                    if row["selected_instance"] is not None and not valid:
                        category = "partial_pose_withheld"
                    elif reason == "missing_observation":
                        category = ("recovery_abandoned_missing_observation" if abandoned else
                                    "initial_target_unconfirmed" if not initial_valid else
                                    "target_unavailable_after_expiry_or_loss" if row["target_id"] is None else
                                    "other_missing_observation")
                    elif reason == "warming_up":
                        category = ("initial_warmup" if origin and origin["initial"] and not ever_known else
                                    "post_gap_reset_warmup" if origin and any("gap" in r for r in origin["reasons"]) else
                                    "post_identity_reset_warmup" if origin and "target_epoch_changed" in origin["reasons"] else
                                    "other_warmup")
                        good = [r for r in history_rows["v3"] if r[2] and t-r[1] <= 2.0]
                        require(len(good) < 3 or t-good[0][1] < .16,
                                f"{fid}: warming_up does not match frozen V3 history condition")
                    elif reason == "low_classifier_confidence":
                        category = "classifier_uncertainty"
                    else:
                        category = "other_unknown"
                    categories[category] += 1
                equal(frame["unknown_category"], category)
                if valid:
                    initial_valid = True
                    last_valid = {"source_frame_id": fid, "time_s": t}
                ever_known = ever_known or pred["label"] != "unknown"
            equal(trial["unknown_categories"], dict(categories))
            equal(trial["unknown_count"], sum(categories.values()))
            equal(trial["unknown_count"], summaries[name]["unknown_processed_frames"])
            equal([f["source_frame_id"] for f in trial["unknown_frames"]],
                  [r["source_frame_id"] for r in logged if r["prediction"]["label"] == "unknown"])
            for version in ("v3", "v4"):
                equal(trial["reset_counts"][version], dict(reset_totals[version]))
            return {"processed": len(logged), "unknown": dict(categories),
                    "v3_v4_history_states_recomputed": True,
                    "unknown_categories_are_accounting_not_randomized_causal_effects": True}
        audit.check(f"{name}/unknown_reason_partition_and_actual_history_state", histories)

    def aggregate():
        grouped = {}
        for name, trial in by_name.items():
            strategy = trial["strategy"]
            bucket = grouped.setdefault(strategy, {"trials": 0, "processed": 0, "unknown": 0, "categories": Counter()})
            bucket["trials"] += 1
            bucket["processed"] += summaries[name]["processed_frames"]
            bucket["unknown"] += summaries[name]["unknown_processed_frames"]
            bucket["categories"].update(trial["unknown_categories"])
        for bucket in grouped.values():
            bucket["categories"] = dict(bucket["categories"])
        equal(replay["aggregate_by_strategy"], grouped)
        return grouped
    audit.check("action_replay_aggregate_unknown_accounting", aggregate)


def validate_report(audit, output, results, entry):
    required = ("comparison.json", "comparison.csv", "all-source-frames.json", "all-source-frames.csv")
    if not all((output / name).exists() for name in required):
        audit.check("report_present", lambda: require(False, "Comparison or complete source inventory absent"))
        return
    comparison = read(output / "comparison.json")
    action = {t["trial"]: t for t in read(output / "action-audit.json")["trials"]}
    metrics = {m["trial"]: m for m in comparison["trial_metrics"]}
    data = {s["trial"]: rows(output / "trials" / s["trial"] / "frames.jsonl") for s in results["trials"]}

    def trial_metrics():
        equal(sorted(metrics), sorted(data))
        equal(comparison["source_reference_frames"], 160)
        equal(comparison["contact_interval_s"], entry["event_landmarks"]["contact"]["time_interval_s"])
        for s in results["trials"]:
            name = s["trial"]
            rs, m, a = data[name], metrics[name], action[name]
            budgets = [r["recovery"]["budget_audit"] for r in rs]
            tail = [r for r in rs if r["source_frame_id"] >= 113]
            gaps = [rs[0]["output_wall_s"]] + [b["output_wall_s"]-a["output_wall_s"] for a,b in zip(rs,rs[1:])]
            recovered = [r for r in rs if r["recovery_selected"]]
            eligible = [r for r in recovered if r["action_eligible"]]
            first = s["first_falling"]
            max_recovery = max(rs, key=lambda r:r["stages_s"]["recovery"])
            expected = {
                "trial": name, "variant": s["strategy"], "repeat": s["repeat"],
                "source_reference_frames": 160, "processed_frames": len(rs), "dropped_frames": 160-len(rs),
                "recovery_max_s": max_recovery["stages_s"]["recovery"],
                "recovery_max_source_frame": max_recovery["source_frame_id"],
                "recovery_total_s": sum(r["stages_s"]["recovery"] for r in rs),
                "recovery_attempt_total_s": sum(r["stages_s"]["recovery"] for r in rs if r["recovery"]["executed"]),
                "recovery_backend_calls_total_s": sum(c["duration_s"] for b in budgets for c in b["calls"]),
                "service_max_s": s["service"]["max_s"], "output_age_max_s": s["output_age"]["max_s"],
                "output_gap_max_s": max(gaps), "output_gap_max_ending_frame": rs[gaps.index(max(gaps))]["source_frame_id"],
                "first_output_wall_s": first["output_wall_s"], "first_source_frame": first["frame"],
                "contact_margin_low_s": first["contact_margin_s"][0], "contact_margin_high_s": first["contact_margin_s"][1],
                "contact_result": s["contact_result"], "event_hit": s["official_fall_event"]["hit"],
                "fall_reference_processed": s["official_fall_event"]["processed_reference_frames"],
                "fall_reference_falling_outputs": len(s["official_fall_event"]["falling_output_frames"]),
                "unknown_outputs": s["unknown_processed_frames"], "unobserved_outputs": s["unobserved_processed_frames"],
                "target_none_outputs": sum(r["target_id"] is None for r in rs),
                "expired_requires_new_confirmation_outputs": sum(r["tracking_reason"] == "expired_requires_new_confirmation" for r in rs),
                "tail_reference_frames": 48, "tail_processed": len(tail), "tail_observed": sum(r["observed"] for r in tail),
                "tail_target_none": sum(r["target_id"] is None for r in tail), "tail_dropped": 48-len(tail),
                "recovery_attempts": sum(r["recovery"]["executed"] for r in rs),
                "recovery_accepted": sum(r["recovery"]["accepted"] for r in rs),
                "recovery_selected": len(recovered), "recovery_action_eligible": len(eligible),
                "budget_aborted_attempts": sum(b["exhausted"] for b in budgets),
                "budget_max_overrun_s": max(b["overrun_s"] for b in budgets),
                "recovery_rtmo_calls": sum(b["rtmo_calls"] for b in budgets),
                "recovery_rtmpose_calls": sum(b["rtmpose_calls"] for b in budgets),
                "recovery_backend_call_max_s": max(b["max_call_s"] for b in budgets),
                "last_frame_observed": int(rs[-1]["observed"]), "last_frame_target_id": rs[-1]["target_id"],
                "max_pending_slots": s["max_pending_slots"], "max_source_gap_s": s["max_source_gap_s"],
                "source_gap_gt_0_35": s["source_gaps_gt_0_35_s"], "source_gap_gt_0_5": s["source_gaps_gt_0_5_s"],
                "provisional_false_falling": len(s["provisional_labeled_false_falling_frames"]),
                "unscored_falling": len(s["unscored_falling_frames"]),
                "processed_label_counts": s["processed_label_counts"], "regions": s["regions"],
                "recovery_attempt_source_ids": s["recovery_statistics"]["attempt_frames"],
                "recovery_accepted_source_ids": s["recovery_statistics"]["accepted_frames"],
                "recovery_selected_source_ids": [r["source_frame_id"] for r in recovered],
                "recovery_action_eligible_source_ids": [r["source_frame_id"] for r in eligible],
                "dropped_source_frame_ids": s["dropped_source_frame_ids"],
                "target_none_source_ids": s["tracking_statistics"]["target_id_null_frames"],
                "tail_observed_source_ids": [r["source_frame_id"] for r in tail if r["observed"]],
                "unknown_fraction_source": s["unknown_processed_frames"]/160,
                "unknown_fraction_processed": s["unknown_processed_frames"]/len(rs),
                "max_valid_action_gap_s": max((f["gap_since_last_valid_s"] for f in a["frames"]
                                               if f["action_observed"] and f["previous_valid"] is not None), default=0.),
                "native_observed": sum(r["observed"] and not r["recovery_selected"] for r in rs),
                "target_expiry_events": len(s["tracking_statistics"]["target_expired_frames"]),
                "unknown_source_ids": [r["source_frame_id"] for r in rs if r["prediction"]["label"] == "unknown"],
                "target_expiry_source_ids": s["tracking_statistics"]["target_expired_frames"],
                "predicted_call_rejections": sum(b["exhaustion_stage"] == "predicted_call_overrun" for b in budgets),
                "elapsed_budget_aborts": sum(b["exhausted"] and b["exhaustion_stage"] != "predicted_call_overrun" for b in budgets),
                "unknown_cause_counts": a["unknown_categories"], "action_audit": a,
            }
            causes = a["unknown_categories"]
            simple = {"unknown_recovery_abandoned":"recovery_abandoned_missing_observation",
                      "unknown_initial_target":"initial_target_unconfirmed",
                      "unknown_target_unavailable":"target_unavailable_after_expiry_or_loss",
                      "unknown_startup":"initial_warmup", "unknown_postgap":"post_gap_reset_warmup",
                      "unknown_low_confidence":"classifier_uncertainty"}
            expected.update({key:causes.get(category,0) for key,category in simple.items()})
            missing_keys = {"initial_target_unconfirmed", "recovery_abandoned_missing_observation",
                            "target_unavailable_after_expiry_or_loss", "other_missing_observation", "partial_pose_withheld"}
            expected["unknown_missing"] = sum(causes.get(key,0) for key in missing_keys)
            expected["unknown_other"] = sum(causes.values()) - sum(expected[k] for k in (
                "unknown_missing", "unknown_startup", "unknown_postgap", "unknown_low_confidence"))
            for version in ("v3", "v4"):
                resets = a["reset_counts"][version]
                expected[f"actual_{version}_resets"] = resets["all_clear_calls"]
                expected[f"{version}_valid_history_resets"] = resets["valid_history_clear_calls"]
                expected[f"{version}_gap_resets"] = resets.get("reason:last_valid_observation_gap_gt_0_35",0)
            equal(m, expected, name)
            equal(set(m), set(expected), f"{name}: every report metric covered")
        return {"trial_rows": 6, "all_report_trial_fields_recomputed": True}
    audit.check("report_trial_metrics_match_validated_logs_and_action_audit", trial_metrics)

    def groups_pairs_csv():
        for cell in comparison["cells"]:
            ms = [m for m in comparison["trial_metrics"] if m["variant"] == cell["variant"]]
            expected = {}
            for key,value in ms[0].items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    vals = [m[key] for m in ms]
                    expected[key] = {"n":len(vals), "median":statistics.median(vals), "min":min(vals), "max":max(vals)}
            equal(cell["statistics"], expected)
            equal(cell["event_hits"], sum(m["event_hit"] for m in ms))
            equal(cell["contact_results"], dict(Counter(m["contact_result"] for m in ms)))
            equal(cell["trials"], [m["trial"] for m in ms])
        for pair in comparison["pairs"]:
            repeat = pair["repeat"]
            a,b = [metrics[f"{variant}-r{repeat}"] for variant in ("budget200ms","costaware200ms")]
            ar,br = [{r["source_frame_id"]:r for r in data[m["trial"]]} for m in (a,b)]
            common = sorted(ar.keys() & br.keys())
            equal(pair, {"repeat": repeat, "unknown_reduction":a["unknown_outputs"]-b["unknown_outputs"],
                         "unknown_source_fraction_reduction":a["unknown_fraction_source"]-b["unknown_fraction_source"],
                         "unknown_processed_fraction_reduction":a["unknown_fraction_processed"]-b["unknown_fraction_processed"],
                         "processed_increase":b["processed_frames"]-a["processed_frames"],
                         "common_source_ids":common,
                         "common_label_change_ids":[i for i in common if ar[i]["prediction"]["label"]!=br[i]["prediction"]["label"]],
                         "common_phase_change_ids":[i for i in common if ar[i]["prediction"]["phase"]!=br[i]["prediction"]["phase"]],
                         "common_observed_change_ids":[i for i in common if ar[i]["observed"]!=br[i]["observed"]],
                         "baseline_only_source_ids":sorted(ar.keys()-br.keys()),
                         "costaware_only_source_ids":sorted(br.keys()-ar.keys())})
        with (output/"comparison.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        equal(len(csv_rows),6)
        for row in csv_rows:
            for key,value in row.items():
                equal(value,str(metrics[row["trial"]][key]))
        return {"aggregate_cells":len(comparison["cells"]),"pairs":len(comparison["pairs"]),"csv_rows":6}
    audit.check("report_aggregate_pairs_and_csv_consistency", groups_pairs_csv)

    def complete_inventory():
        inventory = read(output/"all-source-frames.json")["rows"]
        equal(len(inventory),960)
        seen = set()
        by_source = {name:{r["source_frame_id"]:r for r in rs} for name,rs in data.items()}
        audit_by_source = {name:{r["source_frame_id"]:r for r in a["frames"]} for name,a in action.items()}
        for item in inventory:
            name,fid = item["trial"],item["source_frame_id"]
            require((name,fid) not in seen,"Duplicate source inventory key")
            seen.add((name,fid))
            require(name in data and 1<=fid<=160,"Invalid source inventory key")
            equal(item["source_time_s"],entry["clock"]["source_times_s"][fid-1])
            equal(item["strategy"],metrics[name]["variant"])
            equal(item["repeat"],metrics[name]["repeat"])
            row = by_source[name].get(fid)
            equal(item["processed"],row is not None)
            equal(item["dropped"],row is None)
            if row is None:
                for key in ("prediction","phase","unknown_cause","observed","target_id","tracking_reason",
                            "recovery_attempt","recovery_selected","recovery_exhaustion_stage","recovery_s",
                            "output_wall_s","output_age_s","action_audit"):
                    equal(item[key],None,f"{name}/{fid}: skipped {key}")
            else:
                equal(item,{"prediction":row["prediction"]["label"],"phase":row["prediction"]["phase"],
                            "unknown_cause":audit_by_source[name][fid]["unknown_category"],
                            "observed":row["observed"],"target_id":row["target_id"],
                            "tracking_reason":row["tracking_reason"],"recovery_attempt":row["recovery"]["executed"],
                            "recovery_selected":row["recovery_selected"],
                            "recovery_exhaustion_stage":row["recovery"]["budget_audit"]["exhaustion_stage"],
                            "recovery_s":row["stages_s"]["recovery"],"output_wall_s":row["output_wall_s"],
                            "output_age_s":row["output_age_s"],"action_audit":audit_by_source[name][fid]})
        with (output/"all-source-frames.csv").open() as stream:
            csv_rows=list(csv.DictReader(stream))
        equal(len(csv_rows),960)
        for raw,item in zip(csv_rows,inventory):
            for key,value in item.items():
                if key=="action_audit":
                    equal(json.loads(raw[key]) if raw[key] else None,value)
                else:
                    equal(raw[key],"" if value is None else str(value))
        return {"source_rows":960,"processed_rows":sum(len(rs) for rs in data.values()),
                "skipped_predictions_are_null":True,"csv_equal_json":True}
    audit.check("report_all_960_source_rows_and_csv_keep_drops_separate",complete_inventory)


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
                          for strategy in (("budget200ms", "costaware200ms") if repeat % 2
                                           else ("costaware200ms", "budget200ms"))]
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
        equal(results["historical_benchmark"], "../recovery-budget-20261021/results.json")
        historic = BASE / "recovery-budget-20261021"
        old_rows = []
        for repeat in (1, 2, 3):
            path = historic / "trials" / f"budget200ms-r{repeat}" / "frames.jsonl"
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
    validate_action_audit(audit, output, results)
    validate_report(audit, output, results, entry)
    failures = [check for check in audit.checks if not check["passed"]]
    payload = {"schema": "fallpre.v7.unknown_reduction.independent_validation.v1",
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
    suite = ET.Element("testsuite", name="v7_unknown_reduction_independent_audit",
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
