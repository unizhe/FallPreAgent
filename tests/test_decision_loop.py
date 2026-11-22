from __future__ import annotations

import copy
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from fallpre.care_loop import CareLoop
from fallpre.decision_config import DecisionConfig
from fallpre.decision_contracts import DecisionStatus, ProviderResult
from fallpre.decision_evidence import DecisionEnvelope, synthetic_envelope
from fallpre.decision_loop import (
    DecisionLoop,
    DecisionLoopError,
    DecisionVersionConflict,
    InjectedCrash,
    InjectedStoreCrash,
)
from fallpre.decision_provider import MockDecisionProvider
from fallpre.event_store import AUTHORITY, EventStore
from fallpre.state_store import StateStore


class FakeClock:
    def __init__(self, wall=1_800_000_000.0, mono=100.0):
        self.wall = wall
        self.mono = mono

    def wall_clock(self):
        return self.wall

    def monotonic_clock(self):
        return self.mono

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds


def config(**changes):
    values = dict(provider="mock", model="test-mock", min_interval_s=2.0,
                  max_retries=1, max_reobservations=1, timeout_s=1.0,
                  max_source_age_s=2.5, max_wall_age_s=5.0)
    values.update(changes)
    return DecisionConfig(**values)


def env(t=10.0, *, session="session-a", stream="camera-a", suffix="case", sufficient=True,
        epoch=3, generation=2, event_state="normal", event_key=None, recovery="not_established"):
    return synthetic_envelope(session_id=session, stream_id=stream, source_time_s=float(t), suffix=suffix,
                              sufficient=sufficient, action_epoch=epoch, generation=generation,
                              event_state=event_state, event_key=event_key, recovery_status=recovery)


def make_loop(path, clock=None, **kwargs):
    clock = clock or FakeClock()
    return DecisionLoop(path, run_id="test-run", config=kwargs.pop("config", config()),
                        provider=kwargs.pop("provider", "mock"),
                        monotonic_clock=clock.monotonic_clock, wall_clock=clock.wall_clock,
                        process_id=kwargs.pop("process_id", "process-a"), **kwargs)


def test_complete_normal_window_persists_immutable_result_and_exact_retry(tmp_path):
    clock = FakeClock()
    loop = make_loop(tmp_path / "agent.sqlite3", clock)
    original = env()
    replay = copy.deepcopy(original)
    receipt = loop.observe(original)
    original["perception"]["current_action"]["label"] = "mutated-after-observe"
    outcome = loop.run_one()

    assert outcome["status"] == "accepted"
    duplicate = loop.observe(replay)
    assert duplicate["status"] == "duplicate"
    assert duplicate["decision"]["probability_kind"] == "synthetic_fixture"
    rows = loop.store.observations(session_id="session-a")
    assert len(rows) == 1
    assert rows[0]["envelope"]["perception"]["current_action"]["label"] == "standing"
    assert loop.store.summary_counts()["accepted_results"] == 1
    assert loop.latest("session-a", "camera-a")["current_applicable"] is True
    assert loop.latest("session-a", "camera-a")["fall_probability"] == 0.86
    assert loop.list_decisions("session-a")[0]["decision_id"] == receipt["decision_id"]
    loop.close()


def test_schema_fixture_and_repeatable_versioned_migration(tmp_path):
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "decision-agent-full-envelope.jsonl"
    row = json.loads(fixture.read_text(encoding="utf-8"))
    assert DecisionEnvelope.model_validate(row).evidence_origin == "synthetic_fixture"
    database = tmp_path / "agent.sqlite3"
    for _ in range(3):
        loop = make_loop(database)
        version = loop.store.db.execute("SELECT value FROM decision_schema_meta WHERE key='version'").fetchone()[0]
        assert version == "9"
        loop.close()


def test_schema_v3_migration_keeps_old_results_history_only(tmp_path):
    import sqlite3

    from fallpre.decision_store import DecisionStore

    database = tmp_path / "legacy-v3.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE decision_schema_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("INSERT INTO decision_schema_meta VALUES('version','3')")
    connection.execute("CREATE TABLE decision_results (decision_id TEXT PRIMARY KEY,result_json TEXT NOT NULL)")
    connection.execute("INSERT INTO decision_results VALUES('old-decision','{\"status\":\"ready\",\"probability\":0.86}')")
    connection.commit()
    connection.close()

    for _ in range(2):
        store = DecisionStore(database)
        version = store.db.execute("SELECT value FROM decision_schema_meta WHERE key='version'").fetchone()[0]
        row = store.db.execute("SELECT result_json,service_instance_id FROM decision_results "
                               "WHERE decision_id='old-decision'").fetchone()
        assert version == "9"
        assert row["result_json"] == "{\"status\":\"ready\",\"probability\":0.86}"
        assert row["service_instance_id"] is None
        store.close()


def test_result_cursor_pages_completion_records_and_survives_restart(tmp_path):
    database = tmp_path / "cursor.sqlite3"
    clock = FakeClock()
    loop = make_loop(database, clock)
    first = loop.observe(env(10.0, suffix="cursor-first"))
    loop.run_one()
    clock.advance(2.1)
    second = loop.observe(env(12.1, suffix="cursor-second"))
    loop.run_one()

    page_one = loop.list_decisions_after("test-run", "session-a", "camera-a", limit=1)
    assert len(page_one) == 1
    page_two = loop.list_decisions_after("test-run", "session-a", "camera-a",
                                         after_id=page_one[0]["cursor"], limit=1)
    assert len(page_two) == 1
    assert [row["decision_id"] for row in page_one + page_two] == [first["decision_id"], second["decision_id"]]
    assert page_two[0]["cursor"] > page_one[0]["cursor"]
    assert loop.list_decisions_after("test-run", "session-a", "camera-a",
                                     after_id=page_two[0]["cursor"], limit=1) == []
    loop.close()

    recovered = make_loop(database, clock)
    replay = recovered.list_decisions_after("test-run", "session-a", "camera-a", limit=10)
    assert [row["decision_id"] for row in replay] == [first["decision_id"], second["decision_id"]]
    recovered.close()


def test_incomplete_legacy_observation_is_persisted_with_null_decision(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    value = env()
    value.pop("evidence_summary")
    value["perception"].pop("identity")
    receipt = loop.observe(value)
    assert receipt["task_id"] is None
    assert receipt["decision"] is None
    assert "action_epoch" in receipt["missing_reasons"]
    assert loop.store.summary_counts()["observations"] == 1
    assert loop.run_pending() == []
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    loop.close()


@pytest.mark.parametrize(
    ("section", "field", "value", "summary_field"),
    [
        ("identity", "action_epoch", 4, "action_epoch"),
        ("identity", "generation", 8, "generation"),
        ("identity", "fall_target_epoch", "target-other", "fall_target_epoch"),
        ("identity", "identity_link", "new_target", "identity_link"),
        ("observation", "observable", False, "observable"),
        ("observation", "event_decidable", False, "event_decidable"),
        ("observation", "camera_stable", False, "camera_stable"),
        ("observation", "person_count", 2, "person_count"),
        ("observation", "pose_valid_fraction", 0.4, "pose_valid_fraction"),
        ("observation", "continuous", False, "continuous"),
        ("temporal_action", "source_window", {"start_s": 8.0, "end_s": 10.5}, "source_window"),
        ("temporal_action", "observed_frames", 4, "observed_frames"),
        ("temporal_action", "status", "insufficient", "temporal_status"),
        ("fall_process", "status", "insufficient", "head_status"),
    ],
)
def test_explicit_summary_cannot_override_original_perception(tmp_path, section, field, value, summary_field):
    loop = make_loop(tmp_path / f"summary-conflict-{section}-{field}.sqlite3")
    parent = loop.observe(env(10, suffix="ready-before-summary-conflict"))
    loop.run_one()
    conflicting = env(12.5, suffix=f"raw-{section}-{field}-conflict")
    if section == "fall_process":
        conflicting["perception"]["fall_process"]["head"][field] = value
    else:
        conflicting["perception"][section][field] = value
    receipt = loop.observe(conflicting)
    reason = f"SOURCE_CONFLICT:{summary_field}"
    assert receipt["task_id"] is None
    assert reason in receipt["missing_reasons"]
    saved = loop.store.observations(session_id="session-a")[-1]
    assert saved["envelope"]["perception"] == conflicting["perception"]
    view = loop.latest("session-a", "camera-a")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert reason in view["stale_reason_codes"]
    assert loop.store.get_result(parent["decision_id"]) is not None
    loop.close()


def test_summary_conflict_and_valid_summary_completion_are_distinguished(tmp_path):
    loop = make_loop(tmp_path / "summary-fill.sqlite3")
    summary_conflict = env(10, suffix="summary-side-conflict")
    summary_conflict["evidence_summary"]["person_count"] = 2
    receipt = loop.observe(summary_conflict)
    assert receipt["task_id"] is None
    assert "SOURCE_CONFLICT:person_count" in receipt["missing_reasons"]
    loop.close()

    loop = make_loop(tmp_path / "summary-valid-fill.sqlite3")
    incomplete_raw = env(10, suffix="summary-valid-fill")
    incomplete_raw["perception"].pop("identity")
    receipt = loop.observe(incomplete_raw)
    assert receipt["task_id"] is not None
    assert loop.run_one()["status"] == "accepted"
    assert loop.latest("session-a", "camera-a")["fall_probability"] == 0.86
    assert loop.store.observations(session_id="session-a")[0]["envelope"]["perception"].get("identity") is None
    loop.close()


def test_same_time_conflict_has_no_side_effect_and_new_window_is_not_swallowed(tmp_path):
    clock = FakeClock()
    loop = make_loop(tmp_path / "agent.sqlite3", clock)
    first = env(event_key="camera-a/fall-1")
    loop.observe(first)
    loop.run_one()
    conflicting = copy.deepcopy(first)
    conflicting["perception"]["current_action"]["label"] = "sitting"
    with pytest.raises(ValueError, match="conflicting content"):
        loop.observe(conflicting)
    next_window = env(11.0, suffix="same-event-new-window", event_key="camera-a/fall-1")
    receipt = loop.observe(next_window)
    assert receipt["status"] == "accepted"
    assert receipt["schedule_reason"] == "MIN_SOURCE_INTERVAL"
    assert len(loop.store.observations(session_id="session-a")) == 2
    assert loop.store.summary_counts()["tasks"] == 1
    loop.close()


def test_same_local_event_key_on_two_streams_and_new_producer_session(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    a = env(event_state="confirmed", event_key="event-1")
    b = env(stream="camera-b", event_state="confirmed", event_key="event-1")
    c = env(session="producer-restarted", event_state="confirmed", event_key="event-1")
    receipts = [loop.observe(item) for item in (a, b, c)]
    assert len({item["task_id"] for item in receipts}) == 3
    assert loop.store.summary_counts()["observations"] == 3
    assert loop.store.summary_counts()["tasks"] == 3
    loop.close()


def test_quality_gate_keeps_p0_trigger_freshness_and_current_probability_null(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    value = env()
    value["source_time_s"] = 13.1
    value["perception"]["source_time_s"] = 13.1
    receipt = loop.observe(value)
    assert receipt["schedule_reason"] == "SCHEDULED"
    assert loop.run_one()["status"] == "accepted"
    result = ProviderResult.model_validate(loop.store.get_result(receipt["decision_id"])["provider_result"])
    assert result.status == DecisionStatus.STALE
    assert result.fall_probability is None
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    loop.close()


def test_reobserve_requires_new_window_and_consumes_only_one_budget(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    initial = env(10.0, suffix="observe-more", sufficient=False)
    parent = loop.observe(initial)
    loop.run_one()
    duplicate = loop.observe(copy.deepcopy(initial))
    assert duplicate["status"] == "duplicate"
    state = loop.store.workflow_for(parent["decision_id"])
    assert state["budget_used"] == 0

    too_soon = env(10.5, suffix="same-evidence-too-soon", sufficient=True)
    receipt_soon = loop.observe(too_soon)
    assert receipt_soon["schedule_reason"] == "WAITING_FOR_NEW_SOURCE_EVIDENCE"
    assert loop.store.workflow_for(parent["decision_id"])["budget_used"] == 0

    child = env(11.0, suffix="fresh-evidence", sufficient=True)
    receipt_child = loop.observe(child)
    assert receipt_child["schedule_reason"] == "REOBSERVATION_SCHEDULED"
    assert loop.run_one()["status"] == "accepted"
    workflow = loop.store.workflow_for(parent["decision_id"])
    assert workflow["state"] == "finished"
    assert workflow["budget_total"] == workflow["budget_used"] == 1
    assert loop.store.summary_counts()["tasks"] == 2
    loop.close()


def test_disabled_reobservation_budget_finishes_with_request_review(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3", config=config(max_reobservations=0))
    parent = loop.observe(env(10, suffix="budget-zero", sufficient=False))
    loop.run_one()
    workflow = loop.store.workflow_for(parent["decision_id"])
    assert workflow["state"] == "request_review"
    assert workflow["reason"] == "REOBSERVATION_BUDGET_DISABLED"
    assert workflow["budget_total"] == workflow["budget_used"] == 0
    loop.close()


@pytest.mark.parametrize(
    ("change", "expected_state", "expected_reason"),
    [
        ("target", "cancelled", "REOBSERVATION_TARGET_CHANGED"),
        ("quality", "cancelled", "REOBSERVATION_CONTINUITY_LOST"),
        ("recovery", "finished", "REOBSERVATION_UPSTREAM_RECOVERY"),
    ],
)
def test_target_quality_and_recovery_terminate_reobserve_wait(tmp_path, change, expected_state, expected_reason):
    loop = make_loop(tmp_path / f"{change}.sqlite3")
    parent = loop.observe(env(10, sufficient=False, suffix=f"parent-{change}"))
    loop.run_one()
    if change == "target":
        follow = env(11, sufficient=True, epoch=4, suffix="new-target")
    elif change == "quality":
        follow = env(11, sufficient=True, suffix="lost-continuity")
        follow["evidence_summary"]["continuous"] = False
        follow["perception"]["observation"]["continuous"] = False
    else:
        follow = env(11, sufficient=True, recovery="confirmed", event_state="normal", suffix="recovered")
    receipt = loop.observe(follow)
    workflow = loop.store.workflow_for(parent["decision_id"])
    assert workflow["state"] == expected_state
    assert workflow["reason"] == expected_reason
    assert receipt["task_id"] is None
    loop.close()


def test_eof_stop_and_timeout_close_workflows_without_current_probability(tmp_path):
    clock = FakeClock()
    loop = make_loop(tmp_path / "agent.sqlite3", clock)
    loop.observe(env(10, sufficient=False, suffix="eof"))
    loop.run_one()
    loop.observe(env(11, sufficient=True, suffix="eof-terminal"), end_of_stream=True)
    eof_view = loop.latest("session-a", "camera-a")
    assert eof_view["workflow"]["reason"] == "END_OF_STREAM"
    assert eof_view["current_applicable"] is False
    loop.close()

    loop = make_loop(tmp_path / "timeout.sqlite3", clock)
    timeout_parent = loop.observe(env(10, sufficient=False, suffix="timeout"))
    loop.run_one()
    clock.advance(5.1)
    timed = loop.latest("session-a", "camera-a")
    assert loop.store.workflow_for(timeout_parent["decision_id"])["state"] == "request_review"
    assert timed["current_applicable"] is False
    loop.stop("session-a", "camera-a")
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    loop.close()


def test_new_generation_recovery_and_stop_never_reuse_old_probability(tmp_path):
    clock = FakeClock()
    loop = make_loop(tmp_path / "agent.sqlite3", clock)
    first = env(10, event_state="confirmed", event_key="camera-a/fall-1")
    receipt = loop.observe(first)
    loop.run_one()
    assert loop.latest("session-a", "camera-a")["current_applicable"]
    newer_generation = env(12.1, generation=3, event_state="confirmed", event_key="camera-a/fall-1",
                           suffix="new-generation")
    loop.observe(newer_generation)
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    assert "RESULT_IDENTITY_OR_GENERATION_MISMATCH" in loop.latest("session-a", "camera-a")["stale_reason_codes"]
    late = env(11.5, suffix="late-out-of-order")
    stale_receipt = loop.observe(late)
    assert stale_receipt["stale"] is True
    assert loop.store.get_result(receipt["decision_id"]) is not None
    loop.stop("session-a", "camera-a")
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    loop.close()


@pytest.mark.parametrize(
    ("source_time", "quality_conflict"),
    [(10.2, False), (12.1, False), (12.1, True)],
)
def test_conflict_observation_never_reuses_old_ready_probability(tmp_path, source_time, quality_conflict):
    loop = make_loop(tmp_path / f"conflict-{source_time}-{quality_conflict}.sqlite3")
    original = loop.observe(env(10, suffix="ready-before-conflict"))
    loop.run_one()
    assert loop.latest("session-a", "camera-a")["fall_probability"] == 0.86

    conflict = env(source_time, suffix=f"geometry-conflict-{source_time}")
    conflict["evidence_summary"]["conflict_reason_codes"] = ["GEOMETRY_CONFLICT"]
    conflict["perception"]["fall_process"]["conflict_reason_codes"] = ["GEOMETRY_CONFLICT"]
    if quality_conflict:
        conflict["evidence_summary"].update(observable=False, event_decidable=False)
        conflict["perception"]["observation"].update(observable=False, event_decidable=False)
    receipt = loop.observe(conflict)
    if source_time < 12:
        assert receipt["schedule_reason"] == "MIN_SOURCE_INTERVAL"
        assert receipt["task_id"] is None
    else:
        assert receipt["task_id"] is not None
        assert loop.store.task(receipt["task_id"])["status"] == "pending"
    view = loop.latest("session-a", "camera-a")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert view["historical_result"]["decision_id"] == original["decision_id"]
    assert "GEOMETRY_CONFLICT" in view["stale_reason_codes"]
    if quality_conflict:
        assert "OBSERVATION_UNAVAILABLE" in view["stale_reason_codes"]
        assert "EVENT_NOT_DECIDABLE" in view["stale_reason_codes"]
    loop.close()


def test_conflict_clearing_requires_and_accepts_a_new_qualified_result(tmp_path):
    loop = make_loop(tmp_path / "conflict-cleared.sqlite3")
    loop.observe(env(10, suffix="before-conflict"))
    loop.run_one()
    conflict = env(10.2, suffix="conflict")
    conflict["evidence_summary"]["conflict_reason_codes"] = ["GEOMETRY_CONFLICT"]
    conflict["perception"]["fall_process"]["conflict_reason_codes"] = ["GEOMETRY_CONFLICT"]
    loop.observe(conflict)
    assert loop.latest("session-a", "camera-a")["fall_probability"] is None

    recovered = loop.observe(env(12.5, suffix="conflict-cleared"))
    assert recovered["task_id"] is not None
    result = loop.run_one()
    assert result["status"] == "accepted"
    view = loop.latest("session-a", "camera-a")
    assert view["current_applicable"] is True
    assert view["fall_probability"] == 0.86
    assert view["current"]["decision_id"] == recovered["decision_id"]
    loop.close()


def test_explicit_upstream_recovery_removes_current_projection_but_keeps_history(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    first = env(10, event_state="confirmed", event_key="camera-a/fall-1")
    receipt = loop.observe(first)
    loop.run_one()
    assert loop.latest("session-a", "camera-a")["current_applicable"]
    recovered = env(12.5, event_state="normal", event_key="camera-a/fall-1",
                    recovery="confirmed", suffix="explicit-recovery")
    loop.observe(recovered)
    view = loop.latest("session-a", "camera-a")
    assert view["current_applicable"] is False
    assert view["fall_probability"] is None
    assert "UPSTREAM_RECOVERY_CONFIRMED" in view["stale_reason_codes"]
    assert loop.store.get_result(receipt["decision_id"]) is not None
    loop.close()


def test_source_age_wall_age_and_restart_age_are_separate(tmp_path):
    clock = FakeClock()
    source_database = tmp_path / "source-age.sqlite3"
    loop = make_loop(source_database, clock, process_id="worker-first", service_instance_id="service-first")
    first = env(10)
    loop.observe(first)
    loop.run_one()
    newer = env(12.6, suffix="source-aged")
    loop.observe(newer)
    projection = loop.latest("session-a", "camera-a")
    assert projection["fall_probability"] is None
    assert "RESULT_SOURCE_AGE_EXCEEDED" in projection["stale_reason_codes"]
    loop.close()

    database = tmp_path / "wall-age.sqlite3"
    loop = make_loop(database, clock, process_id="worker-first", service_instance_id="service-first")
    first_receipt = loop.observe(first)
    loop.run_one()
    clock.advance(5.1)
    projection = loop.latest("session-a", "camera-a")
    assert "RESULT_WALL_AGE_EXCEEDED" in projection["stale_reason_codes"]
    loop.close()

    restarted = make_loop(database, clock, process_id="worker-second", service_instance_id="service-second")
    duplicate = restarted.observe(first)
    assert duplicate["status"] == "duplicate"
    assert restarted.run_pending() == []
    assert "RESULT_FROM_PRIOR_PROCESS" in restarted.latest("session-a", "camera-a")["stale_reason_codes"]
    assert restarted.store.summary_counts()["accepted_results"] == 1
    assert restarted.store.get_result(first_receipt["decision_id"]) is not None
    restarted.close()


def test_pending_regular_windows_coalesce_and_critical_transitions_are_preserved(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    first = loop.observe(env(10, suffix="pending-one"))
    second = loop.observe(env(12, suffix="pending-two"))
    assert second["schedule_reason"] == "COALESCED_TO_LATEST_PENDING_WINDOW"
    assert first["task_id"] == second["task_id"]
    critical = env(12.1, suffix="confirmed-transition", event_state="confirmed", event_key="camera-a/event-1")
    third = loop.observe(critical)
    assert third["schedule_reason"] == "CRITICAL_EVENT_TRANSITION"
    assert loop.store.summary_counts()["tasks"] == 2
    assert loop.store.db.execute("SELECT count(*) FROM decision_task_observations").fetchone()[0] == 3
    loop.close()


def test_bounded_queue_recovers_deferred_critical_transition(tmp_path):
    loop = DecisionLoop(tmp_path / "agent.sqlite3", run_id="test-run", config=config(), provider="mock",
                        max_queue=1)
    first = loop.observe(env(10, suffix="queue-capacity-normal"))
    deferred = env(10.1, suffix="queue-capacity-critical", event_state="confirmed",
                   event_key="camera-a/fall-1")
    queued = loop.observe(deferred)
    assert queued["schedule_reason"] == "QUEUE_FULL"
    assert queued["task_id"] is None
    assert loop.store.summary_counts()["tasks"] == 1
    assert loop.run_one()["status"] == "accepted"
    assert loop.store.summary_counts()["tasks"] == 1
    assert loop.run_one()["status"] == "accepted"
    saved = loop.store.observations(session_id="session-a")
    deferred_row = next(row for row in saved if row["envelope"]["source_time_s"] == 10.1)
    assert deferred_row["schedule_reason"] == "BACKPRESSURE_RECOVERED"
    assert deferred_row["task_id"] is not None
    assert loop.store.summary_counts()["tasks"] == 2
    assert loop.store.summary_counts()["accepted_results"] == 2
    assert loop.store.task(first["task_id"])["status"] == "completed"
    loop.close()


def test_duplicate_consumer_restart_does_not_repeat_logical_result(tmp_path):
    clock = FakeClock()
    database = tmp_path / "agent.sqlite3"
    value = env()
    first = make_loop(database, clock, process_id="consumer-one")
    receipt = first.observe(value)
    first.run_pending()
    first.close()

    second = make_loop(database, clock, process_id="consumer-two")
    duplicate = second.observe(value)
    assert duplicate["status"] == "duplicate"
    assert second.run_pending() == []
    assert second.store.summary_counts()["accepted_results"] == 1
    assert second.latest("session-a", "camera-a")["current_applicable"] is True
    assert second.store.get_result(receipt["decision_id"])["accepted"] == 1
    second.close()


def test_crash_between_snapshot_acceptance_and_task_insert_rolls_back_atomically(tmp_path):
    database = tmp_path / "agent.sqlite3"
    value = env()
    first = make_loop(database, process_id="before-atomic-crash")
    with pytest.raises(InjectedStoreCrash):
        first.observe(value, fail_at="after_snapshot_insert_before_task")
    assert first.store.summary_counts()["observations"] == 0
    assert first.store.summary_counts()["tasks"] == 0
    first.close()

    resumed = make_loop(database, process_id="after-atomic-crash")
    receipt = resumed.observe(value)
    assert receipt["status"] == "accepted"
    assert resumed.run_one()["status"] == "accepted"
    assert resumed.store.summary_counts()["observations"] == 1
    assert resumed.store.summary_counts()["accepted_results"] == 1
    resumed.close()


def test_stale_attempt_result_is_history_only_after_lease_recovery(tmp_path):
    clock = FakeClock()
    database = tmp_path / "agent.sqlite3"
    loop_a = make_loop(database, clock, process_id="worker-a")
    receipt = loop_a.observe(env())
    claim_a = loop_a.store.claim_next(now=clock.wall_clock(), lease_s=5, provider="mock", model="test-mock")
    evidence = __import__("fallpre.decision_contracts", fromlist=["DecisionEvidence"]).DecisionEvidence.model_validate(
        claim_a["evidence"])
    late_result = MockDecisionProvider(model="test-mock", clock=clock.monotonic_clock).assess(
        evidence, deadline=clock.monotonic_clock() + 1)

    clock.advance(6)
    loop_b = make_loop(database, clock, process_id="worker-b")
    outcome_b = loop_b.run_one()
    assert outcome_b["status"] == "accepted"
    late = loop_a.accept_result(task_id=receipt["task_id"], attempt=1,
                                lease_token=claim_a["lease_token"], result=late_result)
    assert late["accepted"] is False
    attempts = loop_b.store.attempts(receipt["task_id"])
    assert [item["status"] for item in attempts] == ["late_result", "accepted"]
    assert attempts[0]["error_code"] == "LEASE_EXPIRED"
    assert loop_b.store.summary_counts()["accepted_results"] == 1
    assert loop_b.store.attempts(receipt["task_id"])[0]["result_json"] is not None
    loop_a.close()
    loop_b.close()


def test_expired_lease_retries_obey_persisted_budget_after_restart_and_race(tmp_path):
    clock = FakeClock()
    database = tmp_path / "lease-budget.sqlite3"
    first = make_loop(database, clock, process_id="lease-worker-1")
    receipt = first.observe(env())
    with pytest.raises(InjectedCrash):
        first.run_one(fail_at="after_claim")
    assert first.store.task(receipt["task_id"])["max_attempts"] == 2
    first.close()

    clock.advance(6)
    second = make_loop(database, clock, process_id="lease-worker-2")
    with pytest.raises(InjectedCrash):
        second.run_one(fail_at="after_claim")
    assert [row["attempt_no"] for row in second.store.attempts(receipt["task_id"])] == [1, 2]
    second.close()

    clock.advance(6)
    barrier = threading.Barrier(2)

    def compete(name):
        scheduler = make_loop(database, clock, process_id=name)
        try:
            barrier.wait(timeout=2)
            return scheduler.run_one()
        finally:
            scheduler.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(compete, ("lease-racer-a", "lease-racer-b")))
    assert outcomes == [None, None]
    with make_loop(database, clock) as inspector:
        task = inspector.store.task(receipt["task_id"])
        assert task["status"] == "cancelled"
        assert task["reason"] == "LEASE_EXPIRED_MAX_ATTEMPTS"
        assert [row["status"] for row in inspector.store.attempts(receipt["task_id"])] == [
            "lease_expired", "lease_expired"
        ]
        assert inspector.store.summary_counts()["accepted_results"] == 0


def test_fail_attempt_cannot_override_the_persisted_retry_budget(tmp_path):
    loop = make_loop(tmp_path / "fixed-budget.sqlite3", config=config(max_retries=0))
    receipt = loop.observe(env())
    claim = loop.store.claim_next(now=loop.wall_clock(), lease_s=5,
                                  provider="mock", model="test-mock")
    assert claim["attempt"] == claim["max_attempts"] == 1
    with pytest.raises(TypeError, match="max_attempts"):
        loop.store.fail_attempt(task_id=receipt["task_id"], attempt=1,
                                lease_token=claim["lease_token"], error_code="SYNTHETIC_FAILURE",
                                max_attempts=99, now=loop.wall_clock())
    assert loop.store.task(receipt["task_id"])["status"] == "running"
    assert loop.store.fail_attempt(task_id=receipt["task_id"], attempt=1,
                                   lease_token=claim["lease_token"], error_code="SYNTHETIC_FAILURE",
                                   now=loop.wall_clock())
    task = loop.store.task(receipt["task_id"])
    assert task["status"] == "cancelled"
    assert task["reason"] == "MAX_ATTEMPTS_EXHAUSTED:SYNTHETIC_FAILURE"
    loop.close()


def test_result_returned_after_stop_is_persisted_only_in_attempt_history(tmp_path):
    clock = FakeClock()
    database = tmp_path / "agent.sqlite3"
    worker = make_loop(database, clock, process_id="worker-before-stop")
    receipt = worker.observe(env())
    claim = worker.store.claim_next(now=clock.wall_clock(), lease_s=5, provider="mock", model="test-mock")
    evidence = __import__("fallpre.decision_contracts", fromlist=["DecisionEvidence"]).DecisionEvidence.model_validate(
        claim["evidence"])
    result = MockDecisionProvider(model="test-mock", clock=clock.monotonic_clock).assess(
        evidence, deadline=clock.monotonic_clock() + 1)
    controller = make_loop(database, clock, process_id="controller-stop")
    controller.stop("session-a", "camera-a")
    late = worker.accept_result(task_id=receipt["task_id"], attempt=1,
                                lease_token=claim["lease_token"], result=result)
    assert late["accepted"] is False
    assert worker.store.get_result(receipt["decision_id"]) is None
    history = controller.list_decisions("session-a")[0]["attempt_history"]
    assert history[0]["status"] == "late_result"
    assert history[0]["provider_result"]["fall_probability"] == 0.86
    assert controller.latest("session-a", "camera-a")["fall_probability"] is None
    worker.close()
    controller.close()


def test_two_scheduler_connections_race_to_one_durable_task(tmp_path):
    database = tmp_path / "agent.sqlite3"
    owner = DecisionLoop(database, run_id="test-run", config=config(), provider="mock")
    owner.observe(env())
    owner.close()
    barrier = threading.Barrier(2)

    def run_scheduler(name):
        scheduler = make_loop(database, process_id=name)
        try:
            barrier.wait(timeout=2)
            return scheduler.run_one()
        finally:
            scheduler.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run_scheduler, ("scheduler-a", "scheduler-b")))
    assert sum(outcome is not None for outcome in outcomes) == 1
    with make_loop(database) as inspector:
        assert inspector.store.summary_counts()["accepted_results"] == 1
        assert inspector.store.db.execute("SELECT count(*) FROM decision_attempts").fetchone()[0] == 1


def test_thread_owned_worker_returns_result_through_queue(tmp_path):
    from fallpre.decision_loop import DecisionWorker

    database = tmp_path / "agent.sqlite3"
    owner = DecisionLoop(database, run_id="test-run", config=config(), provider="mock")
    owner.observe(env())
    with DecisionWorker(str(database), config=config(), provider="mock") as worker:
        outcome = worker.run_one(timeout=5)
    assert outcome["status"] == "accepted"
    owner_view = owner.latest("session-a", "camera-a")
    assert owner_view["current_applicable"] is True
    assert owner_view["fall_probability"] == 0.86
    result = owner.store.get_result(outcome["decision_id"])
    assert result["process_id"] != owner.process_id
    assert result["service_instance_id"] == owner.service_instance_id
    owner.close()

    restarted = make_loop(database, service_instance_id="service-after-real-restart")
    duplicate = restarted.observe(env())
    assert duplicate["status"] == "duplicate"
    restarted_view = restarted.latest("session-a", "camera-a")
    assert restarted_view["current_applicable"] is False
    assert restarted_view["fall_probability"] is None
    assert "RESULT_FROM_PRIOR_PROCESS" in restarted_view["stale_reason_codes"]
    assert restarted.store.get_result(outcome["decision_id"]) is not None
    restarted.close()

    with make_loop(database) as inspector:
        assert inspector.store.summary_counts()["accepted_results"] == 1


@pytest.mark.parametrize("fault", ["after_claim", "after_provider_before_write", "after_commit_before_receipt"])
def test_fault_injection_recovery_keeps_one_accepted_result(tmp_path, fault):
    clock = FakeClock()
    database = tmp_path / f"{fault}.sqlite3"
    loop = make_loop(database, clock, process_id="before-crash")
    receipt = loop.observe(env())
    with pytest.raises(InjectedCrash):
        loop.run_one(fail_at=fault)
    before = loop.store.summary_counts()["accepted_results"]
    loop.close()
    if fault != "after_commit_before_receipt":
        clock.advance(6)
    recovered = make_loop(database, clock, process_id="after-crash")
    if fault == "after_commit_before_receipt":
        assert recovered.observe(env())["status"] == "duplicate"
        assert recovered.run_pending() == []
    else:
        assert recovered.run_one()["status"] == "accepted"
    assert recovered.store.summary_counts()["accepted_results"] == 1
    if fault == "after_provider_before_write":
        assert recovered.store.attempts(receipt["task_id"])[1]["status"] == "accepted"
    if fault == "after_commit_before_receipt":
        assert before == 1
    recovered.close()


def test_feedback_is_idempotent_versioned_and_not_provider_input(tmp_path):
    loop = make_loop(tmp_path / "agent.sqlite3")
    receipt = loop.observe(env())
    loop.run_one()
    first = loop.feedback(receipt["decision_id"], label="fall", request_id="review-1", expected_version=0)
    retry = loop.feedback(receipt["decision_id"], label="fall", request_id="review-1", expected_version=0)
    assert first == retry
    assert first["provider_state_changed"] is False
    with pytest.raises(DecisionVersionConflict):
        loop.feedback(receipt["decision_id"], label="non_fall", request_id="review-2", expected_version=0)
    with pytest.raises(ValueError, match="reused"):
        loop.feedback(receipt["decision_id"], label="uncertain", request_id="review-1", expected_version=0)
    saved = loop.store.get_result(receipt["decision_id"])["provider_result"]
    assert "feedback" not in json_keys(saved)
    assert len(loop.store.list_feedback(receipt["decision_id"])) == 1
    loop.close()


def json_keys(value):
    return {key for key in value}


def test_provider_failure_has_bounded_attempts_and_never_fabricates_probability(tmp_path):
    class BrokenProvider:
        def __init__(self):
            self.calls = 0

        def assess(self, evidence, *, deadline):
            del evidence, deadline
            self.calls += 1
            raise RuntimeError("offline fixture failure")

    provider = BrokenProvider()
    loop = make_loop(tmp_path / "agent.sqlite3", provider_impl=provider)
    receipt = loop.observe(env())
    first = loop.run_one()
    second = loop.run_one()
    assert first["status"] == "retry_pending"
    assert second["status"] == "failed"
    assert provider.calls == 2
    assert loop.store.task(receipt["task_id"])["status"] == "cancelled"
    assert loop.store.get_result(receipt["decision_id"]) is None
    assert all(item["error_code"] == "PROVIDER_ERROR_RuntimeError"
               for item in loop.store.attempts(receipt["task_id"]))
    loop.close()


def test_child_provider_failures_exhaust_and_close_workflow_across_restart(tmp_path):
    class BrokenProvider:
        def assess(self, evidence, *, deadline):
            del evidence, deadline
            raise RuntimeError("synthetic child provider failure")

    clock = FakeClock()
    database = tmp_path / "child-failure.sqlite3"
    loop = make_loop(database, clock)
    parent = loop.observe(env(10, sufficient=False, suffix="child-failure-parent"))
    assert loop.run_one()["status"] == "accepted"
    child = loop.observe(env(11, suffix="child-failure-child"))
    assert loop.store.workflow_for(parent["decision_id"])["state"] == "child_pending"
    loop.provider_impl = BrokenProvider()
    assert loop.run_one()["status"] == "retry_pending"
    assert loop.store.workflow_for(parent["decision_id"])["state"] == "child_pending"
    assert loop.run_one()["status"] == "failed"
    workflow = loop.store.workflow_for(parent["decision_id"])
    assert workflow["state"] == "request_review"
    assert workflow["reason"] == "REOBSERVATION_CHILD_FAILED:PROVIDER_ERROR_RuntimeError"
    task = loop.store.task(child["task_id"])
    assert task["status"] == "cancelled"
    assert task["attempt"] == task["max_attempts"] == 2
    attempts_before = len(loop.store.attempts(child["task_id"]))
    service_id = loop.service_instance_id
    loop.close()

    restarted = make_loop(database, clock, service_instance_id="service-after-child-failure")
    assert restarted.store.workflow_for(parent["decision_id"])["state"] == "request_review"
    duplicate = restarted.observe(env(11, suffix="child-failure-child"))
    assert duplicate["status"] == "duplicate"
    assert len(restarted.store.attempts(child["task_id"])) == attempts_before
    later = restarted.observe(env(12.1, suffix="after-child-failure"))
    assert later["task_id"] is None
    assert restarted.store.summary_counts()["tasks"] == 2
    assert restarted.store.task(child["task_id"])["parent_decision_id"] == parent["decision_id"]
    assert service_id != restarted.service_instance_id
    restarted.close()


def test_child_lease_exhaustion_closes_workflow_after_reopen(tmp_path):
    clock = FakeClock()
    database = tmp_path / "child-lease-failure.sqlite3"
    loop = make_loop(database, clock)
    parent = loop.observe(env(10, sufficient=False, suffix="child-lease-parent"))
    loop.run_one()
    child = loop.observe(env(11, suffix="child-lease-child"))
    with pytest.raises(InjectedCrash):
        loop.run_one(fail_at="after_claim")
    clock.advance(6)
    loop.close()

    recovered = make_loop(database, clock)
    with pytest.raises(InjectedCrash):
        recovered.run_one(fail_at="after_claim")
    clock.advance(6)
    assert recovered.run_one() is None
    workflow = recovered.store.workflow_for(parent["decision_id"])
    assert workflow["state"] == "request_review"
    assert workflow["reason"] == "REOBSERVATION_CHILD_LEASE_RETRIES_EXHAUSTED"
    task = recovered.store.task(child["task_id"])
    assert task["status"] == "cancelled"
    assert task["attempt"] == task["max_attempts"] == 2
    assert recovered.store.summary_counts()["accepted_results"] == 1
    recovered.close()

    restarted = make_loop(database, clock, service_instance_id="service-after-child-lease-failure")
    assert restarted.store.workflow_for(parent["decision_id"])["state"] == "request_review"
    assert restarted.run_one() is None
    assert len(restarted.store.attempts(child["task_id"])) == 2
    restarted.close()


@pytest.mark.parametrize("termination", ["deadline", "eof", "stop"])
def test_child_pending_queue_deadline_eof_and_stop_are_terminal(tmp_path, termination):
    clock = FakeClock()
    database = tmp_path / f"child-terminal-{termination}.sqlite3"
    loop = DecisionLoop(database, run_id="test-run", config=config(), provider="mock",
                        monotonic_clock=clock.monotonic_clock, wall_clock=clock.wall_clock,
                        max_queue=1)
    parent = loop.observe(env(10, sufficient=False, suffix=f"child-terminal-{termination}"))
    loop.run_one()
    if termination == "deadline":
        loop.observe(env(10, stream="camera-b", suffix="fills-bounded-queue"))
        child = loop.observe(env(11, suffix="child-waits-in-queue"))
        assert child["schedule_reason"] == "QUEUE_FULL"
        clock.advance(5.1)
        # Claiming freed queue capacity must expire, not revive, the late child.
        assert loop.run_one()["status"] == "accepted"
        workflow = loop.store.workflow_for(parent["decision_id"])
        assert workflow["state"] == "request_review"
        assert workflow["reason"] == "REOBSERVATION_CHILD_DEADLINE_EXCEEDED"
        saved = next(row for row in loop.store.observations(session_id="session-a")
                     if row["envelope"]["source_time_s"] == 11)
        assert saved["schedule_reason"] == "REOBSERVATION_CHILD_DEADLINE_EXCEEDED"
        assert loop.store.db.execute(
            "SELECT count(*) FROM decision_tasks WHERE parent_decision_id=?", (parent["decision_id"],)
        ).fetchone()[0] == 0
    else:
        child = loop.observe(env(11, suffix="child-before-terminal"))
        assert child["task_id"] is not None
        if termination == "eof":
            loop.observe(env(11.2, suffix="end-of-stream"), end_of_stream=True)
            workflow = loop.store.workflow_for(parent["decision_id"])
            assert workflow["state"] == "finished"
            assert workflow["reason"] == "END_OF_STREAM"
            assert loop.store.task(child["task_id"])["status"] == "cancelled"
        else:
            loop.stop("session-a", "camera-a", reason="STOP_REQUESTED")
            workflow = loop.store.workflow_for(parent["decision_id"])
            assert workflow["state"] == "cancelled"
            assert workflow["reason"] == "STOP_REQUESTED"
            assert loop.store.task(child["task_id"])["status"] == "cancelled"
        assert loop.latest("session-a", "camera-a")["fall_probability"] is None
    parent_id = parent["decision_id"]
    service_id = loop.service_instance_id
    loop.close()
    restarted = make_loop(database, clock, service_instance_id="service-after-child-terminal")
    persisted = restarted.store.workflow_for(parent_id)
    expected = {
        "deadline": ("request_review", "REOBSERVATION_CHILD_DEADLINE_EXCEEDED"),
        "eof": ("finished", "END_OF_STREAM"),
        "stop": ("cancelled", "STOP_REQUESTED"),
    }[termination]
    assert (persisted["state"], persisted["reason"]) == expected
    assert service_id != restarted.service_instance_id
    restarted.close()


def test_provider_cannot_bypass_quality_gate_with_ready_result(tmp_path):
    class BypassProvider:
        def assess(self, evidence, *, deadline):
            base = MockDecisionProvider(model="test-mock", clock=lambda: 0).assess(evidence, deadline=deadline)
            values = base.model_dump(mode="json")
            values.update(status="ready", fall_probability=0.99, probability_kind="synthetic_fixture",
                          evidence_sufficiency="sufficient", next_step="emit_result",
                          reason_codes=["INJECTED_BYPASS"])
            return ProviderResult.model_validate(values)

    loop = make_loop(tmp_path / "agent.sqlite3", provider_impl=BypassProvider())
    receipt = loop.observe(env(sufficient=False))
    outcome = loop.run_one()
    assert outcome["status"] == "retry_pending"
    assert outcome["error_code"] == "PROVIDER_RESULT_REJECTED"
    assert loop.store.get_result(receipt["decision_id"]) is None
    assert loop.store.task(receipt["task_id"])["status"] == "pending"
    loop.close()


def test_off_mock_do_not_network_and_aliyun_is_unavailable_in_p1(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network must not be opened in P1")

    monkeypatch.setattr(socket.socket, "connect", denied)
    for provider in ("off", "mock"):
        loop = make_loop(tmp_path / f"{provider}.sqlite3", provider=provider,
                         config=config(provider=provider))
        receipt = loop.observe(env())
        loop.run_one()
        if provider == "off":
            assert loop.store.get_result(receipt["decision_id"])["provider_result"]["status"] == "skipped"
        loop.close()
    with pytest.raises(DecisionLoopError, match="explicit enable_network"):
        DecisionLoop(tmp_path / "aliyun.sqlite3", config=config(provider="aliyun"), provider="aliyun")


def test_existing_care_loop_confirmed_alarm_permission_and_dedup_are_unchanged(tmp_path):
    clock = FakeClock()
    events = EventStore(tmp_path / "events.sqlite3")
    actions = StateStore(str(tmp_path / "actions.sqlite3"))
    care = CareLoop(events, actions)
    decision = make_loop(tmp_path / "decisions.sqlite3", clock)
    first = env(10, event_state="confirmed", event_key="camera-a/fall-1", suffix="confirmed-one")
    first["perception"]["event"]["alarm_authority"] = AUTHORITY
    first["perception"]["event"]["recovery_confirmed"] = False
    care_first = care.observe("session-a", copy.deepcopy(first["perception"]))
    decision_first = decision.observe(first)
    decision.run_pending()
    assert len(care_first["executions"]) == 1
    assert decision.store.get_result(decision_first["decision_id"])["provider_result"]["changes_alarm"] is False

    second = env(12.5, event_state="confirmed", event_key="camera-a/fall-1", suffix="confirmed-next-window")
    second["perception"]["event"]["alarm_authority"] = AUTHORITY
    second["perception"]["event"]["recovery_confirmed"] = False
    care_second = care.observe("session-a", copy.deepcopy(second["perception"]))
    decision.observe(second)
    assert care_second["executions"] == []
    duplicate = care.observe("session-a", copy.deepcopy(second["perception"]))
    assert duplicate["executions"] == []
    assert events.db.execute("SELECT count(*) FROM perception_simulated_deliveries").fetchone()[0] == 1
    care.close()
    actions.close()
    events.close()

    with EventStore(tmp_path / "events.sqlite3") as events:
        actions = StateStore(str(tmp_path / "actions.sqlite3"))
        restarted_care = CareLoop(events, actions)
        third = env(13.0, event_state="confirmed", event_key="camera-a/fall-1", suffix="confirmed-after-restart")
        third["perception"]["event"]["alarm_authority"] = AUTHORITY
        third["perception"]["event"]["recovery_confirmed"] = False
        assert restarted_care.observe("session-a", copy.deepcopy(third["perception"]))["executions"] == []
        assert events.db.execute("SELECT count(*) FROM perception_simulated_deliveries").fetchone()[0] == 1
        restarted_care.close()
        actions.close()
    decision.close()


def test_envelope_rejects_nan_and_incomplete_legacy_fields_are_not_fabricated():
    value = env()
    invalid = copy.deepcopy(value)
    invalid["source_time_s"] = float("nan")
    invalid["perception"]["source_time_s"] = float("nan")
    with pytest.raises(ValidationError):
        DecisionEnvelope.model_validate(invalid)
    partial = copy.deepcopy(value)
    partial.pop("evidence_summary")
    partial["perception"].pop("identity")
    from fallpre.decision_evidence import adapt_envelope

    adapted = adapt_envelope(partial, run_id="run", snapshot_id="snap", decision_id="dec",
                             request_digest="0" * 64)
    assert adapted.evidence is None
    assert "action_epoch" in adapted.missing_fields
    assert "generation" in adapted.missing_fields
