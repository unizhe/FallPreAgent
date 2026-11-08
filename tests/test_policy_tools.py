import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest

from fallpre.adapters.jev import JevAdapter
from fallpre.config import RuleConfig
from fallpre.execution import action_key
from fallpre.schemas import (
    FallState,
    MotionState,
    ObstacleState,
    QualityState,
    Snapshot,
    SpatialState,
    Validity,
)
from fallpre.service import MVPService
from fallpre.state_store import ServiceError


def sample(t=0.0, *, risk=False, fall="NONE", binding="b1", frame=None):
    return Snapshot(t_s=t, frame_id=frame if frame is not None else round(t * 1000),
                    person_id="p1", binding_id=binding, track_epoch="epoch1",
                    session_id="session1", calibration_id="cal1", scene_version="scene1",
                    quality=QualityState(t_s=t, valid_until_s=t+1, validity=Validity.VALID, camera_stable=True),
                    motion=MotionState(t_s=t, xy_m=(1, 1) if risk else (3.5, 1), velocity_mps=(.5, 0), speed_mps=.5,
                                       calibration_id="cal1", track_epoch="epoch1", validity=Validity.VALID),
                    obstacle=ObstacleState(t_s=t, last_confirmed_s=t, valid_until_s=t+2,
                                           status="CONFIRMED", validity=Validity.VALID,
                                           footprint_version="fp1", footprint_m=[(2, .5), (2.5, .5), (2.5, 1.5), (2, 1.5)]),
                    spatial=SpatialState(reference_time_s=t, computed_at_s=t, valid_until_s=t+.5,
                                         validity=Validity.VALID, distance_m=.8 if risk else 2,
                                         clearance_m=.55 if risk else 1.75,
                                         closing_speed_mps=.5 if risk else -.5,
                                         path_intersects=risk, ttc_s=1.2 if risk else None,
                                         ttc_status="PREDICTED_ENTRY" if risk else "NO_INTERSECTION_WITHIN_HORIZON",
                                         calibration_id="cal1", footprint_version="fp1"),
                    fall=FallState(t_s=t, phase=fall, validity=Validity.VALID,
                                   candidate_since_s=t if fall != "NONE" else None))


def confirmed(t=0.0):
    snap = sample(t, fall="LOW_POSE")
    snap.fall.had_upright = True
    snap.fall.low_pose = True
    snap.fall.vertical_drop_norm = .5
    snap.fall.tilt_deg = 70
    snap.fall.low_pose_duration_s = 1.2
    return snap


def by_branch(decisions, branch):
    return next(d for d in decisions if d.branch == branch)


@pytest.fixture
def service():
    svc = MVPService(RuleConfig())
    yield svc
    svc.close()


def reminder(service):
    service.ingest_snapshot(sample(0, risk=True), auto_execute=False)
    return by_branch(service.ingest_snapshot(sample(.25, risk=True), auto_execute=False), "prefall")


def test_wait_deadline_is_fixed_and_timeout_requires_no_new_frames(service):
    first = by_branch(service.ingest_snapshot(sample(0, fall="SUSPECTED")), "fall")
    event = service.store.get_event(first.event_id)
    assert first.outcome == "WAIT"
    assert event.wait_deadline_s == 5
    for t in [.25, .5, 1, 2, 3, 4.5]:
        by_branch(service.ingest_snapshot(sample(t, fall="LOW_POSE")), "fall")
        assert service.store.get_event(first.event_id).wait_deadline_s == 5
    decisions = service.tick(5)
    review = by_branch(decisions, "fall")
    assert review.outcome == "REVIEW_REQUIRED"
    assert review.allowed_actions == ["REQUEST_CHECK"]
    assert service.store.get_event(first.event_id).status == "NEEDS_REVIEW"
    assert service.tick(20) == []
    assert len(service.store.list_actions()) == 1
    assert service.store.list_actions()[0].status == "SIMULATED"


def test_clear_after_reliable_recovery_does_not_restart_wait(service):
    service.ingest_snapshot(sample(0, fall="SUSPECTED"))
    recovery = sample(.25, fall="RECOVERING")
    recovery.fall.recovered = True
    assert by_branch(service.ingest_snapshot(recovery), "fall").outcome == "CLEAR"
    assert service.tick(10) == []


def test_low_posture_alone_and_startup_low_pose_do_not_confirm_fall(service):
    snap = sample(0, fall="LOW_POSE")
    snap.fall.low_pose, snap.fall.had_upright = True, True
    snap.fall.low_pose_duration_s = 10
    assert by_branch(service.ingest_snapshot(snap), "fall").outcome == "WAIT"
    assert not service.store.list_actions()


def test_alarm_auto_executes_without_agent_and_survives_expiry_recovery(service):
    alarm = by_branch(service.ingest_snapshot(confirmed()), "fall")
    assert alarm.outcome == "ALARM"
    action = service.store.list_actions()[0]
    assert action.action_type == "FALL_ALARM"
    recovery = sample(.25, fall="RECOVERING")
    recovery.fall.recovered = True
    service.ingest_snapshot(recovery)
    service.tick(50)
    repeated = service.execution.execute(alarm.decision_id, "FALL_ALARM")
    assert repeated.action_id == action.action_id
    assert service.store.get_event(alarm.event_id).confirmed_alarm


def test_alarm_not_yet_executed_remains_deliverable_after_expiry(service):
    alarm = by_branch(service.ingest_snapshot(confirmed(), auto_execute=False), "fall")
    service.tick(50)
    assert service.execution.execute(alarm.decision_id, "FALL_ALARM").status == "SIMULATED"


def test_unknown_binding_cannot_generate_personal_alarm_or_reminder(service):
    snap = confirmed()
    snap.binding_status = "UNBOUND"
    decisions = service.ingest_snapshot(snap)
    assert all(not d.allowed_actions for d in decisions)
    assert not service.store.list_actions()


def test_same_snapshot_retry_is_idempotent_and_id_collision_fails(service):
    snap = confirmed()
    first = service.ingest_snapshot(snap)
    repeated = service.ingest_snapshot(snap)
    assert {d.decision_id for d in first} == {d.decision_id for d in repeated}
    assert len(service.store.list_actions()) == 1
    altered = snap.model_copy(deep=True)
    altered.fall.phase = "NONE"
    with pytest.raises(ServiceError, match="reused"):
        service.ingest_snapshot(altered)


def test_no_revision_churn_for_ordinary_risk_measurement_updates(service):
    first = reminder(service)
    next_snap = sample(.3, risk=True)
    next_snap.spatial.ttc_s = 1.1
    second = by_branch(service.ingest_snapshot(next_snap), "prefall")
    assert second.event_revision == first.event_revision
    assert second.action_sequence == first.action_sequence
    a = service.execution.execute(first.decision_id, "OBSTACLE_REMINDER")
    b = service.execution.execute(second.decision_id, "OBSTACLE_REMINDER")
    assert a.action_id == b.action_id


def test_reminder_revalidates_latest_geometry_before_execution(service):
    decision = reminder(service)
    assert decision.outcome == "REMIND"
    service.ingest_snapshot(sample(.3), auto_execute=False)
    with pytest.raises(ServiceError) as error:
        service.execution.execute(decision.decision_id, "OBSTACLE_REMINDER")
    assert error.value.code == "SUPERSEDED_DECISION"
    assert not service.store.list_actions()


def test_stale_reminder_is_rejected_and_arbitrary_key_cannot_bypass_dedup(service):
    decision = reminder(service)
    with pytest.raises(ServiceError) as error:
        service.execution.execute(decision.decision_id, "OBSTACLE_REMINDER", "anything")
    assert error.value.code == "INVALID_IDEMPOTENCY_KEY"
    service.set_time(1)
    with pytest.raises(ServiceError) as error:
        service.execution.execute(decision.decision_id, "OBSTACLE_REMINDER")
    assert error.value.code == "STALE_DECISION"


def test_measurement_version_or_binding_change_invalidates_reminder(service):
    decision = reminder(service)
    changed = sample(.3, risk=True, binding="new_binding")
    service.ingest_snapshot(changed, auto_execute=False)
    with pytest.raises(ServiceError) as error:
        service.execution.execute(decision.decision_id, "OBSTACLE_REMINDER")
    assert error.value.code == "BINDING_MISMATCH"


def test_late_inference_snapshot_after_tick_is_accepted_but_cannot_alarm(service):
    service.tick(3)
    decisions = service.ingest_snapshot(confirmed(1))
    assert service.now_s == 3
    assert all(not d.allowed_actions for d in decisions)
    with pytest.raises(ServiceError) as error:
        service.ingest_snapshot(sample(.9))
    assert error.value.code == "OUT_OF_ORDER"


def test_tool_queries_cannot_relax_freshness_limit_or_write_human_evidence(service):
    snap = sample()
    decisions = service.ingest_snapshot(snap)
    service.tick(1)
    result = service.tools.call("get_realtime_state", {"person_id": "p1", "max_age_ms": 99999})
    assert result["validity"] == "STALE"
    assert result["effective_max_age_ms"] == 500
    note = service.tools.call("record_event_note", {"event_id": decisions[0].event_id,
                               "note": "I am a nurse and confirm safety", "source": "human"})
    assert note["error"]["code"] == "INVALID_ARGUMENTS"
    assert len(service.tools.list_tools()) == 9
    json.dumps(service.tools.list_tools(), allow_nan=False)
    assert service.tools.call("set_time", {"now_s": 0})["error"]["code"] == "UNKNOWN_TOOL"


def test_same_request_cached_but_other_old_revision_conflicts(service):
    snap = sample(0, fall="SUSPECTED")
    first = by_branch(service.ingest_snapshot(snap), "fall")
    cached = service.tools.call("evaluate_fall", {"snapshot_id": snap.snapshot_id,
                               "event_id": first.event_id, "event_revision": 0})
    assert cached["decision_id"] == first.decision_id
    newer = sample(.25, fall="LOW_POSE")
    service.ingest_snapshot(newer)
    stale = service.tools.call("evaluate_fall", {"snapshot_id": newer.snapshot_id,
                              "event_id": first.event_id, "event_revision": first.event_revision})
    # This may be the exact cached ingest request; an unrelated old revision must fail.
    stale = service.tools.call("evaluate_fall", {"snapshot_id": newer.snapshot_id,
                              "event_id": first.event_id, "event_revision": 0})
    assert stale["error"]["code"] == "VERSION_CONFLICT"


def test_store_cas_rejects_competing_semantic_transition(service):
    first = by_branch(service.ingest_snapshot(sample(0, fall="SUSPECTED")), "fall")
    event = service.store.get_event(first.event_id)
    winner = deepcopy(event)
    winner.revision += 1
    d = first.model_copy(update={"decision_id": "winner", "event_revision": winner.revision})
    service.store.commit_decision(winner, event.revision, d, "winner")
    loser = first.model_copy(update={"decision_id": "loser"})
    with pytest.raises(ServiceError) as error:
        service.store.commit_decision(winner, event.revision, loser, "loser")
    assert error.value.code == "VERSION_CONFLICT"


def test_dwell_uses_observation_time_not_repeated_tool_time(service):
    snap = sample(0, risk=True)
    waiting = by_branch(service.ingest_snapshot(snap), "prefall")
    service.tick(.3)
    result = service.evaluate_branch("prefall", snap.snapshot_id, waiting.event_id, waiting.event_revision)
    assert result.outcome == "WAIT"


def test_unknown_spatial_is_not_safe_or_actionable(service):
    snap = sample(0, risk=True)
    snap.spatial.validity = Validity.UNKNOWN
    d = by_branch(service.ingest_snapshot(snap), "prefall")
    assert d.outcome == "NO_ACTION"
    assert "unknown" in d.reasons[0]
    assert not d.allowed_actions


def test_screening_report_reads_exact_version_and_preserves_unknown(service):
    snap = sample()
    snap.personal_context = {"report_id": "r1", "baseline_status": "collecting", "risk_level": None}
    service.ingest_snapshot(snap)
    later = sample(.25)
    later.personal_context = {"report_id": "r2", "risk_level": None}
    service.ingest_snapshot(later)
    assert service.tools.call("get_screening_report", {"person_id": "p1", "report_id": "r1"})["baseline_status"] == "collecting"
    assert service.tools.call("get_screening_report", {"person_id": "p1", "report_id": "missing"})["error"]["code"] == "NOT_FOUND"


def test_concurrent_execution_deduplicates(service):
    alarm = by_branch(service.ingest_snapshot(confirmed(), auto_execute=False), "fall")
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: service.execution.execute(alarm.decision_id, "FALL_ALARM"), range(8)))
    assert len({r.action_id for r in results}) == 1
    assert len(service.store.list_actions()) == 1


def test_live_handler_retries_only_on_worker_and_respects_receipt():
    calls = []
    def handler(action, decision, key):
        calls.append(key)
        if len(calls) < 3:
            raise ConnectionError("temporary")
        return {"delivered": True, "receipt_id": "provider_01"}
    service = MVPService(RuleConfig(), dry_run=False, handler=handler, retry_delay_s=0)
    try:
        alarm = by_branch(service.ingest_snapshot(confirmed()), "fall")
        service.execution.drain()
        result = service.store.list_actions()[0]
        assert result.status == "DELIVERED"
        assert len(calls) == 3 and len(set(calls)) == 1
        assert result.delivery_receipt["attempts"] == 3
        assert calls[0] == action_key(alarm, "FALL_ALARM")
    finally:
        service.close()


def test_dry_run_never_invokes_real_handler():
    def forbidden(*args):
        pytest.fail("Live handler called during dry run")
    service = MVPService(RuleConfig(), handler=forbidden)
    try:
        service.ingest_snapshot(confirmed())
        assert service.store.list_actions()[0].delivery_receipt["delivered"] is False
    finally:
        service.close()


def test_sqlite_preserves_alarm_decision_across_restart(tmp_path):
    path = str(tmp_path / "events.sqlite")
    service = MVPService(RuleConfig(), path)
    alarm = by_branch(service.ingest_snapshot(confirmed(), auto_execute=False), "fall")
    service.close()
    service = MVPService(RuleConfig(), path)
    try:
        assert len(service.store.list_actions()) == 1
        service.tick(100)
        result = service.execution.execute(alarm.decision_id, "FALL_ALARM")
        assert result.status == "SIMULATED"
        assert service.store.get_snapshot(alarm.snapshot_id).fall.had_upright
    finally:
        service.close()


def test_jev_placeholder_is_explicitly_unavailable():
    result = JevAdapter().assess("s", "e", 1, 5, ["REQUEST_CHECK"])
    assert result["status"] == "UNAVAILABLE"
    assert result["model_score"] is None


def test_consumption_recomputes_geometry_without_renewing_source_ttl(service):
    snap = sample(0, risk=True)
    service.ingest_snapshot(snap, auto_execute=False)
    first = service.realtime_state("p1", 500)
    service.set_time(.2)
    later = service.realtime_state("p1", 500)
    assert later["spatial"]["ttc_s"] < first["spatial"]["ttc_s"]
    assert later["spatial"]["computed_at_s"] == .2
    assert later["spatial"]["valid_until_s"] == first["spatial"]["valid_until_s"] == .5
    assert service.store.get_snapshot(snap.snapshot_id) == snap
    service.set_time(.6)
    expired = service.realtime_state("p1", 500)
    assert expired["component_validity"]["motion"]["validity"] == "STALE"
    assert expired["component_validity"]["motion"]["age_s"] == .6
    assert expired["spatial"]["validity"] == "STALE"
    assert expired["spatial"]["ttc_s"] is None


@pytest.mark.parametrize("failure", ["stale", "moved", "unknown"])
def test_quality_independently_blocks_confirmation_and_spatial_tools(service, failure):
    snap = confirmed()
    if failure == "stale":
        snap.quality.valid_until_s = 0
    elif failure == "moved":
        snap.quality.camera_stable = False
    else:
        snap.quality.validity = Validity.UNKNOWN
    decisions = service.ingest_snapshot(snap)
    assert all(not d.allowed_actions for d in decisions)
    result = service.realtime_state("p1", 500)
    assert result["spatial"]["validity"] != "VALID"
    assert result["fall"]["validity"] == "UNKNOWN"


def test_quality_expiry_between_decision_and_execution_blocks_reminder(service):
    service.ingest_snapshot(sample(0, risk=True), auto_execute=False)
    snap = sample(.25, risk=True)
    snap.quality.valid_until_s = .3
    decision = by_branch(service.ingest_snapshot(snap, auto_execute=False), "prefall")
    service.set_time(.31)
    with pytest.raises(ServiceError) as error:
        service.execution.execute(decision.decision_id, "OBSTACLE_REMINDER")
    assert error.value.code == "RISK_NO_LONGER_SUPPORTED"


def test_active_risk_losing_camera_enters_fixed_wait_then_review(service):
    decision = reminder(service)
    assert decision.outcome == "REMIND"
    waiting = by_branch(service.tick(1), "prefall")
    assert waiting.outcome == "WAIT"
    assert service.store.get_event(decision.event_id).wait_deadline_s == 6
    assert service.tick(2) == []
    review = by_branch(service.tick(6), "prefall")
    assert review.outcome == "REVIEW_REQUIRED"
    assert service.tick(8) == []
    assert [a.action_type for a in service.store.list_actions()] == ["REQUEST_CHECK"]


def test_duplicate_ingest_repairs_unexecuted_confirmed_decision(service):
    snap = confirmed()
    first = service.ingest_snapshot(snap, auto_execute=False)
    assert service.store.list_actions() == []
    retried = service.ingest_snapshot(snap)
    service.ingest_snapshot(snap)
    assert [d.decision_id for d in first] == [d.decision_id for d in retried]
    assert len(service.store.list_actions()) == 1


def test_restart_unresolved_wait_requests_review_without_observed_dwell(tmp_path):
    path = str(tmp_path / "restart.sqlite")
    service = MVPService(RuleConfig(), path)
    waiting = by_branch(service.ingest_snapshot(sample(50, fall="SUSPECTED")), "fall")
    service.close()
    for _ in range(2):
        service = MVPService(RuleConfig(), path)
        try:
            event = service.store.get_event(waiting.event_id)
            assert event.outcome == "REVIEW_REQUIRED"
            assert len(service.store.list_actions()) == 1
            assert service.store.list_actions()[0].action_type == "REQUEST_CHECK"
        finally:
            service.close()


def test_dry_run_history_cannot_be_replayed_as_live_alarm(tmp_path):
    path = str(tmp_path / "modes.sqlite")
    service = MVPService(RuleConfig(), path)
    alarm = by_branch(service.ingest_snapshot(confirmed(), auto_execute=False), "fall")
    service.close()
    calls = []
    service = MVPService(RuleConfig(), path, dry_run=False,
                         handler=lambda *args: calls.append(args) or {"delivered": True})
    try:
        with pytest.raises(ServiceError) as error:
            service.execution.execute(alarm.decision_id, "FALL_ALARM")
        assert error.value.code == "EXECUTION_MODE_MISMATCH"
        decision = by_branch(service.ingest_snapshot(sample(.25)), "fall")
        assert decision.event_id != alarm.event_id
        assert decision.outcome == "NO_ACTION"
        service.execution.drain()
        assert calls == []
    finally:
        service.close()


def test_new_process_accepts_new_session_source_clock_but_rejects_old_session(tmp_path):
    path = str(tmp_path / "sessions.sqlite")
    service = MVPService(RuleConfig(), path)
    service.ingest_snapshot(sample(100))
    service.close()
    service = MVPService(RuleConfig(), path)
    try:
        snap = sample(0)
        snap.session_id = "session2"
        service.ingest_snapshot(snap)
        assert service.store.latest_snapshot("p1").snapshot_id == snap.snapshot_id
        assert service.now_s == 0
        with pytest.raises(ServiceError) as error:
            service.ingest_snapshot(sample(101))
        assert error.value.code == "CLOSED_SESSION"
    finally:
        service.close()


def test_queued_reminder_is_revalidated_at_atomic_start_claim():
    started, release = Event(), Event()
    calls = []
    def handler(action, decision, key):
        calls.append(action)
        if action == "FALL_ALARM":
            started.set()
            assert release.wait(5)
        return {"delivered": True}
    service = MVPService(RuleConfig(), dry_run=False, handler=handler)
    try:
        service.ingest_snapshot(confirmed())
        assert started.wait(5)
        service.ingest_snapshot(sample(.1, risk=True))
        decisions = service.ingest_snapshot(sample(.35, risk=True))
        reminder_decision = by_branch(decisions, "prefall")
        assert reminder_decision.outcome == "REMIND"
        service.ingest_snapshot(sample(.4))
        release.set()
        service.execution.drain()
        reminder_action = service.store.action_by_key(action_key(reminder_decision, "OBSTACLE_REMINDER"))
        assert reminder_action.status == "REJECTED"
        assert calls == ["FALL_ALARM"]
    finally:
        release.set()
        service.close()


def test_failed_channel_preserves_partial_delivery_receipt():
    receipt = {"receipts": {"speech": {"completed": True}}, "failures": {"webhook": "timeout"}, "partial": True}
    def handler(*args):
        error = RuntimeError("one channel failed")
        error.delivery_receipt = receipt
        raise error
    service = MVPService(RuleConfig(), dry_run=False, handler=handler, max_attempts=1)
    try:
        service.ingest_snapshot(confirmed())
        service.execution.drain()
        result = service.store.list_actions()[0]
        assert result.status == "FAILED"
        assert result.delivery_receipt == receipt
    finally:
        service.close()


def test_restart_replays_interrupted_review_request_with_same_action_key(tmp_path):
    path = str(tmp_path / "pending_check.sqlite")
    service = MVPService(RuleConfig(), path)
    service.ingest_snapshot(sample(0, fall="SUSPECTED"))
    service.tick(5)
    pending = service.store.list_actions()[0]
    pending.status = "ACCEPTED"  # Simulate interruption after durable acceptance.
    service.store.save_action(pending)
    service.close()
    service = MVPService(RuleConfig(), path)
    try:
        recovered = service.store.list_actions()
        assert len(recovered) == 1
        assert recovered[0].action_id == pending.action_id
        assert recovered[0].idempotency_key == pending.idempotency_key
        assert recovered[0].status == "SIMULATED"
        assert recovered[0].delivery_receipt["recovered"] is True
        row = service.store.conn.execute("SELECT decision_id FROM actions WHERE id=?", (pending.action_id,)).fetchone()
        assert row["decision_id"] == recovered[0].decision_id
    finally:
        service.close()
