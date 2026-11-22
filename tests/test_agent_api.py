from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path

import cv2
import numpy as np
import pytest

from fallpre.agent_api import AgentApiError, AgentRunStore, AgentRuntime, _OwnerThread
from fallpre.decision_config import DecisionConfig
from fallpre.decision_evidence import synthetic_envelope
from fallpre.decision_loop import DecisionLoop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location("stage04_webcam_service", ROOT / "scripts/webcam_pose_demo.py")
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


CLIENT_A = "client-session-123456"
CLIENT_B = "client-session-abcdef"


class FakeEngine:
    default_model = "yolo26n-pose"
    default_fall_mode = "pose"
    device = "cpu"
    imgsz = 640
    decision_head_variant = "cd"

    def __init__(self, delay=0.0):
        self.lock = threading.Lock()
        self.calls = 0
        self.delay = delay
        self.streams = set()

    def infer_live(self, image, stream_id, source_time, model_id, source_type, fall_mode):
        time.sleep(self.delay)
        self.calls += 1
        self.streams.add(stream_id)
        return {"source_time": source_time, "width": image.shape[1], "height": image.shape[0],
                "persons": [], "model": {"id": model_id, "sha256": "a" * 64}}

    def stop_stream(self, stream_id):
        self.streams.discard(stream_id)

    def available_models(self):
        return [{"id": self.default_model, "label": "YOLO26n-Pose", "available": True}]

    def temporal_status(self):
        return {"enabled": True, "ready": True, "model": {"id": "stgcn"}}

    def fusion_status(self):
        return {"enabled": False}

    def fall_temporal_status(self):
        return {"enabled": True}

    def fall_event_fusion_status(self):
        return {"enabled": False}


@pytest.fixture
def runtime(tmp_path):
    instance = AgentRuntime(
        DecisionConfig(db_path=str(tmp_path / "agent.sqlite3")), provider="off", enable_network=False,
        resource_status={"fall_head": {"status": "ready", "selected_variant": "cd"}},
    )
    try:
        yield instance
    finally:
        instance.close()


def create_run(runtime, client=CLIENT_A, request_id="req-1", source_type="video"):
    return runtime.create_run(client_session_id=client, request_id=request_id, source_type=source_type)


def test_run_idempotency_conflicts_and_session_ownership(runtime):
    first, duplicate = create_run(runtime)
    same, duplicate_again = create_run(runtime)

    assert first["run_id"] == same["run_id"]
    assert first["session_id"] == same["session_id"]
    assert not duplicate and duplicate_again
    assert first["provider"] == "off"
    with pytest.raises(AgentApiError, match="request_id was reused") as conflict:
        create_run(runtime, request_id="req-1", source_type="camera")
    assert conflict.value.status == 409
    with pytest.raises(AgentApiError) as missing:
        runtime.latest(first["run_id"], CLIENT_B)
    assert missing.value.status == 404


def test_frame_replay_conflict_and_concurrent_duplicate_only_claim_once(runtime):
    run, _ = create_run(runtime, source_type="camera")
    options = {"model_id": "yolo26n-pose", "fall_mode": "pose"}
    status, _, row = runtime.begin_frame(run_id=run["run_id"], client_session_id=CLIENT_A,
                                         stream_id=run["stream_id"], source_type="camera", source_time=1.0,
                                         image_bytes=b"jpeg-1", options=options)
    assert status == "new"

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(runtime.begin_frame, run_id=run["run_id"], client_session_id=CLIENT_A,
                             stream_id=run["stream_id"], source_type="camera", source_time=1.0,
                             image_bytes=b"jpeg-1", options=options)
        time.sleep(0.02)
        waiting_status, future, _ = waiter.result()
        assert waiting_status == "waiting"
        receipt = {"agent": {"status": "pending"}, "source_time": 1.0}
        runtime.finish_frame(row, 1.0, receipt)
        assert future.result(timeout=1) == (200, receipt)

    replay_status, replay, _ = runtime.begin_frame(run_id=run["run_id"], client_session_id=CLIENT_A,
                                                   stream_id=run["stream_id"], source_type="camera", source_time=1.0,
                                                   image_bytes=b"jpeg-1", options=options)
    assert replay_status == "completed"
    assert replay == (200, receipt)
    with pytest.raises(AgentApiError) as image_conflict:
        runtime.begin_frame(run_id=run["run_id"], client_session_id=CLIENT_A,
                            stream_id=run["stream_id"], source_type="camera", source_time=1.0,
                            image_bytes=b"different", options=options)
    assert image_conflict.value.status == 409
    with pytest.raises(AgentApiError) as option_conflict:
        runtime.begin_frame(run_id=run["run_id"], client_session_id=CLIENT_A,
                            stream_id=run["stream_id"], source_type="camera", source_time=1.0,
                            image_bytes=b"jpeg-1", options={**options, "fall_mode": "rgb"})
    assert option_conflict.value.status == 409


def test_restart_stops_visual_run_and_keeps_http_budget(tmp_path):
    database = str(tmp_path / "agent.sqlite3")
    first_runtime = AgentRuntime(DecisionConfig(db_path=database), provider="off", enable_network=False,
                                 resource_status={}, network_budget=2)
    run, _ = first_runtime.create_run(client_session_id=CLIENT_A, request_id="restart-run", source_type="video")
    assert first_runtime.runs.reserve_http_send()
    first_runtime.close()

    restarted = AgentRuntime(DecisionConfig(db_path=database), provider="off", enable_network=False,
                             resource_status={}, network_budget=6)
    try:
        recovered = restarted.runs.get_run(run["run_id"], CLIENT_A)
        assert recovered["status"] == "stopped"
        assert recovered["stop_reason"] == "SERVICE_RESTARTED"
        assert restarted.runs.http_budget() == {"maximum": 2, "used": 1}
        assert restarted.runs.reserve_http_send()
        assert not restarted.runs.reserve_http_send()
    finally:
        restarted.close()


def test_explicit_budget_increase_retains_existing_usage(tmp_path):
    database = tmp_path / "agent.sqlite3"
    original = AgentRunStore(database, network_budget=2)
    assert original.reserve_http_send()
    assert original.reserve_http_send()
    assert not original.reserve_http_send()

    increased = AgentRunStore(database, network_budget=1000, raise_http_budget=True)
    assert increased.http_budget() == {"maximum": 1000, "used": 2}
    assert increased.reserve_http_send()
    assert AgentRunStore(database, network_budget=1000).http_budget() == {"maximum": 1000, "used": 3}


def test_restart_reconciles_registered_pending_task_before_worker_can_claim(tmp_path):
    database = str(tmp_path / "registered-pending.sqlite3")
    config = DecisionConfig(db_path=database, provider="mock")
    first = AgentRuntime(config, provider="mock", enable_network=False, resource_status={})
    run, _ = first.create_run(client_session_id=CLIENT_A, request_id="registered-pending", source_type="video")
    first._scheduler_stop.set()
    first._scheduler_wake.set()
    first._scheduler_thread.join(timeout=2)
    row = first.runs.get_run(run["run_id"], CLIENT_A)
    envelope = synthetic_envelope(session_id=run["session_id"], stream_id=run["stream_id"],
                                  source_time_s=2.5, suffix="registered-pending-restart")
    receipt = first.register_envelope(row, envelope)
    assert receipt["status"] == "pending"
    with sqlite3.connect(database) as db:
        task_id = db.execute("SELECT task_id FROM decision_tasks WHERE run_id=?", (run["run_id"],)).fetchone()[0]
        assert db.execute("SELECT status FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "pending"
    first.close()

    restarted = AgentRuntime(config, provider="mock", enable_network=False, resource_status={})
    try:
        latest = restarted.latest(run["run_id"], CLIENT_A)
        assert latest["run_status"] == "stopped"
        assert latest["projection"]["current_applicable"] is False
        assert latest["projection"]["current"] is None
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT status FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "cancelled"
            assert db.execute("SELECT COUNT(*) FROM decision_attempts WHERE task_id=?", (task_id,)).fetchone()[0] == 0
            assert db.execute("SELECT stopped FROM decision_streams WHERE session_id=? AND stream_id=?",
                              (run["session_id"], run["stream_id"])).fetchone()[0] == 1
    finally:
        restarted.close()

    repeated = AgentRuntime(config, provider="mock", enable_network=False, resource_status={})
    try:
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT status FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()[0] == "cancelled"
            assert db.execute("SELECT COUNT(*) FROM decision_attempts WHERE task_id=?", (task_id,)).fetchone()[0] == 0
    finally:
        repeated.close()


def test_owner_queue_timeout_cancels_queued_command_and_isolates_query_errors(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = DecisionLoop.latest

    def blocked_latest(self, session_id, stream_id):
        if session_id == "blocked-session":
            entered.set()
            assert release.wait(2)
        if session_id == "broken-session":
            raise RuntimeError("injected query fault")
        return original(self, session_id, stream_id)

    monkeypatch.setattr(DecisionLoop, "latest", blocked_latest)
    owner = _OwnerThread(str(tmp_path / "owner.sqlite3"), DecisionConfig(), "off", "owner-test", False,
                         queue_capacity=1)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(owner.call, "latest", "blocked-session", "stream-a", timeout=2)
            assert entered.wait(1)
            with pytest.raises(AgentApiError, match="did not respond"):
                owner.call("latest", "queued-session", "stream-b", timeout=0.02)
            release.set()
            assert first.result(timeout=2)["session_id"] == "blocked-session"

        with pytest.raises(RuntimeError, match="injected query fault"):
            owner.call("latest", "broken-session", "stream-a")
        assert owner.thread.is_alive()
        assert owner.call("latest", "healthy-session", "stream-a")["session_id"] == "healthy-session"
    finally:
        release.set()
        owner.close()


def test_owner_timeout_after_observation_commit_is_audited_and_exact_retry_is_idempotent(
        tmp_path, monkeypatch):
    committed, release = threading.Event(), threading.Event()
    original = DecisionLoop.observe
    first_call = True

    def commit_then_wait(self, *args, **kwargs):
        nonlocal first_call
        receipt = original(self, *args, **kwargs)
        if first_call:
            first_call = False
            committed.set()
            assert release.wait(2)
        return receipt

    monkeypatch.setattr(DecisionLoop, "observe", commit_then_wait)
    database = str(tmp_path / "owner-observe.sqlite3")
    owner = _OwnerThread(database, DecisionConfig(), "off", "owner-observe-test", False)
    envelope = synthetic_envelope(session_id="session-owner", stream_id="stream-owner", source_time_s=1,
                                  suffix="owner-post-commit")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(owner.call, "observe", envelope, run_id="run-owner", timeout=0.05)
            assert committed.wait(1)
            with pytest.raises(AgentApiError, match="did not respond"):
                first.result(timeout=1)
            release.set()
            duplicate = owner.call("observe", envelope, run_id="run-owner", timeout=2)
        assert duplicate["status"] == "duplicate"
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT COUNT(*) FROM decision_observations").fetchone()[0] == 1
        assert owner.thread.is_alive()
    finally:
        release.set()
        owner.close()


def test_feedback_timeout_after_commit_keeps_owner_alive_and_request_idempotent(tmp_path, monkeypatch):
    committed, release = threading.Event(), threading.Event()
    original = DecisionLoop.feedback

    def commit_then_wait(self, decision_id, **kwargs):
        receipt = original(self, decision_id, **kwargs)
        if kwargs.get("request_id") == "owner-feedback-post-commit":
            committed.set()
            assert release.wait(2)
        return receipt

    monkeypatch.setattr(DecisionLoop, "feedback", commit_then_wait)
    database = str(tmp_path / "owner-feedback.sqlite3")
    config = DecisionConfig(db_path=database, provider="mock", model="test-mock")
    owner = _OwnerThread(database, config, "mock", "owner-feedback-test", False)

    class RunStore:
        @staticmethod
        def feedback_belongs_to_run(*_args):
            return True

    owner.attach_run_store(RunStore())
    envelope = synthetic_envelope(session_id="session-feedback", stream_id="stream-feedback", source_time_s=10,
                                  suffix="owner-feedback")
    try:
        observed = owner.call("observe", envelope, run_id="run-feedback")
        assert owner.call("run_one")["status"] == "accepted"
        args = (observed["decision_id"], "run-feedback", "session-feedback", "stream-feedback",
                {"label": "fall", "request_id": "owner-feedback-post-commit", "expected_version": 0})
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(owner.call, "feedback_for_run", *args, timeout=0.05)
            assert committed.wait(1)
            with pytest.raises(AgentApiError, match="did not respond"):
                first.result(timeout=1)
            release.set()
            history = owner.call("feedback_history", observed["decision_id"])
            replay = owner.call("feedback_for_run", *args)
        assert len(history) == 1
        assert replay["version"] == 1
        assert owner.thread.is_alive()
    finally:
        release.set()
        owner.close()


def _headers(port, client=CLIENT_A):
    origin = f"http://127.0.0.1:{port}"
    return {"Host": f"127.0.0.1:{port}", "Origin": origin,
            "X-Agent-Origin": origin, "X-Session-Token": "test-session-token",
            "X-Client-Session-Id": client}


def _dispatch(handler, method, path, body=None, headers=None):
    encoded = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else b""
    request_headers = dict(headers or {})
    if isinstance(body, dict):
        request_headers["Content-Type"] = "application/json"
    if encoded:
        request_headers["Content-Length"] = str(len(encoded))
    request = handler.__new__(handler)
    request.headers = Message()
    for key, value in request_headers.items():
        request.headers[key] = value
    request.path = path
    request.rfile = io.BytesIO(encoded)
    replies = []
    request.reply = lambda code, payload, kind="application/json; charset=utf-8": replies.append((code, payload))
    getattr(request, method)()
    assert replies
    return replies[-1]


def test_http_api_auth_cross_run_frame_dedupe_and_run_header_compatibility(tmp_path):
    runtime = AgentRuntime(DecisionConfig(db_path=str(tmp_path / "http.sqlite3")), provider="off",
                           enable_network=False, resource_status={"fall_head": {"status": "ready"}})
    engine = FakeEngine(delay=0.08)
    port = 18780
    handler = service.make_handler(engine, port, "test-session-token", runtime)
    try:
        status, denied = _dispatch(handler, "do_GET", "/api/v1/agent/runs/run-" + "1" * 32 + "/latest",
                                   headers={"Host": f"127.0.0.1:{port}",
                                            "X-Session-Token": "test-session-token"})
        assert status == 403

        headers = _headers(port)
        status, created = _dispatch(handler, "do_POST", "/api/v1/agent/runs",
                                    {"request_id": "http-run", "source_type": "camera"}, headers)
        assert status == 201
        status, forbidden = _dispatch(handler, "do_GET", f"/api/v1/agent/runs/{created['run_id']}/latest",
                                      headers=_headers(port, CLIENT_B))
        assert status == 404
        status, latest = _dispatch(handler, "do_GET", f"/api/v1/agent/runs/{created['run_id']}/latest",
                                   headers=headers)
        assert status == 200 and latest["projection"]["fall_probability"] is None

        ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), 80, dtype=np.uint8))
        assert ok
        image = encoded.tobytes()
        frame_headers = {**headers, "Content-Type": "image/jpeg", "X-Stream-Id": created["stream_id"],
                         "X-Source-Time": "1.0", "X-Pose-Model": "yolo26n-pose",
                         "X-Source-Type": "camera", "X-Fall-Mode": "pose",
                         "X-Agent-Run-Id": created["run_id"]}
        first = _dispatch(handler, "do_POST", "/infer", image, frame_headers)
        second = _dispatch(handler, "do_POST", "/infer", image, frame_headers)
        assert first[0] == second[0] == 200
        assert engine.calls == 1
        assert first[1]["agent"]["snapshot_id"] == second[1]["agent"]["snapshot_id"]

        ok, other_encoded = cv2.imencode(".jpg", np.full((32, 32, 3), 160, dtype=np.uint8))
        assert ok
        status, conflict = _dispatch(handler, "do_POST", "/infer", other_encoded.tobytes(), frame_headers)
        assert status == 409
        assert engine.calls == 1

        status, stopped = _dispatch(handler, "do_POST", f"/api/v1/agent/runs/{created['run_id']}/stop", {}, headers)
        assert status == 200 and stopped["status"] == "stopped"
    finally:
        runtime.close()


def _post_agent_frame(handler, run, *, port=18781, source_time=1.0, client=CLIENT_A, image=None):
    if image is None:
        ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), 90, dtype=np.uint8))
        assert ok
        image = encoded.tobytes()
    headers = {**_headers(port, client), "Content-Type": "image/jpeg", "X-Stream-Id": run["stream_id"],
               "X-Source-Time": str(source_time), "X-Pose-Model": "yolo26n-pose",
               "X-Source-Type": run["source_type"], "X-Fall-Mode": "pose",
               "X-Agent-Run-Id": run["run_id"]}
    return _dispatch(handler, "do_POST", "/infer", image, headers), image


def test_visual_result_replay_repairs_agent_registration_without_rerunning_inference(tmp_path, monkeypatch):
    database = str(tmp_path / "registration-repair.sqlite3")
    runtime = AgentRuntime(DecisionConfig(db_path=database), provider="off", enable_network=False,
                           resource_status={"fall_head": {"status": "ready"}})
    engine = FakeEngine()
    handler = service.make_handler(engine, 18781, "test-session-token", runtime)
    run, _ = runtime.create_run(client_session_id=CLIENT_A, request_id="registration-repair", source_type="video")
    register = runtime.register_envelope
    injected = True

    def fail_once(*args, **kwargs):
        nonlocal injected
        if injected:
            injected = False
            raise RuntimeError("injected after visual persistence")
        return register(*args, **kwargs)

    monkeypatch.setattr(runtime, "register_envelope", fail_once)
    try:
        (status, first), image = _post_agent_frame(handler, run)
        assert status == 200
        assert first["agent"]["status"] == "pending"
        assert engine.calls == 1
        with sqlite3.connect(database) as db:
            row = db.execute("SELECT status,agent_registration_status,observation_envelope_json "
                             "FROM agent_frame_receipts WHERE run_id=?", (run["run_id"],)).fetchone()
            assert row[0:2] == ("completed", "pending")
            assert row[2]
            assert db.execute("SELECT COUNT(*) FROM decision_observations").fetchone()[0] == 0

        monkeypatch.setattr(runtime, "register_envelope", register)
        second, _ = _post_agent_frame(handler, run, image=image)
        assert second[0] == 200
        assert second[1]["agent"]["status"] in {"skipped", "insufficient"}
        assert engine.calls == 1
        third, _ = _post_agent_frame(handler, run, image=image)
        assert third == second
        with sqlite3.connect(database) as db:
            row = db.execute("SELECT status,agent_registration_status FROM agent_frame_receipts "
                             "WHERE run_id=?", (run["run_id"],)).fetchone()
            assert row == ("completed", "registered")
            assert db.execute("SELECT COUNT(*) FROM decision_observations").fetchone()[0] == 1
    finally:
        runtime.close()


def test_restart_recovers_pending_registration_as_stale_history(tmp_path, monkeypatch):
    database = str(tmp_path / "registration-restart.sqlite3")
    first_runtime = AgentRuntime(DecisionConfig(db_path=database), provider="off", enable_network=False,
                                 resource_status={"fall_head": {"status": "ready"}}, network_budget=6)
    engine = FakeEngine()
    handler = service.make_handler(engine, 18782, "test-session-token", first_runtime)
    run, _ = first_runtime.create_run(client_session_id=CLIENT_A, request_id="registration-restart", source_type="video")

    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("injected before owner registration")

    monkeypatch.setattr(first_runtime, "register_envelope", fail_registration)
    (status, result), _ = _post_agent_frame(handler, run, port=18782)
    assert status == 200 and result["agent"]["status"] == "pending"
    assert engine.calls == 1
    first_runtime.close()

    restarted = AgentRuntime(DecisionConfig(db_path=database), provider="off", enable_network=False,
                             resource_status={"fall_head": {"status": "ready"}}, network_budget=6)
    try:
        recovered = restarted.runs.get_run(run["run_id"], CLIENT_A)
        assert recovered["status"] == "stopped"
        with sqlite3.connect(database) as db:
            row = db.execute("SELECT status,agent_registration_status FROM agent_frame_receipts "
                             "WHERE run_id=?", (run["run_id"],)).fetchone()
            assert row == ("completed", "stale")
            assert db.execute("SELECT COUNT(*) FROM decision_observations WHERE stale=1").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM decision_tasks").fetchone()[0] == 0
        latest = restarted.latest(run["run_id"], CLIENT_A)
        assert latest["run_status"] == "stopped"
        assert latest["projection"]["current_applicable"] is False
        assert engine.calls == 1
        assert restarted.runs.http_budget() == {"maximum": 6, "used": 0}
    finally:
        restarted.close()
