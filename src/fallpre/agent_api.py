"""Local authenticated API for durable, asynchronous video decision runs."""
from __future__ import annotations

import hashlib
import json
import queue
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapters.aliyun_decision import DecisionProviderError
from .decision_config import DecisionConfig
from .decision_evidence import canonical_json
from .decision_loop import DecisionLoop, DecisionWorker
from .decision_trace import trace_decision
from .live_decision import live_observation_envelope

_CLIENT_ID = re.compile(r"^[A-Za-z0-9-]{16,64}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RUN_ID = re.compile(r"^run-[a-f0-9]{32}$")
_DECISION_ID = re.compile(r"^decision-[a-f0-9]{32}$")
DEFAULT_HTTP_SENDS = 6
MAX_HTTP_SENDS = 1000


class AgentApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        super().__init__(message)


class AgentRunStore:
    """Small local run registry and frame idempotency log in the decision DB."""

    def __init__(self, database_path: str | Path, *, network_budget: int = DEFAULT_HTTP_SENDS,
                 raise_http_budget: bool = False) -> None:
        self.path = str(database_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize(network_budget, raise_http_budget)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _initialize(self, network_budget: int, raise_http_budget: bool) -> None:
        db = self._connect()
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    client_session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    source_type TEXT NOT NULL CHECK(source_type IN ('camera','video')),
                    request_digest TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    created_at REAL NOT NULL,
                    receipt_json TEXT,
                    UNIQUE(client_session_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS agent_runs_owner ON agent_runs(client_session_id, created_at);
                CREATE TABLE IF NOT EXISTS agent_frame_receipts (
                    run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
                    source_time REAL NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('in_flight','completed','failed','interrupted')),
                    response_json TEXT,
                    observation_envelope_json TEXT,
                    agent_registration_status TEXT NOT NULL DEFAULT 'not_started',
                    agent_registration_error TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    PRIMARY KEY(run_id, source_time)
                );
                CREATE TABLE IF NOT EXISTS agent_http_budget (
                    budget_key TEXT PRIMARY KEY,
                    maximum INTEGER NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
            """)
            receipt_columns = {row["name"] for row in db.execute("PRAGMA table_info(agent_frame_receipts)")}
            for name, declaration in (
                ("observation_envelope_json", "TEXT"),
                ("agent_registration_status", "TEXT NOT NULL DEFAULT 'not_started'"),
                ("agent_registration_error", "TEXT"),
            ):
                if name not in receipt_columns:
                    db.execute(f"ALTER TABLE agent_frame_receipts ADD COLUMN {name} {declaration}")
            now = time.time()
            maximum = max(0, min(MAX_HTTP_SENDS, int(network_budget)))
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT maximum,used FROM agent_http_budget WHERE budget_key='stage04'").fetchone()
            if row is None:
                db.execute("INSERT INTO agent_http_budget VALUES('stage04',?,0,?)",
                           (maximum, now))
            elif raise_http_budget:
                db.execute("UPDATE agent_http_budget SET maximum=max(maximum,?),updated_at=? "
                           "WHERE budget_key='stage04'", (maximum, now))
            else:
                db.execute("UPDATE agent_http_budget SET maximum=min(maximum,?),updated_at=? "
                           "WHERE budget_key='stage04'", (maximum, now))
            # Live visual state is intentionally not recovered across service boots.
            db.execute("UPDATE agent_runs SET status='stopped',stop_reason='SERVICE_RESTARTED' "
                       "WHERE status='active'")
            db.execute("UPDATE agent_frame_receipts SET status='interrupted',completed_at=? "
                       "WHERE status='in_flight'", (now,))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, *, client_session_id: str, request_id: str, source_type: str,
                   receipt: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        digest = hashlib.sha256(canonical_json({"source_type": source_type}).encode()).hexdigest()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM agent_runs WHERE client_session_id=? AND request_id=?",
                               (client_session_id, request_id)).fetchone()
            if prior:
                if prior["request_digest"] != digest:
                    raise AgentApiError(409, "REQUEST_ID_CONFLICT", "request_id was reused with different content")
                row = dict(prior)
                saved = json.loads(row["receipt_json"]) if row.get("receipt_json") else None
                db.commit()
                return (saved or self.run_receipt(row, receipt["public_status"]), True)
            run_id = "run-" + uuid.uuid4().hex
            session_id = "session-" + uuid.uuid4().hex
            stream_id = uuid.uuid4().hex
            db.execute("INSERT INTO agent_runs(run_id,client_session_id,request_id,source_type,request_digest,"
                       "session_id,stream_id,status,created_at) VALUES(?,?,?,?,?,?,?,'active',?)",
                       (run_id, client_session_id, request_id, source_type, digest, session_id, stream_id, time.time()))
            row = db.execute("SELECT * FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
            result = self.run_receipt(dict(row), receipt["public_status"])
            db.execute("UPDATE agent_runs SET receipt_json=? WHERE run_id=?",
                       (canonical_json(result), run_id))
            db.commit()
            return result, False
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def run_receipt(row: dict[str, Any], public_status: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "agent-run.v1",
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "stream_id": row["stream_id"],
            "source_type": row["source_type"],
            "provider": public_status["provider"],
            "model": public_status["model"],
            "provider_status": public_status,
            "stages": public_status.get("stages", {}),
            "status": row["status"],
            "stop_reason": row.get("stop_reason"),
        }

    def get_run(self, run_id: str, client_session_id: str, *, require_active: bool = False) -> dict[str, Any]:
        db = self._connect()
        try:
            row = db.execute("SELECT * FROM agent_runs WHERE run_id=? AND client_session_id=?",
                             (run_id, client_session_id)).fetchone()
            if row is None:
                raise AgentApiError(404, "RUN_NOT_FOUND", "Agent run was not found")
            result = dict(row)
            if require_active and result["status"] != "active":
                raise AgentApiError(409, "RUN_STOPPED", "Agent run is stopped; create a new run")
            return result
        finally:
            db.close()

    def stopped_runs(self) -> list[dict[str, Any]]:
        """Return every non-active run whose decision stream must stay terminal."""
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT run_id,session_id,stream_id,status,stop_reason FROM agent_runs "
                "WHERE status!='active' ORDER BY created_at,run_id").fetchall()
            return [dict(row) for row in rows]
        finally:
            db.close()

    def stop_run(self, run_id: str, client_session_id: str, reason: str = "STOP_REQUESTED") -> dict[str, Any]:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_runs WHERE run_id=? AND client_session_id=?",
                             (run_id, client_session_id)).fetchone()
            if row is None:
                raise AgentApiError(404, "RUN_NOT_FOUND", "Agent run was not found")
            db.execute("UPDATE agent_runs SET status='stopped',stop_reason=? WHERE run_id=?",
                       (reason, run_id))
            db.commit()
            return dict(row)
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def claim_frame(self, *, run_id: str, source_time: float, image_sha256: str,
                    options: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        options_json = canonical_json(options)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_frame_receipts WHERE run_id=? AND source_time=?",
                             (run_id, source_time)).fetchone()
            if row is not None:
                if row["image_sha256"] != image_sha256 or row["options_json"] != options_json:
                    raise AgentApiError(409, "FRAME_TIME_CONFLICT",
                                        "source_time was already used with different image or inference options")
                db.commit()
                if (row["status"] == "completed" and row["response_json"]
                        and row["agent_registration_status"] == "pending"
                        and row["observation_envelope_json"]):
                    http_status, payload = self._decode_frame_response(row["response_json"])
                    return "registration_pending", {
                        "http_status": http_status,
                        "payload": payload,
                        "envelope": json.loads(row["observation_envelope_json"]),
                    }
                if row["status"] == "completed" and row["response_json"]:
                    return "completed", self._decode_frame_response(row["response_json"])
                if row["status"] in {"failed", "interrupted"}:
                    return row["status"], self._decode_frame_response(row["response_json"]) \
                        if row["response_json"] else None
                return "in_flight", None
            db.execute("INSERT INTO agent_frame_receipts(run_id,source_time,image_sha256,options_json,status,created_at) "
                       "VALUES(?,?,?,?, 'in_flight',?)", (run_id, source_time, image_sha256, options_json, time.time()))
            db.commit()
            return "new", None
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _decode_frame_response(encoded: str) -> tuple[int, dict[str, Any]]:
        value = json.loads(encoded)
        if isinstance(value, dict) and value.get("schema_version") == "agent-frame-response.v1":
            return int(value["http_status"]), value["payload"]
        # Retain compatibility with receipts written by an early P3 preview.
        return 200, value

    def finish_frame(self, *, run_id: str, source_time: float, response: dict[str, Any],
                     failed: bool = False, http_status: int = 200) -> None:
        db = self._connect()
        try:
            db.execute("UPDATE agent_frame_receipts SET status=?,response_json=?,completed_at=? "
                       "WHERE run_id=? AND source_time=? AND status='in_flight'",
                       ("failed" if failed else "completed", canonical_json({
                           "schema_version": "agent-frame-response.v1", "http_status": http_status,
                           "payload": response,
                       }), time.time(), run_id, source_time))
        finally:
            db.close()

    def persist_visual_result(self, *, run_id: str, source_time: float, response: dict[str, Any],
                              envelope: dict[str, Any]) -> None:
        """Durably separate a completed visual pass from Agent registration."""
        encoded_response = canonical_json({
            "schema_version": "agent-frame-response.v1", "http_status": 200, "payload": response,
        })
        encoded_envelope = canonical_json(envelope)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,observation_envelope_json FROM agent_frame_receipts "
                             "WHERE run_id=? AND source_time=?", (run_id, source_time)).fetchone()
            if row is None:
                raise AgentApiError(404, "FRAME_RECEIPT_NOT_FOUND", "frame receipt was not found")
            if row["status"] == "completed" and row["observation_envelope_json"]:
                if row["observation_envelope_json"] != encoded_envelope:
                    raise AgentApiError(409, "FRAME_ENVELOPE_CONFLICT", "frame envelope already differs")
                db.commit()
                return
            if row["status"] != "in_flight":
                raise AgentApiError(409, "FRAME_NOT_IN_FLIGHT", "frame receipt is no longer in flight")
            db.execute("UPDATE agent_frame_receipts SET status='completed',response_json=?,"
                       "observation_envelope_json=?,agent_registration_status='pending',"
                       "agent_registration_error=NULL,completed_at=? WHERE run_id=? AND source_time=?",
                       (encoded_response, encoded_envelope, time.time(), run_id, source_time))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def finish_agent_registration(self, *, run_id: str, source_time: float, response: dict[str, Any],
                                  agent_receipt: dict[str, Any]) -> None:
        response = dict(response)
        response["agent"] = agent_receipt
        registration_status = "stale" if agent_receipt.get("status") == "stale" else "registered"
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE agent_frame_receipts SET response_json=?,agent_registration_status=?,"
                       "agent_registration_error=NULL WHERE run_id=? AND source_time=? "
                       "AND status='completed' AND agent_registration_status='pending'",
                       (canonical_json({"schema_version": "agent-frame-response.v1", "http_status": 200,
                                        "payload": response}), registration_status, run_id, source_time))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def keep_agent_registration_pending(self, *, run_id: str, source_time: float,
                                        response: dict[str, Any], error_code: str) -> None:
        response = dict(response)
        response["agent"] = {"schema_version": "agent-receipt.v1", "status": "pending",
                             "reason_codes": ["LOCAL_DECISION_REGISTRATION_PENDING"]}
        db = self._connect()
        try:
            db.execute("UPDATE agent_frame_receipts SET response_json=?,agent_registration_status='pending',"
                       "agent_registration_error=? WHERE run_id=? AND source_time=? AND status='completed'",
                       (canonical_json({"schema_version": "agent-frame-response.v1", "http_status": 200,
                                        "payload": response}), error_code[:64], run_id, source_time))
        finally:
            db.close()

    def pending_registrations(self) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT f.run_id,f.source_time,f.response_json,f.observation_envelope_json,"
                "r.client_session_id,r.session_id,r.stream_id,r.source_type,r.status AS run_status "
                "FROM agent_frame_receipts f JOIN agent_runs r USING(run_id) "
                "WHERE f.status='completed' AND f.agent_registration_status='pending' "
                "AND f.observation_envelope_json IS NOT NULL ORDER BY f.created_at,f.source_time").fetchall()
            output = []
            for row in rows:
                item = dict(row)
                item["response"] = self._decode_frame_response(item.pop("response_json"))
                item["envelope"] = json.loads(item.pop("observation_envelope_json"))
                output.append(item)
            return output
        finally:
            db.close()

    def delete_unstarted_frame(self, *, run_id: str, source_time: float) -> None:
        db = self._connect()
        try:
            db.execute("DELETE FROM agent_frame_receipts WHERE run_id=? AND source_time=? AND status='in_flight'",
                       (run_id, source_time))
        finally:
            db.close()

    def reserve_http_send(self) -> bool:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT maximum,used FROM agent_http_budget WHERE budget_key='stage04'").fetchone()
            if row is None or row["used"] >= row["maximum"]:
                db.commit()
                return False
            db.execute("UPDATE agent_http_budget SET used=used+1,updated_at=? WHERE budget_key='stage04'",
                       (time.time(),))
            db.commit()
            return True
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def http_budget(self) -> dict[str, int]:
        db = self._connect()
        try:
            row = db.execute("SELECT maximum,used FROM agent_http_budget WHERE budget_key='stage04'").fetchone()
            return {"maximum": int(row["maximum"]), "used": int(row["used"])} if row else {
                "maximum": 0, "used": 0}
        finally:
            db.close()

    def feedback_belongs_to_run(self, decision_id: str, run_id: str, session_id: str, stream_id: str) -> bool:
        db = self._connect()
        try:
            row = db.execute("SELECT 1 FROM decision_tasks WHERE decision_id=? AND run_id=? AND session_id=? "
                             "AND stream_id=? UNION SELECT 1 FROM decision_observations WHERE decision_id=? "
                             "AND run_id=? AND session_id=? AND stream_id=? LIMIT 1",
                             (decision_id, run_id, session_id, stream_id, decision_id, run_id, session_id, stream_id)).fetchone()
            return row is not None
        finally:
            db.close()


class _OwnerThread:
    """Own all request-side DecisionLoop/SQLite calls on one thread."""

    DEFAULT_QUEUE_CAPACITY = 64

    def __init__(self, database_path: str, config: DecisionConfig, provider: str,
                 service_instance_id: str, enable_network: bool,
                 *, queue_capacity: int = DEFAULT_QUEUE_CAPACITY) -> None:
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        self.database_path = database_path
        self.config = config
        self.provider = provider
        self.service_instance_id = service_instance_id
        self.enable_network = enable_network
        self.commands: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._submit_lock = threading.Lock()
        self._closing = threading.Event()
        self._ready = Future()
        self.thread = threading.Thread(target=self._serve, name="fallpre-agent-api-owner", daemon=True)
        self.thread.start()
        self._ready.result(timeout=15)

    def _serve(self) -> None:
        loop = None
        try:
            loop = DecisionLoop(self.database_path, config=self.config, provider=self.provider,
                                service_instance_id=self.service_instance_id,
                                enable_network=self.enable_network)
            self._ready.set_result(True)
        except BaseException as error:
            self._ready.set_exception(error)
            return
        try:
            while True:
                item = self.commands.get()
                if item is None:
                    return
                method, args, kwargs, future = item
                # Once execution starts, caller timeouts must not cancel the
                # Future: an in-flight SQLite mutation still has to be recorded.
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if method == "feedback_for_run":
                        decision_id, run_id, session_id, stream_id, payload = args
                        if not self._runtime_store.feedback_belongs_to_run(
                                decision_id, run_id, session_id, stream_id):
                            raise AgentApiError(404, "DECISION_NOT_FOUND", "decision was not found in this run")
                        result = loop.feedback(decision_id, **payload)
                    elif method == "decisions_after":
                        result = loop.list_decisions_after(*args, **kwargs)
                    elif method == "feedback_history":
                        result = loop.store.list_feedback(*args, **kwargs)
                    else:
                        result = getattr(loop, method)(*args, **kwargs)
                    future.set_result(result)
                except BaseException as error:
                    # A failing command is isolated to its caller. In
                    # particular, never let an invalid Future state terminate
                    # the database-owning thread.
                    if not future.done():
                        future.set_exception(error)
        finally:
            loop.close()

    def attach_run_store(self, store: AgentRunStore) -> None:
        self._runtime_store = store

    def call(self, method: str, *args: Any, timeout: float = 3.0, **kwargs: Any) -> Any:
        future: Future = Future()
        with self._submit_lock:
            if self._closing.is_set():
                raise AgentApiError(503, "LOCAL_DECISION_CLOSED", "local decision store is closing")
            try:
                self.commands.put_nowait((method, args, kwargs, future))
            except queue.Full:
                raise AgentApiError(503, "LOCAL_DECISION_BUSY", "local decision queue is full") from None
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # If still queued this prevents execution. If already running,
            # cancel() safely returns False and the owner completes the audit.
            future.cancel()
            raise AgentApiError(503, "LOCAL_DECISION_BUSY", "local decision store did not respond in time") from None

    def close(self) -> None:
        with self._submit_lock:
            if self._closing.is_set():
                return
            self._closing.set()
            while True:
                try:
                    item = self.commands.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[3].cancel()
            self.commands.put_nowait(None)
        self.thread.join(timeout=5)


class _PersistedBudgetTransport:
    def __init__(self, store: AgentRunStore, inner: Any) -> None:
        self.store = store
        self.inner = inner

    def post_json(self, **kwargs: Any) -> Any:
        if not self.store.reserve_http_send():
            raise DecisionProviderError("RUN_HTTP_BUDGET_EXHAUSTED", retryable=False)
        return self.inner.post_json(**kwargs)


class AgentRuntime:
    """Run/session API, frame dedupe, durable loop and asynchronous provider."""

    def __init__(self, config: DecisionConfig, *, provider: str, enable_network: bool,
                 resource_status: dict[str, Any], network_budget: int = DEFAULT_HTTP_SENDS,
                 raise_http_budget: bool = False,
                 stop_stream: Any | None = None) -> None:
        database_path = Path(config.db_path)
        if not database_path.is_absolute():
            database_path = Path(__file__).resolve().parents[2] / database_path
        self.database_path = str(database_path)
        self.config = replace(config, db_path=self.database_path, provider=provider)
        self.provider = provider
        self.enable_network = enable_network
        self.resource_status = resource_status
        self.stop_stream_callback = stop_stream
        if provider == "aliyun":
            self.config.require_aliyun_configuration(provider=provider)
        self.runs = AgentRunStore(self.database_path, network_budget=network_budget,
                                  raise_http_budget=raise_http_budget)
        self.service_instance_id = uuid.uuid4().hex
        self.worker: DecisionWorker | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_stop = threading.Event()
        self._scheduler_wake = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame_futures: dict[tuple[str, float], Future] = {}
        self.owner = _OwnerThread(self.database_path, self.config, provider,
                                  self.service_instance_id, enable_network)
        self.owner.attach_run_store(self.runs)
        try:
            # Reconcile stream/task state before a worker can claim any durable work.
            self._recover_stopped_runs()
            self._recover_pending_registrations()
        except BaseException:
            self.owner.close()
            raise
        if provider in {"mock", "aliyun"}:
            transport = None
            if provider == "aliyun":
                from .adapters.aliyun_decision import HTTPSJsonTransport
                transport = _PersistedBudgetTransport(self.runs, HTTPSJsonTransport())
            self.worker = DecisionWorker(self.database_path, config=self.config, provider=provider,
                                         service_instance_id=self.service_instance_id,
                                         enable_network=enable_network, provider_transport=transport)
            self._scheduler_thread = threading.Thread(target=self._schedule, name="fallpre-agent-scheduler", daemon=True)
            self._scheduler_thread.start()

    def _schedule(self) -> None:
        while not self._scheduler_stop.is_set():
            self._scheduler_wake.wait(timeout=0.25)
            self._scheduler_wake.clear()
            if self.worker is None:
                continue
            while not self._scheduler_stop.is_set():
                try:
                    result = self.worker.run_one(timeout=None)
                except BaseException:
                    break
                if result is None:
                    break

    def public_status(self) -> dict[str, Any]:
        public = self.config.public_status()
        public["provider"] = self.provider
        public["network_enabled"] = self.enable_network
        public["decision_timeout_s"] = self.config.timeout_s
        public["http_budget"] = self.runs.http_budget()
        public["stages"] = self.resource_status
        return public

    def create_run(self, *, client_session_id: str, request_id: str, source_type: str) -> tuple[dict[str, Any], bool]:
        if not _CLIENT_ID.fullmatch(client_session_id):
            raise AgentApiError(400, "INVALID_CLIENT_SESSION", "invalid client session")
        if not _REQUEST_ID.fullmatch(request_id) or source_type not in {"camera", "video"}:
            raise AgentApiError(422, "INVALID_RUN_REQUEST", "request_id or source_type is invalid")
        status = self.public_status()
        result, duplicate = self.runs.create_run(client_session_id=client_session_id, request_id=request_id,
                                                 source_type=source_type, receipt={"public_status": status})
        if duplicate:
            # Keep the original public configuration snapshot, but expose current run state.
            row = self.runs.get_run(result["run_id"], client_session_id)
            result["status"] = row["status"]
            result["stop_reason"] = row["stop_reason"]
        return result, duplicate

    def begin_frame(self, *, run_id: str, client_session_id: str, stream_id: str,
                    source_type: str, source_time: float, image_bytes: bytes,
                    options: dict[str, Any]) -> tuple[str, dict[str, Any] | Future | None, dict[str, Any]]:
        run = self.runs.get_run(run_id, client_session_id, require_active=True)
        if run["stream_id"] != stream_id or run["source_type"] != source_type:
            raise AgentApiError(409, "RUN_STREAM_MISMATCH", "run, stream, and source type must match")
        key = (run_id, source_time)
        digest = hashlib.sha256(image_bytes).hexdigest()
        with self._frame_lock:
            status, cached = self.runs.claim_frame(run_id=run_id, source_time=source_time,
                                                   image_sha256=digest, options=options)
            if status == "new":
                future: Future = Future()
                self._frame_futures[key] = future
                return "new", None, run
            if status == "registration_pending":
                return "repair", cached, run
            if status == "in_flight":
                future = self._frame_futures.get(key)
                if future is not None:
                    return "waiting", future, run
                return "interrupted", None, run
            return status, cached, run

    def finish_frame(self, run: dict[str, Any], source_time: float, response: dict[str, Any], *,
                     failed: bool = False, http_status: int = 200) -> None:
        key = (run["run_id"], source_time)
        self.runs.finish_frame(run_id=run["run_id"], source_time=source_time, response=response,
                               failed=failed, http_status=http_status)
        with self._frame_lock:
            future = self._frame_futures.pop(key, None)
            if future is not None and not future.done():
                future.set_result((http_status, response))

    def _finish_frame_waiter(self, run_id: str, source_time: float, http_status: int,
                             response: dict[str, Any]) -> None:
        key = (run_id, source_time)
        with self._frame_lock:
            future = self._frame_futures.pop(key, None)
            if future is not None and not future.done():
                future.set_result((http_status, response))

    def abort_unstarted_frame(self, run: dict[str, Any], source_time: float) -> None:
        key = (run["run_id"], source_time)
        self.runs.delete_unstarted_frame(run_id=run["run_id"], source_time=source_time)
        with self._frame_lock:
            self._frame_futures.pop(key, None)

    def observe_live(self, run: dict[str, Any], result: dict[str, Any], *, source_time: float,
                     head_variant: str = "cd") -> dict[str, Any]:
        envelope, _ = self.make_live_envelope(run, result, source_time=source_time,
                                              head_variant=head_variant)
        return self.register_envelope(run, envelope, result=result)

    @staticmethod
    def make_live_envelope(run: dict[str, Any], result: dict[str, Any], *, source_time: float,
                           head_variant: str = "cd") -> tuple[dict[str, Any], dict[str, Any]]:
        return live_observation_envelope(
            result, run_id=run["run_id"], session_id=run["session_id"], stream_id=run["stream_id"],
            source_time_s=source_time, head_variant=head_variant)

    def persist_visual_result(self, run: dict[str, Any], result: dict[str, Any], envelope: dict[str, Any],
                              *, source_time: float) -> None:
        self.runs.persist_visual_result(run_id=run["run_id"], source_time=source_time,
                                        response=result, envelope=envelope)

    def register_envelope(self, run: dict[str, Any], envelope: dict[str, Any], *,
                          result: dict[str, Any] | None = None) -> dict[str, Any]:
        current_run = self.runs.get_run(run["run_id"], run["client_session_id"])
        allow_task = current_run["status"] == "active"
        receipt = self.owner.call(
            "observe", envelope, run_id=run["run_id"], allow_task=allow_task,
            force_stale=not allow_task,
            task_suppression_reason="RUN_STOPPED_BEFORE_REGISTRATION" if not allow_task else None,
        )
        if receipt.get("task_id"):
            self._scheduler_wake.set()
        projection = self.owner.call("latest", run["session_id"], run["stream_id"])
        status = "stale" if not allow_task or receipt.get("stale") else "pending" if receipt.get("task_id") else (
            "insufficient" if receipt.get("missing_reasons") else
            "skipped" if receipt.get("schedule_reason") == "PROVIDER_OFF" else "unavailable")
        if allow_task and projection.get("current_applicable"):
            status = "ready"
        elif allow_task and projection.get("latest_task_status") in {"pending", "running"}:
            status = "pending"
        agent_receipt = {
            "schema_version": "agent-receipt.v1",
            "run_id": run["run_id"],
            "snapshot_id": receipt.get("snapshot_id"),
            "decision_id": receipt.get("decision_id"),
            "task_id": receipt.get("task_id"),
            "status": status,
            "schedule_reason": receipt.get("schedule_reason"),
            "missing_reasons": receipt.get("missing_reasons", []),
            "trace": envelope.get("evidence_summary"),
            "visual_ms": (result or {}).get("inference_ms"),
            "projection_current": projection.get("current") if allow_task else None,
        }
        evidence = agent_receipt["trace"] or {}
        trace_decision("observation_registered", run_id=run["run_id"],
                       **{k: agent_receipt[k] for k in ("snapshot_id", "decision_id", "task_id", "status",
                                                        "schedule_reason", "missing_reasons")},
                       source_time_s=envelope["source_time_s"],
                       **{k: evidence.get(k) for k in ("source_window", "observed_frames",
                                                       "person_count", "pose_valid_fraction", "yolo_model_version",
                                                       "temporal_status", "temporal_score", "head_status",
                                                       "head_version", "head_raw_score")})
        return agent_receipt

    def finish_agent_registration(self, run: dict[str, Any], source_time: float,
                                  response: dict[str, Any], agent_receipt: dict[str, Any]) -> dict[str, Any]:
        updated = dict(response)
        updated["agent"] = agent_receipt
        self.runs.finish_agent_registration(run_id=run["run_id"], source_time=source_time,
                                            response=response, agent_receipt=agent_receipt)
        self._finish_frame_waiter(run["run_id"], source_time, 200, updated)
        return updated

    def keep_registration_pending(self, run: dict[str, Any], source_time: float,
                                  response: dict[str, Any], error: BaseException) -> dict[str, Any]:
        self.runs.keep_agent_registration_pending(run_id=run["run_id"], source_time=source_time,
                                                  response=response, error_code=type(error).__name__)
        updated = dict(response)
        updated["agent"] = {"schema_version": "agent-receipt.v1", "status": "pending",
                             "reason_codes": ["LOCAL_DECISION_REGISTRATION_PENDING"]}
        self._finish_frame_waiter(run["run_id"], source_time, 200, updated)
        return updated

    def repair_pending_frame(self, run: dict[str, Any], cached: dict[str, Any], source_time: float
                             ) -> tuple[int, dict[str, Any]]:
        response = cached["payload"]
        try:
            agent_receipt = self.register_envelope(run, cached["envelope"], result=response)
            response = self.finish_agent_registration(run, source_time, response, agent_receipt)
        except BaseException as error:
            response = self.keep_registration_pending(run, source_time, response, error)
        return int(cached["http_status"]), response

    def _recover_pending_registrations(self) -> None:
        """Reconcile persisted visual snapshots after a service restart as stale history."""
        for item in self.runs.pending_registrations():
            run = {key: item[key] for key in (
                "run_id", "client_session_id", "session_id", "stream_id", "source_type", "run_status")}
            run["status"] = item["run_status"]
            if run["status"] != "active":
                try:
                    self.owner.call("stop", run["session_id"], run["stream_id"],
                                    reason="REGISTRATION_RECOVERY_AFTER_STOP")
                except BaseException:
                    pass
            try:
                receipt = self.register_envelope(run, item["envelope"], result=item["response"][1])
                self.runs.finish_agent_registration(run_id=run["run_id"], source_time=item["source_time"],
                                                    response=item["response"][1], agent_receipt=receipt)
            except BaseException as error:
                self.runs.keep_agent_registration_pending(run_id=run["run_id"],
                                                          source_time=item["source_time"],
                                                          response=item["response"][1],
                                                          error_code=type(error).__name__)

    def _recover_stopped_runs(self) -> None:
        """Cancel durable stream work for stopped runs before the worker starts."""
        for run in self.runs.stopped_runs():
            reason = run.get("stop_reason") or "SERVICE_RESTARTED"
            self.owner.call("stop", run["session_id"], run["stream_id"], reason=reason)

    def latest(self, run_id: str, client_session_id: str) -> dict[str, Any]:
        run = self.runs.get_run(run_id, client_session_id)
        projection = self.owner.call("latest", run["session_id"], run["stream_id"])
        decision_id = ((projection.get("current") or {}).get("decision_id")
                      or (projection.get("latest_result") or {}).get("decision_id"))
        current = projection.get("current") or {}
        trace_decision("latest_projection", run_id=run_id, decision_id=current.get("decision_id"),
                       current_applicable=projection.get("current_applicable"),
                       fall_probability=current.get("fall_probability"))
        feedback = self.owner.call("feedback_history", decision_id) if decision_id else []
        return {"schema_version": "agent-latest.v1", "run_id": run_id, "run_status": run["status"],
                "source_type": run["source_type"], "feedback": feedback,
                "provider_status": self.public_status(),
                "projection": projection}

    def decisions_after(self, run_id: str, client_session_id: str, *, after_id: int, limit: int) -> dict[str, Any]:
        run = self.runs.get_run(run_id, client_session_id)
        rows = self.owner.call("decisions_after", run_id, run["session_id"], run["stream_id"],
                               after_id=after_id, limit=limit)
        for row in rows:
            row["feedback"] = self.owner.call("feedback_history", row["decision_id"])
        return {"schema_version": "agent-decisions-page.v1", "run_id": run_id,
                "items": rows, "next_after_id": rows[-1]["cursor"] if rows else after_id,
                "has_more": len(rows) == limit}

    def stop_run(self, run_id: str, client_session_id: str) -> dict[str, Any]:
        run = self.runs.stop_run(run_id, client_session_id)
        stopped = self.owner.call("stop", run["session_id"], run["stream_id"])
        if run["stream_id"] and self.stop_stream_callback:
            self.stop_stream_callback(run["stream_id"])
        return {"run_id": run_id, "status": "stopped", "stop": stopped}

    def feedback(self, run_id: str, client_session_id: str, decision_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not _DECISION_ID.fullmatch(decision_id):
            raise AgentApiError(422, "INVALID_DECISION_ID", "decision_id is invalid")
        run = self.runs.get_run(run_id, client_session_id)
        result = self.owner.call("feedback_for_run", decision_id, run_id, run["session_id"], run["stream_id"],
                                 payload)
        return result

    def feedback_history(self, decision_id: str, run_id: str, client_session_id: str) -> list[dict[str, Any]]:
        run = self.runs.get_run(run_id, client_session_id)
        if not self.runs.feedback_belongs_to_run(decision_id, run_id, run["session_id"], run["stream_id"]):
            return []
        return self.owner.call("feedback_history", decision_id)

    def close(self) -> None:
        self._scheduler_stop.set()
        self._scheduler_wake.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self.worker:
            self.worker.close(timeout=5)
        self.owner.close()
