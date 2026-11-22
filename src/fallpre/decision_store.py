"""Versioned SQLite journal for local decision observations, tasks and history.

Each ``DecisionStore`` owns one thread-bound SQLite connection. Provider work is
never performed by this module or while a transaction is open. A separate
store instance is required for each worker thread/process.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .decision_evidence import canonical_json


class DecisionConflict(ValueError):
    """An idempotency key was reused for different durable content."""


class DecisionVersionConflict(DecisionConflict):
    """Feedback expected a different historical feedback version."""


class InjectedStoreCrash(RuntimeError):
    """A test/demo crash inside the atomic observation+task transaction."""


class DecisionStore:
    SCHEMA_VERSION = 9

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._migrate()

    def _migrate(self) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("CREATE TABLE IF NOT EXISTS decision_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = self.db.execute("SELECT value FROM decision_schema_meta WHERE key='version'").fetchone()
            version = int(row["value"]) if row else 0
            if version > self.SCHEMA_VERSION:
                raise RuntimeError(f"decision database schema {version} is newer than supported")
            if version < 1:
                statements = (
                    """CREATE TABLE decision_runs (
                        run_id TEXT NOT NULL, session_id TEXT NOT NULL, created_at REAL NOT NULL,
                        PRIMARY KEY(run_id, session_id))""",
                    """CREATE TABLE decision_streams (
                        session_id TEXT NOT NULL, stream_id TEXT NOT NULL,
                        latest_observation_id INTEGER, latest_source_time REAL,
                        stopped INTEGER NOT NULL DEFAULT 0, stop_reason TEXT,
                        PRIMARY KEY(session_id, stream_id))""",
                    """CREATE TABLE decision_observations (
                        id INTEGER PRIMARY KEY, snapshot_id TEXT NOT NULL UNIQUE,
                        run_id TEXT NOT NULL, session_id TEXT NOT NULL,
                        stream_id TEXT NOT NULL, source_time REAL NOT NULL, digest TEXT NOT NULL,
                        envelope_json TEXT NOT NULL, evidence_json TEXT, evidence_digest TEXT NOT NULL,
                        missing_reasons_json TEXT NOT NULL, stale INTEGER NOT NULL,
                        decision_id TEXT, request_digest TEXT, task_id TEXT, schedule_reason TEXT,
                        provider TEXT, model TEXT, max_attempts INTEGER, max_reobservations INTEGER,
                        parent_decision_id TEXT, critical INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        UNIQUE(session_id, stream_id, source_time))""",
                    "CREATE INDEX decision_observation_latest ON decision_observations(session_id,stream_id,source_time DESC,id DESC)",
                    """CREATE TABLE decision_tasks (
                        task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, session_id TEXT NOT NULL,
                        stream_id TEXT NOT NULL, snapshot_id TEXT NOT NULL, decision_id TEXT NOT NULL,
                        evidence_json TEXT NOT NULL, evidence_digest TEXT NOT NULL, request_digest TEXT NOT NULL,
                        provider TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL
                            CHECK(status IN ('pending','running','completed','cancelled')),
                        attempt INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
                        max_reobservations INTEGER NOT NULL DEFAULT 1,
                        lease_token TEXT, lease_until REAL, available_at REAL NOT NULL,
                        source_time REAL NOT NULL, parent_decision_id TEXT,
                        critical INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
                        reason TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                        UNIQUE(session_id,stream_id,request_digest,provider,model))""",
                    "CREATE INDEX decision_task_claim ON decision_tasks(status,available_at,created_at)",
                    "CREATE INDEX decision_task_stream ON decision_tasks(session_id,stream_id,status,source_time DESC)",
                    """CREATE TABLE decision_task_observations (
                        task_id TEXT NOT NULL REFERENCES decision_tasks(task_id),
                        observation_id INTEGER NOT NULL REFERENCES decision_observations(id),
                        disposition TEXT NOT NULL, PRIMARY KEY(task_id,observation_id))""",
                    """CREATE TABLE decision_attempts (
                        task_id TEXT NOT NULL REFERENCES decision_tasks(task_id), attempt_no INTEGER NOT NULL,
                        lease_token TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
                        finished_at REAL, error_code TEXT, result_json TEXT,
                        PRIMARY KEY(task_id,attempt_no))""",
                    """CREATE TABLE decision_results (
                        decision_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES decision_tasks(task_id),
                        attempt_no INTEGER NOT NULL, result_json TEXT NOT NULL, source_time REAL NOT NULL,
                        window_end REAL NOT NULL, completed_at REAL NOT NULL, process_id TEXT NOT NULL,
                        accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
                        session_id TEXT NOT NULL, stream_id TEXT NOT NULL)""",
                    "CREATE UNIQUE INDEX decision_one_accepted_result ON decision_results(task_id) WHERE accepted=1",
                    "CREATE INDEX decision_result_order ON decision_results(session_id,stream_id,source_time DESC)",
                    """CREATE TABLE decision_workflows (
                        parent_decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, session_id TEXT NOT NULL,
                        stream_id TEXT NOT NULL, action_epoch INTEGER NOT NULL, generation INTEGER NOT NULL,
                        fall_target_epoch TEXT, trigger_source_time REAL NOT NULL, deadline_at REAL NOT NULL,
                        budget_total INTEGER NOT NULL, budget_used INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL CHECK(state IN ('waiting_observation','child_pending','finished','request_review','cancelled')),
                        child_decision_id TEXT, reason TEXT, updated_at REAL NOT NULL)""",
                    "CREATE INDEX decision_workflow_stream ON decision_workflows(session_id,stream_id,state,deadline_at)",
                    """CREATE TABLE decision_feedback_versions (
                        decision_id TEXT PRIMARY KEY, version INTEGER NOT NULL)""",
                    """CREATE TABLE decision_feedback_requests (
                        request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt_json TEXT NOT NULL)""",
                    """CREATE TABLE decision_feedback (
                        id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, run_id TEXT NOT NULL,
                        session_id TEXT NOT NULL, stream_id TEXT NOT NULL, label TEXT NOT NULL,
                        request_id TEXT NOT NULL UNIQUE, expected_version INTEGER NOT NULL,
                        version INTEGER NOT NULL, recorded_at REAL NOT NULL)""",
                )
                for statement in statements:
                    self.db.execute(statement)
                self.db.execute("INSERT INTO decision_schema_meta(key,value) VALUES('version','1')")
                version = 1
            if version < 2:
                observation_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_observations)")}
                if "snapshot_id" not in observation_columns:
                    self.db.execute("ALTER TABLE decision_observations ADD COLUMN snapshot_id TEXT")
                    self.db.execute("UPDATE decision_observations SET snapshot_id='snapshot-'||substr(digest,1,24) "
                                    "WHERE snapshot_id IS NULL")
                self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS decision_observation_snapshot "
                                "ON decision_observations(snapshot_id)")
                task_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_tasks)")}
                if "max_reobservations" not in task_columns:
                    self.db.execute("ALTER TABLE decision_tasks ADD COLUMN max_reobservations INTEGER NOT NULL DEFAULT 1")
                result_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_results)")}
                if "session_id" not in result_columns:
                    self.db.execute("ALTER TABLE decision_results ADD COLUMN session_id TEXT")
                if "stream_id" not in result_columns:
                    self.db.execute("ALTER TABLE decision_results ADD COLUMN stream_id TEXT")
                self.db.execute("UPDATE decision_results SET session_id=(SELECT session_id FROM decision_tasks t "
                                "WHERE t.task_id=decision_results.task_id) WHERE session_id IS NULL")
                self.db.execute("UPDATE decision_results SET stream_id=(SELECT stream_id FROM decision_tasks t "
                                "WHERE t.task_id=decision_results.task_id) WHERE stream_id IS NULL")
                self.db.execute("CREATE INDEX IF NOT EXISTS decision_result_order "
                                "ON decision_results(session_id,stream_id,source_time DESC)")
                self.db.execute("UPDATE decision_schema_meta SET value='2' WHERE key='version'")
                version = 2
            if version < 3:
                observation_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_observations)")}
                additions = {
                    "provider": "TEXT",
                    "model": "TEXT",
                    "max_attempts": "INTEGER",
                    "max_reobservations": "INTEGER",
                    "parent_decision_id": "TEXT",
                    "critical": "INTEGER NOT NULL DEFAULT 0",
                }
                for name, declaration in additions.items():
                    if name not in observation_columns:
                        self.db.execute(f"ALTER TABLE decision_observations ADD COLUMN {name} {declaration}")
                self.db.execute("UPDATE decision_schema_meta SET value='3' WHERE key='version'")
                version = 3
            if version < 4:
                result_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_results)")}
                if "service_instance_id" not in result_columns:
                    self.db.execute("ALTER TABLE decision_results ADD COLUMN service_instance_id TEXT")
                # Existing rows remain NULL and are history-only after this migration.
                self.db.execute("UPDATE decision_schema_meta SET value='4' WHERE key='version'")
                version = 4
            if version < 5:
                table_exists = self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_tasks'"
                ).fetchone()
                if table_exists:
                    task_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_tasks)")}
                    additions = {
                        "network_started_at": "REAL",
                        "network_deadline_at": "REAL",
                    }
                    for name, declaration in additions.items():
                        if name not in task_columns:
                            self.db.execute(f"ALTER TABLE decision_tasks ADD COLUMN {name} {declaration}")
                self.db.execute("UPDATE decision_schema_meta SET value='5' WHERE key='version'")
                version = 5
            if version < 6:
                table_exists = self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_attempts'"
                ).fetchone()
                if table_exists:
                    attempt_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_attempts)")}
                    additions = {
                        "request_id": "TEXT",
                        "http_status": "INTEGER",
                        "client_latency_ms": "REAL",
                        "service_latency_ms": "REAL",
                        "input_tokens": "INTEGER",
                        "retry_after_s": "REAL",
                    }
                    for name, declaration in additions.items():
                        if name not in attempt_columns:
                            self.db.execute(f"ALTER TABLE decision_attempts ADD COLUMN {name} {declaration}")
                self.db.execute("UPDATE decision_schema_meta SET value='6' WHERE key='version'")
                version = 6
            if version < 7:
                table_exists = self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_tasks'"
                ).fetchone()
                if table_exists:
                    task_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_tasks)")}
                    if "network_remaining_s" not in task_columns:
                        self.db.execute("ALTER TABLE decision_tasks ADD COLUMN network_remaining_s REAL")
                self.db.execute("UPDATE decision_schema_meta SET value='7' WHERE key='version'")
                version = 7
            if version < 8:
                table_exists = self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_attempts'"
                ).fetchone()
                if table_exists:
                    attempt_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_attempts)")}
                    additions = {"diagnostic_code": "TEXT", "field_path": "TEXT"}
                    for name, declaration in additions.items():
                        if name not in attempt_columns:
                            self.db.execute(f"ALTER TABLE decision_attempts ADD COLUMN {name} {declaration}")
                self.db.execute("UPDATE decision_schema_meta SET value='8' WHERE key='version'")
                version = 8
            if version < 9:
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS decision_result_cursors ("
                    "cursor INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL UNIQUE, "
                    "run_id TEXT NOT NULL, session_id TEXT NOT NULL, stream_id TEXT NOT NULL, "
                    "source_time REAL NOT NULL, created_at REAL NOT NULL)"
                )
                self.db.execute("CREATE INDEX IF NOT EXISTS decision_result_cursor_run "
                                "ON decision_result_cursors(run_id,session_id,stream_id,cursor)")
                # Preserve results created before this migration in a deterministic order.
                tables = {row[0] for row in self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                result_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(decision_results)")}
                task_columns = ({row["name"] for row in self.db.execute("PRAGMA table_info(decision_tasks)")}
                                if "decision_tasks" in tables else set())
                required_result = {"decision_id", "task_id", "session_id", "stream_id", "source_time", "completed_at", "accepted"}
                if "decision_tasks" in tables and required_result <= result_columns and "run_id" in task_columns:
                    old_results = self.db.execute(
                        "SELECT r.decision_id,t.run_id,r.session_id,r.stream_id,r.source_time,r.completed_at "
                        "FROM decision_results r JOIN decision_tasks t ON t.task_id=r.task_id "
                        "WHERE r.accepted=1 ORDER BY r.completed_at,r.decision_id"
                    ).fetchall()
                else:
                    old_results = []
                for result in old_results:
                    self.db.execute("INSERT OR IGNORE INTO decision_result_cursors(decision_id,run_id,session_id,"
                                    "stream_id,source_time,created_at) VALUES(?,?,?,?,?,?)",
                                    tuple(result))
                self.db.execute("UPDATE decision_schema_meta SET value='9' WHERE key='version'")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> DecisionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def latest_observation(self, session_id: str, stream_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM decision_observations WHERE session_id=? AND stream_id=? AND stale=0 "
            "ORDER BY source_time DESC,id DESC LIMIT 1", (session_id, stream_id)).fetchone()
        return self._observation_dict(row) if row else None

    def session_ids(self) -> list[str]:
        return [row[0] for row in self.db.execute(
            "SELECT DISTINCT session_id FROM decision_observations ORDER BY session_id")]

    def observations(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self.db.execute("SELECT * FROM decision_observations ORDER BY session_id,stream_id,source_time,id")
        else:
            rows = self.db.execute("SELECT * FROM decision_observations WHERE session_id=? "
                                   "ORDER BY stream_id,source_time,id", (session_id,))
        return [self._observation_dict(row) for row in rows]

    def summary_counts(self) -> dict[str, int]:
        names = {
            "observations": "decision_observations",
            "tasks": "decision_tasks",
            "accepted_results": "decision_results WHERE accepted=1",
            "attempts": "decision_attempts",
            "feedback": "decision_feedback",
            "workflows": "decision_workflows",
        }
        counts = {}
        for key, table in names.items():
            counts[key] = self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for row in self.db.execute("SELECT status,count(*) AS n FROM decision_tasks GROUP BY status"):
            counts[f"tasks_{row['status']}"] = row["n"]
        return counts

    @staticmethod
    def _observation_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["envelope"] = json.loads(data.pop("envelope_json"))
        evidence_json = data.pop("evidence_json")
        data["evidence"] = json.loads(evidence_json) if evidence_json else None
        data["missing_reasons"] = json.loads(data.pop("missing_reasons_json"))
        data["stale"] = bool(data["stale"])
        return data

    def active_workflow(self, session_id: str, stream_id: str, now: float) -> dict[str, Any] | None:
        self.expire_workflows(now)
        row = self.db.execute(
            "SELECT * FROM decision_workflows WHERE session_id=? AND stream_id=? "
            "AND state='waiting_observation' ORDER BY updated_at DESC LIMIT 1", (session_id, stream_id)).fetchone()
        return dict(row) if row else None

    def record_observation(
        self, *, run_id: str, envelope: dict[str, Any], evidence: dict[str, Any] | None,
        evidence_digest: str, missing_reasons: list[str], task: dict[str, Any] | None,
        min_interval_s: float, max_queue: int = 1000, now: float, end_of_stream: bool = False,
        fail_at: str | None = None, force_stale: bool = False,
        task_suppression_reason: str | None = None,
    ) -> dict[str, Any]:
        session_id = envelope["session_id"]
        stream_id = envelope["stream_id"]
        source_time = envelope["source_time_s"]
        encoded_envelope = canonical_json(envelope)
        digest = __import__("hashlib").sha256(encoded_envelope.encode("utf-8")).hexdigest()
        encoded_evidence = canonical_json(evidence) if evidence is not None else None
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute(
                "SELECT * FROM decision_observations WHERE session_id=? AND stream_id=? AND source_time=?",
                (session_id, stream_id, source_time)).fetchone()
            if prior:
                if prior["digest"] != digest:
                    raise DecisionConflict("conflicting content at an already persisted source timestamp")
                self.db.commit()
                prior_dict = self._observation_dict(prior)
                return {"status": "duplicate", "snapshot_id": prior["snapshot_id"],
                        "observation_id": prior["id"], "task_id": prior["task_id"],
                        "decision_id": prior["decision_id"], "request_digest": prior["request_digest"],
                        "stale": prior_dict["stale"], "decision": None,
                        "missing_reasons": prior_dict["missing_reasons"],
                        "schedule_reason": prior["schedule_reason"]}

            stream = self.db.execute("SELECT * FROM decision_streams WHERE session_id=? AND stream_id=?",
                                     (session_id, stream_id)).fetchone()
            stale = bool(force_stale or (stream and stream["latest_source_time"] is not None
                                         and source_time < stream["latest_source_time"]))
            snapshot_id = task["snapshot_id"] if task else f"snapshot-{digest[:24]}"
            self.db.execute("INSERT OR IGNORE INTO decision_runs VALUES(?,?,?)", (run_id, session_id, now))
            cur = self.db.execute(
                "INSERT INTO decision_observations(snapshot_id,run_id,session_id,stream_id,source_time,digest,envelope_json,"
                "evidence_json,evidence_digest,missing_reasons_json,stale,provider,model,max_attempts,max_reobservations,"
                "parent_decision_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, run_id, session_id, stream_id, source_time, digest, encoded_envelope, encoded_evidence,
                 evidence_digest, canonical_json(missing_reasons), int(stale), task.get("provider") if task else None,
                 task.get("model") if task else None, task.get("max_attempts") if task else None,
                 task.get("max_reobservations") if task else None,
                 task.get("parent_decision_id") if task else None, now))
            observation_id = cur.lastrowid
            if not stream:
                self.db.execute("INSERT INTO decision_streams(session_id,stream_id,latest_observation_id,latest_source_time) "
                                "VALUES(?,?,?,?)", (session_id, stream_id, observation_id,
                                                     None if stale else source_time))
            elif not stale:
                self.db.execute("UPDATE decision_streams SET latest_observation_id=?,latest_source_time=? "
                                "WHERE session_id=? AND stream_id=? AND stopped=0",
                                (observation_id, source_time, session_id, stream_id))

            if fail_at == "after_snapshot_insert_before_task":
                raise InjectedStoreCrash("injected crash after observation insert and before task registration")

            reason = None
            task_id = None
            decision_id = task.get("decision_id") if task else None
            request_digest = task.get("request_digest") if task else None
            parent = None
            critical = False
            waiting = self.db.execute(
                "SELECT * FROM decision_workflows WHERE session_id=? AND stream_id=? "
                "AND state='waiting_observation' ORDER BY updated_at DESC LIMIT 1", (session_id, stream_id)).fetchone()
            child_pending = self.db.execute(
                "SELECT * FROM decision_workflows WHERE session_id=? AND stream_id=? "
                "AND state='child_pending' ORDER BY updated_at DESC LIMIT 1", (session_id, stream_id)).fetchone()
            if stream and stream["stopped"]:
                reason = "STREAM_STOPPED"
            elif stale:
                reason = "OUT_OF_ORDER_OBSERVATION"
            elif end_of_stream and waiting:
                parent = waiting["parent_decision_id"]
                reason = "END_OF_STREAM"
                self.db.execute("UPDATE decision_workflows SET state='finished',reason=?,updated_at=? "
                                "WHERE parent_decision_id=?", (reason, now, parent))
            elif end_of_stream and child_pending:
                parent = child_pending["parent_decision_id"]
                reason = "END_OF_STREAM"
                self._cancel_child_tasks(parent, reason=reason, now=now)
                self.db.execute("UPDATE decision_workflows SET state='finished',reason=?,updated_at=? "
                                "WHERE parent_decision_id=? AND state='child_pending'", (reason, now, parent))
            elif task_suppression_reason:
                reason = task_suppression_reason
            elif evidence is None:
                reason = "EVIDENCE_INCOMPLETE"
                if waiting:
                    parent = waiting["parent_decision_id"]
                    reason = "REOBSERVATION_EVIDENCE_INCOMPLETE"
                    self.db.execute("UPDATE decision_workflows SET state='cancelled',reason=?,updated_at=? "
                                    "WHERE parent_decision_id=?", (reason, now, parent))
            elif task is None:
                reason = "PROVIDER_OFF"
            else:
                wf = waiting
                if wf:
                    parent = wf["parent_decision_id"]
                    same_identity = (evidence["action_epoch"] == wf["action_epoch"]
                                     and evidence["generation"] == wf["generation"]
                                     and evidence.get("fall_target_epoch") == wf["fall_target_epoch"])
                    if evidence.get("recovery_status") == "confirmed":
                        reason = "REOBSERVATION_UPSTREAM_RECOVERY"
                        self.db.execute("UPDATE decision_workflows SET state='finished',reason=?,updated_at=? "
                                        "WHERE parent_decision_id=?", (reason, now, parent))
                    elif (not evidence.get("observable") or not evidence.get("continuous")
                          or not evidence.get("event_decidable")):
                        reason = "REOBSERVATION_CONTINUITY_LOST"
                        self.db.execute("UPDATE decision_workflows SET state='cancelled',reason=?,updated_at=? "
                                        "WHERE parent_decision_id=?", (reason, now, parent))
                    elif not same_identity:
                        reason = "REOBSERVATION_TARGET_CHANGED"
                        self.db.execute("UPDATE decision_workflows SET state='cancelled',reason=?,updated_at=? "
                                        "WHERE parent_decision_id=?", (reason, now, parent))
                    elif source_time - wf["trigger_source_time"] < 1.0:
                        reason = "WAITING_FOR_NEW_SOURCE_EVIDENCE"
                    elif wf["budget_used"] >= wf["budget_total"]:
                        reason = "REOBSERVATION_BUDGET_EXHAUSTED"
                        self.db.execute("UPDATE decision_workflows SET state='request_review',reason=?,updated_at=? "
                                        "WHERE parent_decision_id=?", (reason, now, parent))
                    else:
                        task["parent_decision_id"] = parent
                        reason = "REOBSERVATION_SCHEDULED"
                else:
                    previous = self.db.execute(
                        "SELECT envelope_json FROM decision_observations WHERE session_id=? AND stream_id=? "
                        "AND id<? AND stale=0 ORDER BY source_time DESC,id DESC LIMIT 1",
                        (session_id, stream_id, observation_id)).fetchone()
                    critical = self._is_critical_transition(envelope, previous)
                    running = self.db.execute(
                        "SELECT 1 FROM decision_tasks WHERE session_id=? AND stream_id=? AND status='running' LIMIT 1",
                        (session_id, stream_id)).fetchone()
                    pending = self.db.execute(
                        "SELECT * FROM decision_tasks WHERE session_id=? AND stream_id=? AND status='pending' "
                        "AND parent_decision_id IS NULL AND critical=0 ORDER BY created_at LIMIT 1",
                        (session_id, stream_id)).fetchone()
                    if pending and not critical and not running:
                        task_id = pending["task_id"]
                        try:
                            self.db.execute(
                                "UPDATE decision_tasks SET snapshot_id=?,decision_id=?,evidence_json=?,evidence_digest=?,"
                                "request_digest=?,source_time=?,updated_at=? WHERE task_id=?",
                                (task["snapshot_id"], task["decision_id"], encoded_evidence,
                                 evidence_digest, task["request_digest"], source_time, now, task_id))
                            self.db.execute("INSERT OR IGNORE INTO decision_task_observations VALUES(?,?,?)",
                                            (task_id, observation_id, "coalesced_latest"))
                            self.db.execute("UPDATE decision_observations SET decision_id=?,request_digest=?,"
                                            "schedule_reason='COALESCED_TO_LATEST_PENDING_WINDOW' WHERE task_id=?",
                                            (task["decision_id"], task["request_digest"], task_id))
                            reason = "COALESCED_TO_LATEST_PENDING_WINDOW"
                        except sqlite3.IntegrityError:
                            reason = "TASK_ALREADY_REGISTERED"
                    else:
                        last = self.db.execute(
                            "SELECT source_time FROM decision_tasks WHERE session_id=? AND stream_id=? "
                            "ORDER BY source_time DESC LIMIT 1", (session_id, stream_id)).fetchone()
                        if not critical and last and source_time - last["source_time"] < min_interval_s:
                            reason = "MIN_SOURCE_INTERVAL"
                        else:
                            reason = "CRITICAL_EVENT_TRANSITION" if critical else "SCHEDULED"
                if reason in {"SCHEDULED", "CRITICAL_EVENT_TRANSITION", "REOBSERVATION_SCHEDULED"}:
                    active_count = self.db.execute(
                        "SELECT count(*) FROM decision_tasks WHERE status IN ('pending','running')").fetchone()[0]
                    if active_count >= max_queue:
                        reason = "QUEUE_FULL"
                        if task.get("parent_decision_id"):
                            self.db.execute("UPDATE decision_workflows SET budget_used=budget_used+1,state='child_pending',"
                                            "child_decision_id=?,updated_at=? WHERE parent_decision_id=? "
                                            "AND state='waiting_observation'",
                                            (task["decision_id"], now, task["parent_decision_id"]))
                    else:
                        task_id = task.get("task_id") or f"task-{uuid.uuid4().hex}"
                        try:
                            self.db.execute(
                                "INSERT INTO decision_tasks(task_id,run_id,session_id,stream_id,snapshot_id,decision_id,"
                                "evidence_json,evidence_digest,request_digest,provider,model,status,attempt,max_attempts,"
                                "max_reobservations,available_at,source_time,parent_decision_id,critical,created_at,updated_at) "
                                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'pending',0,?,?,?,?,?,?,?,?)",
                                (task_id, run_id, session_id, stream_id, task["snapshot_id"], task["decision_id"],
                                 encoded_evidence, evidence_digest, task["request_digest"], task["provider"], task["model"],
                                 task["max_attempts"], task.get("max_reobservations", 1), now, source_time,
                                 task.get("parent_decision_id"), int(critical), now, now))
                            self.db.execute("INSERT INTO decision_task_observations VALUES(?,?,?)",
                                            (task_id, observation_id, "scheduled"))
                            if task.get("parent_decision_id"):
                                self.db.execute("UPDATE decision_workflows SET budget_used=budget_used+1,state='child_pending',"
                                                "child_decision_id=?,updated_at=? WHERE parent_decision_id=? "
                                                "AND state='waiting_observation'",
                                                (task["decision_id"], now, task["parent_decision_id"]))
                        except sqlite3.IntegrityError:
                            existing = self.db.execute(
                                "SELECT task_id FROM decision_tasks WHERE session_id=? AND stream_id=? "
                                "AND request_digest=? AND provider=? AND model=?",
                                (session_id, stream_id, task["request_digest"], task["provider"], task["model"])).fetchone()
                            if existing:
                                task_id = existing["task_id"]
                                reason = "TASK_ALREADY_REGISTERED"
                            else:
                                raise
            self.db.execute(
                "UPDATE decision_observations SET decision_id=?,request_digest=?,task_id=?,schedule_reason=?,"
                "parent_decision_id=?,critical=? WHERE id=?",
                (decision_id, request_digest, task_id, reason, parent or (task.get("parent_decision_id") if task else None),
                 int(critical), observation_id))
            self.db.commit()
            return {"status": "stale" if stale else "accepted", "snapshot_id": snapshot_id,
                    "observation_id": observation_id, "task_id": task_id, "decision_id": decision_id,
                    "request_digest": request_digest, "stale": stale, "decision": None,
                    "missing_reasons": missing_reasons, "schedule_reason": reason}
        except BaseException:
            self.db.rollback()
            raise

    @staticmethod
    def _is_critical_transition(envelope: dict[str, Any], previous: sqlite3.Row | None) -> bool:
        event = envelope["perception"].get("event", {})
        if not isinstance(event, dict):
            return False
        transitions = event.get("transitions")
        if isinstance(transitions, list) and transitions:
            return True
        if previous:
            prior = json.loads(previous["envelope_json"]).get("perception", {}).get("event", {})
            return any(prior.get(key) != event.get(key) for key in ("state", "event_key", "recovery_status"))
        return event.get("state") in {"suspected", "confirmed"} or event.get("recovery_confirmed") is True

    def _finish_child_workflow(self, task: sqlite3.Row, *, state: str, reason: str, now: float) -> None:
        parent_id = task["parent_decision_id"]
        if parent_id:
            self.db.execute(
                "UPDATE decision_workflows SET state=?,reason=?,updated_at=? "
                "WHERE parent_decision_id=? AND state='child_pending'",
                (state, reason, now, parent_id),
            )

    def _cancel_child_tasks(self, parent_id: str, *, reason: str, now: float) -> None:
        self.db.execute(
            "UPDATE decision_tasks SET status='cancelled',cancel_requested=1,lease_token=NULL,lease_until=NULL,"
            "reason=?,updated_at=? WHERE parent_decision_id=? AND status IN ('pending','running')",
            (reason, now, parent_id),
        )
        self.db.execute(
            "UPDATE decision_attempts SET status='cancelled',finished_at=?,error_code=? "
            "WHERE task_id IN (SELECT task_id FROM decision_tasks WHERE parent_decision_id=?) "
            "AND status='running'",
            (now, reason, parent_id),
        )
        self.db.execute(
            "UPDATE decision_observations SET schedule_reason=? WHERE parent_decision_id=? "
            "AND schedule_reason='QUEUE_FULL'",
            (reason, parent_id),
        )

    def expired_aliyun_tasks(self, *, now: float) -> list[dict[str, Any]]:
        """Return Aliyun tasks whose persisted budget/lease cannot be resumed safely."""
        rows = self.db.execute(
            "SELECT * FROM decision_tasks WHERE provider='aliyun' AND ("
            "(status='pending' AND (attempt>=max_attempts OR "
            "(network_remaining_s IS NOT NULL AND network_remaining_s<=0) OR "
            "(network_deadline_at IS NOT NULL AND network_deadline_at<=?))) OR "
            "(status='running' AND lease_until<=?)) ORDER BY created_at,task_id",
            (now, now),
        )
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            result.append(item)
        return result

    def complete_unavailable_without_attempt(self, *, task_id: str, result_json: str,
                                             error_code: str, now: float, process_id: str,
                                             service_instance_id: str) -> bool:
        """Persist terminal expiry after restart without resending an uncertain request."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            task = self.db.execute("SELECT * FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()
            if task is None or task["provider"] != "aliyun":
                self.db.commit()
                return False
            expired = (
                task["status"] == "pending"
                and (task["attempt"] >= task["max_attempts"]
                     or (task["network_remaining_s"] is not None and task["network_remaining_s"] <= 0)
                     or (task["network_deadline_at"] is not None and task["network_deadline_at"] <= now))
            ) or (task["status"] == "running" and task["lease_until"] is not None
                  and task["lease_until"] <= now)
            existing = self.db.execute(
                "SELECT 1 FROM decision_results WHERE task_id=? AND accepted=1", (task_id,)
            ).fetchone()
            if not expired or existing:
                self.db.commit()
                return False
            result = json.loads(result_json)
            self.db.execute(
                "INSERT INTO decision_results(decision_id,task_id,attempt_no,result_json,source_time,window_end,"
                "completed_at,process_id,accepted,session_id,stream_id,service_instance_id) "
                "VALUES(?,?,?,?,?,?,?,?,1,?,?,?)",
                (result["decision_id"], task_id, task["attempt"], result_json, task["source_time"],
                 result["source_window"]["end_s"], now, process_id, task["session_id"], task["stream_id"],
                 service_instance_id),
            )
            self._insert_result_cursor(task, result["decision_id"], task["source_time"], now)
            self.db.execute(
                "UPDATE decision_tasks SET status='completed',lease_token=NULL,lease_until=NULL,reason=?,updated_at=? "
                "WHERE task_id=?",
                (error_code, now, task_id),
            )
            self.db.execute(
                "UPDATE decision_attempts SET status='unavailable',finished_at=COALESCE(finished_at,?),"
                "error_code=COALESCE(error_code,?) WHERE task_id=? AND attempt_no=?",
                (now, error_code, task_id, task["attempt"]),
            )
            self._finish_child_workflow(task, state="request_review",
                                        reason=f"REOBSERVATION_CHILD_FAILED:{error_code}", now=now)
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise
    def claim_next(self, *, now: float, lease_s: float, provider: str | None = None,
                   model: str | None = None, max_queue: int = 1000,
                   network_budget_s: float = 3.0) -> dict[str, Any] | None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            expired = list(self.db.execute(
                "SELECT * FROM decision_tasks WHERE status='running' AND lease_until<=?", (now,)))
            for row in expired:
                self.db.execute("UPDATE decision_attempts SET status='lease_expired',finished_at=?,error_code='LEASE_EXPIRED' "
                                "WHERE task_id=? AND attempt_no=? AND lease_token=? AND status='running'",
                                (now, row["task_id"], row["attempt"], row["lease_token"]))
                if row["attempt"] < row["max_attempts"] and not row["cancel_requested"]:
                    self.db.execute("UPDATE decision_tasks SET status='pending',lease_token=NULL,lease_until=NULL,"
                                    "reason='LEASE_EXPIRED',updated_at=? WHERE task_id=? AND status='running' AND lease_token=?",
                                    (now, row["task_id"], row["lease_token"]))
                else:
                    self.db.execute("UPDATE decision_tasks SET status='cancelled',lease_token=NULL,lease_until=NULL,"
                                    "reason='LEASE_EXPIRED_MAX_ATTEMPTS',updated_at=? "
                                    "WHERE task_id=? AND status='running' AND lease_token=?",
                                    (now, row["task_id"], row["lease_token"]))
                    self._finish_child_workflow(row, state="request_review",
                                                reason="REOBSERVATION_CHILD_LEASE_RETRIES_EXHAUSTED", now=now)

            exhausted = list(self.db.execute(
                "SELECT * FROM decision_tasks WHERE status='pending' AND attempt>=max_attempts"))
            for row in exhausted:
                self.db.execute("UPDATE decision_tasks SET status='cancelled',reason='MAX_ATTEMPTS_EXHAUSTED',"
                                "updated_at=? WHERE task_id=? AND status='pending'", (now, row["task_id"]))
                self._finish_child_workflow(row, state="request_review",
                                            reason="REOBSERVATION_CHILD_RETRIES_EXHAUSTED", now=now)
            # Lease expiry is classified first so an exhausted attempt retains
            # its precise terminal reason even when its workflow deadline also elapsed.
            self._expire_workflows_tx(now)
            if provider is not None and model is not None:
                active = self.db.execute("SELECT count(*) FROM decision_tasks "
                                         "WHERE status IN ('pending','running')").fetchone()[0]
                capacity = max(0, max_queue - active)
                deferred = list(self.db.execute(
                    "SELECT * FROM decision_observations WHERE schedule_reason='QUEUE_FULL' "
                    "AND stale=0 AND evidence_json IS NOT NULL AND provider=? AND model=? "
                    "ORDER BY critical DESC,source_time,id LIMIT ?", (provider, model, capacity)))
                for item in deferred:
                    if item["parent_decision_id"]:
                        workflow = self.db.execute(
                            "SELECT state,child_decision_id FROM decision_workflows WHERE parent_decision_id=?",
                            (item["parent_decision_id"],)).fetchone()
                        if not workflow or workflow["state"] != "child_pending" or workflow["child_decision_id"] != item["decision_id"]:
                            self.db.execute("UPDATE decision_observations SET schedule_reason='REOBSERVATION_WORKFLOW_TERMINATED' "
                                            "WHERE id=?", (item["id"],))
                            continue
                    existing = self.db.execute(
                        "SELECT task_id FROM decision_tasks WHERE session_id=? AND stream_id=? "
                        "AND request_digest=? AND provider=? AND model=?",
                        (item["session_id"], item["stream_id"], item["request_digest"], provider, model)).fetchone()
                    if existing:
                        self.db.execute("UPDATE decision_observations SET task_id=?,schedule_reason='TASK_ALREADY_REGISTERED' "
                                        "WHERE id=?", (existing["task_id"], item["id"]))
                        continue
                    task_id = f"task-{item['request_digest'][:32]}"
                    self.db.execute(
                        "INSERT INTO decision_tasks(task_id,run_id,session_id,stream_id,snapshot_id,decision_id,"
                        "evidence_json,evidence_digest,request_digest,provider,model,status,attempt,max_attempts,"
                        "max_reobservations,available_at,source_time,parent_decision_id,critical,created_at,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,'pending',0,?,?,?,?,?,?,?,?)",
                        (task_id, item["run_id"], item["session_id"], item["stream_id"], item["snapshot_id"],
                         item["decision_id"], item["evidence_json"], item["evidence_digest"], item["request_digest"],
                         provider, model, item["max_attempts"] or 1, item["max_reobservations"] or 0,
                         now, item["source_time"], item["parent_decision_id"], item["critical"], now, now))
                    self.db.execute("INSERT INTO decision_task_observations VALUES(?,?,?)",
                                    (task_id, item["id"], "backpressure_recovered"))
                    self.db.execute("UPDATE decision_observations SET task_id=?,schedule_reason='BACKPRESSURE_RECOVERED' "
                                    "WHERE id=?", (task_id, item["id"]))
                    active += 1
            filters = ["t.status='pending'", "t.attempt<t.max_attempts", "t.available_at<=?",
                       "NOT EXISTS(SELECT 1 FROM decision_tasks r WHERE r.session_id=t.session_id "
                       "AND r.stream_id=t.stream_id AND r.status='running')"]
            params: list[Any] = [now]
            if provider is not None:
                filters.append("t.provider=?")
                params.append(provider)
            if model is not None:
                filters.append("t.model=?")
                params.append(model)
            row = self.db.execute(
                "SELECT t.* FROM decision_tasks t WHERE " + " AND ".join(filters)
                + " ORDER BY t.created_at,t.task_id LIMIT 1", params).fetchone()
            if not row:
                self.db.commit()
                return None
            token = uuid.uuid4().hex
            attempt = row["attempt"] + 1
            network_started_at = row["network_started_at"]
            network_deadline_at = row["network_deadline_at"]
            network_remaining_s = row["network_remaining_s"]
            if provider == "aliyun" and network_deadline_at is None:
                network_started_at = now
                network_deadline_at = now + network_budget_s
                network_remaining_s = network_budget_s
            elif provider == "aliyun" and network_remaining_s is None:
                network_remaining_s = max(0.0, min(network_budget_s, network_deadline_at - now))
            changed = self.db.execute(
                "UPDATE decision_tasks SET status='running',attempt=?,lease_token=?,lease_until=?,updated_at=?,"
                "network_started_at=COALESCE(network_started_at,?),"
                "network_deadline_at=COALESCE(network_deadline_at,?),"
                "network_remaining_s=COALESCE(network_remaining_s,?) "
                "WHERE task_id=? AND status='pending'",
                (attempt, token, now + lease_s, now, network_started_at, network_deadline_at,
                 network_remaining_s, row["task_id"])).rowcount
            if not changed:
                self.db.commit()
                return None
            self.db.execute("INSERT INTO decision_attempts(task_id,attempt_no,lease_token,status,started_at) "
                            "VALUES(?,?,?,'running',?)", (row["task_id"], attempt, token, now))
            if row["parent_decision_id"]:
                remaining_attempts = max(0, row["max_attempts"] - row["attempt"])
                child_deadline = now + lease_s * remaining_attempts
                self.db.execute("UPDATE decision_workflows SET deadline_at=max(deadline_at,?),updated_at=? "
                                "WHERE parent_decision_id=? AND state='child_pending'",
                                (child_deadline, now, row["parent_decision_id"]))
            self.db.commit()
            claimed = dict(row)
            claimed.update(status="running", attempt=attempt, lease_token=token, lease_until=now + lease_s,
                           network_started_at=network_started_at, network_deadline_at=network_deadline_at,
                           network_remaining_s=network_remaining_s)
            claimed["evidence"] = json.loads(claimed.pop("evidence_json"))
            return claimed
        except BaseException:
            self.db.rollback()
            raise

    def complete_result(self, *, task_id: str, attempt: int, lease_token: str,
                        result_json: str, process_id: str, service_instance_id: str,
                        now: float) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            task = self.db.execute("SELECT * FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()
            if task is None:
                raise KeyError(task_id)
            attempt_row = self.db.execute("SELECT * FROM decision_attempts WHERE task_id=? AND attempt_no=?",
                                          (task_id, attempt)).fetchone()
            if attempt_row is None or attempt_row["lease_token"] != lease_token:
                raise DecisionConflict("unknown task attempt or lease token")
            existing = self.db.execute("SELECT * FROM decision_results WHERE task_id=? AND accepted=1", (task_id,)).fetchone()
            if existing:
                if attempt_row["status"] == "accepted":
                    if existing["result_json"] != result_json:
                        raise DecisionConflict("logical task already has a different final result")
                    self.db.commit()
                    return {"accepted": True, "duplicate": True, "decision_id": existing["decision_id"]}
                self.db.execute("UPDATE decision_attempts SET status='late_result',finished_at=?,result_json=? "
                                "WHERE task_id=? AND attempt_no=?", (now, result_json, task_id, attempt))
                self.db.commit()
                return {"accepted": False, "duplicate": False, "decision_id": json.loads(result_json)["decision_id"]}
            accepted = (task["status"] == "running" and task["lease_token"] == lease_token
                        and attempt_row["lease_token"] == lease_token and not task["cancel_requested"])
            result_obj = json.loads(result_json)
            if accepted:
                self.db.execute(
                    "INSERT INTO decision_results(decision_id,task_id,attempt_no,result_json,source_time,window_end,"
                    "completed_at,process_id,accepted,session_id,stream_id,service_instance_id) "
                    "VALUES(?,?,?,?,?,?,?,?,1,?,?,?)",
                    (result_obj["decision_id"], task_id, attempt, result_json, task["source_time"],
                     result_obj["source_window"]["end_s"], now, process_id, task["session_id"], task["stream_id"],
                     service_instance_id))
                self._insert_result_cursor(task, result_obj["decision_id"], task["source_time"], now)
                self.db.execute("UPDATE decision_tasks SET status='completed',lease_token=NULL,lease_until=NULL,"
                                "reason=NULL,updated_at=? WHERE task_id=?", (now, task_id))
                provider_metadata = result_obj.get("provider_metadata") or {}
                self.db.execute(
                    "UPDATE decision_attempts SET status='accepted',finished_at=?,result_json=?,request_id=?,"
                    "http_status=?,client_latency_ms=?,service_latency_ms=?,input_tokens=?,diagnostic_code=?,field_path=? "
                    "WHERE task_id=? AND attempt_no=?",
                    (now, result_json, provider_metadata.get("request_id"), provider_metadata.get("http_status"),
                     provider_metadata.get("client_latency_ms"), provider_metadata.get("service_latency_ms"),
                     provider_metadata.get("input_tokens"), provider_metadata.get("diagnostic_code"),
                     provider_metadata.get("field_path"), task_id, attempt),
                )
                if result_obj.get("next_step") == "observe_more" and not task["parent_decision_id"]:
                    ev = json.loads(task["evidence_json"])
                    workflow_state = "waiting_observation" if task["max_reobservations"] > 0 else "request_review"
                    self.db.execute(
                        "INSERT OR IGNORE INTO decision_workflows(parent_decision_id,run_id,session_id,stream_id,"
                        "action_epoch,generation,fall_target_epoch,trigger_source_time,deadline_at,budget_total,state,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (result_obj["decision_id"], task["run_id"], task["session_id"], task["stream_id"],
                         ev["action_epoch"], ev["generation"], ev.get("fall_target_epoch"), task["source_time"],
                         now + 5.0, task["max_reobservations"], workflow_state, now))
                    if task["max_reobservations"] == 0:
                        self.db.execute("UPDATE decision_workflows SET reason='REOBSERVATION_BUDGET_DISABLED' "
                                        "WHERE parent_decision_id=?", (result_obj["decision_id"],))
                elif task["parent_decision_id"]:
                    state = "finished" if result_obj.get("status") == "ready" else "request_review"
                    why = "REOBSERVATION_COMPLETED" if state == "finished" else "REOBSERVATION_INSUFFICIENT"
                    self.db.execute("UPDATE decision_workflows SET state=?,reason=?,updated_at=? "
                                    "WHERE parent_decision_id=?", (state, why, now, task["parent_decision_id"]))
            else:
                self.db.execute("UPDATE decision_attempts SET status='late_result',finished_at=?,result_json=? "
                                "WHERE task_id=? AND attempt_no=?", (now, result_json, task_id, attempt))
            self.db.commit()
            return {"accepted": accepted, "duplicate": False, "decision_id": result_obj["decision_id"]}
        except BaseException:
            self.db.rollback()
            raise

    def fail_attempt(self, *, task_id: str, attempt: int, lease_token: str, error_code: str,
                     now: float, retry_delay_s: float = 0.0,
                     network_remaining_s: float | None = None,
                     terminal_result_json: str | None = None, process_id: str = "unknown",
                     service_instance_id: str = "unknown",
                     attempt_metadata: dict[str, Any] | None = None) -> bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            task = self.db.execute("SELECT * FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()
            if (not task or task["status"] != "running" or task["lease_token"] != lease_token
                    or task["attempt"] != attempt):
                self.db.commit()
                return False
            metadata = attempt_metadata or {}
            self.db.execute(
                "UPDATE decision_attempts SET status='failed',finished_at=?,error_code=?,request_id=?,http_status=?,"
                "client_latency_ms=?,service_latency_ms=?,input_tokens=?,retry_after_s=?,diagnostic_code=?,field_path=? "
                "WHERE task_id=? AND attempt_no=? AND lease_token=?",
                (now, error_code, metadata.get("request_id"), metadata.get("http_status"),
                 metadata.get("client_latency_ms"), metadata.get("service_latency_ms"),
                 metadata.get("input_tokens"), metadata.get("retry_after_s"), metadata.get("diagnostic_code"),
                 metadata.get("field_path"), task_id, attempt, lease_token),
            )
            retry_before_deadline = (task["network_deadline_at"] is None
                                     or now + retry_delay_s < task["network_deadline_at"])
            retry_before_budget = network_remaining_s is None or network_remaining_s > 0
            if (attempt < task["max_attempts"] and not task["cancel_requested"]
                    and retry_before_deadline and retry_before_budget and terminal_result_json is None):
                self.db.execute("UPDATE decision_tasks SET status='pending',lease_token=NULL,lease_until=NULL,"
                                "available_at=?,network_remaining_s=COALESCE(?,network_remaining_s),reason=?,updated_at=? "
                                "WHERE task_id=?",
                                (now + retry_delay_s, network_remaining_s, error_code, now, task_id))
            else:
                if terminal_result_json is not None:
                    result_obj = json.loads(terminal_result_json)
                    self.db.execute(
                        "INSERT INTO decision_results(decision_id,task_id,attempt_no,result_json,source_time,window_end,"
                        "completed_at,process_id,accepted,session_id,stream_id,service_instance_id) "
                        "VALUES(?,?,?,?,?,?,?,?,1,?,?,?)",
                        (result_obj["decision_id"], task_id, attempt, terminal_result_json, task["source_time"],
                         result_obj["source_window"]["end_s"], now, process_id, task["session_id"],
                         task["stream_id"], service_instance_id))
                    self._insert_result_cursor(task, result_obj["decision_id"], task["source_time"], now)
                    terminal_reason = error_code
                    self.db.execute("UPDATE decision_tasks SET status='completed',lease_token=NULL,lease_until=NULL,"
                                    "reason=?,updated_at=? WHERE task_id=?", (terminal_reason, now, task_id))
                    self.db.execute("UPDATE decision_attempts SET status='unavailable',result_json=? "
                                    "WHERE task_id=? AND attempt_no=?", (terminal_result_json, task_id, attempt))
                else:
                    terminal_reason = f"MAX_ATTEMPTS_EXHAUSTED:{error_code}"
                    self.db.execute("UPDATE decision_tasks SET status='cancelled',lease_token=NULL,lease_until=NULL,"
                                    "reason=?,updated_at=? WHERE task_id=?", (terminal_reason, now, task_id))
                self._finish_child_workflow(task, state="request_review",
                                            reason=f"REOBSERVATION_CHILD_FAILED:{error_code}", now=now)
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def task(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM decision_tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["evidence"] = json.loads(data.pop("evidence_json"))
        return data

    def expire_workflows(self, now: float) -> int:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            count = self._expire_workflows_tx(now)
            self.db.commit()
            return count
        except BaseException:
            self.db.rollback()
            raise

    def _expire_workflows_tx(self, now: float) -> int:
        expired = list(self.db.execute(
            "SELECT * FROM decision_workflows WHERE state IN ('waiting_observation','child_pending') "
            "AND deadline_at<=?", (now,)))
        for workflow in expired:
            if workflow["state"] == "waiting_observation":
                state, reason = "request_review", "OBSERVATION_TIMEOUT"
            else:
                state, reason = "request_review", "REOBSERVATION_CHILD_DEADLINE_EXCEEDED"
                self._cancel_child_tasks(workflow["parent_decision_id"], reason=reason, now=now)
            self.db.execute("UPDATE decision_workflows SET state=?,reason=?,updated_at=? "
                            "WHERE parent_decision_id=? AND state=?",
                            (state, reason, now, workflow["parent_decision_id"], workflow["state"]))
        return len(expired)

    def stop(self, *, session_id: str, stream_id: str, reason: str, now: float) -> dict[str, int]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            workflows = self.db.execute("UPDATE decision_workflows SET state='cancelled',reason=?,updated_at=? "
                                        "WHERE session_id=? AND stream_id=? AND state IN ('waiting_observation','child_pending')",
                                        (reason, now, session_id, stream_id)).rowcount
            pending = self.db.execute("UPDATE decision_tasks SET status='cancelled',reason=?,updated_at=? "
                                      "WHERE session_id=? AND stream_id=? AND status='pending'",
                                      (reason, now, session_id, stream_id)).rowcount
            running = self.db.execute("UPDATE decision_tasks SET status='cancelled',cancel_requested=1,"
                                      "lease_token=NULL,lease_until=NULL,reason=?,updated_at=? "
                                      "WHERE session_id=? AND stream_id=? AND status='running'",
                                      (reason, now, session_id, stream_id)).rowcount
            self.db.execute("UPDATE decision_attempts SET status='cancelled',finished_at=?,error_code=? "
                            "WHERE status='running' AND task_id IN (SELECT task_id FROM decision_tasks "
                            "WHERE session_id=? AND stream_id=? AND status='cancelled')",
                            (now, reason, session_id, stream_id))
            self.db.execute("UPDATE decision_observations SET schedule_reason=? WHERE session_id=? AND stream_id=? "
                            "AND schedule_reason='QUEUE_FULL' AND parent_decision_id IN "
                            "(SELECT parent_decision_id FROM decision_workflows WHERE session_id=? AND stream_id=? "
                            "AND state='cancelled')", (reason, session_id, stream_id, session_id, stream_id))
            self.db.execute("UPDATE decision_streams SET stopped=1,stop_reason=? WHERE session_id=? AND stream_id=?",
                            (reason, session_id, stream_id))
            self.db.commit()
            return {"workflows_cancelled": workflows, "pending_tasks_cancelled": pending,
                    "running_tasks_marked": running}
        except BaseException:
            self.db.rollback()
            raise

    def get_result(self, decision_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM decision_results WHERE decision_id=? AND accepted=1", (decision_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["provider_result"] = json.loads(result.pop("result_json"))
        return result

    def _insert_result_cursor(self, task: sqlite3.Row, decision_id: str, source_time: float, now: float) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO decision_result_cursors(decision_id,run_id,session_id,stream_id,source_time,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (decision_id, task["run_id"], task["session_id"], task["stream_id"], source_time, now),
        )

    def list_results_after(self, *, run_id: str, session_id: str, stream_id: str,
                           after_id: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        if type(after_id) is not int or after_id < 0:
            raise ValueError("after_id must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        rows = self.db.execute(
            "SELECT c.cursor,c.decision_id,c.run_id,c.session_id,c.stream_id,c.source_time,c.created_at AS cursor_created_at,"
            "r.result_json,r.completed_at,r.task_id,t.status AS task_status "
            "FROM decision_result_cursors c JOIN decision_results r ON r.decision_id=c.decision_id AND r.accepted=1 "
            "JOIN decision_tasks t ON t.task_id=r.task_id "
            "WHERE c.run_id=? AND c.session_id=? AND c.stream_id=? AND c.cursor>? "
            "ORDER BY c.cursor LIMIT ?",
            (run_id, session_id, stream_id, after_id, limit),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json"))
            attempts = self.db.execute(
                "SELECT attempt_no,status,started_at,finished_at,error_code,request_id,http_status,"
                "client_latency_ms,service_latency_ms,input_tokens,diagnostic_code,field_path "
                "FROM decision_attempts WHERE task_id=? ORDER BY attempt_no", (item["task_id"],)
            ).fetchall()
            item["attempt_history"] = [dict(attempt) for attempt in attempts]
            items.append(item)
        return items

    def list_results(self, *, session_id: str, stream_id: str | None = None,
                     after_source_time: float | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        clauses = ["session_id=?", "accepted=1"]
        params: list[Any] = [session_id]
        if stream_id is not None:
            clauses.append("stream_id=?")
            params.append(stream_id)
        if after_source_time is not None:
            clauses.append("source_time>?")
            params.append(after_source_time)
        rows = self.db.execute("SELECT * FROM decision_results WHERE " + " AND ".join(clauses)
                               + " ORDER BY source_time DESC,decision_id LIMIT ?", (*params, limit))
        result = []
        for row in rows:
            item = dict(row)
            item["provider_result"] = json.loads(item.pop("result_json"))
            result.append(item)
        return result

    def attempts(self, task_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM decision_attempts WHERE task_id=? ORDER BY attempt_no", (task_id,))]

    def workflow_for(self, decision_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM decision_workflows WHERE parent_decision_id=?", (decision_id,)).fetchone()
        return dict(row) if row else None

    def workflow_status(self, session_id: str, stream_id: str, now: float) -> dict[str, Any] | None:
        self.expire_workflows(now)
        row = self.db.execute("SELECT * FROM decision_workflows WHERE session_id=? AND stream_id=? "
                              "ORDER BY updated_at DESC LIMIT 1", (session_id, stream_id)).fetchone()
        return dict(row) if row else None

    def list_decisions(self, *, session_id: str, stream_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        clauses = ["o.session_id=?"]
        params: list[Any] = [session_id]
        if stream_id is not None:
            clauses.append("o.stream_id=?")
            params.append(stream_id)
        rows = self.db.execute(
            "SELECT o.id AS observation_id,o.source_time,o.stale,o.schedule_reason,o.task_id,o.decision_id,"
            "o.missing_reasons_json,t.status AS task_status,t.reason AS task_reason,t.parent_decision_id,"
            "r.result_json,r.accepted,"
            "r.completed_at,r.service_instance_id "
            "FROM decision_observations o LEFT JOIN decision_tasks t ON o.task_id=t.task_id "
            "LEFT JOIN decision_results r ON t.task_id=r.task_id AND r.accepted=1 WHERE "
            + " AND ".join(clauses) + " ORDER BY o.source_time DESC,o.id DESC LIMIT ?", (*params, limit))
        items = []
        seen = set()
        for row in rows:
            item = dict(row)
            item["missing_reasons"] = json.loads(item.pop("missing_reasons_json"))
            item["provider_result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
            if item.get("task_id"):
                attempts = self.db.execute("SELECT attempt_no,status,started_at,finished_at,error_code,result_json,"
                                           "request_id,http_status,client_latency_ms,service_latency_ms,input_tokens,"
                                           "retry_after_s,diagnostic_code,field_path "
                                           "FROM decision_attempts WHERE task_id=? ORDER BY attempt_no",
                                           (item["task_id"],))
                item["attempt_history"] = [
                    {**dict(attempt), "provider_result": json.loads(attempt["result_json"])
                     if attempt["result_json"] else None}
                    for attempt in attempts
                ]
            else:
                item["attempt_history"] = []
            # Coalesced observations point to one task; expose the one logical result once.
            dedupe_key = item["provider_result"].get("decision_id") if item["provider_result"] else None
            if dedupe_key and dedupe_key in seen:
                item["provider_result"] = None
                item["result_alias_of"] = dedupe_key
            elif dedupe_key:
                seen.add(dedupe_key)
            items.append(item)
        return items

    def feedback(self, *, decision_id: str, label: str, request_id: str, expected_version: int,
                 now: float) -> dict[str, Any]:
        if label not in {"fall", "non_fall", "uncertain"}:
            raise ValueError("label must be fall, non_fall, or uncertain")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be nonempty")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("expected_version must be a nonnegative integer")
        result = self.get_result(decision_id)
        if result is None:
            raise KeyError(decision_id)
        payload = {"decision_id": decision_id, "label": label, "request_id": request_id,
                   "expected_version": expected_version}
        digest = __import__("hashlib").sha256(canonical_json(payload).encode()).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute("SELECT digest,receipt_json FROM decision_feedback_requests WHERE request_id=?",
                                    (request_id,)).fetchone()
            if prior:
                if prior["digest"] != digest:
                    raise DecisionConflict("feedback request_id was reused for different content")
                self.db.commit()
                return json.loads(prior["receipt_json"])
            version_row = self.db.execute("SELECT version FROM decision_feedback_versions WHERE decision_id=?",
                                          (decision_id,)).fetchone()
            current = version_row["version"] if version_row else 0
            if current != expected_version:
                raise DecisionVersionConflict(f"expected feedback version {expected_version}; current {current}")
            version = current + 1
            result_row = self.db.execute("SELECT t.run_id,t.session_id,t.stream_id FROM decision_tasks t "
                                         "JOIN decision_results r ON r.task_id=t.task_id WHERE r.decision_id=? "
                                         "AND r.accepted=1", (decision_id,)).fetchone()
            receipt = {"schema_version": "decision-feedback.v1", "decision_id": decision_id,
                       "request_id": request_id, "label": label, "version": version,
                       "expected_version": expected_version, "provider_state_changed": False}
            self.db.execute("INSERT INTO decision_feedback_versions VALUES(?,?) ON CONFLICT(decision_id) "
                            "DO UPDATE SET version=excluded.version", (decision_id, version))
            self.db.execute("INSERT INTO decision_feedback(run_id,session_id,stream_id,decision_id,label,request_id,"
                            "expected_version,version,recorded_at) VALUES(?,?,?,?,?,?,?,?,?)",
                            (result_row["run_id"], result_row["session_id"], result_row["stream_id"], decision_id,
                             label, request_id, expected_version, version, now))
            self.db.execute("INSERT INTO decision_feedback_requests VALUES(?,?,?)",
                            (request_id, digest, canonical_json(receipt)))
            self.db.commit()
            return receipt
        except BaseException:
            self.db.rollback()
            raise

    def list_feedback(self, decision_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute(
            "SELECT decision_id,label,request_id,expected_version,version,recorded_at "
            "FROM decision_feedback WHERE decision_id=? ORDER BY version", (decision_id,))]


__all__ = ["DecisionConflict", "DecisionStore", "DecisionVersionConflict", "InjectedStoreCrash"]
