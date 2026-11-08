"""SQLite journal. Event transitions use compare-and-swap inside transactions."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any

from .schemas import ExecutionResult, RiskDecision, Snapshot, new_id


class ServiceError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.details = code, details


@dataclass
class EventRecord:
    event_id: str
    person_id: str
    binding_id: str
    branch: str
    created_at_s: float
    revision: int = 0
    status: str = "NONE"
    outcome: str = "NO_ACTION"
    partition: str = "idle"
    semantic_key: str = ""
    latest_snapshot_id: str | None = None
    latest_decision_id: str | None = None
    wait_deadline_s: float | None = None
    risk_since_s: float | None = None
    clear_since_s: float | None = None
    last_observation_s: float | None = None
    confirmed_alarm: bool = False
    action_sequence: int = 0
    action_slots: dict[str, int] = field(default_factory=dict)


class StateStore:
    def __init__(self, path: str = ":memory:"):
        self.lock = RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots(
              id TEXT PRIMARY KEY, person TEXT NOT NULL, t REAL NOT NULL,
              frame INTEGER NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS snapshot_time ON snapshots(person,t DESC,frame DESC);
            CREATE TABLE IF NOT EXISTS events(
              id TEXT PRIMARY KEY, person TEXT NOT NULL, binding TEXT NOT NULL,
              branch TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS event_history(
              event_id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL,
              PRIMARY KEY(event_id,revision));
            CREATE TABLE IF NOT EXISTS decisions(
              id TEXT PRIMARY KEY, event_id TEXT NOT NULL, cache_key TEXT UNIQUE NOT NULL,
              body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS decision_modes(
              decision_id TEXT PRIMARY KEY, mode TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actions(
              id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, event_id TEXT NOT NULL,
              action_key TEXT UNIQUE NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS notes(
              id TEXT PRIMARY KEY, event_id TEXT NOT NULL, body TEXT NOT NULL);
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def put_snapshot(self, snapshot: Snapshot) -> bool:
        """False for an exact retry; reject ID reuse and per-person source reordering."""
        with self.transaction():
            row = self.conn.execute("SELECT body FROM snapshots WHERE id=?", (snapshot.snapshot_id,)).fetchone()
            if row:
                if Snapshot.model_validate_json(row["body"]) != snapshot:
                    raise ServiceError("ID_CONFLICT", "snapshot_id was reused with different content")
                return False
            previous = self.latest_snapshot(snapshot.person_id)
            if previous:
                if snapshot.session_id == previous.session_id:
                    if (snapshot.t_s, snapshot.frame_id) <= (previous.t_s, previous.frame_id):
                        raise ServiceError("OUT_OF_ORDER", "Snapshot is not newer than the previous source observation")
                else:
                    earlier = self.conn.execute("SELECT body FROM snapshots WHERE person=?", (snapshot.person_id,)).fetchall()
                    if any(Snapshot.model_validate_json(r["body"]).session_id == snapshot.session_id for r in earlier):
                        raise ServiceError("CLOSED_SESSION", "Cannot resume a superseded observation session")
            self.conn.execute("INSERT INTO snapshots VALUES(?,?,?,?,?)", (
                snapshot.snapshot_id, snapshot.person_id, snapshot.t_s, snapshot.frame_id,
                snapshot.model_dump_json()))
            return True

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        with self.lock:
            row = self.conn.execute("SELECT body FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            raise ServiceError("NOT_FOUND", "Unknown snapshot", snapshot_id=snapshot_id)
        return Snapshot.model_validate_json(row["body"])

    def latest_snapshot(self, person_id: str) -> Snapshot | None:
        with self.lock:
            row = self.conn.execute("SELECT body FROM snapshots WHERE person=? ORDER BY rowid DESC LIMIT 1", (person_id,)).fetchone()
        return Snapshot.model_validate_json(row["body"]) if row else None

    def create_event(self, snapshot: Snapshot, branch: str, now_s: float) -> EventRecord:
        event = EventRecord(new_id("evt"), snapshot.person_id, snapshot.binding_id, branch, now_s)
        body = json.dumps(asdict(event))
        with self.transaction():
            self.conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (
                event.event_id, event.person_id, event.binding_id, branch, event.revision, body))
            self.conn.execute("INSERT INTO event_history VALUES(?,?,?)", (event.event_id, 0, body))
        return event

    def get_event(self, event_id: str, revision: int | None = None) -> EventRecord:
        with self.lock:
            if revision is None:
                row = self.conn.execute("SELECT body FROM events WHERE id=?", (event_id,)).fetchone()
            else:
                row = self.conn.execute("SELECT body FROM event_history WHERE event_id=? AND revision=?", (event_id, revision)).fetchone()
        if not row:
            raise ServiceError("NOT_FOUND", "Unknown event or event revision", event_id=event_id, event_revision=revision)
        return EventRecord(**json.loads(row["body"]))

    def list_events(self, person_id: str | None = None) -> list[EventRecord]:
        with self.lock:
            rows = self.conn.execute("SELECT body FROM events" + (" WHERE person=?" if person_id else "") + " ORDER BY rowid", (person_id,) if person_id else ()).fetchall()
        return [EventRecord(**json.loads(row["body"])) for row in rows]

    def current_event(self, snapshot: Snapshot, branch: str) -> EventRecord | None:
        events = [e for e in self.list_events(snapshot.person_id) if e.binding_id == snapshot.binding_id and e.branch == branch]
        return events[-1] if events else None

    def cached_decision(self, cache_key: str) -> RiskDecision | None:
        with self.lock:
            row = self.conn.execute("SELECT body FROM decisions WHERE cache_key=?", (cache_key,)).fetchone()
        return RiskDecision.model_validate_json(row["body"]) if row else None

    def commit_decision(self, event: EventRecord, expected_revision: int,
                        decision: RiskDecision, cache_key: str, execution_mode: str = "dry_run") -> None:
        with self.transaction():
            body = json.dumps(asdict(event))
            result = self.conn.execute("UPDATE events SET revision=?,body=? WHERE id=? AND revision=?", (
                event.revision, body, event.event_id, expected_revision))
            if result.rowcount != 1:
                current = self.get_event(event.event_id)
                raise ServiceError("VERSION_CONFLICT", "Event changed during evaluation", latest_revision=current.revision)
            self.conn.execute("INSERT OR IGNORE INTO event_history VALUES(?,?,?)", (event.event_id, event.revision, body))
            self.conn.execute("INSERT INTO decisions VALUES(?,?,?,?)", (
                decision.decision_id, event.event_id, cache_key, decision.model_dump_json()))
            self.conn.execute("INSERT INTO decision_modes VALUES(?,?)", (decision.decision_id, execution_mode))

    def decision_mode(self, decision_id: str) -> str:
        with self.lock:
            row = self.conn.execute("SELECT mode FROM decision_modes WHERE decision_id=?", (decision_id,)).fetchone()
        return row["mode"] if row else "unknown"

    def list_decisions(self) -> list[RiskDecision]:
        with self.lock:
            rows = self.conn.execute("SELECT body FROM decisions ORDER BY rowid").fetchall()
        return [RiskDecision.model_validate_json(row["body"]) for row in rows]

    def get_decision(self, decision_id: str) -> RiskDecision:
        with self.lock:
            row = self.conn.execute("SELECT body FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if not row:
            raise ServiceError("NOT_FOUND", "Unknown decision", decision_id=decision_id)
        return RiskDecision.model_validate_json(row["body"])

    def decisions_for_snapshot(self, snapshot_id: str) -> list[RiskDecision]:
        with self.lock:
            rows = self.conn.execute("SELECT body FROM decisions ORDER BY rowid").fetchall()
        return [d for row in rows if (d := RiskDecision.model_validate_json(row["body"])).snapshot_id == snapshot_id]

    def action_by_key(self, key: str) -> ExecutionResult | None:
        with self.lock:
            row = self.conn.execute("SELECT body FROM actions WHERE action_key=?", (key,)).fetchone()
        return ExecutionResult.model_validate_json(row["body"]) if row else None

    def get_action(self, action_id: str) -> ExecutionResult:
        with self.lock:
            row = self.conn.execute("SELECT body FROM actions WHERE id=?", (action_id,)).fetchone()
        if not row:
            raise ServiceError("NOT_FOUND", "Unknown action", action_id=action_id)
        return ExecutionResult.model_validate_json(row["body"])

    def save_action(self, result: ExecutionResult) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO actions VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET decision_id=excluded.decision_id,body=excluded.body", (
                result.action_id, result.decision_id, result.event_id, result.idempotency_key, result.model_dump_json()))

    def list_actions(self) -> list[ExecutionResult]:
        with self.lock:
            rows = self.conn.execute("SELECT body FROM actions ORDER BY rowid").fetchall()
        return [ExecutionResult.model_validate_json(row["body"]) for row in rows]

    def add_note(self, event_id: str, note: str, source: str, now_s: float) -> dict[str, Any]:
        self.get_event(event_id)
        body = {"note_id": new_id("note"), "event_id": event_id, "note": note,
                "source": source, "t_s": now_s, "authoritative_observation": False}
        with self.lock:
            self.conn.execute("INSERT INTO notes VALUES(?,?,?)", (body["note_id"], event_id, json.dumps(body)))
        return body

    def close(self) -> None:
        with self.lock:
            self.conn.close()
