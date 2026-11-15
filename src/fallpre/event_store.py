"""Durable perception events and a local-only simulated notification outbox.

Namespaced tables can coexist with the existing MVP StateStore. This bridge
does not replace agent.py or register tools, and is not an alarm detector
or an LLM. A caller must reuse the same session_id across consumer restarts and
start a NEW session when the perception producer resets its event counter/clock.
Never infer recovery from a missing person, stale frame or non-fall head score.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

AUTHORITY = 'validated_handoff_and_constrained_ST_policy'


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be an explicit nonempty string')
    return value


class EventStore:
    """SQLite transaction per observation; no network or external delivery.

    Example::

        with EventStore('perception_events.sqlite') as store:
            result = store.ingest('camera-boot-20261109-01', perception_output)
            for item in store.pending_notifications():
                receipt = store.simulate_delivery(item['id'])

    ``event_key`` is a JSON tuple [session_id, stream_id, upstream_event_key].
    It is opaque: use it unchanged in queries and acknowledgements.
    """

    def __init__(self, path):
        self.db = sqlite3.connect(str(Path(path)), isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS perception_streams (
                session_id TEXT, stream_id TEXT, latest_time REAL NOT NULL,
                PRIMARY KEY(session_id, stream_id));
            CREATE TABLE IF NOT EXISTS perception_snapshots (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, stream_id TEXT NOT NULL,
                source_time REAL NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL,
                stale INTEGER NOT NULL, event_key TEXT,
                UNIQUE(session_id, stream_id, source_time));
            CREATE TABLE IF NOT EXISTS perception_events (
                event_key TEXT PRIMARY KEY, session_id TEXT NOT NULL, stream_id TEXT NOT NULL,
                source_event_key TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL,
                confirmed INTEGER NOT NULL, recovery_confirmed INTEGER NOT NULL,
                latest_snapshot INTEGER NOT NULL REFERENCES perception_snapshots(id));
            CREATE TABLE IF NOT EXISTS perception_outbox (
                id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE REFERENCES perception_events(event_key),
                status TEXT NOT NULL CHECK(status IN ('pending','acknowledged')),
                attempts INTEGER NOT NULL DEFAULT 0, recommendation TEXT NOT NULL,
                receipt TEXT, last_error TEXT);
            CREATE TABLE IF NOT EXISTS perception_simulated_deliveries (
                event_key TEXT PRIMARY KEY REFERENCES perception_events(event_key), receipt TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def ingest(self, session_id, payload):
        """Persist stale evidence without regressing the current stream or event.

        A delayed first confirmed event still queues a historical-evidence review;
        it does not assert a currently active fall. Known recovery suppresses it.

        Exact frame retries are idempotent. Different content at the same source
        timestamp is rejected rather than silently replacing earlier evidence.
        First-seen confirmed events are queued even if upstream's in-memory
        ``new_event``/``emit`` flags are false. Suspected/unknown frames never
        create a notification, irrespective of raw model score.
        """
        session_id = _name(session_id, 'session_id')
        stream_id = _name(payload.get('stream_id'), 'stream_id')
        t = payload.get('source_time_s')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError('source_time_s must be finite and nonnegative')
        if payload.get('schema_version') != 1:
            raise ValueError('Only PerceptionChain schema_version 1 is supported')
        event = payload.get('event')
        if not isinstance(event, dict):
            raise ValueError('event must be an object')
        upstream_key = event.get('event_key')
        key = None
        if upstream_key is not None:
            _name(upstream_key, 'event.event_key')
            if not upstream_key.startswith(stream_id + '/') or upstream_key == stream_id + '/':
                raise ValueError('event_key must belong to the payload stream')
            key = _json([session_id, stream_id, upstream_key])
        encoded = _json(payload)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute(
                'SELECT digest,event_key,stale FROM perception_snapshots WHERE session_id=? AND stream_id=? AND source_time=?',
                (session_id, stream_id, t)).fetchone()
            if prior:
                if prior['digest'] != digest:
                    raise ValueError('Conflicting payload at an already persisted source timestamp')
                self.db.commit()
                return dict(status='duplicate', event_key=prior['event_key'], outbox_created=False)
            clock = self.db.execute('SELECT latest_time FROM perception_streams WHERE session_id=? AND stream_id=?',
                                    (session_id, stream_id)).fetchone()
            stale = bool(clock and t < clock['latest_time'])
            snapshot = self.db.execute(
                'INSERT INTO perception_snapshots(session_id,stream_id,source_time,digest,payload,stale,event_key) '
                'VALUES(?,?,?,?,?,?,?)', (session_id, stream_id, t, digest, encoded, int(stale), key)).lastrowid
            created = False
            if not stale:
                self.db.execute('INSERT INTO perception_streams VALUES(?,?,?) ON CONFLICT(session_id,stream_id) '
                                'DO UPDATE SET latest_time=excluded.latest_time', (session_id, stream_id, t))
            confirmed = event.get('state') == 'confirmed' and event.get('alarm_authority') == AUTHORITY
            if key and (not stale or confirmed):
                recovered = (not stale and event.get('recovery_confirmed') is True
                             and event.get('recovery_status') == 'confirmed'
                             and event.get('alarm_authority') == AUTHORITY
                             and any(tr.get('kind') == 'rearmed'
                                     and tr.get('recovery_scope') == 'standing_or_seated'
                                     and stream_id + '/' + str(tr.get('previous_event_id')) == upstream_key
                                     for tr in event.get('transitions', []) if isinstance(tr, dict)))
                self.db.execute('''INSERT INTO perception_events VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_key) DO UPDATE SET first_seen=MIN(perception_events.first_seen,excluded.first_seen),
                    last_seen=MAX(perception_events.last_seen,excluded.last_seen),
                    confirmed=MAX(perception_events.confirmed,excluded.confirmed),
                    recovery_confirmed=MAX(perception_events.recovery_confirmed,excluded.recovery_confirmed),
                    latest_snapshot=CASE WHEN excluded.last_seen>=perception_events.last_seen
                    THEN excluded.latest_snapshot ELSE perception_events.latest_snapshot END''',
                                (key, session_id, stream_id, upstream_key, t, t,
                                 int(confirmed), int(recovered), snapshot))
                event_record = self.db.execute('SELECT recovery_confirmed FROM perception_events WHERE event_key=?',
                                               (key,)).fetchone()
                if confirmed and not event_record['recovery_confirmed']:
                    recommendation = dict(
                        kind='inspect_delayed_fall_evidence' if stale else 'inspect_confirmed_fall',
                        delayed_evidence=stale, mode='simulation_only',
                        event_key=key, evidence_snapshot_id=snapshot,
                        text=('收到迟到的跌倒事件证据；先查询最新观测和恢复状态，不据此认定当前仍倒地。'
                              if stale else '查看跌倒事件证据并请求人工确认；由明确的处置策略决定后续联系。'),
                        external_action_performed=False)
                    created = bool(self.db.execute(
                        "INSERT OR IGNORE INTO perception_outbox(event_key,status,recommendation) VALUES(?,'pending',?)",
                        (key, _json(recommendation))).rowcount)
            self.db.commit()
            return dict(status=('stale_event_recorded' if created else 'stale_audit_only')
                        if stale else 'accepted', event_key=key,
                        outbox_created=created)
        except BaseException:
            self.db.rollback()
            raise

    def list_events(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM perception_events ORDER BY first_seen,event_key')]

    def get_event(self, event_key):
        row = self.db.execute('SELECT * FROM perception_events WHERE event_key=?', (event_key,)).fetchone()
        return dict(row) if row else None

    def evidence(self, event_key):
        """Return original snapshots, including transitions and stale markers."""
        return [dict(id=r['id'], source_time_s=r['source_time'], stale=bool(r['stale']),
                     payload=json.loads(r['payload'])) for r in self.db.execute(
                         'SELECT * FROM perception_snapshots WHERE event_key=? ORDER BY source_time,id', (event_key,))]

    def pending_notifications(self):
        rows = self.db.execute("SELECT * FROM perception_outbox WHERE status='pending' ORDER BY id")
        return [dict(dict(r), recommendation=json.loads(r['recommendation']),
                     current_event=self.get_event(r['event_key'])) for r in rows]

    def record_failure(self, outbox_id, error):
        """Leave a failed simulation pending; acknowledged records cannot regress."""
        error = _name(error, 'error')
        return bool(self.db.execute("UPDATE perception_outbox SET attempts=attempts+1,last_error=? "
                                    "WHERE id=? AND status='pending'", (error, outbox_id)).rowcount)

    def simulate_delivery(self, outbox_id):
        """Atomically record one LOCAL receipt and acknowledge it, retry safely.

        This does not claim exactly-once network delivery. Real delivery needs a
        recipient supporting idempotency, durable leases and an explicit policy.
        """
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM perception_outbox WHERE id=?', (outbox_id,)).fetchone()
            if row is None:
                raise KeyError(outbox_id)
            receipt = _json(dict(mode='simulation_only', event_key=row['event_key'],
                                 external_action_performed=False))
            self.db.execute('INSERT OR IGNORE INTO perception_simulated_deliveries VALUES(?,?)',
                            (row['event_key'], receipt))
            self.db.execute("UPDATE perception_outbox SET status='acknowledged',receipt=?,attempts=attempts+1,"
                            "last_error=NULL WHERE id=? AND status='pending'", (receipt, outbox_id))
            self.db.commit()
            return json.loads(receipt)
        except BaseException:
            self.db.rollback()
            raise
