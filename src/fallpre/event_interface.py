"""Small, platform-neutral query and handling interface for perception events.

No LLM, jev SDK, network endpoint or orchestration is assumed. Processing state
is a caller's work record, never a physical recovery, detector label or delivery
receipt. Actor names are caller-supplied audit data, not verified identities.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .event_store import EventStore, _json, _name

PROCESSING_STATES = ('unhandled', 'acknowledged', 'in_review', 'resolved')


class ProcessingConflict(ValueError):
    """Caller must reread the event, not silently overwrite another handler."""


def _limit(value):
    if type(value) is not int or not 1 <= value <= 500:
        raise ValueError('limit must be an integer from 1 to 500')
    return value


def _version(value):
    if type(value) is not int or value < 0:
        raise ValueError('expected_version must be a nonnegative integer')
    return value


class EventInterface:
    """Use a caller-owned EventStore; operations return JSON-serializable dicts.

    ``request_id`` is globally unique within this database. An exact retry returns
    its original receipt, even if processing later advanced. Read get_event() for
    the current state. Different request content reusing that ID is rejected.
    """

    def __init__(self, store: EventStore):
        self.store = store
        self.db = store.db
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS perception_processing (
                event_key TEXT PRIMARY KEY REFERENCES perception_events(event_key),
                status TEXT NOT NULL CHECK(status IN ('unhandled','acknowledged','in_review','resolved')),
                version INTEGER NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS perception_processing_history (
                event_key TEXT NOT NULL REFERENCES perception_events(event_key),
                version INTEGER NOT NULL, request_id TEXT NOT NULL UNIQUE,
                prior_status TEXT NOT NULL, status TEXT NOT NULL,
                actor TEXT NOT NULL, note TEXT NOT NULL, recorded_at TEXT NOT NULL,
                PRIMARY KEY(event_key,version));
            CREATE TABLE IF NOT EXISTS perception_processing_requests (
                request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt TEXT NOT NULL);
        ''')

    def _processing(self, event_key):
        row = self.db.execute('SELECT * FROM perception_processing WHERE event_key=?', (event_key,)).fetchone()
        return dict(row) if row else dict(event_key=event_key, status='unhandled', version=0,
                                         actor=None, note='', updated_at=None)

    def get_event(self, event_key):
        _name(event_key, 'event_key')
        # A read transaction keeps perception metadata, evidence and workflow
        # coherent if a different DB connection writes at the same time.
        self.db.execute('BEGIN')
        try:
            event = self.store.get_event(event_key)
            if event is None:
                raise KeyError(event_key)
            row = self.db.execute('SELECT payload FROM perception_snapshots WHERE id=?',
                                  (event['latest_snapshot'],)).fetchone()
            result = dict(schema_version=1, event_key=event_key, perception=event,
                          latest_observation=json.loads(row['payload']), processing=self._processing(event_key))
            self.db.commit()
            return result
        except BaseException:
            self.db.rollback()
            raise

    def list_events(self, *, session_id=None, stream_id=None, processing_status=None,
                    after_event_key=None, limit=100):
        """Keyset page ordered by opaque event_key; filters use exact values.

        Pages are not a frozen multi-request snapshot. Concurrent inserts with an
        earlier key require a fresh scan. No pagination cursor grants access.
        """
        limit = _limit(limit)
        where, parameters = [], []
        for field, value in [('session_id', session_id), ('stream_id', stream_id)]:
            if value is not None:
                where.append(f'e.{field}=?')
                parameters.append(_name(value, field))
        if processing_status is not None:
            if processing_status not in PROCESSING_STATES:
                raise ValueError('Unknown processing_status')
            where.append("COALESCE(p.status,'unhandled')=?")
            parameters.append(processing_status)
        if after_event_key is not None:
            where.append('e.event_key>?')
            parameters.append(_name(after_event_key, 'after_event_key'))
        sql = ('SELECT e.*,COALESCE(p.status,\'unhandled\') AS processing_status,'
               'COALESCE(p.version,0) AS processing_version FROM perception_events e '
               'LEFT JOIN perception_processing p ON e.event_key=p.event_key')
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        rows = [dict(r) for r in self.db.execute(sql + ' ORDER BY e.event_key LIMIT ?',
                                                (*parameters, limit + 1))]
        items = rows[:limit]
        return dict(schema_version=1, items=items,
                    next_after_event_key=items[-1]['event_key'] if len(rows) > limit else None)

    def get_evidence(self, event_key, *, after_id=0, limit=100):
        """Page original snapshots in ingestion-ID order, with source time/staleness."""
        _name(event_key, 'event_key')
        limit = _limit(limit)
        if type(after_id) is not int or after_id < 0:
            raise ValueError('after_id must be a nonnegative integer')
        if self.store.get_event(event_key) is None:
            raise KeyError(event_key)
        rows = list(self.db.execute('SELECT * FROM perception_snapshots WHERE event_key=? AND id>? '
                                    'ORDER BY id LIMIT ?', (event_key, after_id, limit + 1)))
        items = [dict(id=r['id'], source_time_s=r['source_time'], stale=bool(r['stale']),
                      payload=json.loads(r['payload'])) for r in rows[:limit]]
        return dict(schema_version=1, event_key=event_key, items=items,
                    next_after_id=items[-1]['id'] if len(rows) > limit else None)

    def set_processing(self, event_key, status, *, expected_version, request_id, actor, note=''):
        """Persist a work-status transition without touching detection or notification.

        Any explicit status is allowed, including reopening resolved -> in_review.
        Every new request increments version and retains an audit row. Resolution
        requires a note but never establishes physical recovery or clears alarms.
        """
        _name(event_key, 'event_key')
        _name(request_id, 'request_id')
        _name(actor, 'actor')
        _version(expected_version)
        if status not in PROCESSING_STATES:
            raise ValueError('Unknown processing status')
        if not isinstance(note, str) or len(note) > 4000:
            raise ValueError('note must be a string of at most 4000 characters')
        if status == 'resolved' and not note.strip():
            raise ValueError('resolved requires a handling note, not a recovery assertion')
        request = dict(event_key=event_key, status=status, expected_version=expected_version,
                       request_id=request_id, actor=actor, note=note)
        digest = hashlib.sha256(_json(request).encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT * FROM perception_processing_requests WHERE request_id=?',
                                    (request_id,)).fetchone()
            if prior:
                if prior['digest'] != digest:
                    raise ProcessingConflict('request_id was reused for different processing content')
                self.db.commit()
                return json.loads(prior['receipt'])
            if self.store.get_event(event_key) is None:
                raise KeyError(event_key)
            previous = self._processing(event_key)
            if previous['version'] != expected_version:
                raise ProcessingConflict(f"Expected version {expected_version}; current {previous['version']}")
            recorded_at = datetime.now(timezone.utc).isoformat()
            version = expected_version + 1
            self.db.execute('''INSERT INTO perception_processing VALUES(?,?,?,?,?,?)
                ON CONFLICT(event_key) DO UPDATE SET status=excluded.status,version=excluded.version,
                actor=excluded.actor,note=excluded.note,updated_at=excluded.updated_at''',
                            (event_key, status, version, actor, note, recorded_at))
            self.db.execute('INSERT INTO perception_processing_history VALUES(?,?,?,?,?,?,?,?)',
                            (event_key, version, request_id, previous['status'], status, actor, note, recorded_at))
            receipt = dict(schema_version=1, event_key=event_key, request_id=request_id,
                           processing=dict(status=status, version=version, actor=actor, note=note,
                                           updated_at=recorded_at),
                           perception_changed=False, notification_changed=False)
            self.db.execute('INSERT INTO perception_processing_requests VALUES(?,?,?)',
                            (request_id, digest, _json(receipt)))
            self.db.commit()
            return receipt
        except BaseException:
            self.db.rollback()
            raise

    def processing_history(self, event_key, *, after_version=0, limit=100):
        _name(event_key, 'event_key')
        _version(after_version)
        limit = _limit(limit)
        if self.store.get_event(event_key) is None:
            raise KeyError(event_key)
        rows = [dict(r) for r in self.db.execute(
            'SELECT * FROM perception_processing_history WHERE event_key=? AND version>? '
            'ORDER BY version LIMIT ?', (event_key, after_version, limit + 1))]
        items = rows[:limit]
        return dict(schema_version=1, event_key=event_key, items=items,
                    next_after_version=items[-1]['version'] if len(rows) > limit else None)
