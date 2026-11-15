"""Single Agent bridge. Source decisions are authoritative; execution is dry-run only."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .event_interface import EventInterface
from .event_store import AUTHORITY, EventStore
from .execution import ExecutionService
from .schemas import RiskDecision, Snapshot
from .state_store import StateStore


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def perception_payload(stream, row):
    """Adapt recorded live-engine output without rerunning any alarm policy."""
    f = row['primary']
    key = f'{stream}/{f["event_id"]}' if f.get('event_id') else None
    rearmed = next((x for x in f.get('transitions', []) if x.get('kind') == 'rearmed'
                    and x.get('recovery_scope') == 'standing_or_seated'), None)
    visible = (row.get('camera_stable') is True and row.get('pose_count') == 1
               and row.get('target_epoch') is not None and row.get('temporal_status') == 'ready')
    return dict(schema_version=1, stream_id=stream, source_time_s=row['source_time_s'],
                identity=dict(action_epoch=row.get('target_epoch')),
                observation=dict(observable=visible, event_decidable=f['state'] != 'unobservable'),
                current_action=dict(label=row['action']),
                fall_process=dict(head=deepcopy(row.get('head')), changes_alarm=False),
                event=dict(state=f['state'], event_key=key, lifecycle=f.get('lifecycle'),
                           active_event_key=key if f.get('current_event_id') else None,
                           transitions=deepcopy(f.get('transitions', [])),
                           recovery_confirmed=True if rearmed else None,
                           recovery_status='confirmed' if rearmed else 'not_established',
                           alarm_authority=AUTHORITY),
                notification=dict(emit=bool(f.get('new_event')), external_action_performed=False))


def live_perception_payload(stream, result):
    """Project the live engine's primary result onto the Agent's schema v1."""
    return perception_payload(stream, dict(
        primary=result['fall'], source_time_s=result['source_time'],
        action=result['action']['label'],
        target_epoch=result['action'].get('selection', {}).get('target_epoch'),
        camera_stable=result['quality'].get('camera_stable'),
        pose_count=len(result['persons']),
        temporal_status=result.get('temporal', {}).get('status'),
        head=result.get('tail_process_shadow', {}).get('head'),
    ))


class CareLoop:
    """One synchronous owner per database; optional model runs in a separate comparison job.

    Reminder scenario: one simulated stand-steady cue following observed seated →
    standing_up, with a 30 source-second cooldown. This is an activity cue,
    not a validated instability detector or pre-fall prediction.
    """
    def __init__(self, events: EventStore, execution_store: StateStore):
        self.events = events
        self.api = EventInterface(events)
        self.actions = execution_store
        self.tools = self
        self.executor = ExecutionService(execution_store, self._validate, dry_run=True)
        events.db.executescript('''
            CREATE TABLE IF NOT EXISTS care_feedback(key TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS care_progress(key TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS care_suggestions(key TEXT PRIMARY KEY, body TEXT NOT NULL);
        ''')

    def list_tools(self):
        return [dict(name=name) for name in ("get_event", "get_evidence", "list_events", "processing_history")]

    def call(self, name, arguments):
        if name not in {x["name"] for x in self.list_tools()}:
            raise ValueError("Unsupported tool")
        return getattr(self.api, name)(**arguments)

    def _validate(self, decision, action):
        if action == 'FALL_ALARM':
            if not self.events.get_event(decision.binding_id)['confirmed']:
                raise ValueError('No authoritative confirmed event')
        elif action != 'STAND_STEADY_REMINDER' or decision.rule_version != 'care-simulation-v1':
            raise ValueError('Unsupported simulated action')

    def _execute(self, key, stream, t, action):
        # StateStore's unique cache key survives a restart between persistence
        # and dispatch. Reuse its execution journal and action idempotency.
        decision = self.actions.cached_decision(key)
        if decision is None:
            snap = Snapshot(t_s=t, person_id=stream, binding_id=key)
            event = self.actions.create_event(snap, 'alarm' if action == 'FALL_ALARM' else 'reminder', t)
            decision = RiskDecision(event_id=event.event_id, event_revision=1,
                                    snapshot_id=snap.snapshot_id, person_id=stream, binding_id=key,
                                    branch=event.branch, outcome=action, allowed_actions=[action],
                                    issued_at_s=t, expires_at_s=t+1, action_sequence=1,
                                    confirmed_alarm=action == 'FALL_ALARM', rule_version='care-simulation-v1')
            event.revision = 1
            event.latest_decision_id = decision.decision_id
            event.confirmed_alarm = decision.confirmed_alarm
            self.actions.commit_decision(event, 0, decision, key)
        return self.executor.execute(decision.decision_id, action).model_dump()

    def _suggest(self, key, branch, payload, baseline, context):
        # No video, local paths, or personal identity is sent in exported state.
        state = dict(branch=branch, source_time_s=payload['source_time_s'],
                     observation=payload['observation'], current_action=payload['current_action'],
                     event_state=payload['event']['state'], recovery=payload['event'].get('recovery_status'),
                     fall_process=payload.get('fall_process'), baseline=baseline,
                     prediction_validated=False, recent_history=context.get("history", []),
                     previous_reminder_time_s=context.get("last_cue") if context.get("last_cue", -1e9) >= 0 else None)
        body = dict(request_id=digest(key), state=state, changes_alarm=False,
                    model_status='NOT_RUN', evidence_digest=digest(payload))
        self.events.db.execute('INSERT OR IGNORE INTO care_suggestions VALUES(?,?)',
                               (key, json.dumps(body)))

    def observe(self, session, payload):
        result = self.events.ingest(session, payload)
        receipts = []
        # Durable confirmed-event outbox is drained even on duplicate retries.
        for item in self.events.pending_notifications():
            if item['recommendation']['delayed_evidence']:
                continue  # historical review must not become a current alarm
            key = item['event_key']
            receipt = self._execute(key, item['current_event']['stream_id'],
                                    item['current_event']['last_seen'], 'FALL_ALARM')
            current = self.api.get_event(key)
            if current['processing']['version'] == 0:
                self.api.set_processing(key, 'in_review', expected_version=0,
                                        request_id=digest(['start', key]), actor='care-agent-simulation',
                                        note='模拟报警已执行；待处理，不代表已恢复或实际送达。')
            self.events.simulate_delivery(item["id"])
            receipts.append(receipt)
        if result['status'] != 'accepted':
            return dict(ingest=result, executions=receipts)
        stream, t = payload['stream_id'], payload['source_time_s']
        progress_key = json.dumps([session, stream])
        prior = self.events.db.execute('SELECT body FROM care_progress WHERE key=?', (progress_key,)).fetchone()
        p = json.loads(prior[0]) if prior else {}
        epoch = payload.get('identity', {}).get('action_epoch')
        valid = payload['observation']['observable'] is True and epoch is not None
        state = payload['event']['state']
        label = payload['current_action']['label']
        continuous = valid and p.get('epoch') == epoch and 0 < t-p.get('t', -1) <= .35
        seated = p.get('seated') if continuous else None
        last_cue = p.get('last_cue', -1e9)
        cue = (seated is not None and label == 'standing_up' and state == 'normal'
               and t-last_cue >= 30)
        pending = p.get('pending_cue')
        if pending:
            outcome = ('fall_confirmed' if state == 'confirmed' else
                       'observation_lost' if not continuous else
                       'observed_standing' if label == 'standing' else
                       'observation_timeout' if t-pending['t'] >= 5 else None)
            if outcome:
                feedback = dict(cue_time_s=pending['t'], observed_at_s=t, outcome=outcome,
                                intervention_effect_proven=False, physically_recovered=None)
                self.events.db.execute('INSERT OR REPLACE INTO care_feedback VALUES(?,?)',
                                       (pending['key'], json.dumps(feedback)))
                pending = None
        if cue:
            key = json.dumps([session, stream, epoch, 'rise', seated])
            receipts.append(self._execute(key, stream, t, 'STAND_STEADY_REMINDER'))
            self._suggest(key, 'reminder', payload, 'STAND_STEADY_REMINDER_SIMULATED', p)
            pending = dict(key=key, t=t)
            last_cue = t
        if state == 'suspected' and p.get('state') != 'suspected':
            self._suggest(json.dumps([session, stream, t, 'suspected']), 'alarm', payload, 'WAIT', p)
        history = [h for h in p.get('history', []) if continuous and 0 <= t-h['t'] <= 2.5]
        history.append(dict(t=t, label=label, observable=valid, event_state=state))
        next_p = dict(history=history[-32:], t=t, epoch=epoch, state=state, last_cue=last_cue, pending_cue=pending,
                      seated=(seated if seated is not None else t)
                      if valid and label == 'seated' and state == 'normal' else None)
        self.events.db.execute('INSERT OR REPLACE INTO care_progress VALUES(?,?)',
                               (progress_key, json.dumps(next_p)))
        return dict(ingest=result, executions=receipts)

    def observe_live(self, session, stream, result):
        """Opt-in integration hook for PoseEngine; never mutates its output."""
        return self.observe(session, live_perception_payload(stream, result))

    def close(self):
        self.executor.close()
