from copy import deepcopy

import pytest

from fallpre.event_store import AUTHORITY, EventStore


def observation(t=1, stream='camera-a', state='confirmed', event_id='fall-1', **event_updates):
    event = dict(state=state, event_key=f'{stream}/{event_id}' if event_id else None,
                 new_event=True, recovery_confirmed=None, recovery_status='not_established',
                 alarm_authority=AUTHORITY, transitions=[])
    event.update(event_updates)
    return dict(schema_version=1, stream_id=stream, source_time_s=t,
                observation=dict(observable=True), event=event,
                fall_process=dict(label='fall_process', score=.99))


def test_restart_replay_is_idempotent_and_ack_is_durable(tmp_path):
    path = tmp_path/'perception_events.db'
    with EventStore(path) as store:
        first = store.ingest('producer-boot-1', observation())
        assert first['outbox_created']
        notification = store.pending_notifications()[0]
        assert store.record_failure(notification['id'], 'simulated failure')
    with EventStore(path) as store:
        assert store.ingest('producer-boot-1', observation())['status'] == 'duplicate'
        assert len(store.pending_notifications()) == 1
        assert store.pending_notifications()[0]['attempts'] == 1
        receipt = store.simulate_delivery(notification['id'])
        assert receipt['external_action_performed'] is False
    with EventStore(path) as store:
        assert store.simulate_delivery(notification['id']) == receipt
        assert not store.pending_notifications()
        assert not store.record_failure(notification['id'], 'late retry')
        assert store.db.execute('SELECT COUNT(*) FROM perception_simulated_deliveries').fetchone()[0] == 1


def test_stale_and_missing_observation_cannot_recover_event(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        key = store.ingest('boot', observation(2))['event_key']
        stale = observation(1, state='clear', recovery_confirmed=True, recovery_status='confirmed')
        assert store.ingest('boot', stale)['status'] == 'stale_audit_only'
        unknown = observation(3, state='unobservable')
        unknown['observation']['observable'] = False
        store.ingest('boot', unknown)
        assert store.get_event(key)['recovery_confirmed'] == 0
        assert store.get_event(key)['last_seen'] == 3
        assert store.evidence(key)[0]['stale'] is True
        store.ingest('boot', observation(4, state='clear', recovery_confirmed=True,
                                        recovery_status='confirmed', transitions=[dict(kind='rearmed', previous_event_id='fall-1',
                                                                                    recovery_scope='standing_or_seated')]))
        assert store.get_event(key)['recovery_confirmed'] == 1


def test_new_event_flag_not_required_when_first_seen_confirmed(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        result = store.ingest('boot', observation(new_event=False))
        assert result['outbox_created']
        assert not store.ingest('boot', observation(2, new_event=False))['outbox_created']


def test_stream_and_producer_session_separate_event_namespaces(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        keys = {store.ingest('boot-1', observation())['event_key'],
                store.ingest('boot-2', observation())['event_key'],
                store.ingest('boot-1', observation(stream='camera-b'))['event_key']}
        assert len(keys) == len(store.list_events()) == len(store.pending_notifications()) == 3


@pytest.mark.parametrize('state', ['suspected', 'unobservable', 'clear'])
def test_high_head_score_never_creates_alarm(tmp_path, state):
    with EventStore(tmp_path/'perception_events.db') as store:
        assert not store.ingest('boot', observation(state=state))['outbox_created']
        assert not store.pending_notifications()


def test_bad_authority_cannot_enqueue_or_recover(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        p = observation(alarm_authority='experimental_head', recovery_confirmed=True,
                        recovery_status='confirmed')
        result = store.ingest('boot', p)
        assert not result['outbox_created']
        assert store.get_event(result['event_key'])['recovery_confirmed'] == 0


def test_conflicting_retry_rolls_back_without_replacing_evidence(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        p = observation()
        original = deepcopy(p)
        key = store.ingest('boot', p)['event_key']
        p['event']['state'] = 'clear'
        with pytest.raises(ValueError, match='Conflicting'):
            store.ingest('boot', p)
        assert store.evidence(key)[0]['payload'] == original
        assert len(store.pending_notifications()) == 1


def test_validation_requires_explicit_session_and_matching_event_key(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        with pytest.raises(ValueError):
            store.ingest('', observation())
        with pytest.raises(ValueError):
            store.ingest('boot', observation(float('nan')))
        p = observation()
        p['event']['event_key'] = 'camera-b/fall-1'
        with pytest.raises(ValueError):
            store.ingest('boot', p)
        assert store.db.execute('SELECT COUNT(*) FROM perception_snapshots').fetchone()[0] == 0


def test_recovery_must_reference_the_same_event_and_cannot_reopen(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        other = dict(kind='rearmed', previous_event_id='fall-2', recovery_scope='standing_or_seated')
        p = observation(state='normal', recovery_confirmed=True, recovery_status='confirmed',
                        transitions=[other])
        key = store.ingest('boot', p)['event_key']
        assert store.get_event(key)['recovery_confirmed'] == 0
        p = observation(2, state='normal', recovery_confirmed=True, recovery_status='confirmed',
                        transitions=[dict(other, previous_event_id='fall-1')])
        store.ingest('boot', p)
        assert store.get_event(key)['recovery_confirmed'] == 1
        assert not store.ingest('boot', observation(3))['outbox_created']
        assert not store.pending_notifications()


def test_delayed_first_alarm_is_preserved_without_regressing_stream(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        store.ingest('boot', observation(3, state='normal', event_id=None))
        result = store.ingest('boot', observation(2))
        assert result['status'] == 'stale_event_recorded'
        assert result['outbox_created']
        assert store.db.execute('SELECT latest_time FROM perception_streams').fetchone()[0] == 3
        item = store.pending_notifications()[0]
        assert item['recommendation']['delayed_evidence']
        assert item['recommendation']['kind'] == 'inspect_delayed_fall_evidence'
        assert store.evidence(result['event_key'])[0]['stale']
        assert not store.ingest('boot', observation(2))['outbox_created']


def test_delayed_confirmed_history_cannot_reopen_recovered_event(tmp_path):
    with EventStore(tmp_path/'perception_events.db') as store:
        recovered = observation(3, state='normal', recovery_confirmed=True, recovery_status='confirmed',
                                transitions=[dict(kind='rearmed', previous_event_id='fall-1',
                                                  recovery_scope='standing_or_seated')])
        key = store.ingest('boot', recovered)['event_key']
        assert not store.ingest('boot', observation(2))['outbox_created']
        event = store.get_event(key)
        assert event['confirmed'] == event['recovery_confirmed'] == 1
        assert event['last_seen'] == 3
        latest = next(x for x in store.evidence(key) if x['id'] == event['latest_snapshot'])
        assert latest['payload']['event']['state'] == 'normal'
        assert not store.pending_notifications()


def test_namespaced_schema_coexists_with_existing_state_store(tmp_path):
    from fallpre.state_store import StateStore

    path = tmp_path/'shared.db'
    existing = StateStore(str(path))
    schema = existing.conn.execute("SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0]
    with EventStore(path) as store:
        store.ingest('boot', observation())
        assert store.db.execute("SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0] == schema
        assert store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 0
        assert len(store.list_events()) == 1
    existing.conn.close()
