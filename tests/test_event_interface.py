import pytest

from fallpre.event_interface import EventInterface, ProcessingConflict
from fallpre.event_store import AUTHORITY, EventStore


def add_event(store, event_id='fall-1', stream='camera-a', session='boot', t=1):
    payload = dict(schema_version=1, stream_id=stream, source_time_s=t,
                   observation=dict(observable=True),
                   event=dict(state='confirmed', event_key=f'{stream}/{event_id}',
                              recovery_confirmed=None, recovery_status='not_established',
                              alarm_authority=AUTHORITY, transitions=[]))
    return store.ingest(session, payload)['event_key']


def test_resolution_is_only_handling_not_recovery_or_notification(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        key = add_event(store)
        api = EventInterface(store)
        before = api.get_event(key)
        pending = store.pending_notifications()
        assert before['processing']['status'] == 'unhandled'
        receipt = api.set_processing(key, 'resolved', expected_version=0, request_id='r1',
                                     actor='local-reviewer', note='已记录核查结果，后续另行处理')
        assert receipt['perception_changed'] is receipt['notification_changed'] is False
        after = api.get_event(key)
        assert after['processing']['status'] == 'resolved'
        assert after['perception'] == before['perception']
        assert after['latest_observation'] == before['latest_observation']
        assert after['perception']['recovery_confirmed'] == 0
        assert store.pending_notifications() == pending


def test_workflow_restart_and_exact_retry_return_original_receipt(tmp_path):
    path = tmp_path/'events.sqlite'
    with EventStore(path) as store:
        key = add_event(store)
        api = EventInterface(store)
        first = api.set_processing(key, 'acknowledged', expected_version=0, request_id='r1', actor='operator')
        api.set_processing(key, 'in_review', expected_version=1, request_id='r2', actor='operator')
    with EventStore(path) as store:
        api = EventInterface(store)
        assert api.set_processing(key, 'acknowledged', expected_version=0,
                                  request_id='r1', actor='operator') == first
        assert api.get_event(key)['processing']['version'] == 2
        assert len(api.processing_history(key)['items']) == 2


def test_compare_and_swap_between_connections_prevents_lost_updates(tmp_path):
    path = tmp_path/'events.sqlite'
    with EventStore(path) as a, EventStore(path) as b:
        key = add_event(a)
        first, second = EventInterface(a), EventInterface(b)
        first.set_processing(key, 'in_review', expected_version=0, request_id='r1', actor='a')
        with pytest.raises(ProcessingConflict, match='current 1'):
            second.set_processing(key, 'resolved', expected_version=0, request_id='r2', actor='b', note='checked')
        assert len(second.processing_history(key)['items']) == 1
        assert b.db.execute('SELECT COUNT(*) FROM perception_processing_requests').fetchone()[0] == 1
        second.set_processing(key, 'resolved', expected_version=1, request_id='r2', actor='b', note='checked')
        assert first.get_event(key)['processing']['version'] == 2


def test_reusing_request_id_for_other_content_is_rejected(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        a = add_event(store)
        b = add_event(store, 'fall-2', t=2)
        api = EventInterface(store)
        api.set_processing(a, 'acknowledged', expected_version=0, request_id='r1', actor='operator')
        with pytest.raises(ProcessingConflict, match='reused'):
            api.set_processing(b, 'acknowledged', expected_version=0, request_id='r1', actor='operator')
        assert api.get_event(b)['processing']['version'] == 0


def test_reopen_and_perception_updates_preserve_independent_workflow(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        key = add_event(store)
        api = EventInterface(store)
        api.set_processing(key, 'resolved', expected_version=0, request_id='r1', actor='operator', note='checked')
        add_event(store, t=2)
        assert api.get_event(key)['processing']['status'] == 'resolved'
        api.set_processing(key, 'in_review', expected_version=1, request_id='r2', actor='operator', note='new context')
        assert api.get_event(key)['perception']['last_seen'] == 2
        history = api.processing_history(key, limit=1)
        assert history['next_after_version'] == 1
        last = api.processing_history(key, after_version=1, limit=1)
        assert last['items'][0]['prior_status'] == 'resolved'
        assert last['next_after_version'] is None


def test_query_filters_and_keyset_pagination(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        keys = [add_event(store), add_event(store, 'fall-2', t=2), add_event(store, stream='camera-b')]
        api = EventInterface(store)
        api.set_processing(keys[1], 'in_review', expected_version=0, request_id='r1', actor='operator')
        first = api.list_events(session_id='boot', stream_id='camera-a', limit=1)
        second = api.list_events(session_id='boot', stream_id='camera-a', limit=1,
                                 after_event_key=first['next_after_event_key'])
        assert [first['items'][0]['event_key'], second['items'][0]['event_key']] == sorted(keys[:2])
        assert second['next_after_event_key'] is None
        assert len(api.list_events(processing_status='unhandled')['items']) == 2
        assert api.list_events(processing_status='in_review')['items'][0]['event_key'] == keys[1]
        assert api.list_events(session_id='other')['items'] == []


def test_evidence_pagination_keeps_original_payloads_and_stale_markers(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        key = add_event(store, t=2)
        add_event(store, t=1)
        api = EventInterface(store)
        first = api.get_evidence(key, limit=1)
        second = api.get_evidence(key, after_id=first['next_after_id'], limit=1)
        assert first['items'][0]['source_time_s'] == 2
        assert second['items'][0]['source_time_s'] == 1
        assert second['items'][0]['stale'] is True
        assert second['next_after_id'] is None


@pytest.mark.parametrize('arguments', [
    dict(status='resolved'), dict(status='safe'), dict(expected_version=True),
    dict(actor=''), dict(note='x'*4001), dict(request_id=''),
])
def test_invalid_processing_requests_do_not_write(tmp_path, arguments):
    with EventStore(tmp_path/'events.sqlite') as store:
        key = add_event(store)
        api = EventInterface(store)
        kwargs = dict(status='acknowledged', expected_version=0, request_id='r1', actor='operator')
        kwargs.update(arguments)
        with pytest.raises(ValueError):
            api.set_processing(key, **kwargs)
        assert api.get_event(key)['processing']['version'] == 0
        assert api.processing_history(key)['items'] == []


def test_missing_event_and_invalid_query_are_explicit(tmp_path):
    with EventStore(tmp_path/'events.sqlite') as store:
        api = EventInterface(store)
        for query in [api.get_event, api.get_evidence, api.processing_history]:
            with pytest.raises(KeyError):
                query('missing')
        with pytest.raises(KeyError):
            api.set_processing('missing', 'in_review', expected_version=0, request_id='r1', actor='operator')
        for limit in [0, 501, True]:
            with pytest.raises(ValueError):
                api.list_events(limit=limit)
        assert store.db.execute('SELECT COUNT(*) FROM perception_processing_requests').fetchone()[0] == 0
