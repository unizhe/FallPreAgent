import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from fallpre.adapters.jev_suggestions import assess
from fallpre.agent import FallPreventionAgent
from fallpre.care_loop import CareLoop, perception_payload
from fallpre.event_store import EventStore
from fallpre.state_store import StateStore

ROOT = Path(__file__).resolve().parents[1]

def row(t, state='normal', action='standing', **kw):
    value = dict(source_time_s=t, primary=dict(state=state, event_id='e1' if state == 'confirmed' else None,
                 current_event_id='e1' if state == 'confirmed' else None, transitions=[], new_event=state == 'confirmed'),
                 action=action, target_epoch=1, camera_stable=True, pose_count=1, temporal_status='ready',
                 head={'score': .999})
    value.update(kw)
    return perception_payload('s', value)


@pytest.fixture
def loop(tmp_path):
    e = EventStore(tmp_path/'e.sqlite')
    x = StateStore(str(tmp_path/'x.sqlite'))
    c = CareLoop(e, x)
    yield c
    c.close()
    x.close()
    e.close()


def test_alarm_idempotence_and_processing(loop):
    p = row(1, 'confirmed')
    original = deepcopy(p)
    a = FallPreventionAgent(loop)
    out = a.observe('session', p)
    assert len(out['executions']) == 1
    assert out['executions'][0]['status'] == 'SIMULATED'
    assert not a.observe('session', p)['executions']
    assert p == original
    record = loop.api.get_event(out['ingest']['event_key'])
    assert record['processing']['status'] == 'in_review'
    assert not record['perception']['recovery_confirmed']


def test_reminder_does_not_suppress_alarm(loop):
    loop.observe('session', row(1, action='seated'))
    out = loop.observe('session', row(1.1, action='standing_up'))
    assert out['executions'][0]['action_type'] == 'STAND_STEADY_REMINDER'
    assert loop.observe('session', row(1.2, 'confirmed'))['executions'][0]['action_type'] == 'FALL_ALARM'


@pytest.mark.parametrize('changes', [dict(pose_count=2), dict(camera_stable=False),
                                   dict(target_epoch=2), dict(temporal_status='unobservable')])
def test_invalid_observation_cannot_remind(loop, changes):
    loop.observe('session', row(1, action='seated'))
    assert not loop.observe('session', row(1.1, action='standing_up', **changes))['executions']


def test_missing_and_scores_do_not_alarm(loop):
    assert not loop.observe('session', row(1, 'unobservable'))['executions']
    assert not loop.observe('session', row(2, 'suspected'))['executions']
    assert not loop.events.pending_notifications()


def test_late_confirmation_is_review_only(loop):
    loop.observe('session', row(3))
    assert not loop.observe('session', row(1, 'confirmed'))['executions']
    assert loop.events.pending_notifications()[0]['recommendation']['delayed_evidence']


def test_restart_deduplicates(tmp_path):
    for _ in range(2):
        with EventStore(tmp_path/'e.sqlite') as e:
            x = StateStore(str(tmp_path/'x.sqlite'))
            c = CareLoop(e, x)
            c.observe('session', row(1, 'confirmed'))
            assert len(x.list_actions()) == 1
            c.close()
            x.close()


def test_jev_is_advisory_and_validated():
    request = dict(state=dict(branch='reminder'))
    assert assess(request)['status'] == 'NOT_RUN'
    def good(_):
        return {'model': 'test', 'answers': {'reminder_needed': {'type': 'noul', 'noul': .8}}}
    assert assess(request, enabled=True, transport=good)['score'] == .8
    def bad(_):
        return {'model': 'test', 'answers': {'reminder_needed': {'type': 'noul', 'noul': float('nan')}}}
    assert assess(request, enabled=True, transport=bad)['status'] == 'UNAVAILABLE'


def test_feedback_is_observation_not_intervention_success(loop):
    loop.observe('session', row(1, action='seated'))
    loop.observe('session', row(1.1, action='standing_up'))
    loop.observe('session', row(1.2, action='standing'))
    feedback = json.loads(loop.events.db.execute('SELECT body FROM care_feedback').fetchone()[0])
    assert feedback['outcome'] == 'observed_standing'
    assert feedback['intervention_effect_proven'] is False


def test_live_hook_and_query_tools(loop):
    live = dict(fall=dict(state='confirmed', event_id='x', current_event_id='x'),
                source_time=1, action=dict(label='unknown', selection={}),
                quality={}, persons=[])
    original = deepcopy(live)
    result = loop.observe_live('session', 's', live)
    key = result['ingest']['event_key']
    agent = FallPreventionAgent(loop)
    assert agent.invoke('get_event', {'event_key': key})['perception']['confirmed'] == 1
    assert live == original


def test_bundled_demo_runs_through_cli(tmp_path):
    output = tmp_path / 'demo'
    subprocess.run([sys.executable, str(ROOT / 'scripts/replay_care_agent.py'),
                    '--demo', '--output', str(output)], check=True, cwd=ROOT)
    summary = json.loads((output / 'summary.json').read_text())
    trace = [json.loads(line) for line in (output / 'trace.jsonl').read_text().splitlines()]
    assert [(r['source_time_s'], r['result']['action_type'], r['result']['status']) for r in trace] == [
        (.2, 'STAND_STEADY_REMINDER', 'SIMULATED'), (1.2, 'FALL_ALARM', 'SIMULATED'),
    ]
    assert summary['counts']['duplicate_retries_checked'] == 7
    assert summary['external_actions'] == summary['pending_notifications'] == 0


def test_shadow_log_can_replay_without_using_shadow_alarm(tmp_path, monkeypatch):
    # Exercise the logger and CLI together, without model weights or hardware.
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from webcam_pose_demo import PoseEngine
    from webcam_tail_shadow import TailShadowEngine

    def infer(_engine, _image, _stream, t, *_args):
        return dict(fall_mode='rgb', source_time=t,
                    fall=dict(state='normal', new_event=False, current_event_id=None),
                    action=dict(label='standing', selection=dict(target_epoch=1)),
                    quality=dict(camera_stable=True), persons=[{}], temporal=dict(status='ready'))

    class Shadow:
        def update(self, frame, timestamp):
            return dict(head={'status': 'ready', 'fall_process_score': .99},
                        evidence=dict(association='matched', code='test', support=True),
                        simulated_fall=dict(state='confirmed', new_event=True, current_event_id='shadow'))

    monkeypatch.setattr(PoseEngine, 'infer_live', infer)
    incoming = tmp_path / 'shadow.jsonl'
    engine = TailShadowEngine.__new__(TailShadowEngine)
    engine.shadow_streams = {'s': Shadow()}
    engine.log_error = None
    with incoming.open('w') as engine.shadow_log:
        engine.infer_live(None, 's', 1.2)
    logged = json.loads(incoming.read_text())
    assert logged['shadow_new_event'] is True
    assert logged['perception']['event']['state'] == 'normal'
    output = tmp_path / 'replay'
    subprocess.run([sys.executable, str(ROOT / 'scripts/replay_care_agent.py'),
                    '--input', str(incoming), '--output', str(output)], check=True, cwd=ROOT)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['counts']['frames'] == 1
    assert summary['confirmed_events'] == summary['execution_count'] == 0
