from copy import deepcopy

from test_stgcn_fall_evidence import result, temporal

from fallpre.weekly.perception_chain import PerceptionChain


def source(t, label='standing'):
    r = result()
    r['quality']['reason'] = 'stable'
    r['action'].update(label=label, text='站立' if label == 'standing' else '未知')
    r['pose_fall_observation'] = dict(person=r['persons'][0], selection=dict(target_epoch=7))
    r['temporal'] = temporal(t)
    r['temporal'].update(window_end_s=t, fall_heads=dict(st_geometry=dict(status='ready', fall_process_score=.99)))
    return r


def test_known_action_is_green_even_when_fall_is_undecidable():
    chain = PerceptionChain('camera-a')
    r = source(0)
    original = deepcopy(r)
    out = chain.update(r, 0)
    assert r == original
    assert out['current_action']['text'] == '站立'
    assert out['display']['color'] == 'green'
    assert out['event']['state'] == 'unobservable'
    assert not out['notification']['emit']


def test_binary_high_score_never_emits_event_without_witnessed_process():
    chain = PerceptionChain('camera-a')
    for i in range(30):
        out = chain.update(source(i / 10), i / 10)
        assert not out['notification']['emit']
        assert out['fall_process']['changes_alarm'] is False


def test_missing_subject_is_unknown_not_nonfall():
    chain = PerceptionChain('camera-a')
    r = source(0)
    r['persons'] = []
    r['action']['target_index'] = None
    r['pose_fall_observation']['person'] = None
    out = chain.update(r, 0)
    assert out['fall_process']['label'] == 'unknown'
    assert out['display']['color'] == 'gray'
    assert out['event']['recovery_confirmed'] is None


def test_future_history_is_not_displayed_as_current_classification():
    chain = PerceptionChain('camera-a')
    for i in range(20):
        t = i / 10
        r = source(t)
        r['temporal']['window_end_s'] = t + 1
        out = chain.update(r, t)
        assert out['temporal_action']['top3'] == []
        assert out['fall_process']['label'] == 'unknown'


def test_notifications_are_deduplicated_within_stream_not_across_streams():
    from types import SimpleNamespace

    def confirmed(*args):
        return dict(fall=dict(state='confirmed', lifecycle='new_confirmed', event_id='rgb-fall-0001',
                             current_event_id='rgb-fall-0001', new_event=True, reason='test'),
                    observation=dict(person=None), temporal_accelerated=False)
    one, two = PerceptionChain('camera-a'), PerceptionChain('camera-b')
    one.events = SimpleNamespace(update=confirmed)
    two.events = SimpleNamespace(update=confirmed)
    first = one.update(source(0), 0)
    repeat = one.update(source(.1), .1)
    other = two.update(source(0), 0)
    assert first['notification']['emit'] and not repeat['notification']['emit']
    assert other['notification']['emit']
    assert first['event']['event_key'] != other['event']['event_key']


def test_process_classifier_does_not_require_a_positive_rgb_fall_detection():
    chain = PerceptionChain('camera-a')
    for i in range(30):
        t = i / 10
        r = source(t)
        r['rgb_raw_candidates']['fallen'] = []
        r['temporal']['fall_heads']['st_geometry']['fall_process_score'] = .1
        out = chain.update(r, t)
    assert out['fall_process']['label'] == 'non_fall_activity'
    assert not out['notification']['emit']


def test_second_person_blocks_binary_admission_even_with_high_head_score():
    chain = PerceptionChain('camera-a')
    for i in range(30):
        t = i / 10
        r = source(t)
        r['rgb_raw_candidates']['persons'].append(dict(box=[550, 30, 630, 300], confidence=.4))
        out = chain.update(r, t)
    assert out['fall_process']['label'] == 'unknown'
    assert not out['notification']['emit']
