"""Fusion must respect current geometry, identity, causality and alarm ownership."""
from copy import deepcopy

import pytest

from fallpre.weekly.stgcn_fusion import ConstrainedSTGCNFusion, FusionConfig
from fallpre.weekly.stgcn_temporal import MODEL_ID


def result(progress=0, label='seated', phase='seated', epoch=1):
    p = [[220., 100., .95] for _ in range(17)]
    for s, h, k, a, x in [(5, 11, 13, 15, 200), (6, 12, 14, 16, 240)]:
        p[s] = [x, 180 - 80 * progress, .95]
        p[h] = [x, 280 - 80 * progress, .95]
        p[k] = [x + 60 * (1 - progress), 340 - 30 * progress, .95]
        p[a] = [x, 430, .95]
    return dict(width=640, height=480, persons=[dict(keypoints=p)],
                quality={'camera_stable': True},
                action=dict(label=label, text=label, target_index=0, phase=phase,
                            resolved_phase=phase, selection={'target_epoch': epoch, 'state': 'locked'},
                            decision_stage='model', decision_code='model_unknown'),
                fall=dict(state='normal', new_event=False, event_id=None),
                behavior={'locomotion': {'label': 'stationary'}, 'turning': {'label': 'not_turning'}})


def temporal(t, index=8, epoch=1, generation=0, score=.85):
    scores = [(1 - score) / 59] * 60
    scores[index] = score
    return dict(status='ready', model_id=MODEL_ID, target_epoch=epoch, generation=generation,
                all_scores=scores, window_end_s=t,
                window=dict(start_s=max(0, t - 2.5), end_s=t, requested_window_s=2.5,
                            causal=True, temporal_wrap=False, sample_count=100,
                            resampling='linear_supported'))


def warmed(progress=0, label='seated', phase='seated'):
    fusion = ConstrainedSTGCNFusion()
    for i in range(20):
        fusion.update(result(progress, label, phase), i / 10, temporal(i / 10))
    return fusion


def rise(fusion, mutate=None):
    output = None
    for i in range(5):
        t = 2 + i / 10
        current = result((i + 1) * .06, 'unknown', 'rising')
        prediction = temporal(t)
        if mutate:
            mutate(current, prediction)
        output = fusion.update(current, t, prediction)
    return output


def test_supported_unknown_rise_candidate_preserves_inputs_alarm_and_other_axes():
    fusion = warmed()
    output = rise(fusion)
    assert output['changed'] and output['label'] == 'standing_up'
    assert output['axes']['locomotion']['label'] == 'stationary'
    assert output['axes']['turning']['label'] == 'not_turning'
    assert output['changes_primary_decision'] is False
    assert output['changes_alarm'] is False
    current = result(.36, 'unknown', 'rising')
    before = deepcopy(current)
    fusion.update(current, 2.5, temporal(2.5))
    assert current == before


def test_supported_descent_requires_standing_origin_and_downward_motion():
    fusion = warmed(1, 'standing', 'standing')
    for i in range(5):
        t = 2 + i / 10
        output = fusion.update(result(1 - (i + 1) * .06, 'unknown', 'descending'), t, temporal(t, 7))
    assert output['changed'] and output['label'] == 'sitting_down'


@pytest.mark.parametrize('label', ['standing', 'seated', 'bending', 'squatting', 'walking', 'standing_up'])
def test_established_primary_actions_are_retained(label):
    fusion = warmed()
    output = fusion.update(result(.3, label, 'rising'), 2, temporal(2))
    assert not output['changed'] and output['label'] == label


@pytest.mark.parametrize('state', ['candidate', 'suspected', 'confirmed', 'recovering'])
def test_alarm_lifecycle_and_repeated_event_are_not_rewritten(state):
    fusion = warmed()
    r = result(.3, 'unknown', 'rising')
    r['fall'].update(state=state, event_id='fall-1', new_event=state == 'confirmed')
    before = deepcopy(r)
    output = fusion.update(r, 2, temporal(2))
    assert output['code'] == 'fall_state_protected' and not output['changed']
    assert r == before
    for t in (2.1, 2.2):
        out = fusion.update(r, t, temporal(t))
        assert out['changes_alarm'] is False and 'new_event' not in out
    r['fall'].update(state='normal', new_event=False)
    output = fusion.update(r, 2.3, temporal(2.3))
    assert output['code'] != 'fall_state_protected'


def test_st_falling_alone_neither_creates_lying_nor_alarm():
    fusion = warmed()
    r = result()
    before = deepcopy(r)
    output = fusion.update(r, 2, temporal(2, 42, score=.99))
    assert output['code'] == 'unsupported_class'
    assert output['label'] == 'seated' and r == before
    assert 'event' not in output and 'fall' not in output


def test_squat_fall_conflict_remains_explicit():
    fusion = warmed()
    output = fusion.update(result(.1, 'squatting', 'unknown'), 2, temporal(2, 42, score=.99))
    assert output['status'] == 'conflict'
    assert output['conflicts'][0]['code'] == 'squat_vs_falling'
    assert output['label'] == 'squatting' and not output['changed']


def test_pickup_does_not_become_bending():
    fusion = warmed()
    output = fusion.update(result(.1, 'unknown', 'unknown'), 2, temporal(2, 5, score=.99))
    assert output['label'] == 'unknown' and output['code'] == 'unsupported_class'


@pytest.mark.parametrize('kind', ['epoch', 'generation', 'stale', 'future', 'protocol', 'noncausal'])
def test_async_identity_age_and_protocol_gates(kind):
    fusion = warmed()
    p = temporal(2)
    if kind == 'epoch':
        p['target_epoch'] = 2
    elif kind == 'generation':
        p['generation'] = 100
    elif kind == 'stale':
        p['window']['end_s'] = 1.3
    elif kind == 'future':
        p['window']['end_s'] = 2.01
    elif kind == 'protocol':
        p['window']['requested_window_s'] = 1.5
    else:
        p['window']['causal'] = False
    output = fusion.update(result(.1, 'unknown', 'rising'), 2, p)
    assert not output['changed']
    assert output['code'] in {'identity_mismatch', 'temporal_stale_or_future', 'protocol_mismatch'}


def test_identity_switch_cannot_inherit_seat_anchor():
    fusion = warmed()
    output = fusion.update(result(.2, 'unknown', 'rising', epoch=2), 2, temporal(2))
    assert not output['changed'] and output['code'] == 'identity_mismatch'
    assert fusion.anchor is None


@pytest.mark.parametrize('kind', ['missing', 'quality', 'camera', 'ambiguous'])
def test_current_missing_or_bad_observation_discards_anchor(kind):
    fusion = warmed()
    r = result(.1, 'unknown', 'rising')
    if kind == 'missing':
        r['persons'] = []
    elif kind == 'quality':
        r['persons'][0]['keypoints'][15][2] = .1
    elif kind == 'camera':
        r['quality']['camera_stable'] = False
    else:
        r['action']['selection']['state'] = 'ambiguous'
    output = fusion.update(r, 2, temporal(2))
    assert not output['changed'] and fusion.anchor is None
    output = rise_after_gap(fusion)
    assert not output['changed']


def rise_after_gap(fusion):
    for i in range(5):
        t = 2.1 + i / 10
        output = fusion.update(result(.1 + i * .06, 'unknown', 'rising'), t,
                               temporal(t, generation=fusion.buffer.generation))
    return output


def test_long_gap_clears_identity_generation_and_anchors():
    fusion = warmed()
    output = fusion.update(result(.3, 'unknown', 'rising'), 3, temporal(3))
    assert not output['changed'] and fusion.anchor is None
    assert output['code'] == 'identity_mismatch'


def test_repeated_or_nonfinite_timestamp_fails_closed_then_rewarms():
    fusion = warmed()
    for bad in (1.9, float('nan')):
        output = fusion.update(result(), bad, temporal(2))
        assert output['code'] == 'invalid_timestamp'
        assert not output['changed'] and not fusion.history
    output = fusion.update(result(.2, 'unknown', 'rising'), 2, temporal(2))
    assert not output['changed']


@pytest.mark.parametrize('kind', ['static', 'opposite_phase', 'stopped', 'feet', 'score'])
def test_recent_transition_score_cannot_override_current_completion_or_motion(kind):
    fusion = warmed()

    def mutate(r, p):
        if kind == 'static':
            r['action']['static_evidence'] = {'stable': True}
        elif kind == 'opposite_phase':
            r['action']['resolved_phase'] = 'descending'
        elif kind == 'stopped':
            r['action']['decision_code'] = 'rise_motion_stopped'
        elif kind == 'feet':
            for j in (15, 16):
                r['persons'][0]['keypoints'][j][0] += (280 - r['persons'][0]['keypoints'][11][1]) * 2
        else:
            p.update(temporal(p['window']['end_s'], score=.4))
    assert not rise(fusion, mutate)['changed']


def test_same_causal_prefix_is_identical_irrespective_of_future_motion():
    a, b = warmed(), warmed()
    assert rise(a) == rise(b)
    a.update(result(1, 'standing', 'standing'), 2.5, temporal(2.5))
    b.update(result(0, 'squatting', 'unknown'), 2.5, temporal(2.5, 42))
    # The earlier outputs were returned value snapshots, not mutable state views.
    assert a.anchor != b.anchor


def test_configuration_validation():
    with pytest.raises(ValueError):
        FusionConfig(minimum_score=float('nan'))


@pytest.mark.parametrize('field,value', [('width', None), ('height', 0), ('width', '640')])
def test_invalid_dimensions_cannot_crash_optional_branch(field, value):
    fusion = warmed()
    r = result()
    r[field] = value
    output = fusion.update(r, 2, temporal(2))
    assert not output['changed'] and output['code'] == 'pose_unreliable'


def test_timestamp_error_cannot_reuse_an_old_async_generation():
    fusion = warmed()
    old = temporal(3.6)
    output = fusion.update(result(), 1.9, temporal(1.9))
    assert output['code'] == 'invalid_timestamp'
    for i in range(20, 37):
        output = fusion.update(result(), i / 10, old)
    assert output['code'] == 'identity_mismatch'


@pytest.mark.parametrize('scores', [[.5] * 60, [float('nan')] * 60, [1] * 59])
def test_invalid_closed_set_distribution_is_rejected(scores):
    fusion = warmed()
    p = temporal(2)
    p['all_scores'] = scores
    assert fusion.update(result(), 2, p)['code'] == 'invalid_scores'


@pytest.mark.parametrize('state', [None, 'confirming', 'searching'])
def test_missing_or_unconfirmed_selection_state_is_not_admitted(state):
    fusion = warmed()
    r = result(.2, 'unknown', 'rising')
    r['action']['selection']['state'] = state
    output = fusion.update(r, 2, temporal(2))
    assert output['code'] == 'target_unavailable' and not output['changed']
    assert fusion.anchor is None
