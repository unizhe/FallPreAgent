"""Fall temporal scores stay descriptive and cannot own event lifecycle."""
from copy import deepcopy

import numpy as np
import pytest

from fallpre.weekly.stgcn_fall_evidence import (
    FallEvidenceConfig,
    FallTemporalEvidence,
    fall_geometry_features,
)
from fallpre.weekly.stgcn_temporal import MODEL_ID


def result(step=0, *, epoch=1, rgb_epoch='rgb-3'):
    radians = np.deg2rad(min(step * 12, 78))
    hip = np.array([260., 260. + step * 8])
    shoulder = hip - np.array([np.sin(radians), np.cos(radians)]) * 100
    points = np.tile([260., 200., .95], (17, 1))
    for s, h, k, a, x in [(5, 11, 13, 15, -20), (6, 12, 14, 16, 20)]:
        points[s, :2], points[h, :2] = shoulder + [x, 0], hip + [x, 0]
        points[k, :2], points[a, :2] = hip + [x + 20, 60], hip + [x + 10, 100]
    for j in (7, 8, 9, 10):
        points[j, :2] = shoulder + [j, 20]
    box = [float(points[5:, 0].min() - 20), float(points[5:, 1].min() - 20),
           float(points[5:, 0].max() + 20), float(points[5:, 1].max() + 20)]
    person = dict(box=box, confidence=.95, keypoints=points.tolist())
    rgb = dict(box=list(box), confidence=.95, keypoints=[], label='fallen', detector='person')
    return dict(width=640, height=540, persons=[person], source_time=None,
                quality={'camera_stable': True}, fall_mode='rgb',
                action=dict(label='unknown', raw_label='unknown', target_index=0,
                            selection=dict(target_index=0, target_epoch=epoch, state='locked')),
                fall=dict(state='normal', observed=True, target_epoch=rgb_epoch, new_event=False,
                          event_id=None, summary={'confirmed_events': []}),
                fall_observation=dict(source='rgb_fall', person=rgb,
                                      selection=dict(state='locked', target_epoch=rgb_epoch)),
                rgb_diagnostics={'reference_pose_epoch': 999},
                rgb_raw_candidates={'persons': [dict(box=list(box), confidence=.95)],
                                    'fallen': [dict(box=list(box), confidence=.90)]})


def temporal(t, *, epoch=1, generation=0, score=.9, index=42):
    scores = [(1 - score) / 59] * 60
    scores[index] = score
    return dict(status='ready', model_id=MODEL_ID, target_epoch=epoch, generation=generation,
                all_scores=scores,
                window=dict(start_s=max(0, t - 2.5), end_s=t, requested_window_s=2.5,
                            sample_count=100, resampling='linear_supported', causal=True, temporal_wrap=False))


def warmed():
    branch = FallTemporalEvidence()
    for i in range(20):
        branch.update(result(), i / 10, temporal(i / 10))
    return branch


def falling(branch, mutate=None):
    outputs = []
    for step in range(1, 8):
        t = 1.9 + step / 10
        r, p = result(step), temporal(t)
        if mutate:
            mutate(r, p)
        before = deepcopy(r)
        out = branch.update(r, t, p)
        assert r == before
        outputs.append(out)
    return outputs


def test_current_descending_low_pose_with_unique_rgb_target_only_supports_evidence():
    outputs = falling(warmed())
    out = outputs[-1]
    assert out['status'] == 'support' and out['support']
    assert out['observation_quality']['status'] == 'observable'
    assert out['model_evidence']['falling_score'] == .9
    assert out['association']['action_target_epoch'] == 1
    assert out['association']['rgb_target_epoch'] == 'rgb-3'
    assert out['association']['namespaces_distinct']
    assert out['geometry']['descending'] and out['geometry']['low_pose']
    for field in ('changes_alarm', 'changes_event', 'changes_primary_decision', 'negative_example'):
        assert out[field] is False
    assert 'event_id' not in out and 'new_event' not in out


def test_relative_and_global_motion_are_both_preserved_without_changing_input():
    p = np.asarray(result()['persons'][0]['keypoints'])
    observations = np.stack([p, p + [0, 10, 0], p + [0, 20, 0]])
    before = observations.copy()
    out = fall_geometry_features(observations, [0, .1, .2], 640, 540)
    assert out['quality'] == 'observable'
    assert out['metrics']['center_dy_image'] > 0
    assert out['metrics']['hip_drop_reference_torso'] == pytest.approx(.2)
    assert out['metrics']['relative_pose_motion_rms'] == pytest.approx(0)
    assert len(out['vector']) == len(out['feature_names'])
    np.testing.assert_array_equal(observations, before)


@pytest.mark.parametrize('kind', ['different_box', 'missing_rgb', 'unlocked_rgb', 'no_rgb_epoch',
                                 'no_current_fall_observation', 'no_fallen', 'weak_fallen',
                                 'multiple_pose', 'multiple_rgb', 'rgb_ambiguity', 'pose_mode'])
def test_no_cross_namespace_fusion_without_unique_current_geometry(kind):
    def mutate(r, p):
        if kind == 'different_box':
            r['fall_observation']['person']['box'] = [10, 10, 100, 100]
            # Even deliberately equal epoch values cannot establish identity.
            r['fall_observation']['selection']['target_epoch'] = 1
            r['fall']['target_epoch'] = 1
        elif kind == 'missing_rgb':
            r.pop('fall_observation')
        elif kind == 'unlocked_rgb':
            r['fall_observation']['selection']['state'] = 'provisional'
        elif kind == 'no_rgb_epoch':
            r['fall_observation']['selection']['target_epoch'] = None
        elif kind == 'no_current_fall_observation':
            r['fall']['observed'] = False
        elif kind == 'no_fallen':
            r['rgb_raw_candidates']['fallen'] = []
        elif kind == 'weak_fallen':
            r['rgb_raw_candidates']['fallen'][0]['confidence'] = .4
        elif kind == 'multiple_pose':
            r['persons'].append(deepcopy(r['persons'][0]))
        elif kind == 'multiple_rgb':
            r['rgb_raw_candidates']['fallen'].append(deepcopy(r['rgb_raw_candidates']['fallen'][0]))
        elif kind == 'rgb_ambiguity':
            r['rgb_diagnostics']['person_ambiguity_retained'] = True
        else:
            r['fall_mode'] = 'pose'
    out = falling(warmed(), mutate)[-1]
    assert out['status'] == 'abstain' and not out['support']
    assert not out['association']['associated']


def test_distant_second_person_does_not_create_false_local_ambiguity():
    def mutate(r, p):
        other = deepcopy(r['persons'][0])
        other['box'] = [500, 10, 630, 250]
        r['persons'].append(other)
    out = falling(warmed(), mutate)[-1]
    assert out['support']


@pytest.mark.parametrize('kind', ['stale', 'future', 'action_epoch', 'generation', 'protocol', 'noncausal'])
def test_st_window_must_match_action_identity_time_and_protocol(kind):
    def mutate(r, p):
        if kind == 'stale':
            p['window']['end_s'] -= .7
        elif kind == 'future':
            p['window']['end_s'] += .01
        elif kind == 'action_epoch':
            p['target_epoch'] = 'rgb-3'
        elif kind == 'generation':
            p['generation'] = 100
        elif kind == 'protocol':
            p['window']['requested_window_s'] = 1.5
        else:
            p['window']['causal'] = False
    out = falling(warmed(), mutate)[-1]
    assert out['status'] == 'abstain' and not out['support']


@pytest.mark.parametrize('kind', ['missing_target', 'unlocked_target', 'torso_missing', 'camera', 'invalid_box'])
def test_bad_current_observation_is_unknown_never_a_nonfall_example(kind):
    branch = warmed()
    r = result(6)
    if kind == 'missing_target':
        r['persons'] = []
    elif kind == 'unlocked_target':
        r['action']['selection']['state'] = 'ambiguous'
    elif kind == 'torso_missing':
        r['persons'][0]['keypoints'][11][2] = .1
    elif kind == 'camera':
        r['quality']['camera_stable'] = False
    else:
        r['persons'][0]['box'] = [0, 0, 0, 0]
    out = branch.update(r, 2, temporal(2))
    assert not out['support'] and out['negative_example'] is False
    assert out['status'] == 'abstain'


def test_rgb_identity_change_requires_new_current_association_history():
    branch = warmed()
    falling(branch)
    r = result(8, rgb_epoch='rgb-new')
    out = branch.update(r, 2.7, temporal(2.7))
    assert out['code'] == 'association_warmup' and not out['support']


def test_action_identity_change_or_gap_rejects_old_temporal_result():
    branch = warmed()
    out = branch.update(result(6, epoch=2), 2, temporal(2))
    assert out['code'] == 'current_geometry_unavailable' and not out['support']
    branch = warmed()
    out = branch.update(result(6), 3, temporal(3))
    assert not out['support']
    assert branch.buffer.generation == 1


def test_short_target_loss_clears_association_accumulation():
    branch = warmed()
    falling(branch)
    r = result(8)
    r['persons'] = []
    branch.update(r, 2.7, temporal(2.7))
    out = branch.update(result(9), 2.8, temporal(2.8))
    assert not out['support']
    assert len(branch.geometry_rows) <= 1


def test_squat_conflict_is_explicit_and_never_cancels_existing_red_event():
    def mutate(r, p):
        r['action']['label'] = 'squatting'
        r['fall'].update(state='confirmed', new_event=False, event_id='existing-event')
    out = falling(warmed(), mutate)[-1]
    assert out['status'] == 'conflict' and not out['support']
    assert out['conflicts'][0]['code'] == 'squat_vs_st_falling'
    assert out['changes_event'] is False


@pytest.mark.parametrize('state', ['normal', 'suspected', 'confirmed', 'unobservable'])
def test_primary_event_and_recovery_lifecycle_are_immutable(state):
    def mutate(r, p):
        r['fall'].update(state=state, new_event=state == 'confirmed', event_id='fall-0001',
                         lifecycle='recovering', summary={'confirmed_events': [{'event_id': 'fall-0001'}]})
    outputs = falling(warmed(), mutate)
    assert all(o['changes_event'] is False and o['changes_alarm'] is False for o in outputs)


def test_sustained_floor_pose_is_not_a_new_descent_and_cannot_repeat_events():
    branch = warmed()
    assert falling(branch)[-1]['support']
    for i in range(27, 40):
        out = branch.update(result(7), i / 10, temporal(i / 10))
        assert not out['support'] and out['changes_event'] is False
    assert out['code'] == 'current_fall_geometry_insufficient'


def test_nonfall_closed_set_vote_does_not_become_negative_truth():
    out = falling(warmed(), lambda r, p: p.update(temporal(p['window']['end_s'], index=7)))[-1]
    assert out['code'] == 'model_fall_support_insufficient'
    assert out['negative_example'] is False


def test_prefix_result_is_independent_of_future_motion():
    a, b = warmed(), warmed()
    prefix_a, prefix_b = falling(a), falling(b)
    assert prefix_a == prefix_b
    a.update(result(8), 2.7, temporal(2.7))
    b.update(result(), 2.7, temporal(2.7, index=7))
    assert prefix_a == prefix_b


@pytest.mark.parametrize('times', [[0, 0, .2], [0, float('nan'), .2], [.2, .1, .3]])
def test_geometry_helper_rejects_noncausal_timestamps(times):
    p = np.asarray(result()['persons'][0]['keypoints'])
    out = fall_geometry_features(np.stack([p, p, p]), times, 640, 540)
    assert out['quality'] == 'unobservable' and out['vector'] is None


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        FallEvidenceConfig(minimum_score=float('nan'))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1, None, '2.0'])
def test_invalid_time_fails_closed_without_crashing_or_reusing_generation(invalid):
    branch = warmed()
    out = branch.update(result(6), invalid, temporal(2))
    assert out['code'] == 'invalid_timestamp'
    assert not out['support'] and not branch.geometry_rows
    assert branch.buffer.generation > 0


def test_duplicate_time_resets_st_identity_generation():
    branch = warmed()
    out = branch.update(result(6), 1.9, temporal(2))
    assert out['code'] == 'invalid_timestamp'
    assert branch.buffer.generation == 1


@pytest.mark.parametrize('scores', [[float('nan')] * 60, [.9] * 60, [1.] * 59])
def test_invalid_class_distribution_cannot_support_fall(scores):
    def mutate(r, p):
        p['all_scores'] = scores
    out = falling(warmed(), mutate)[-1]
    assert out['code'] == 'invalid_model_scores'
    assert not out['support']


def test_third_pose_epoch_is_recorded_but_never_compared_to_action_epoch():
    out = falling(warmed())[-1]
    assert out['association']['fall_pose_reference_epoch'] == 999
    assert out['association']['action_target_epoch'] == 1
    assert out['association']['rgb_target_epoch'] == 'rgb-3'
    assert out['support']
