import copy
import json

import pytest
from test_webcam_actions import detection

from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.human import ROOT
from fallpre.weekly.webcam_actions import LiveActions
from fallpre.weekly.webcam_diagnostics import SessionDiagnostics


def action(label='unknown', **changes):
    return dict(dict(label=label, raw_label='background', phase='standing',
                     reason='等待静态确认', decision_code='static_confirmation',
                     decision_stage='display', target_index=None, evidence_s=.2,
                     selection={'state': 'locked'}, reset_reasons=[],
                     reconciliation={'changed': False, 'code': 'warming_up'}), **changes)


def test_observation_counts_reasons_and_resets_have_distinct_denominators():
    stats = SessionDiagnostics()
    first = stats.update(0, action(reset_reasons=['source_gap', 'source_gap', 'target_changed']), [], 640, 480)
    stats.update(.1, action('standing', decision_code='accepted'), [], 640, 480)
    out = stats.update(.5, action(raw_label='unknown', decision_code='model_unknown'), [], 640, 480)
    summary = out['summary']
    assert summary['total_frames'] == 3
    assert summary['unknown_frames'] == 2 and summary['unknown_rate'] == pytest.approx(2/3)
    assert sum(r['count'] for r in summary['unknown_reasons']) == 2
    assert sum(r['rate'] for r in summary['unknown_reasons']) == 1
    assert {r['code']: r['count'] for r in summary['resets']} == {'source_gap': 1, 'target_changed': 1}
    assert summary['max_gap_s'] == .4 and summary['long_gap_frames'] == 1
    assert first['summary']['total_frames'] == 1  # No alias to mutable counters.
    json.dumps(out, allow_nan=False)


def test_raw_result_and_recovery_survive_in_scalar_diagnostics():
    stats = SessionDiagnostics()
    out = stats.update(0, action('seated', raw_label='sitting_down', phase='seated',
                                resolved_label='background', resolved_phase='seated',
                                reconciliation=dict(changed=True, code='static_confirmed', evidence={'stable': True}),
                                visible_seat=dict(seat_recovery_s=.4, seat_recovered=True, veto=False, reason='恢复')),
                       [], 640, 480)
    assert out['sample']['raw_label'] == 'sitting_down'
    assert out['sample']['resolved_label'] == 'background'
    assert out['sample']['seat_recovered'] and out['sample']['seat_recovery_s'] == .4
    assert out['summary']['conflict_frames'] == out['summary']['resolved_frames'] == 1
    assert not any(key in json.dumps(out) for key in ['keypoints', 'kpts', 'pixels', 'image/jpeg'])


def test_rise_provenance_survives_with_static_recovery_and_excludes_coordinates():
    evidence = dict(code='rise_confirmed', reason='观察到离座抬髋', rising=True,
                    veto_seated=True, stop_rise=False, applied=True,
                    anchor_available=True, anchor_kind='observed_descent', anchor_age_s=.4,
                    hip_up=.11, shoulder_up=.08, hip_absolute_up=.12, shoulder_absolute_up=.09,
                    horizontal=.02, thigh_extension=.05, knee_opening=.06,
                    motion_duration_s=.266, positive_steps=2, additional_up=.012, current_continues=True, current_feet_stable=True,
                    foot_motion=.01, reset_reason=None, standing_complete=False,
                    keypoints=[[100, 200, .9]], pixels='do not export', unknown_field='private')
    out = SessionDiagnostics().update(0, action('standing_up', rise_evidence=evidence,
                                              visible_seat=dict(seat_recovered=True)), [], 640, 480)
    exported = out['sample']['rise_evidence']
    assert exported['code'] == 'rise_confirmed'
    assert exported['applied'] and exported['veto_seated']
    assert exported['positive_steps'] == 2 and exported['motion_duration_s'] == .266
    assert exported['additional_up'] == .012
    assert exported['current_feet_stable'] is True
    assert exported['hip_up'] == .11 and exported['standing_complete'] is False
    assert exported['hip_absolute_up'] == .12 and exported['shoulder_absolute_up'] == .09
    assert exported['horizontal'] == .02
    assert out['sample']['seat_recovered']  # Existing independent diagnostics remain available.
    assert not {'keypoints', 'pixels', 'unknown_field'} & exported.keys()
    evidence['code'] = 'changed'
    assert exported['code'] == 'rise_confirmed'
    json.dumps(out, allow_nan=False)


def test_rise_provenance_absent_or_nonfinite_is_json_safe():
    absent = SessionDiagnostics().update(0, action(), [], 640, 480)
    assert absent['sample']['rise_evidence'] is None
    out = SessionDiagnostics().update(0, action(rise_evidence=dict(
        hip_up=float('nan'), shoulder_up=float('inf'), positive_steps=[1, 2],
        reason={'keypoints': [1, 2, 3]})), [], 640, 480)
    for key in ['hip_up', 'shoulder_up', 'positive_steps', 'reason']:
        assert key not in out['sample']['rise_evidence']
    json.dumps(out, allow_nan=False)


@pytest.fixture(scope='module')
def model():
    return CompletionModel.load(ROOT/'runs/action-v7/training/model.json')


def test_live_statistics_survive_evidence_resets_but_are_isolated_per_session(model):
    live = LiveActions(model)
    points = detection()
    original = copy.deepcopy(points)
    for i in range(15):
        out = live.update([points], i/30, 640, 480)
    assert out['diagnostics']['sample']['quality']['complete_side']
    assert points == original
    out = live.update([points], 2, 640, 480)
    assert out['diagnostics']['summary']['total_frames'] == 16
    assert 'source_gap' in out['diagnostics']['sample']['reset_reasons']
    with pytest.raises(ValueError):
        live.update([points], 2, 640, 480)
    assert live.diagnostics.labels.total() == 16
    independent = LiveActions(model).update([points], 0, 640, 480)
    assert independent['diagnostics']['summary']['total_frames'] == 1


def test_partial_chain_and_seat_only_resets_are_observable(model):
    live = LiveActions(model)
    for i in range(20):
        live.update([detection()], i/30, 640, 480)
    p = detection()
    p['keypoints'][6][2] = 0  # Left complete chain survives; seat needs both shoulders.
    out = live.update([p], 20/30, 640, 480)
    sample = out['diagnostics']['sample']
    assert sample['quality']['complete_side']
    assert sample['quality']['shoulders'] == 1
    assert 'seat_joints_missing' in sample['reset_reasons']
    p = detection()
    for j in (15, 16):
        p['keypoints'][j][2] = .29
    out = live.update([p], 21/30, 640, 480)
    sample = out['diagnostics']['sample']
    assert sample['reason_code'] == 'joint_chain_missing'
    assert 'joint_chain_missing' in sample['reset_reasons']
    assert not sample['quality']['complete_side']


def test_handoff_and_target_hold_export_scalars_without_pose_arrays():
    out = SessionDiagnostics().update(0, action(
        static_evidence=dict(code='static_continuity', stable=True, points=[[1,2,3]]),
        target_continuity=dict(history_retained=True, gap_s=.15)), [], 640, 480)
    assert out['sample']['static_evidence'] == dict(code='static_continuity', stable=True)
    assert out['sample']['target_continuity'] == dict(history_retained=True, gap_s=.15)


def test_seat_rise_handoff_export_is_scalar_and_excludes_pose():
    proof=dict(code='pelvis_ascent',applied=True,support_feet=1,positive_steps=True,
               hip_up=.12,extension=.16,keypoints=[[1,2,.9]],private='not exported')
    out=SessionDiagnostics().update(0,action('standing_up',seat_rise_handoff=proof),[],640,480)
    assert out['sample']['seat_rise_handoff']=={k:v for k,v in proof.items() if k not in {'keypoints','private'}}
