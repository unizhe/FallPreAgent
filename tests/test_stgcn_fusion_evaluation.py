"""Event metrics must not turn red continuation or later recovery into a hit."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('fusion_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_stgcn_fusion.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def row(t, event=None, new=False, state='normal', expected=None):
    return dict(video_time_s=t, fall=dict(event_id=event, new_event=new, state=state), expected=expected)


def test_old_red_is_not_new_hit_and_recovery_does_not_cross_next_event():
    samples = [row(1, 'a', True, 'confirmed'), row(2, 'a', False, 'confirmed'),
               row(3, 'a', False, 'confirmed'), row(4, 'a', False, 'confirmed'), row(5)]
    events = [dict(id='first', window=[.5, 2], recovery_start_s=2),
              dict(id='second', window=[3, 4], recovery_start_s=4)]
    result = evaluation.event_metrics(samples, events)
    assert result['events'][0]['hit']
    assert not result['events'][0]['recovery_observed']
    assert not result['events'][1]['hit']
    assert result['events'][1]['recovery_to_normal_delay_s'] == 1


def test_duplicate_emissions_and_normal_walking_false_events_are_explicit():
    result = evaluation.event_metrics([row(1, 'a', True, 'confirmed', 'walking_turning'),
                                        row(2, 'a', True, 'confirmed', 'walking_turning')], [])
    assert len(result['independent_new_red_events']) == 1
    assert len(result['duplicate_emissions_same_event_id']) == 1
    assert len(result['normal_activity_false_events']) == 1


def test_unknown_remains_in_recall_denominator():
    samples = [dict(expected='seated', A='unknown', scored_duration_s=.1),
               dict(expected='seated', A='seated', scored_duration_s=.1)]
    report = evaluation.classify_metrics(samples, 'A')
    assert report['per_class']['seated']['recall'] == .5
    assert report['stable_unknown_s']['seated'] == .1
