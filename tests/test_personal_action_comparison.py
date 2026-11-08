"""Scoring must not turn misses or boundary-adjacent errors into improvements."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'personal_action_comparison',
    Path(__file__).resolve().parents[1] / 'scripts/compare_personal_actions.py',
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def row(t, label):
    return dict(video_time_s=t, result={'action': {'label': label}})


def segment(start, end, label, identifier=1):
    return dict(start_s=start, end_s=end, label=label, id=identifier)


def test_four_class_unknown_is_distinct_from_wrong_known_and_excludes_other():
    segments = [segment(0, 1, 'standing'), segment(1, 2, 'other', 2)]
    rows = [row(.1, 'standing'), row(.2, 'unknown'), row(.3, 'seated'), row(1.2, 'unknown')]
    stats = comparison.class_stats(rows, segments)
    assert stats['four_class']['n'] == 3
    assert stats['four_class']['unknown'] == 1
    assert stats['four_class']['unknown_rate'] == pytest.approx(1 / 3)
    assert stats['four_class']['provisional_match'] == 1
    assert stats['four_class']['wrong_known'] == 1
    assert stats['seated']['n'] == 0
    assert stats['seated']['unknown_rate'] is None


def test_boundary_observation_belongs_only_to_following_segment_in_both_variants():
    segments = [segment(0, 1, 'standing'), segment(1, 2, 'seated', 2)]
    before = [row(.9, 'standing'), row(1., 'standing'), row(2., 'seated')]
    after = [row(.9, 'standing'), row(1., 'seated'), row(2., 'seated')]
    old = comparison.class_stats(before, segments)
    new = comparison.class_stats(after, segments)
    assert old['standing']['n'] == new['standing']['n'] == 1
    assert old['seated']['n'] == new['seated']['n'] == 1
    assert old['seated']['wrong_known'] == 1
    assert new['seated']['provisional_match'] == 1
    assert old['four_class']['n'] == new['four_class']['n'] == 2
    assert comparison.segment_latency(before, segments)[1]['missed'] is True
    assert comparison.segment_latency(after, segments)[1]['first_correct_latency_s'] == 0


def test_delayed_output_after_interval_never_converts_a_miss_to_zero_latency():
    segments = [segment(0, 1, 'standing_up'), segment(1, 2, 'standing', 2)]
    rows = [row(.2, 'unknown'), row(.8, 'unknown'), row(1.1, 'standing_up'), row(1.4, 'standing')]
    rising, standing = comparison.segment_latency(rows, segments)
    assert rising['missed'] is True
    assert rising['first_correct_latency_s'] is None
    assert rising['two_consecutive_latency_s'] is None
    assert standing['missed'] is False
    assert standing['first_correct_latency_s'] == pytest.approx(.4)
    assert standing['last_previous_label_offset_s'] == pytest.approx(.1)


def test_confirmation_requires_adjacent_correct_observations_within_gap_limit():
    segments = [segment(0, 2, 'standing')]
    rows = [row(.1, 'standing'), row(.5, 'standing'), row(.6, 'unknown'),
            row(.7, 'standing'), row(.8, 'standing')]
    result = comparison.segment_latency(rows, segments)[0]
    assert result['first_correct_latency_s'] == .1
    assert result['two_consecutive_latency_s'] == .8
    no_confirmation = comparison.segment_latency(rows[:4], segments)[0]
    assert no_confirmation['missed'] is False
    assert no_confirmation['two_consecutive_latency_s'] is None


def test_empty_interval_remains_missed_without_a_numeric_latency():
    result = comparison.segment_latency([], [segment(4, 4.4, 'sitting_down')])[0]
    assert result['missed'] is True
    assert result['first_correct_latency_s'] is None
    assert result['two_consecutive_latency_s'] is None
    assert result['last_previous_label_offset_s'] is None


def test_transition_audit_exposes_false_rises_and_lost_static_states():
    segments = [segment(0, 1, 'seated'), segment(1, 2, 'standing_up', 2),
                segment(2, 3, 'other', 3)]
    before = [row(.5, 'seated'), row(1.2, 'unknown'), row(2.5, 'unknown')]
    after = [row(.5, 'standing_up'), row(1.2, 'standing_up'), row(2.5, 'standing_up')]
    for r in before + after:
        r['result']['action'].update(raw_label='background', phase='seated')
    result = comparison.transition_audit(before, after, segments)
    assert result['added_rise_by_prelabel'] == {'seated': 1, 'standing_up': 1, 'other': 1}
    assert len(result['lost_prelabel_matches']) == 1
    assert result['lost_prelabel_matches'][0]['time_s'] == .5
    assert result['raw_prediction_difference_times'] == []
    after[0]['result']['action']['raw_label'] = 'standing_up'
    assert comparison.transition_audit(before, after, segments)['raw_prediction_difference_times'] == [.5]
