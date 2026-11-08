"""Event metrics must not hide missing observations, repeated alarms or label gaps."""
from copy import deepcopy

import pytest

from fallpre.weekly.fall_event_metrics import evaluate_fall_events

REVIEW = {'review_status': 'human_confirmed', 'reviewer': 'test-reviewer'}


def row(t, *, observable=True, event_id=None, new=False, state='normal', output=None, recovered=None):
    return dict(time_s=t, output_time_s=output, observable=observable,
                fall=dict(state=state, event_id=event_id, new_event=new, recovered_event_id=recovered))


def event(eid='gt-1', start=1., end=3., **extra):
    return dict(id=eid, motion_start_s=start, match_end_s=end, **REVIEW, **extra)


def normal(start, end, verified=True):
    return dict(start_s=start, end_s=end, **(REVIEW if verified else {'review_status': 'ai_provisional'}))


def evaluate(rows=(), events=(), normals=(), duration=10, gap=.45):
    return evaluate_fall_events(rows, events, normals, camera_id='camera-1', recording_start_s=0,
                                recording_end_s=duration, max_observation_gap_s=gap)


def test_missing_observations_do_not_remove_confirmed_events_from_miss_denominator():
    r = evaluate([row(0, observable=False), row(9, observable=False)], [event()])
    assert r['verified']['annotated_events'] == r['verified']['misses'] == 1
    assert r['verified']['recall'] == 0
    assert r['verified']['output_confirmation_delay']['n'] == 0
    assert r['coverage']['unobservable_or_unknown_seconds'] == 10
    assert r['events'][0]['observable_seconds'] == 0


def test_empty_verified_evidence_is_unavailable_not_zero_false_alarm_success():
    r = evaluate()
    for key in ('recall', 'false_alarm_events', 'false_alarm_events_per_camera_hour'):
        assert r['verified'][key] is None
    assert r['verified']['effective_normal_monitoring_hours'] == 0


def test_rate_uses_reviewed_normal_and_observable_continuous_time_only():
    rows = [row(i / 10) for i in range(21)]
    rows[5] = row(.5, event_id='false', new=True, state='confirmed')
    r = evaluate(rows, normals=[normal(0, 1), normal(1, 2, verified=False)], duration=2)
    assert r['coverage']['observable_seconds'] == pytest.approx(2)
    assert r['verified']['effective_normal_monitoring_seconds'] == pytest.approx(1)
    assert r['provisional']['effective_normal_monitoring_seconds'] == pytest.approx(1)
    assert r['verified']['false_alarm_events'] == 1
    assert r['verified']['false_alarm_events_per_camera_hour'] == pytest.approx(3600)


def test_no_unlabelled_or_long_gap_time_can_become_normal_exposure():
    rows = [row(0), row(.2), row(.4, observable=False), row(.6), row(5), row(5.2)]
    r = evaluate(rows, normals=[normal(0, 4)], duration=6)
    assert r['coverage']['observable_seconds'] == pytest.approx(.4)
    assert r['verified']['effective_normal_monitoring_seconds'] == pytest.approx(.2)
    assert r['coverage']['unobservable_or_unknown_seconds'] == pytest.approx(5.6)


def test_reviewer_is_required_and_reviewed_by_is_compatible():
    e = event()
    e.pop('reviewer')
    r = evaluate(events=[e])
    assert r['provisional']['misses'] == 1 and r['verified']['annotated_events'] == 0
    e['reviewed_by'] = 'another-reviewer'
    assert evaluate(events=[e])['verified']['misses'] == 1


def test_output_delay_is_not_fabricated_from_source_time():
    r = evaluate([row(1.4, state='confirmed', event_id='a', new=True)], [event()])
    e = r['events'][0]
    assert e['hit'] and e['matching_basis'] == 'source_time_proxy'
    assert e['source_confirmation_delay_s'] == pytest.approx(.4)
    assert e['output_confirmation_delay_s'] is None
    assert r['verified']['matched_without_output_timestamp'] == 1


def test_real_output_latency_and_late_confirmation_are_explicit():
    r = evaluate([row(1.4, state='confirmed', event_id='a', new=True, output=1.8)], [event()])
    assert r['events'][0]['output_confirmation_delay_s'] == pytest.approx(.8)
    late = evaluate([row(2.9, state='confirmed', event_id='a', new=True, output=3.1)], [event()])
    assert late['verified']['misses'] == 1 and len(late['unscored_new_alarms']) == 1


def test_alarm_from_before_motion_cannot_be_assigned_by_late_return():
    r = evaluate([row(.8, state='confirmed', event_id='a', new=True, output=1.2)], [event()])
    assert r['verified']['misses'] == 1


def test_old_red_same_id_and_duplicate_emissions_never_become_a_new_hit():
    rows = [row(1.2, state='confirmed', event_id='a', new=True),
            row(1.4, state='confirmed', event_id='a', new=True),
            row(4.2, state='confirmed', event_id='a'),
            row(4.4, state='confirmed', event_id='a', new=True)]
    r = evaluate(rows, [event(), event('gt-2', 4, 6)])
    assert r['verified']['hits'] == r['verified']['misses'] == 1
    assert len(r['alarms']) == 1 and len(r['duplicate_emissions_same_id']) == 2
    assert len(r['events'][0]['duplicate_emissions_same_id']) == 1


def test_carry_in_red_cannot_be_reissued_as_new_event():
    rows = [row(0, state='confirmed', event_id='old'),
            row(1.2, state='confirmed', event_id='old', new=True)]
    r = evaluate(rows, [event()])
    assert r['verified']['misses'] == 1 and len(r['carry_in_red_events']) == 1
    assert len(r['alarms']) == 0 and len(r['duplicate_emissions_same_id']) == 1


def test_new_ids_within_one_event_are_duplicates_not_extra_successes():
    rows = [row(1.2, state='confirmed', event_id='a', new=True),
            row(1.4, state='confirmed', event_id='b', new=True)]
    r = evaluate(rows, [event()])
    assert r['verified']['hits'] == 1
    assert r['verified']['duplicate_new_ids_within_gt_events'] == 1


def test_one_alarm_cannot_satisfy_overlapping_gt_events():
    r = evaluate([row(2, state='confirmed', event_id='a', new=True)],
                 [event(), event('gt-2', 1.5, 4)])
    assert r['verified']['hits'] == 1 and r['verified']['misses'] == 1


def test_overlapping_normal_labels_do_not_double_count_or_include_fall_windows():
    rows = [row(i / 10) for i in range(51)]
    r = evaluate(rows, [event()], [normal(0, 5), normal(0, 4), normal(0, 5, verified=False)], duration=5)
    assert r['verified']['effective_normal_monitoring_seconds'] == pytest.approx(3)
    assert r['provisional']['effective_normal_monitoring_seconds'] == 0
    assert r['normal_positive_label_conflict_seconds']['verified'] == 2


def test_unlabelled_alarm_not_counted_as_false_or_success():
    r = evaluate([row(4, state='confirmed', event_id='a', new=True)])
    assert len(r['unscored_new_alarms']) == 1
    assert r['verified']['false_alarm_events'] is None


def test_repeated_false_emissions_separate_from_independent_false_events():
    rows = [row(0), row(.1, state='confirmed', event_id='a', new=True),
            row(.2, state='confirmed', event_id='a', new=True), row(.3)]
    r = evaluate(rows, normals=[normal(0, 1)], duration=1)
    assert r['verified']['false_alarm_events'] == 1
    assert r['verified']['false_alarm_emissions'] == 2


def test_normal_state_never_invents_recovery_truth_or_recovery_identity():
    rows = [row(1.2, state='confirmed', event_id='a', new=True), row(4)]
    assert evaluate(rows, [event()])['events'][0]['recovery'] is None
    e = event(recovery_truth=dict(recovered_at_s=3, window_end_s=5, **REVIEW))
    result = evaluate(rows, [e])['events'][0]['recovery']
    assert result['identified_recovery_observed'] is False and result['output_delay_s'] is None


def test_explicit_recovery_and_lifecycle_ground_truth_only():
    e = event(recovery_truth=dict(recovered_at_s=3, window_end_s=5, **REVIEW),
              red_forbidden_intervals=[normal(3, 5)], must_remain_active_intervals=[normal(1.3, 2.5)])
    rows = [row(1.2, state='confirmed', event_id='a', new=True), row(1.5),
            row(3.2, state='confirmed', event_id='a'), row(3.5, recovered='a', output=3.7)]
    result = evaluate(rows, [e])['events'][0]
    assert result['recovery']['source_delay_s'] == .5
    assert result['recovery']['output_delay_s'] == pytest.approx(.7)
    assert result['lifecycle']['red_forbidden_intervals']['error_samples'] == 1
    assert result['lifecycle']['must_remain_active_intervals']['error_samples'] == 1


def test_labels_and_rows_are_not_mutated_and_all_counts_are_serializable():
    import json
    rows = [row(1.2, state='confirmed', event_id='a', new=True)]
    events = [event()]
    normals = [normal(4, 6)]
    original = deepcopy((rows, events, normals))
    json.dumps(evaluate(rows, events, normals), allow_nan=False)
    assert (rows, events, normals) == original


@pytest.mark.parametrize('rows,events,normals', [
    ([row(1), row(1)], [], []), ([row(float('nan'))], [], []),
    ([row(1, output=.9)], [], []), ([row(1, new=True)], [], []),
    ([], [event(), event()], []), ([], [], [normal(3, 2)]),
])
def test_invalid_clock_or_event_contract_fails_closed(rows, events, normals):
    with pytest.raises(ValueError):
        evaluate(rows, events, normals)
