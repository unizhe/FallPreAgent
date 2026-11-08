"""Strict, per-camera fall-event metrics with explicit annotation provenance.

All times use one recording-relative clock. Rows require ``time_s`` and an
explicit ``observable`` boolean (missing means unknown), plus ``fall`` with
state/event_id/new_event. Optional ``output_time_s`` is the actual availability
time on that same clock; never fill it with a model's standalone inference time.

Events require id/motion_start_s/match_end_s. Normal intervals require
start_s/end_s. ``review_status='human_confirmed'`` and a nonempty ``reviewer``
(or reviewed_by) enable verified metrics. Everything else is provisional.
No model output, visibility filter or unlabelled interval becomes ground truth.

Recovery requires event.recovery_truth with recovered_at_s/window_end_s and
an explicit fall.recovered_event_id in an output row. A normal state alone is
not treated as an identified recovery. Optional event.red_forbidden_intervals
and event.must_remain_active_intervals supply explicit lifecycle ground truth.
"""
from __future__ import annotations

import math

_VERIFIED = {'human_confirmed', 'human_reviewed', 'verified'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _verified(label):
    reviewer = label.get('reviewer', label.get('reviewed_by'))
    return label.get('review_status') in _VERIFIED and isinstance(reviewer, str) and bool(reviewer.strip())


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _intersect(a, b):
    return _union((max(s, x), min(e, y)) for s, e in a for x, y in b if max(s, x) < min(e, y))


def _subtract(intervals, excluded):
    result = list(intervals)
    for x, y in _union(excluded):
        result = [(a, b) for s, e in result for a, b in ((s, min(e, x)), (max(s, y), e)) if b > a]
    return _union(result)


def _seconds(intervals):
    return sum(e - s for s, e in intervals)


def _inside(t, intervals):
    return any(s <= t < e for s, e in intervals)


def _interval(label, start_key, end_key, bounds):
    start = _number(label[start_key], start_key)
    end = _number(label[end_key], end_key)
    if not bounds[0] <= start < end <= bounds[1]:
        raise ValueError(f'Invalid or out-of-recording interval: {start}, {end}')
    return start, end


def _delay_summary(values):
    values = sorted(values)
    return {'n': len(values), 'mean_s': sum(values) / len(values) if values else None,
            'p95_s': values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)] if values else None}


def evaluate_fall_events(rows, events, normal_intervals, *, camera_id,
                         recording_start_s, recording_end_s, max_observation_gap_s=.45):
    """Evaluate one camera and one non-resetting session; never mutate inputs.

    Matching is one-to-one, earliest eligible alarm to the earliest-ending GT
    window. Both source time and actual output time, when known, must lie in the
    window. A source-only match is disclosed as a proxy, with output delay null.
    Every annotated event remains in its cohort denominator, even if every
    observation is missing. Verified and provisional cohorts are always separate.
    """
    start = _number(recording_start_s, 'recording_start_s')
    end = _number(recording_end_s, 'recording_end_s')
    gap = _number(max_observation_gap_s, 'max_observation_gap_s')
    if end <= start or gap <= 0 or not isinstance(camera_id, str) or not camera_id:
        raise ValueError('Require camera_id, a positive recording duration and a positive gap limit')
    bounds = start, end
    normalized = []
    previous = None
    for source in rows:
        t = _number(source['time_s'], 'time_s')
        if not start <= t <= end or previous is not None and t <= previous:
            raise ValueError('Rows must have strictly increasing in-recording source times')
        previous = t
        output = source.get('output_time_s')
        if output is not None:
            output = _number(output, 'output_time_s')
            if output < t:
                raise ValueError('An output cannot be available before its input observation')
        fall = source.get('fall') or {}
        if not isinstance(fall, dict):
            raise ValueError('fall must be a mapping')
        if fall.get('new_event') and (fall.get('state') != 'confirmed' or not isinstance(
                fall.get('event_id'), str) or not fall['event_id']):
            raise ValueError('New alarms require confirmed state and a nonempty string event_id')
        normalized.append({'time_s': t, 'output_time_s': output, 'observable': source.get('observable') is True,
                           'fall': fall})
    observable = _union((a['time_s'], b['time_s']) for a, b in zip(normalized, normalized[1:])
                        if a['observable'] and b['observable'] and b['time_s'] - a['time_s'] <= gap + 1e-9)
    continuous = _union((a['time_s'], b['time_s']) for a, b in zip(normalized, normalized[1:])
                        if b['time_s'] - a['time_s'] <= gap + 1e-9)
    truth = []
    ids = set()
    for event in events:
        eid = event.get('id')
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError('GT event ids must be distinct nonempty strings')
        ids.add(eid)
        s, e = _interval(event, 'motion_start_s', 'match_end_s', bounds)
        truth.append({'id': eid, 'motion_start_s': s, 'match_end_s': e, 'verified': _verified(event),
                      'annotation': event})
    positive = _union((e['motion_start_s'], e['match_end_s']) for e in truth)
    normal = {'verified': [], 'provisional': []}
    for interval in normal_intervals:
        span = _interval(interval, 'start_s', 'end_s', bounds)
        normal['verified' if _verified(interval) else 'provisional'].append(span)
    normal = {k: _union(v) for k, v in normal.items()}
    # Duplicated provisional labels cannot increase verified exposure or be counted twice.
    normal['provisional'] = _subtract(normal['provisional'], normal['verified'])
    conflicts = {k: _seconds(_intersect(v, positive)) for k, v in normal.items()}
    normal = {k: _subtract(v, positive) for k, v in normal.items()}
    effective = {k: _intersect(v, observable) for k, v in normal.items()}
    alarms, repeats, carry_in, known = [], [], [], set()
    for index, row in enumerate(normalized):
        f = row['fall']
        eid = f.get('event_id')
        if f.get('state') == 'confirmed' and eid:
            emission = {'event_id': eid, 'time_s': row['time_s'], 'output_time_s': row['output_time_s'],
                        'matching_time_s': row['output_time_s'] if row['output_time_s'] is not None else row['time_s'],
                        'row_index': index}
            if f.get('new_event'):
                (repeats if eid in known else alarms).append(emission)
            elif eid not in known:
                carry_in.append(emission)
            known.add(eid)
    alarms.sort(key=lambda a: (a['matching_time_s'], a['time_s']))
    matched = {}
    assigned = set()
    def eligible(alarm, event):
        return (event['motion_start_s'] <= alarm['time_s'] < event['match_end_s'] and
                event['motion_start_s'] <= alarm['matching_time_s'] < event['match_end_s'])
    for index, alarm in enumerate(alarms):
        candidates = [e for e in truth if e['id'] not in matched and eligible(alarm, e)]
        if candidates:
            event = min(candidates, key=lambda e: (e['match_end_s'], e['motion_start_s'], e['id']))
            matched[event['id']] = index
            assigned.add(index)
    duplicates = {e['id']: [] for e in truth}
    duplicate_indices = set()
    for index, alarm in enumerate(alarms):
        if index in assigned:
            continue
        candidates = [e for e in truth if e['id'] in matched and eligible(alarm, e)]
        if candidates:
            event = min(candidates, key=lambda e: (e['match_end_s'], e['motion_start_s'], e['id']))
            duplicates[event['id']].append(alarm)
            duplicate_indices.add(index)
    checks = []
    for event in sorted(truth, key=lambda e: (e['motion_start_s'], e['id'])):
        alarm = alarms[matched[event['id']]] if event['id'] in matched else None
        interval = [(event['motion_start_s'], event['match_end_s'])]
        associated = [alarm] + duplicates[event['id']] if alarm else []
        predicted_ids = {a['event_id'] for a in associated}
        repeated = [a for a in repeats if a['event_id'] in predicted_ids and eligible(a, event)]
        check = {'id': event['id'], 'verified': event['verified'], 'motion_start_s': event['motion_start_s'],
                 'match_end_s': event['match_end_s'], 'hit': alarm is not None, 'miss': alarm is None,
                 'matched_alarm': alarm, 'matching_basis': ('actual_output' if alarm['output_time_s'] is not None
                                                          else 'source_time_proxy') if alarm else None,
                 'source_confirmation_delay_s': alarm['time_s'] - event['motion_start_s'] if alarm else None,
                 'output_confirmation_delay_s': (alarm['output_time_s'] - event['motion_start_s']
                                                if alarm and alarm['output_time_s'] is not None else None),
                 'observable_seconds': _seconds(_intersect(interval, observable)),
                 'duplicate_new_ids': duplicates[event['id']], 'duplicate_emissions_same_id': repeated,
                 'recovery': None, 'lifecycle': {}}
        annotation = event['annotation']
        recovery = annotation.get('recovery_truth')
        if recovery is not None:
            rs, re = _interval(recovery, 'recovered_at_s', 'window_end_s', bounds)
            releases = [r for r in normalized if r['fall'].get('recovered_event_id') in predicted_ids
                        and rs <= r['time_s'] < re and (r['output_time_s'] is None or r['output_time_s'] < re)]
            release = min(releases, key=lambda r: r['output_time_s'] if r['output_time_s'] is not None
                          else r['time_s']) if releases else None
            check['recovery'] = {'verified': _verified(recovery), 'recovered_at_s': rs, 'window_end_s': re,
                                 'identified_recovery_observed': release is not None,
                                 'source_delay_s': release['time_s'] - rs if release else None,
                                 'output_delay_s': release['output_time_s'] - rs
                                 if release and release['output_time_s'] is not None else None,
                                 'censored': release is None}
        for key, state in [('red_forbidden_intervals', 'confirmed'), ('must_remain_active_intervals', 'normal')]:
            labels = annotation.get(key)
            if labels is None:
                check['lifecycle'][key] = None
                continue
            spans = [_interval(x, 'start_s', 'end_s', bounds) for x in labels]
            bad = [r for r in normalized if _inside(r['time_s'], spans) and r['fall'].get('state') == state
                   and (state == 'normal' or r['fall'].get('event_id') in predicted_ids)]
            check['lifecycle'][key] = {'verified': bool(labels) and all(_verified(x) for x in labels),
                                      'error_samples': len(bad), 'times_s': [r['time_s'] for r in bad],
                                      'labelled_seconds': _seconds(_union(spans)),
                                      'continuous_observation_seconds': _seconds(_intersect(spans, continuous))}
        checks.append(check)
    result = {'schema_version': 1, 'camera_id': camera_id, 'events': checks,
              'alarms': alarms, 'duplicate_emissions_same_id': repeats, 'carry_in_red_events': carry_in,
              'coverage': {'recording_seconds': end - start, 'observable_seconds': _seconds(observable),
                           'unobservable_or_unknown_seconds': end - start - _seconds(observable),
                           'observable_fraction': _seconds(observable) / (end - start),
                           'observations': len(normalized),
                           'observable_observations': sum(r['observable'] for r in normalized),
                           'max_observation_gap_s': gap},
              'normal_positive_label_conflict_seconds': conflicts,
              'limitations': ['One camera and one session; aggregate rates by summing counts and exposure hours.',
                              'Unlabelled time is never a normal negative; missing coverage never removes a GT event.',
                              'Only human-confirmed labels with a named reviewer enter verified metrics.',
                              'Source-time matches are disclosed proxies; actual output delay is null if unavailable.',
                              'Short development recordings do not establish deployment or care-home hourly performance.']}
    for cohort in ('verified', 'provisional'):
        subset = [e for e in checks if e['verified'] == (cohort == 'verified')]
        hours = _seconds(effective[cohort]) / 3600
        false = [a for a in alarms if _inside(a['time_s'], effective[cohort])]
        false_repeats = [a for a in repeats if _inside(a['time_s'], effective[cohort])]
        hit = sum(e['hit'] for e in subset)
        result[cohort] = {'annotated_events': len(subset), 'hits': hit, 'misses': len(subset) - hit,
                          'recall': hit / len(subset) if subset else None,
                          'source_confirmation_delay': _delay_summary([e['source_confirmation_delay_s'] for e in subset
                                                                      if e['hit']]),
                          'output_confirmation_delay': _delay_summary([e['output_confirmation_delay_s'] for e in subset
                                                                      if e['output_confirmation_delay_s'] is not None]),
                          'matched_without_output_timestamp': sum(e['hit'] and e['output_confirmation_delay_s'] is None
                                                                  for e in subset),
                          'normal_labelled_seconds': _seconds(normal[cohort]),
                          'effective_normal_monitoring_seconds': hours * 3600,
                          'effective_normal_monitoring_hours': hours,
                          'false_alarm_events': len(false) if hours else None,
                          'false_alarm_event_details': false,
                          'false_alarm_events_per_camera_hour': len(false) / hours if hours else None,
                          'false_alarm_emissions': len(false) + len(false_repeats) if hours else None,
                          'false_alarm_emissions_per_camera_hour': (len(false) + len(false_repeats)) / hours if hours else None,
                          'duplicate_new_ids_within_gt_events': sum(len(e['duplicate_new_ids']) for e in subset)}
    result['unscored_new_alarms'] = [a for i, a in enumerate(alarms) if i not in assigned | duplicate_indices and
                                    not any(_inside(a['time_s'], spans) for spans in effective.values())]
    result['verification'] = {'has_verified_events': bool(result['verified']['annotated_events']),
                              'has_verified_normal_exposure': result['verified']['effective_normal_monitoring_hours'] > 0,
                              'labels_overlap': any(conflicts.values()),
                              'independent_deployment_validation': False}
    return result
