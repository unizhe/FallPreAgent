from copy import deepcopy

import numpy as np
import pytest

from fallpre.weekly.stgcn_fall_head import (
    CLASSES,
    binary_metrics,
    endpoint_region,
    grouped_folds,
    guard_destination,
    leakage_groups,
    reviewed_regions,
)

MANIFEST = {'clips': [{'id': 'a', 'duration_s': 10, 'video': {'sha256': 'hash'}}]}


def event():
    return dict(clip_id='a', video_sha256='hash', event_id='e1', event_kind='fall',
                onset_s=2., landing_s=3., recovery_start_s=5., recovery_end_s=7.,
                uncertainty_s=dict(onset=.1, landing=.1, recovery_start=.2, recovery_end=.2),
                quality=dict(process='observable', lying='observable', recovery='observable'),
                review_status='human_confirmed', reviewed_by='test reviewer', reviewed_at='2026-11-07')


def negative():
    return dict(clip_id='a', video_sha256='hash', start_s=7., end_s=9.,
                boundary_uncertainty_s=.1, label='non_fall_activity', activity_note='walking',
                quality='observable', review_status='human_confirmed',
                reviewed_by='test reviewer', reviewed_at='2026-11-07')


def record(t):
    return dict(clip_id='a', video_time_s=t, source_time_s=t + .01, window_end_s=t + .01,
                window=dict(causal=True, temporal_wrap=False))


def test_label_is_only_current_process_and_never_entire_event_lying_or_recovery():
    regions, _ = reviewed_regions({'events': [event()], 'non_fall_intervals': [negative()]}, MANIFEST)
    assert endpoint_region(regions, record(2.5))['label'] == 'fall_process'
    assert endpoint_region(regions, record(2.05)) is None
    assert endpoint_region(regions, record(2.95)) is None
    assert endpoint_region(regions, record(4.)) is None
    assert endpoint_region(regions, record(6.)) is None
    assert endpoint_region(regions, record(8.))['label'] == 'non_fall_activity'


@pytest.mark.parametrize('kind', ['intentional_lying', 'uncertain'])
def test_active_lying_or_uncertain_never_becomes_fall_truth(kind):
    e = dict(event(), event_kind=kind)
    regions, events = reviewed_regions({'events': [e]}, MANIFEST)
    assert regions == [] and len(events) == 1


def test_ai_and_low_quality_are_not_training_classes():
    e = dict(event(), review_status='ai_provisional_pending_human')
    n = dict(negative(), quality='occluded')
    assert reviewed_regions({'events': [e], 'non_fall_intervals': [n]}, MANIFEST)[0] == []
    with pytest.raises(ValueError, match='No human-reviewed'):
        grouped_folds([])


@pytest.mark.parametrize('field,value', [('landing_s', 1), ('onset_s', float('nan')),
                                       ('video_sha256', 'wrong'), ('reviewed_by', ''),
                                       ('event_kind', 'normal'), ('recovery_end_s', 11)])
def test_invalid_reviews_fail_closed(field, value):
    e = dict(event(), **{field: value})
    with pytest.raises(ValueError):
        reviewed_regions({'events': [e]}, MANIFEST)


def test_censored_recovery_allowed_but_overlap_and_future_features_rejected():
    e = dict(event(), recovery_start_s=None, recovery_end_s=None)
    regions, _ = reviewed_regions({'events': [e]}, MANIFEST)
    assert endpoint_region(regions, record(2.5))
    future = dict(record(2.5), window_end_s=4.)
    with pytest.raises(ValueError, match='current source time'):
        endpoint_region(regions, future)
    n = dict(negative(), start_s=2., end_s=4.)
    with pytest.raises(ValueError, match='overlap'):
        reviewed_regions({'events': [event()], 'non_fall_intervals': [n]}, MANIFEST)


def samples():
    return [dict(clip_id=f'{g}-{label}', video_sha256=f'{g}-{label}', group=g, label=label,
                 subject_id=None, camera_setup_id=None, session_id=g)
            for g in ['day1', 'day2', 'day3'] for label in CLASSES for _ in range(2)]


def test_unknown_person_view_does_not_block_dev_but_cannot_establish_independence():
    rows = samples()
    folds = grouped_folds(rows)
    assert len(folds) == 3
    for _, train, val in folds:
        assert not {rows[i]['group'] for i in train} & {rows[i]['group'] for i in val}
    with pytest.raises(ValueError, match='Unknown grouping'):
        grouped_folds(rows, strict=True)


def test_duplicate_raw_video_or_export_cannot_leak_across_groups():
    rows = samples()
    rows[-1]['video_sha256'] = rows[0]['video_sha256']
    groups = leakage_groups(rows)
    assert groups[-1] == groups[0]
    with pytest.raises(ValueError, match='Insufficient'):
        grouped_folds(rows)


def test_strict_components_merge_people_and_views_across_sessions():
    rows = deepcopy(samples())
    for r in rows:
        r['subject_id'] = 'same-person'
        r['camera_setup_id'] = r['group']
    assert len(set(leakage_groups(rows, strict=True))) == 1


def test_shared_historical_links_and_parent_traversal_are_read_only(tmp_path):
    runs = tmp_path / 'runs'
    shared = runs / '_shared-assets'
    shared.mkdir(parents=True)
    (runs / 'history').symlink_to(shared, target_is_directory=True)
    guard_destination(runs / 'new/data', runs)
    for path in [shared / 'model.npz', runs / 'history/model.npz', runs / '../outside']:
        with pytest.raises(ValueError):
            guard_destination(path, runs)


def test_metrics_keep_misses_and_undefined_precision_explicit():
    metrics = binary_metrics(np.array([0, 1, 1]), np.array([0, 0, 0]))
    assert metrics['confusion_matrix'] == [[1, 0], [2, 0]]
    assert metrics['per_class']['fall_process']['recall'] == 0
    assert metrics['per_class']['fall_process']['precision'] is None
