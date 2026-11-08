"""Training must stay blocked unless provenance, quality and grouping are valid."""
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fallpre.weekly.stgcn_home import (
    endpoint_label,
    grouped_folds,
    predict_linear,
    proposed_label,
    train_linear,
    validate_reviews,
)


def review():
    return dict(clip_id='a', video_sha256='sha', start_s=1., end_s=3.,
                label='standing_up', boundary_uncertainty_s=.2,
                observation_quality='observable', review_status='human_confirmed',
                reviewed_by='test reviewer', reviewed_at='2026-11-06')


MANIFEST = {'clips': [{'id': 'a', 'duration_s': 10, 'video': {'sha256': 'sha'}}]}


def test_ai_and_uncertain_quality_cannot_become_training_truth():
    ai, occluded = review(), review()
    ai['review_status'] = 'ai_prelabel_pending_human_review'
    occluded['observation_quality'] = 'occluded'
    assert validate_reviews({'segments': [ai, occluded]}, MANIFEST) == []
    with pytest.raises(ValueError, match='No human-reviewed'):
        grouped_folds([])


@pytest.mark.parametrize('field,value', [('video_sha256', 'other'), ('reviewed_by', ''),
                                       ('end_s', 11), ('boundary_uncertainty_s', float('nan')),
                                       ('label', 'unknown'), ('observation_quality', None)])
def test_bad_provenance_or_ambiguous_truth_is_rejected(field, value):
    row = review()
    row[field] = value
    with pytest.raises(ValueError):
        validate_reviews({'segments': [row]}, MANIFEST)


def test_overlap_is_rejected_and_endpoint_excludes_uncertain_boundaries():
    row = review()
    assert endpoint_label([row], 'a', 1.1) is None
    assert endpoint_label([row], 'a', 1.3) == 'standing_up'
    assert endpoint_label([row], 'a', 2.9) is None
    assert endpoint_label([row], 'a', 3.1) is None
    assert endpoint_label([row], 'other', 2) is None
    second = dict(row, start_s=2, end_s=4, label='standing')
    with pytest.raises(ValueError, match='overlap'):
        validate_reviews({'segments': [row, second]}, MANIFEST)


def records():
    return [dict(clip_id=f'{group}-{label}', group=group, video_sha256=f'{group}-{label}', label=label)
            for group in ['day1', 'day2', 'day3'] for label in ['standing', 'seated'] for _ in range(4)]


def test_cv_keeps_whole_capture_days_and_exports_together():
    rows = records()
    folds = grouped_folds(rows)
    assert len(folds) == 3
    for group, train, val in folds:
        assert {rows[i]['group'] for i in val} == {group}
        assert group not in {rows[i]['group'] for i in train}
    leaked = deepcopy(rows)
    leaked[-1]['video_sha256'] = leaked[0]['video_sha256']
    with pytest.raises(ValueError, match='crosses capture groups'):
        grouped_folds(leaked)


def test_missing_class_across_groups_fails_closed():
    with pytest.raises(ValueError, match='Insufficient'):
        grouped_folds([r for r in records() if r['group'] != 'day3'])


def test_floor_or_broad_squat_is_not_invented_as_fall_or_phase_truth():
    assert proposed_label({'label': 'other', 'subaction': 'squat'}) is None
    assert proposed_label({'label': 'other', 'subaction': 'controlled_floor_descent'}) is None
    assert proposed_label({'label': 'other', 'subaction': 'floor_lying'}) == 'lying'
    assert proposed_label({'label': 'other', 'subaction': 'squat_hold'}) == 'squatting'


def test_small_head_trains_independently_and_is_seed_reproducible():
    # A plumbing check only, not evidence of performance on human recordings.
    x = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    first = train_linear(x, y, seed=17, epochs=30)
    second = train_linear(x, y, seed=17, epochs=30)
    np.testing.assert_array_equal(first['weight'], second['weight'])
    assert predict_linear(first, x).argmax(1).tolist() == y.tolist()


def test_cli_rejects_shared_assets_traversal_and_historical_links(tmp_path):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/stgcn_home_adaptation.py'))
    guard = namespace['ensure_artifact_destination']
    runs = tmp_path / 'runs'
    runs.mkdir()
    shared = runs / '_shared-assets'
    shared.mkdir()
    historical = runs / 'historical'
    historical.mkdir()
    (historical / 'weekly').symlink_to(shared, target_is_directory=True)
    guard(runs / 'new-experiment/data', runs)
    for path in [shared / 'head.npz', historical / 'weekly/head.npz',
                 runs / '../outside', tmp_path / 'outside']:
        with pytest.raises(ValueError):
            guard(path, runs)
