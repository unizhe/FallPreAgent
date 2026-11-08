"""Persistence and training boundaries for real human review, without labelling source data."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('fall_review', ROOT / 'scripts/serve_fall_candidate_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module.ReviewStore(tmp_path / 'runs/review-test')


def payload(store):
    return dict(reviewed_by='AUTOMATED TEST ONLY', revision=None, entries=[dict(
        clip_id=c['clip_id'], event_id=c['event_id'], action_type='simulated_fall',
        quality=dict.fromkeys(('process', 'lying', 'recovery'), 'observable'),
        **c['initial_times'], boundary_resolution='confirmed', boundary_uncertainty_s=.2,
        confirmed=True, note='Synthetic software test; not human annotation')
        for c in store.bootstrap()['candidates']])


def test_original_video_references_and_clip_tail(store):
    b = store.bootstrap()
    assert len(b['candidates']) == 14
    assert b['candidates'][3]['initial_times']['recovery_start_s'] is None
    assert all(p.is_symlink() for p in store.media.iterdir())
    assert b['draft'] is None


def test_draft_resume_and_stale_tab_rejected(store):
    p = payload(store)
    p['entries'][0]['confirmed'] = False
    saved = store.save(p)
    assert store.bootstrap()['draft']['entries'][0]['confirmed'] is False
    assert saved['revision'] == store.bootstrap()['revision']
    with pytest.raises(FileExistsError):
        store.save(p)


def test_final_requires_every_confirmation_and_valid_times(store):
    p = payload(store)
    p['entries'][0]['confirmed'] = False
    with pytest.raises(ValueError, match='尚未确认'):
        store.save(p, final=True)
    assert not store.draft.exists()
    p['entries'][0]['confirmed'] = True
    p['entries'][0]['landing_s'] = 0
    with pytest.raises(ValueError):
        store.save(p, final=True)
    assert not store.draft.exists()


def test_unresolved_and_intentional_lying_are_not_fall_training_regions(store):
    p = payload(store)
    p['entries'][0]['boundary_resolution'] = 'unresolved'
    p['entries'][0]['onset_s'] = None
    p['entries'][1]['action_type'] = 'intentional_lying'
    saved = store.save(p, final=True)
    result = json.loads(Path(saved['saved_path']).read_text())
    regions, accepted = module.reviewed_regions(result, store.manifest)
    assert saved['confirmed_events'] == 13 and saved['unresolved_events'] == 1
    assert result['events'][0]['review_status'] == 'human_reviewed_unresolved'
    assert result['events'][1]['event_kind'] == 'intentional_lying'
    assert len(accepted) == 13 and len(regions) == 12
    assert result['non_fall_intervals'] == []
    assert all(e['demonstration'] for e in result['events'])
    assert all(e['video_sha256'] for e in result['events'])
    original = Path(saved['saved_path']).read_bytes()
    p['revision'] = saved['revision']
    again = store.save(p, final=True)
    assert again['saved_path'] != saved['saved_path']
    assert Path(saved['saved_path']).read_bytes() == original


def test_forged_candidate_and_quality_rejected(store):
    p = payload(store)
    p['entries'][0]['event_id'] = 'another-person'
    with pytest.raises(ValueError, match='身份'):
        store.save(p)
    p = payload(store)
    p['entries'][0]['quality']['process'] = ''
    with pytest.raises(ValueError, match='缺少'):
        store.save(p, final=True)
