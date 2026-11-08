"""Official state proxies must not silently expand into normal/fall truth."""
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
proxy_label = runpy.run_path(str(ROOT / 'scripts/extract_stgcn_urfd_official.py'))['official_proxy_label']


def test_only_explicit_simulated_fall_transition_can_be_positive():
    states = {('fall-01', 1): -1, ('fall-01', 2): 0, ('fall-01', 3): 1}
    assert proxy_label('fall-01', 1, states)[0] == 'non_fall_activity'
    assert proxy_label('fall-01', 2, states)[0] == 'fall_process'
    assert proxy_label('fall-01', 3, states)[0] is None
    assert proxy_label('fall-01', 4, states)[0] is None


def test_transition_in_adl_excludes_whole_ambiguous_session():
    states = {('adl-10', 1): -1, ('adl-10', 2): 0, ('adl-10', 3): 1}
    assert all(proxy_label('adl-10', frame, states)[0] is None for frame in (1, 2, 3))


def test_source_group_name_does_not_fill_missing_labels():
    assert proxy_label('adl-04', 1, {})[0] is None
    assert proxy_label('fall-01', 1, {})[0] is None


def test_clear_adl_states_remain_separate_from_observation_quality():
    # This produces only a label proposal. Extraction separately requires a
    # ready, same-target causal window; missing poses are not labeled negative.
    states = {('adl-04', 1): -1}
    assert proxy_label('adl-04', 1, states)[0] == 'non_fall_activity'
