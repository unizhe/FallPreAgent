from types import SimpleNamespace

import numpy as np
import pytest
from test_action_v2 import person, row
from test_action_v4 import stream as phase_stream

from fallpre.weekly.action_v5 import OnlineCompletion, terminal_features


def make_stream(prob=0.9, label="standing_up", phase="rising"):
    s = OnlineCompletion.__new__(OnlineCompletion)
    # Initialize with a valid frozen-V4-shaped stub, then isolate completion tests.
    stub = phase_stream().model
    model = SimpleNamespace(
        base=stub,
        config=dict(threshold=0.6, window_s=0.2, vote_fraction=0.65, minimum_rise_s=0.15),
        probability=lambda x: prob,
    )
    OnlineCompletion.__init__(s, model)
    s.baseline.update = lambda r: dict(
        label=label, phase=phase, v3_label="background", phase_scores={}, phase_up=0.5, phase_opening=0.2
    )
    s.feature_provider = lambda h, p, t: np.zeros(54)
    return s


def test_sustained_completion_ends_rise_and_latches():
    s = make_stream()
    labels = [s.update(row(i, [person()]))["label"] for i in range(20)]
    assert labels[0] == "standing_up" and labels[-1] == "background"
    s.model.probability = lambda x: 0
    assert s.update(row(21, [person()]))["completion_latched"]


def test_low_probability_slow_ascent_not_forced_to_end():
    s = make_stream(0.1)
    assert all(s.update(row(i, [person()]))["label"] == "standing_up" for i in range(250))


@pytest.mark.parametrize("label", ["walking", "bending", "falling", "sitting_down", "background", "unknown"])
def test_only_existing_rising_output_can_change(label):
    s = make_stream(label=label)
    assert all(s.update(row(i, [person()]))["label"] == label for i in range(30))


def test_minimum_initial_rise_is_retained():
    s = make_stream()
    s.model.config["minimum_rise_s"] = 0.3
    assert all(s.update(row(i, [person()]))["label"] == "standing_up" for i in range(8))


def test_identity_change_resets_terminal_latch():
    s = make_stream()
    for i in range(20):
        s.update(row(i, [person()]))
    r = row(21, [person()])
    r["target_epoch"] = 2
    assert not s.update(r)["completion_latched"]


def test_new_cycle_can_rise_after_completed_cycle():
    s = make_stream()
    for i in range(20):
        s.update(row(i, [person()]))
    s.baseline.update = lambda r: dict(label="background", phase="seated")
    s.update(row(21, [person()]))
    s.baseline.update = lambda r: dict(label="standing_up", phase="rising")
    assert s.update(row(22, [person()]))["label"] == "standing_up"


def test_terminal_features_ignore_labels_and_references():
    history = [row(i, [person()]) for i in range(20)]
    p = dict(phase_up=0.5, phase_opening=0.2, phase_scores={})
    a = terminal_features(history, p, 0)
    for r in history:
        r.update(target="standing", reference_phase="standing", subject_id="secret")
    assert np.array_equal(a, terminal_features(history, p, 0))


def test_missing_current_frame_provides_no_completion_features():
    history = [row(i, [person()]) for i in range(20)] + [row(21, [])]
    assert terminal_features(history, {}, 0) is None
