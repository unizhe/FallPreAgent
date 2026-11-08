"""Live completion must use real elapsed time and independent observations."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest
from test_action_v2 import person, row
from test_action_v5 import make_stream as stream_v5
from test_action_v6 import stream as stream_v6
from test_action_v7 import stream as stream_v7

from fallpre.weekly.action_v5 import live_terminal_features, terminal_features
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.completion_timing import completion_vote


def make_live(version):
    stream = {5: stream_v5, 6: stream_v6, 7: stream_v7}[version]()
    stream.live_timing = True
    stream.model.config["window_s"] = 0.1 if version == 5 else 0.067
    return stream


def timed_row(t, index=0):
    return dict(row(index, [person()]), t=t)


@pytest.mark.parametrize("version", [5, 6, 7])
@pytest.mark.parametrize("fps", [7.5, 10, 30])
def test_supported_completion_at_real_camera_cadences(version, fps):
    stream = make_live(version)
    predictions = [stream.update(timed_row(i / fps, i)) for i in range(int(fps) + 1)]
    assert predictions[0]["label"] == "standing_up"
    assert predictions[-1]["label"] == "background"
    first = next(i for i, pred in enumerate(predictions) if pred["label"] == "background")
    assert first >= 1
    assert first / fps >= 0.15
    assert len(stream.votes) >= 2


@pytest.mark.parametrize("version", [5, 6, 7])
def test_completion_with_jitter_needs_independent_recent_votes(version):
    stream = make_live(version)
    timestamps = [0, 0.112, 0.263, 0.391, 0.57, 0.69, 0.91]
    results = [stream.update(timed_row(t, i)) for i, t in enumerate(timestamps)]
    assert results[0]["label"] == results[1]["label"] == "standing_up"
    assert results[-1]["label"] == "background"
    assert len({t for t, _ in stream.votes}) == len(stream.votes)


@pytest.mark.parametrize("version", [5, 6, 7])
@pytest.mark.parametrize("change", ["identity", "gap", "missing", "feature_missing"])
def test_live_completion_cannot_reuse_evidence_after_discontinuity(version, change):
    stream = make_live(version)
    for i in range(8):
        stream.update(timed_row(i / 7.5, i))
    assert stream.completed
    current = timed_row(8 / 7.5, 8)
    if change == "identity":
        current["target_epoch"] = 9
    elif change == "gap":
        current["t"] += 0.36
    elif change == "missing":
        current["instances"] = []
    else:
        stream.feature_provider = lambda *_: None
    prediction = stream.update(current)
    assert not stream.completed
    assert len(stream.votes) <= 1
    assert not prediction.get("completion_latched", False)
    assert not prediction.get("fast_completion_latched", False)
    assert not prediction.get("motion_completion_latched", False)


@pytest.mark.parametrize("version", [5, 6, 7])
@pytest.mark.parametrize("invalid_t", [0.1, 0.4, -1, float("nan"), float("inf")])
def test_invalid_clock_clears_votes_and_latch_before_rejection(version, invalid_t):
    stream = make_live(version)
    for i in range(4):
        stream.update(timed_row(i / 7.5, i))
    assert stream.completed
    with pytest.raises(ValueError, match="source time"):
        stream.update(timed_row(invalid_t, 4))
    assert not stream.completed
    assert not stream.votes
    assert not stream.history


def test_one_positive_sample_after_negative_is_not_completion():
    votes = deque()
    assert completion_vote(votes, 0, False, 0.067, live_timing=True) == (False, 0)
    enough, fraction = completion_vote(votes, 0.133, True, 0.067, live_timing=True)
    assert enough and fraction < 0.65
    enough, fraction = completion_vote(votes, 0.266, True, 0.067, live_timing=True)
    assert enough and fraction == 1


def test_gap_does_not_create_confirmation_from_single_vote():
    votes = deque([(0, True)])
    assert completion_vote(votes, 0.36, True, 0.067, live_timing=True) == (False, 1)
    assert list(votes) == [(0.36, True)]


@pytest.mark.parametrize("fps", [7.5, 10])
def test_low_rate_constant_ascent_has_real_velocity_and_motion_veto(fps):
    history = [dict(timed_row(i / fps, i), instances=[person(y=100 - 20 * i / fps)]) for i in range(5)]
    features = live_terminal_features(history, {}, 0)
    assert features is not None
    assert features[10] < -0.1 and features[11] < -0.1
    if fps == 7.5:
        frozen = terminal_features(history, {}, 0)
        assert frozen[10] == frozen[11] == 0
    stream = make_live(7)
    stream.feature_provider = live_terminal_features
    for observation in history:
        prediction = stream.update(observation)
    assert prediction["completion_motion_veto"]
    assert prediction["label"] == "standing_up"


def test_dense_cadence_features_and_vote_decisions_preserve_frozen_behavior():
    history = [dict(timed_row(i / 30, i), instances=[person(y=100 - i)]) for i in range(20)]
    assert np.array_equal(terminal_features(history, {}, 0), live_terminal_features(history, {}, 0))
    frozen, live = deque(), deque()
    for i in range(100):
        vote = i % 7 > 1
        assert completion_vote(frozen, i / 30, vote, 0.067) == completion_vote(
            live, i / 30, vote, 0.067, live_timing=True
        )


@pytest.mark.parametrize("change", ["missing", "identity", "clock", "gap", "nan_pose"])
def test_terminal_features_never_bridge_invalid_or_unobserved_history(change):
    history = [timed_row(i / 7.5, i) for i in range(4)]
    if change == "missing":
        history[-2]["instances"] = []
    elif change == "identity":
        history[-1]["target_epoch"] = 2
    elif change == "clock":
        history[-2]["t"] = history[-1]["t"]
    elif change == "gap":
        history[-1]["t"] += 0.36
    else:
        history[-1]["instances"][0]["kpts"][11][1] = float("nan")
    assert live_terminal_features(history, {}, 0) is None


def test_live_constructor_option_propagates_without_changing_artifact_defaults():
    path = Path(__file__).parents[1] / "runs/action-v7/training/model.json"
    model = CompletionModel.load(path)
    for enabled in [False, True]:
        stream = OnlineCompletion(model, live_timing=enabled)
        for layer in [stream, stream.baseline, stream.baseline.baseline]:
            assert layer.live_timing is enabled
            assert layer.feature_provider is (live_terminal_features if enabled else terminal_features)
