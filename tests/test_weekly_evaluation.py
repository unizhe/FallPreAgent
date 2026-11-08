"""The evaluator must not fabricate action GT or hide missing state labels."""
from copy import deepcopy

import pytest

from fallpre.weekly.evaluate_human import evaluate_records, parse_states, summarize_states


def fixture_records():
    meta = {"clip_id": "fall-01", "labels_used_by_models": False}
    rows = [{"clip_id": "fall-01", "source_frame_id": i, "t": i / 30, "instances": [{}]}
            for i in range(1, 6)]
    action = {"source_frame_ids": [2, 3, 4], "source_times_s": [2/30, 3/30, 4/30], "frames": 3,
              "window_start_s": 2/30, "window_end_s": 4/30, "top5": [{"label": "falling", "score": .6}]}
    metrics = {"clip_id": "fall-01", "frame_count": 5, "action_windows": 1}
    states = {("fall-01", 1): -1, ("fall-01", 2): 0, ("fall-01", 4): 1, ("fall-01", 5): 1}
    return meta, rows, [action], metrics, states


def test_missing_state_is_unknown_not_negative_and_lying_is_not_falling():
    result = evaluate_records(*fixture_records())
    assert result["run_state_composition"]["missing_frame_ids"] == [3]
    composition = result["action_windows"][0]["official_state_composition"]
    assert composition["counts"] == {"not_lying": 0, "fall_transition": 1, "lying_on_ground": 1, "unknown": 1}
    assert composition["fractions"]["unknown"] == pytest.approx(1/3)
    assert result["five_action_accuracy"] is None
    assert result["coverage"]["transition_window_coverage"] == 1
    assert result["clip_level_descriptive_probe"]["any_window_top1_is_falling"] is True


@pytest.mark.parametrize("mutation", ["timestamp", "endpoint", "missing_frame", "duplicate", "out_of_order"])
def test_rejects_misaligned_model_windows(mutation):
    arguments = list(deepcopy(fixture_records()))
    action = arguments[2][0]
    if mutation == "timestamp":
        action["source_times_s"][1] += .1
    elif mutation == "endpoint":
        action["window_end_s"] += .1
    elif mutation == "missing_frame":
        action["source_frame_ids"] = [4, 5, 6]
    elif mutation == "duplicate":
        action["source_frame_ids"] = [2, 2, 4]
    else:
        action["source_frame_ids"] = [4, 3, 2]
    with pytest.raises(ValueError):
        evaluate_records(*arguments)


def test_no_window_and_no_transition_are_null_not_negative_or_perfect():
    arguments = list(fixture_records())
    arguments[2] = []
    arguments[3]["action_windows"] = 0
    arguments[4] = {("fall-01", i): -1 for i in range(1, 6)}
    result = evaluate_records(*arguments)
    assert result["coverage"]["classified_window_source_frame_coverage"] == 0
    assert result["coverage"]["transition_window_coverage"] is None
    assert result["clip_level_descriptive_probe"]["any_window_top1_is_falling"] is None


def test_overlapping_windows_count_unique_source_support():
    arguments = list(fixture_records())
    arguments[2].append(deepcopy(arguments[2][0]))
    arguments[3]["action_windows"] = 2
    result = evaluate_records(*arguments)
    assert result["coverage"]["frames_in_any_classified_window"] == 3
    assert result["coverage"]["transition_frames_in_any_classified_window"] == 1


def test_official_parser_ignores_features_but_rejects_duplicate_labels():
    data = b"adl-01,6,-1,99,1\nadl-01,8,-1,23,0\n"
    states = parse_states(data)
    assert summarize_states("adl-01", [6, 7, 8], states)["missing_frame_ids"] == [7]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_states(data + b"adl-01,6,-1,99\n")


def test_evaluator_requires_explicit_label_independence():
    arguments = list(fixture_records())
    arguments[0]["labels_used_by_models"] = True
    with pytest.raises(ValueError, match="did not use labels"):
        evaluate_records(*arguments)
