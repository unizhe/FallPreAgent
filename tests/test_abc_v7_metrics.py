"""Regression tests for ABC evaluation clocks, abstention and event accounting."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_abc_v7", ROOT / "scripts/evaluate_abc_v7.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def rows(times, labels, phases=None):
    return [
        dict(
            source_frame_id=i + 1, source_time_s=t, observed=True,
            target_id=1, epoch=1, presence_only=False,
            prediction=dict(label=label, phase=(phases or ["other"] * len(times))[i]),
        )
        for i, (t, label) in enumerate(zip(times, labels))
    ]


def annotation(clip="example", group="event-1", primary=True, actions=None, states=None):
    return dict(
        clip_id=clip, event_group_id=group, primary_event_representation=primary,
        action_segments=actions or [], state_segments=states or [], visibility_segments=[],
    )


def segment(start, end, label):
    return dict(start_frame=start, end_frame=end, label=label, human_reviewed=False)


def test_irregular_source_clock_conserves_span_without_inventing_tail():
    weights, gaps = evaluation.sample_weights([0.0, 0.1, 0.3, 1.0])
    assert weights == pytest.approx([0.1, 0.2, 0.35, 0.0])
    assert gaps == pytest.approx(0.35)
    assert sum(weights) + gaps == pytest.approx(1.0)
    assert evaluation.sample_weights([8.2]) == ([0.0], 0)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, None], [0, float("nan")]])
def test_invalid_source_clocks_are_rejected(times):
    with pytest.raises(ValueError):
        evaluation.sample_weights(times)


def test_unknown_keeps_reference_support_and_terminal_unknown_has_zero_seconds():
    observed = rows([0.0, 0.1, 0.3, 1.0], ["walking", "unknown", "background", "unknown"])
    ann = annotation(actions=[segment(1, 4, "background")])
    metrics, _, _ = evaluation.evaluate_clip("A", observed, ann)
    assert metrics["unknown_samples"] == 2
    assert metrics["unknown_s"] == pytest.approx(0.2)
    assert metrics["annotated_action_s"] == pytest.approx(0.65)
    assert metrics["per_class"]["background"]["support_s"] == pytest.approx(0.65)
    assert metrics["per_class"]["background"]["recall"] == pytest.approx(0.35 / 0.65)
    assert metrics["confusion_duration_s"]["background"]["unknown"] == pytest.approx(0.2)


def test_large_clock_gap_does_not_bridge_an_episode():
    observed = rows([0.0, 0.1, 1.0, 1.1], ["unknown"] * 4)
    weights, _ = evaluation.sample_weights([r["source_time_s"] for r in observed])
    episodes = evaluation.runs_of_true([True] * 4, observed, weights)
    assert [(e["start_frame"], e["end_frame"]) for e in episodes] == [(1, 2), (3, 4)]
    assert [e["duration_s"] for e in episodes] == pytest.approx([0.45, 0.1])


def test_false_episodes_break_on_abstention_and_reference_gaps():
    observed = rows(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        ["walking", "walking", "unknown", "walking", "bending", "walking"],
    )
    ann = annotation(actions=[segment(1, 4, "background"), segment(6, 6, "background")])
    metrics, _, details = evaluation.evaluate_clip("A", observed, ann)
    walking = details["false_action_runs"]["walking"]
    assert [(e["start_frame"], e["end_frame"]) for e in walking] == [(1, 2), (4, 4), (6, 6)]
    assert metrics["false_action_episodes"]["walking"] == 3
    assert metrics["false_action_s"]["walking"] == pytest.approx(0.3)
    assert metrics["false_action_episodes"]["bending"] == 0
    assert metrics["unknown_s"] == pytest.approx(0.1)


def test_event_requires_hit_inside_inclusive_frame_interval():
    ann = annotation(actions=[segment(2, 3, "falling")])
    outside = rows([0.0, 0.1, 0.2, 0.3], ["falling", "unknown", "background", "falling"])
    _, events, _ = evaluation.evaluate_clip("A", outside, ann)
    assert not events[0]["detected"]
    assert events[0]["first_hit_frame"] is None
    at_end = rows([0.0, 0.1, 0.2, 0.3], ["falling", "unknown", "falling", "falling"])
    _, events, _ = evaluation.evaluate_clip("A", at_end, ann)
    assert events[0]["detected"]
    assert events[0]["first_hit_frame"] == 3
    assert events[0]["first_hit_delay_s"] == pytest.approx(0.1)


def test_same_event_original_and_preview_count_once_in_aggregate():
    observed = rows([0.0, 0.1, 0.2], ["falling"] * 3)
    metrics, events = [], []
    for cid, primary in [("urfd-fall-01-original", True), ("urfd-fall-01", False)]:
        ann = annotation(cid, group="urfd-fall-01", primary=primary,
                         actions=[segment(1, 3, "falling")])
        metric, hits, _ = evaluation.evaluate_clip("A", observed, ann)
        metrics.append(metric)
        events.extend(hits)
    result = evaluation.aggregate(metrics, events)["A"]
    assert result["unique_clocked_groups"] == 1
    assert result["observations"] == 3
    assert result["action_event_total"] == result["action_event_hits"] == 1
    assert result["annotated_action_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("primaries", [(True, True), (False, False)])
def test_same_event_requires_exactly_one_primary_representation(primaries):
    observed = rows([0.0, 0.1], ["falling"] * 2)
    metrics, events = [], []
    for cid, primary in zip(("original", "preview"), primaries):
        ann = annotation(cid, group="same-fall", primary=primary,
                         actions=[segment(1, 2, "falling")])
        metric, hits, _ = evaluation.evaluate_clip("A", observed, ann)
        metrics.append(metric)
        events.extend(hits)
    with pytest.raises(ValueError, match="exactly one primary"):
        evaluation.aggregate(metrics, events)


def test_multicam_counts_observations_but_excludes_actions_and_seconds():
    observed = rows([None, None, None], ["disabled_frame_only"] * 3, ["disabled"] * 3)
    observed[1]["observed"] = False
    ann = annotation("multicam", actions=[segment(1, 3, "falling")])
    metrics, events, _ = evaluation.evaluate_clip("A", observed, ann)
    assert metrics["frame_only"]
    assert metrics["target_unavailable_samples"] == 1
    for key in ("unknown_s", "target_unavailable_s", "held_observation_s",
                "source_span_s", "action_event_hits", "action_event_total"):
        assert metrics[key] is None
    assert events == []
    result = evaluation.aggregate([metrics], events)["A"]
    assert result["unique_clocked_groups"] == result["action_event_total"] == 0


def test_seated_and_coarse_metadata_do_not_create_five_action_events():
    observed = rows([0.0, 0.1, 0.2], ["standing_up"] * 3, ["forward_lean"] * 3)
    ann = annotation(states=[dict(start_frame=1, end_frame=3, state="seated")])
    ann["official_coarse_segments"] = [segment(1, 3, "falling")]
    metrics, events, _ = evaluation.evaluate_clip("A", observed, ann)
    assert metrics["seated_false_rise_samples"] == 3
    assert metrics["seated_forward_lean_samples"] == 3
    assert metrics["annotated_action_s"] == 0
    assert metrics["confusion_duration_s"] == {}
    assert events == []
