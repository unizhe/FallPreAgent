"""Queue-policy safeguards; no model loading or inference."""

import math

import pytest

from fallpre.weekly.replay_scheduling import ReplaySchedule


@pytest.mark.parametrize("policy", ["fifo", "latest"])
def test_boundary_arrival_is_available_but_future_frame_is_not(policy):
    schedule = ReplaySchedule([0, 0.033, 0.067], policy)
    assert schedule.next(0).index == 0
    assert schedule.next(math.nextafter(0.033, -math.inf)) is None
    selected = schedule.next(0.033)
    assert selected.index == 1
    assert selected.arrival_time_s == 0.033
    assert schedule.next(0.033) is None
    assert schedule.cursor == 2


def test_latest_drops_only_older_arrived_frames_on_irregular_clock():
    schedule = ReplaySchedule([0, 0.017, 0.041, 0.146, 0.3], "latest")
    assert schedule.next(0).index == 0
    selected = schedule.next(0.1)
    assert selected.index == 2
    assert selected.arrival_time_s == 0.041
    assert selected.dropped_indices == (1,)
    assert selected.arrived_pending_before == 2
    assert selected.pending_after_selection == 0
    assert schedule.next_arrival_s == 0.146
    assert schedule.next(0.14) is None


def test_fifo_retains_pending_while_latest_consumes_pending_prefix():
    times = [0, 0.03, 0.07, 0.1]
    fifo, latest = ReplaySchedule(times, "fifo"), ReplaySchedule(times, "latest")
    old, new = fifo.next(0.08), latest.next(0.08)
    assert (old.index, old.dropped_indices, old.pending_after_selection) == (0, (), 2)
    assert (new.index, new.dropped_indices, new.pending_after_selection) == (2, (0, 1), 0)
    assert old.arrived_pending_before == new.arrived_pending_before == 3
    assert fifo.next(0.08).index == 1
    assert fifo.next(0.08).index == 2
    assert fifo.next(0.08) is latest.next(0.08) is None


@pytest.mark.parametrize("policy", ["fifo", "latest"])
def test_processed_and_dropped_form_exact_partition_of_160_frames(policy):
    times = [i / 30 for i in range(160)]
    schedule = ReplaySchedule(times, policy)
    elapsed, processed, dropped = 0.0, [], []
    while not schedule.complete:
        selected = schedule.next(elapsed)
        if selected is None:
            elapsed = schedule.next_arrival_s
            continue
        processed.append(selected.index)
        dropped.extend(selected.dropped_indices)
        assert selected.arrival_time_s <= elapsed
        # Deliberately include stalls, including one longer than model gap limits.
        elapsed += 0.8 if len(processed) % 11 == 0 else 0.08
    assert len(processed) + len(dropped) == 160
    assert sorted(processed + dropped) == list(range(160))
    assert set(processed).isdisjoint(dropped)
    assert processed == sorted(set(processed))
    assert processed[-1] == 159
    assert bool(dropped) == (policy == "latest")
    assert schedule.next(elapsed) is None
    assert schedule.next_arrival_s is None


def test_completion_of_current_work_does_not_invent_future_arrivals():
    schedule = ReplaySchedule([0, 0.1, 1.0], "latest")
    assert schedule.next(0).index == 0
    assert schedule.next(0.1).index == 1
    assert schedule.next(0.2) is None
    assert not schedule.complete
    assert schedule.next_arrival_s == 1.0
    assert schedule.next(0.999) is None
    assert schedule.next(1.0).index == 2
    assert schedule.complete


def test_scheduling_never_retimes_source_gap_to_worker_elapsed_time():
    original_times = [0, 0.03, 0.07, 0.8, 0.84]
    schedule = ReplaySchedule(original_times, "latest")
    first = schedule.next(0)
    after_stall = schedule.next(0.91)
    assert first.arrival_time_s == 0
    assert after_stall.index == 4
    assert after_stall.arrival_time_s == original_times[4] == 0.84
    assert after_stall.arrival_time_s != 0.91
    assert after_stall.arrival_time_s - first.arrival_time_s > 0.5
    assert schedule.times == tuple(original_times)


@pytest.mark.parametrize("times", [[], [1], [0, 0], [0, -0.1], [0, 0.2, 0.1], [0, float("inf")], [0, float("nan")]])
def test_invalid_arrival_clock_is_rejected(times):
    with pytest.raises(ValueError):
        ReplaySchedule(times)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="policy"):
        ReplaySchedule([0], "drop_every_other")


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf")])
def test_invalid_elapsed_time_is_rejected_without_consuming_frames(elapsed):
    schedule = ReplaySchedule([0, 1])
    with pytest.raises(ValueError):
        schedule.next(elapsed)
    assert schedule.cursor == 0


def test_backwards_clock_is_rejected_and_equal_elapsed_is_allowed():
    schedule = ReplaySchedule([0, 0.1, 0.2])
    assert schedule.next(0.1).index == 0
    with pytest.raises(ValueError, match="backwards"):
        schedule.next(0.099)
    assert schedule.cursor == 1
    assert schedule.next(0.1).index == 1
