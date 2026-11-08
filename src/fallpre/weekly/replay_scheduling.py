"""Causal selection from a known frame-arrival schedule, without reading images.

Arrival times are seconds relative to replay start. They remain distinct from
the wall-clock time supplied to ``next``; callers retain each selected frame's
original source timestamp when updating tracking and action state.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Literal


@dataclass(frozen=True)
class ScheduledSelection:
    index: int
    arrival_time_s: float
    dropped_indices: tuple[int, ...]
    arrived_pending_before: int
    pending_after_selection: int


class ReplaySchedule:
    """One serial worker with FIFO or one latest pending frame.

    ``next(elapsed_s)`` is called only when the worker can begin another frame.
    It consumes a frame only after its scheduled arrival. The latest policy
    discards older pending frames at that point; it never cancels an in-flight
    frame. A long service time can therefore still produce a stale output.
    """

    def __init__(self, times: Iterable[float], policy: Literal["fifo", "latest"] = "fifo"):
        self.times = tuple(float(t) for t in times)
        if not self.times or self.times[0] != 0.0:
            raise ValueError("Arrival times must be nonempty and start at zero")
        if not all(math.isfinite(t) for t in self.times):
            raise ValueError("Arrival times must be finite")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("Arrival times must strictly increase")
        if policy not in {"fifo", "latest"}:
            raise ValueError("Expected fifo or latest scheduling policy")
        self.policy = policy
        self.cursor = 0
        self._last_elapsed_s = None

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.times)

    @property
    def next_arrival_s(self) -> float | None:
        return None if self.complete else self.times[self.cursor]

    def next(self, elapsed_s: float) -> ScheduledSelection | None:
        elapsed_s = float(elapsed_s)
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        if self._last_elapsed_s is not None and elapsed_s < self._last_elapsed_s:
            raise ValueError("Elapsed time must not move backwards")
        self._last_elapsed_s = elapsed_s
        arrived_end = bisect_right(self.times, elapsed_s, lo=self.cursor)
        pending = arrived_end - self.cursor
        if pending == 0:
            return None
        selected = self.cursor if self.policy == "fifo" else arrived_end - 1
        result = ScheduledSelection(
            index=selected,
            arrival_time_s=self.times[selected],
            dropped_indices=tuple(range(self.cursor, selected)),
            arrived_pending_before=pending,
            pending_after_selection=arrived_end - selected - 1,
        )
        self.cursor = selected + 1
        return result
