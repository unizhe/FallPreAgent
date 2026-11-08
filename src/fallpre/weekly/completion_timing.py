"""Causal completion evidence for irregular live-camera observation times.

Frozen offline models retain their original windows. Live mode may retain the
immediately preceding real observation when the camera cadence exceeds that
window; it never interpolates observations or duplicates a vote.
"""

import math

MAX_OBSERVATION_GAP_S = 0.35


def clear_completion(stream):
    stream.history.clear()
    stream.rise_start = None
    stream.completed = False
    stream.votes.clear()


def validate_live_time(stream, t):
    if not getattr(stream, "live_timing", False):
        return
    if not math.isfinite(t) or t < 0 or stream.last_t is not None and t <= stream.last_t:
        clear_completion(stream)
        nested = getattr(stream, "baseline", None)
        while nested is not None and hasattr(nested, "votes"):
            clear_completion(nested)
            nested = getattr(nested, "baseline", None)
        raise ValueError("Live completion requires finite, nonnegative, increasing source time")


def completion_vote(votes, t, vote, window_s, *, live_timing=False):
    """Append one observation and return support-span and positive fraction.

    At dense cadence this is the frozen window. When fewer than two observations
    fit, live mode retains the preceding one, subject to the same continuity
    limit as the live pipeline. A single sample can never confirm completion.
    """
    if live_timing and votes and (t <= votes[-1][0] or t - votes[-1][0] > MAX_OBSERVATION_GAP_S):
        votes.clear()
    votes.append((t, bool(vote)))
    minimum = 2 if live_timing else 0
    while len(votes) > minimum and votes[0][0] < t - window_s:
        votes.popleft()
    enough = bool(votes) and t - votes[0][0] >= window_s * 0.65
    if live_timing:
        enough = enough and len(votes) >= 2
    fraction = sum(v for _, v in votes) / max(len(votes), 1)
    return enough, fraction


def continuous_live_history(history):
    """Return only the present track's contiguous, observed causal suffix."""
    if not history:
        return []
    current = history[-1]
    now = current["t"]
    if not math.isfinite(now) or now < 0 or not current["instances"]:
        return []
    recent = [current]
    for row in reversed(history[:-1]):
        t = row["t"]
        gap = recent[-1]["t"] - t
        if (
            not math.isfinite(t)
            or t < 0
            or not row["instances"]
            or row.get("target_epoch") != current.get("target_epoch")
            or not 0 < gap <= MAX_OBSERVATION_GAP_S
        ):
            break
        if now - t > 0.65:
            break
        recent.append(row)
    return list(reversed(recent))
