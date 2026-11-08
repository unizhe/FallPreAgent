"""Motion-vetoed completion: short-window probability needs independent cessation evidence.

Reconsiders only V6's additional termination of V5 rises. No action can be
created outside the frozen V5 onset path. All features use the current/past track.
"""

import json
from collections import deque
from pathlib import Path

from .action_v5 import live_terminal_features, terminal_features
from .action_v6 import CompletionModel as V6Model
from .action_v6 import OnlineCompletion as V6Stream
from .completion_timing import completion_vote, validate_live_time


class CompletionModel:
    def __init__(self, artifact):
        self.artifact = artifact
        self.base = V6Model(artifact["frozen_v6"])
        self.config = artifact["completion_config"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


class OnlineCompletion:
    def __init__(self, model, *, live_timing=False):
        self.model = model
        self.live_timing = live_timing
        self.baseline = V6Stream(model.base, live_timing=live_timing)
        self.history = deque()
        self.epoch = None
        self.last_t = None
        self.rise_start = None
        self.completed = False
        self.votes = deque()
        self.feature_provider = live_terminal_features if live_timing else terminal_features

    def update(self, row):
        t = row["t"]
        validate_live_time(self, t)
        epoch = row.get("target_epoch")
        if (
            epoch != self.epoch
            or self.last_t is not None
            and (t <= self.last_t or t - self.last_t > 0.35)
            or not row["instances"]
        ):
            self.history.clear()
            self.rise_start = None
            self.completed = False
            self.votes.clear()
        self.epoch, self.last_t = epoch, t
        self.history.append(row)
        while self.history and self.history[0]["t"] < t - 0.7:
            self.history.popleft()
        old = self.baseline.update(row)
        result = dict(
            old,
            v6_label=old["label"],
            v6_phase=old["phase"],
            motion_completion_latched=False,
            completion_motion_veto=None,
            current_up_speed=None,
        )
        if old.get("v4_phase") != "rising" or not row["instances"]:
            self.rise_start = None
            self.completed = False
            self.votes.clear()
            return result
        if self.rise_start is None:
            self.rise_start = t
        # Restore V5 output until V7 independently confirms completion.
        if old["v5_label"] == "standing_up":
            result.update(
                label="standing_up",
                phase=old["v5_phase"],
                branch="v5_rise_pending_completion",
                reason="completion_not_confirmed",
            )
        x = self.feature_provider(list(self.history), old, self.rise_start)
        if x is None:
            if getattr(self, "live_timing", False):
                self.votes.clear()
                self.completed = False
            return result
        cfg = self.model.config
        speed = max(0.0, -float(x[10]), -float(x[11]))
        veto = speed > cfg["max_up_speed"]
        prob = old["fast_completion_probability"]
        vote = prob is not None and prob >= cfg["threshold"] and not veto and x[1] >= cfg["minimum_knee"]
        enough, fraction = completion_vote(
            self.votes, t, vote, cfg["window_s"], live_timing=getattr(self, "live_timing", False)
        )
        if t - self.rise_start >= 0.15 and enough and fraction >= 0.65:
            self.completed = True
        result.update(
            motion_completion_latched=self.completed, completion_motion_veto=veto, current_up_speed=speed
        )
        if self.completed and old["v5_label"] == "standing_up":
            result.update(
                label="background",
                phase="standing",
                branch="motion_confirmed_completion",
                reason="short_probability_and_motion_cessation",
                confidence=None,
            )
        return result


def load_manifest(path):
    from .action_v6 import load_manifest as previous

    m = previous(path)
    if any(e.get("legacy_v6") and e["split"] == "test" for e in m["entries"]):
        raise ValueError("V6 data cannot be independent V7 test")
    return m


def read_observations(entry, cache=None):
    from .action_v6 import read_observations as previous
    from .human import ROOT

    cache = Path(cache) if cache is not None else ROOT / "runs/action-v7/poses"
    if (cache / entry["clip_id"] / "observations.jsonl").exists():
        return previous(entry, cache)
    return previous(entry)


def iter_predictions(entry, model, cache=None):
    from .action_v2 import PersonTracker

    meta, rows = read_observations(entry, cache)
    tracker = PersonTracker(meta["fps"])
    online = OnlineCompletion(model)
    for row in rows:
        tracked = tracker.update(row)
        yield row, tracked, online.update(tracked)
