"""Short-window terminal evidence alongside frozen V5; only removes rise tails."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from .action_v5 import CompletionModel as V5Model
from .action_v5 import OnlineCompletion as V5Stream
from .action_v5 import live_terminal_features, terminal_features
from .completion_timing import completion_vote, validate_live_time


class CompletionModel:
    def __init__(self, artifact):
        self.artifact = artifact
        self.base = V5Model(artifact["frozen_v5"])
        self.config = artifact["completion_config"]
        self.indices = artifact["feature_indices"]
        self.forest = V5Model(dict(artifact, frozen_v4=artifact["frozen_v5"]["frozen_v4"]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def probability(self, x):
        return self.forest.probability(np.asarray(x)[self.indices])


class OnlineCompletion:
    def __init__(self, model, *, live_timing=False):
        self.model = model
        self.live_timing = live_timing
        self.baseline = V5Stream(model.base, live_timing=live_timing)
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
            v5_label=old["label"],
            v5_phase=old["phase"],
            fast_completion_probability=None,
            fast_completion_latched=False,
        )
        if old.get("v4_phase") != "rising" or not row["instances"]:
            self.rise_start = None
            self.completed = False
            self.votes.clear()
            return result
        if self.rise_start is None:
            self.rise_start = t
        x = self.feature_provider(list(self.history), old, self.rise_start)
        if x is None:
            if getattr(self, "live_timing", False):
                self.votes.clear()
                self.completed = False
            return result
        prob = self.model.probability(x)
        cfg = self.model.config
        enough, fraction = completion_vote(
            self.votes, t, prob >= cfg["threshold"], cfg["window_s"],
            live_timing=getattr(self, "live_timing", False),
        )
        if (
            t - self.rise_start >= cfg["minimum_rise_s"]
            and enough
            and fraction >= cfg["vote_fraction"]
        ):
            self.completed = True
        result.update(fast_completion_probability=prob, fast_completion_latched=self.completed)
        if self.completed and old["label"] == "standing_up":
            result.update(
                label="background",
                phase="standing",
                branch="short_window_completion",
                confidence=None,
                reason="causal_short_window_terminal_evidence",
            )
        return result


def load_manifest(path):
    from .action_v5 import load_manifest as previous

    m = previous(path)
    if any(e.get("legacy_v5") and e["split"] == "test" for e in m["entries"]):
        raise ValueError("V5 data cannot be independent V6 test")
    return m


def read_observations(entry, cache=None):
    from .action_v5 import read_observations as previous
    from .human import ROOT

    cache = Path(cache) if cache is not None else ROOT / "runs/action-v6/poses"
    if entry.get("input_mode") == "public_openpose_skeleton" and entry.get("pose_cache_path"):
        from .human import sha256

        p = ROOT / entry["pose_cache_path"]
        if sha256(p) != entry["pose_cache_sha256"]:
            raise ValueError("Skeleton checksum mismatch")
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        if len(rows) != entry["frame_count"]:
            raise ValueError("Incomplete skeleton source")
        for i, r in enumerate(rows):
            if r["source_frame_id"] != i + 1 or abs(r["t"] - i / entry["fps"]) > 1e-7:
                raise ValueError("Invalid source clock")
        return dict(fps=entry["fps"]), rows
    if (cache / entry["clip_id"] / "observations.jsonl").exists():
        return previous(entry, cache)
    return previous(entry)


def iter_predictions(entry, model, cache=None):
    from .action_v2 import PersonTracker

    meta, rows = read_observations(entry, cache)
    tracker = PersonTracker(meta["fps"])
    stream = OnlineCompletion(model)
    for row in rows:
        tr = tracker.update(row)
        yield row, tr, stream.update(tr)
