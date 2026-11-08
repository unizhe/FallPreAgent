"""Causal terminal-standing detector on the immutable V4 rise-onset path.

The completion model can only end an existing rise. It cannot create an action
or change walking/bending/falling. Completed state is latched until the V4
upstream phase leaves rising; missing observations/identity gaps clear evidence.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

from .action_v3 import signals
from .action_v4 import PROTECTED, OnlinePhase, PhaseModel
from .completion_timing import completion_vote, continuous_live_history, validate_live_time


def terminal_features(history, prediction, rise_start, *, live_timing=False):
    if not history or not history[-1]["instances"]:
        return None
    now = history[-1]["t"]
    recent = (
        continuous_live_history(history)
        if live_timing
        else [r for r in history if r["instances"] and now - r["t"] <= 0.65]
    )
    if len(recent) < 3 or now - recent[0]["t"] < 0.12:
        return None
    tt = np.array([r["t"] for r in recent])
    values = [signals(r["instances"][0]) for r in recent]
    shapes = np.stack([v[0] for v in values])
    roots = np.stack([v[1] for v in values])
    scale = max(float(np.median([v[2] for v in values])), 1)
    if live_timing and not (np.isfinite(shapes).all() and np.isfinite(roots).all() and np.isfinite(scale)):
        return None
    result = []
    for span in [0.12, 0.25, 0.5]:
        ix = tt >= now - span
        if live_timing and np.count_nonzero(ix) < 2:
            # Use the actual past pair, including its actual elapsed time. A
            # one-point regression otherwise invents zero current velocity.
            ix[-2:] = True
        a = shapes[ix, :3]
        pos = roots[ix, :3] / scale
        t = tt[ix]
        ct = t - t.mean()
        slope = ct @ pos / max(float(ct @ ct), 1e-6)
        result.extend(np.r_[a[-1], np.median(a, axis=0), a.std(0), slope, pos[-1] - pos[0]])
    result.extend(
        [prediction.get("phase_up", 0), prediction.get("phase_opening", 0), max(0, now - rise_start)]
    )
    result.extend(
        prediction.get("phase_scores", {}).get(k, 0)
        for k in ["seated", "forward_lean", "rising", "standing", "descending", "other"]
    )
    return np.nan_to_num(np.clip(result, -30, 30)).astype(np.float32)


def live_terminal_features(history, prediction, rise_start):
    return terminal_features(history, prediction, rise_start, live_timing=True)


class CompletionModel:
    def __init__(self, artifact):
        self.artifact = artifact
        self.base = PhaseModel(artifact["frozen_v4"])
        self.classes = artifact["classes"]
        self.trees = artifact["trees"]
        self.config = artifact["completion_config"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def probability(self, x):
        out = np.zeros(len(self.classes))
        for tree in self.trees:
            i = 0
            while tree["left"][i] != -1:
                i = tree["left"][i] if x[tree["feature"][i]] <= tree["threshold"][i] else tree["right"][i]
            out += tree["value"][i]
        return float(out[self.classes.index("standing")] / len(self.trees))


class OnlineCompletion:
    def __init__(self, model, *, live_timing=False):
        self.model = model
        self.live_timing = live_timing
        self.baseline = OnlinePhase(model.base)
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
            or self.last_t is not None and t - self.last_t > 0.35
            or self.live_timing and not row["instances"]
        ):
            self.history.clear()
            self.rise_start = None
            self.completed = False
            self.votes.clear()
        self.epoch = epoch
        self.last_t = t
        self.history.append(row)
        while self.history and self.history[0]["t"] < t - 0.7:
            self.history.popleft()
        old = self.baseline.update(row)
        result = dict(
            old,
            v4_label=old["label"],
            v4_phase=old["phase"],
            completion_probability=None,
            completion_latched=False,
        )
        if old["phase"] != "rising" or self.live_timing and not row["instances"]:
            self.rise_start = None
            self.completed = False
            self.votes.clear()
            return result
        if self.rise_start is None:
            self.rise_start = t
        x = self.feature_provider(list(self.history), old, self.rise_start)
        if x is None:
            if self.live_timing:
                self.votes.clear()
                self.completed = False
            return result
        prob = self.model.probability(x)
        cfg = self.model.config
        enough, fraction = completion_vote(
            self.votes, t, prob >= cfg["threshold"], cfg["window_s"], live_timing=self.live_timing
        )
        if t - self.rise_start >= cfg["minimum_rise_s"] and enough and fraction >= cfg["vote_fraction"]:
            self.completed = True
        result.update(completion_probability=prob, completion_latched=self.completed)
        if self.completed:
            result["phase"] = "standing"
            if old["label"] == "standing_up":
                result.update(
                    label="background",
                    branch="terminal_standing_confirmed",
                    confidence=None,
                    reason="causal_completion_evidence",
                )
        assert old["label"] not in PROTECTED or result["label"] == old["label"]
        return result


def load_manifest(path):
    from .action_v4 import load_manifest as load_v4

    m = load_v4(path)
    if any(e.get("legacy_v4") and e["split"] == "test" for e in m["entries"]):
        raise ValueError("V4 data cannot be new V5 test")
    fingerprints = {}
    for e in m["entries"]:
        fingerprint = e.get("source", {}).get("member_sha256")
        if fingerprint:
            previous = fingerprints.setdefault(fingerprint, e["split"])
            if previous != e["split"]:
                raise ValueError("Duplicate skeleton across splits")
    return m


def read_observations(entry, cache=None):
    from .action_v4 import read_observations as read_v4
    from .human import ROOT, sha256

    cache = Path(cache) if cache is not None else ROOT / "runs/action-v5/poses"
    if entry.get("input_mode") == "public_openpose_skeleton":
        p = ROOT / "runs/action-v5/poses" / entry["clip_id"] / "observations.jsonl"
        if sha256(p) != entry["pose_cache_sha256"]:
            raise ValueError("Skeleton checksum mismatch")
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        if len(rows) != entry["frame_count"]:
            raise ValueError("Incomplete skeleton source")
        for i, r in enumerate(rows):
            if r["source_frame_id"] != i + 1 or abs(r["t"] - i / entry["fps"]) > 1e-7:
                raise ValueError("Invalid original source clock")
        return dict(fps=entry["fps"], input_mode="public_openpose_skeleton"), rows
    if (cache / entry["clip_id"] / "observations.jsonl").exists():
        return read_v4(entry, cache)
    return read_v4(entry)


def iter_predictions(entry, model, cache=None):
    from .action_v2 import PersonTracker

    meta, rows = read_observations(entry, cache)
    tracker = PersonTracker(meta["fps"])
    stream = OnlineCompletion(model)
    for row in rows:
        tr = tracker.update(row)
        yield row, tr, stream.update(tr)
