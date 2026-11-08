"""Learned phase emissions with causal duration/seat-anchor state transitions.

All classification consumes current/past pose only. A phase is a visual proxy,
not force-plate seat-off. The immutable V3 baseline keeps protected decisions.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .action_v2 import PersonTracker
from .action_v3 import MotionForest, OnlineMotion, observations, signals
from .action_v3 import load_manifest as load_v3_manifest
from .human import ROOT

CACHE = ROOT / "runs/action-v4/poses"
PROTECTED = {"walking", "bending", "falling"}
PHASES = ("seated", "forward_lean", "rising", "standing", "descending", "other", "unknown")


def load_manifest(path):
    m = load_v3_manifest(path)
    for e in m["entries"]:
        if e.get("legacy_v3") and e["split"] == "test":
            raise ValueError("Historical V3 data is not a new test")
        end = 0.0
        for segment in e.get("phase_segments", []):
            lo, hi = segment["start_s"], segment["end_s"]
            if not (
                np.isfinite(lo)
                and np.isfinite(hi)
                and lo >= end
                and hi > lo
                and hi <= e["duration_s"]
                and segment["label"] in PHASES
            ):
                raise ValueError("Invalid phase interval")
            end = hi
    return m


def read_observations(entry, cache=CACHE):
    if (Path(cache) / entry["clip_id"] / "observations.jsonl").exists():
        return observations(entry, cache)
    return observations(entry)


def phase_features(history):
    good = [r for r in history if r["instances"]]
    if not good or not history[-1]["instances"] or len(good) < 3:
        return None, None
    now = good[-1]["t"]
    if now - good[0]["t"] < 0.16:
        return None, None
    vals = [signals(r["instances"][0]) for r in good]
    shapes = np.stack([v[0] for v in vals])
    roots = np.stack([v[1] for v in vals])
    scale = float(np.median([v[2] for v in vals]))
    tt = np.array([r["t"] for r in good])
    norm = (roots - roots[-1]) / scale
    a = np.column_stack((shapes, norm))
    x = []
    for span in [0.25, 0.75, 2.0, 4.0]:
        m = tt >= now - span
        v = a[m]
        t = tt[m]
        ct = t - t.mean()
        x.extend(np.r_[v[-1], v.mean(0), v.std(0), v[-1] - v[0], ct @ v / max(float(ct @ ct), 1e-6)])
    recent = tt >= now - 0.3
    dt = max(tt[-1] - tt[recent][0], 0.05)
    info = dict(
        t=now,
        hip_y=float(roots[-1, 1]),
        shoulder_y=float(roots[-1, 2]),
        scale=scale,
        knee=float(shapes[-1, 1]),
        hip_angle=float(shapes[-1, 2]),
        tilt=float(shapes[-1, 0]),
        reliable=bool(shapes[-1, 4] and shapes[-1, 5] and shapes[-1, 6]),
        hip_speed=float((roots[recent][0, 1] - roots[-1, 1]) / scale / dt),
    )
    return np.nan_to_num(np.clip(x, -30, 30)).astype(np.float32), info


class PhaseModel:
    def __init__(self, artifact):
        self.artifact = artifact
        self.base = MotionForest(artifact["frozen_v3"])
        self.classes = artifact["classes"]
        self.trees = artifact["trees"]
        self.config = artifact["phase_config"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def scores(self, x):
        out = np.zeros(len(self.classes))
        for tree in self.trees:
            i = 0
            while tree["left"][i] != -1:
                i = tree["left"][i] if x[tree["feature"][i]] <= tree["threshold"][i] else tree["right"][i]
            out += tree["value"][i]
        return dict(zip(self.classes, map(float, out / len(self.trees))))


class OnlinePhase:
    def __init__(self, model):
        self.model = model
        self.baseline = OnlineMotion(model.base)
        self.feature_provider = phase_features
        self.history = deque()
        self.epoch = None
        self.last_t = None
        self.last_seen = None
        self.clear_phase()

    def clear_phase(self):
        self.phase = "unknown"
        self.anchor = None
        self.seat_since = None
        self.candidate_since = None
        self.stable_since = None
        self.rise_since = None

    def reset(self):
        self.history.clear()
        self.clear_phase()
        self.last_seen = None

    def update(self, row):
        t = row["t"]
        if self.last_t is not None and t <= self.last_t:
            raise ValueError("Non-increasing source time")
        reset = row.get("target_epoch") != self.epoch or (self.last_t is not None and t - self.last_t > 0.35)
        if row["instances"] and self.last_seen is not None and t - self.last_seen > 0.35:
            reset = True
        if reset:
            self.reset()
        self.epoch = row.get("target_epoch")
        self.last_t = t
        if row["instances"]:
            self.last_seen = t
        self.history.append(row)
        while self.history and self.history[0]["t"] < t - 4.05:
            self.history.popleft()
        base = self.baseline.update(row)
        x, info = self.feature_provider(list(self.history))
        if x is None or not info["reliable"]:
            if not row["instances"] and self.last_seen is not None and t - self.last_seen > 0.35:
                self.clear_phase()
            return dict(
                base,
                phase="unknown",
                phase_scores={},
                phase_reason="missing_or_unreliable_phase_observation",
                v3_label=base["label"],
                scores_semantics="frozen_v3_action_scores; use phase_scores for V4 stage emissions",
                branch="v3_preserved_phase_unavailable",
            )
        scores = self.model.scores(x)
        top = max(scores, key=scores.get)
        knee = info["knee"]
        speed = info["hip_speed"]
        cfg = self.model.config
        pseat = scores.get("seated", 0)
        pstand = scores.get("standing", 0)
        # A sustained observed seated posture is necessary to arm a new rise.
        seated_emission = pseat >= cfg["seat_probability"]
        seated_geometry = (
            knee < 0.78
            and info["hip_angle"] < 0.82
            and pseat + scores.get("descending", 0) + scores.get("forward_lean", 0) > 0.6
            and abs(speed) < 0.10
        )
        seated = (seated_emission or seated_geometry) and knee < 0.83 and abs(speed) < 0.22
        if self.phase not in {"rising", "forward_lean"} and seated:
            if self.seat_since is None:
                self.seat_since = t
            if t - self.seat_since >= 0.16:
                self.phase = "seated"
                if self.anchor is None:
                    self.anchor = dict(info)
                elif (
                    abs(speed) < 0.04
                    and abs(self.anchor["hip_y"] - info["hip_y"]) / self.anchor["scale"] < 0.03
                ):
                    self.anchor["hip_y"] = 0.9 * self.anchor["hip_y"] + 0.1 * info["hip_y"]
                    self.anchor["t"] = t
        elif self.phase not in {"seated", "forward_lean", "rising"}:
            self.seat_since = None
            self.phase = "standing" if pstand >= 0.5 else top if top in {"other", "descending"} else "unknown"
        if self.anchor and t - self.anchor["t"] > 10 and self.phase != "rising":
            self.anchor = None
            self.phase = "unknown"
            self.seat_since = None
        up = (self.anchor["hip_y"] - info["hip_y"]) / self.anchor["scale"] if self.anchor else 0.0
        opening = knee - self.anchor["knee"] if self.anchor else 0.0
        if self.phase in {"seated", "forward_lean"} and self.anchor:
            if top == "forward_lean" or info["hip_angle"] < self.anchor["hip_angle"] - 0.045:
                self.phase = "forward_lean"
            moving = up >= cfg["min_up"] and opening >= cfg["min_opening"] and speed > 0.015
            if moving:
                if self.candidate_since is None:
                    self.candidate_since = t
                if t - self.candidate_since >= cfg["confirm_s"]:
                    self.phase = "rising"
                    self.rise_since = t
                    self.stable_since = None
            else:
                self.candidate_since = None
            if seated and abs(up) < 0.05:
                self.phase = "seated"
        if self.phase == "rising":
            stable = (pstand > 0.45 or knee > 0.93) and abs(speed) < cfg["stable_speed"]
            if stable:
                if self.stable_since is None:
                    self.stable_since = t
            else:
                self.stable_since = None
            if self.stable_since is not None and t - self.stable_since >= cfg["stable_s"]:
                self.phase = "standing"
                self.anchor = None
                self.seat_since = None
            elif speed < -0.15 or t - self.rise_since > 8:
                self.phase = "descending" if speed < -0.15 else "unknown"
                self.anchor = None
                self.seat_since = None
        if self.phase == "forward_lean" and speed < -0.3:
            self.phase = "descending"
            self.anchor = None
            self.seat_since = None
        result = dict(
            base,
            phase=self.phase,
            phase_scores=scores,
            raw_phase=top,
            v3_label=base["label"],
            scores_semantics="frozen_v3_action_scores; use phase_scores for V4 stage emissions",
            phase_up=float(up),
            phase_opening=float(opening),
            phase_reason=None,
            branch="v3_preserved",
        )
        # Retain every protected V3 decision, including its existing false positives.
        if base["label"] in PROTECTED:
            return result
        if self.phase == "rising":
            result.update(
                label="standing_up",
                v3_confidence=base.get("confidence"),
                confidence=None,
                branch="causal_phase_rise",
                reason=None,
                quality="available",
                model_inference_executed=True,
            )
        elif base["label"] == "standing_up" and self.phase in {"seated", "forward_lean", "standing"}:
            result.update(
                label="background", branch="phase_reject_unconfirmed_rise", reason="no_confirmed_seat_to_rise"
            )
        return result


def iter_predictions(entry, model, cache=CACHE):
    meta, rows = read_observations(entry, cache)
    tracker = PersonTracker(meta["fps"])
    stream = OnlinePhase(model)
    for row in rows:
        tracked = tracker.update(row)
        yield row, tracked, stream.update(tracked)
