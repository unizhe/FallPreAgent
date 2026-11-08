"""Reviewed-label gates and frozen ST-GCN++ home-action features.

This experimental adapter has no authority over the live alarm or action. It
uses the production trailing window and preserves separate observation quality.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .stgcn_temporal import prepare_tensor

HOME_CLASSES = (
    'standing', 'seated', 'sitting_down', 'standing_up', 'squatting_down',
    'squatting', 'squat_rising', 'bending', 'straightening', 'lying',
    'floor_recovery', 'other_observed',
)
SUBACTION_MAP = {
    'squat_descent': 'squatting_down', 'squat_hold': 'squatting',
    'squat_rise': 'squat_rising', 'floor_lying': 'lying', 'lying': 'lying',
    'floor_recovery': 'floor_recovery',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def proposed_label(segment):
    """Semantic suggestions only; never promote AI labels to reviewed truth."""
    label = segment.get('label')
    return label if label in HOME_CLASSES else SUBACTION_MAP.get(segment.get('subaction'))


def validate_reviews(reviews, manifest):
    """Require original-video identity, human provenance and explicit quality."""
    clips = {c['id']: c for c in manifest['clips']}
    accepted = []
    for item in reviews.get('segments', []):
        if item.get('review_status') != 'human_confirmed':
            continue
        clip = clips.get(item.get('clip_id'))
        if clip is None or item.get('video_sha256') != clip['video']['sha256']:
            raise ValueError('Reviewed label must bind to the original video SHA-256')
        if not item.get('reviewed_by') or not item.get('reviewed_at'):
            raise ValueError('Human confirmation requires reviewed_by and reviewed_at')
        start, end = item.get('start_s'), item.get('end_s')
        uncertainty = item.get('boundary_uncertainty_s')
        if not all(isinstance(x, (int, float)) and math.isfinite(x)
                   for x in (start, end, uncertainty)):
            raise ValueError('Review times and boundary uncertainty must be finite')
        if not 0 <= start < end <= clip['duration_s'] + 1e-6 or uncertainty < 0:
            raise ValueError('Reviewed interval is outside the original video')
        if item.get('observation_quality') not in {'observable', 'occluded', 'unobservable', 'uncertain'}:
            raise ValueError('Observation quality must be separate from the action label')
        if item['observation_quality'] != 'observable':
            continue
        if item.get('label') not in HOME_CLASSES:
            raise ValueError('Unsupported home action; unknown is not a negative class')
        accepted.append(dict(item))
    for clip_id in clips:
        rows = sorted((s for s in accepted if s['clip_id'] == clip_id), key=lambda s: s['start_s'])
        for a, b in zip(rows, rows[1:]):
            if a['end_s'] > b['start_s'] + 1e-9:
                raise ValueError('Reviewed single-layer action intervals must not overlap')
    return accepted


def endpoint_label(segments, clip_id, video_time):
    """Label the current endpoint, not the window majority or future ending."""
    matches = [s for s in segments if s['clip_id'] == clip_id
               and s['start_s'] + s['boundary_uncertainty_s'] <= video_time
               < s['end_s'] - s['boundary_uncertainty_s']]
    if len(matches) > 1:
        raise ValueError('Ambiguous endpoint labels')
    return matches[0]['label'] if matches else None


def grouped_folds(records, min_groups=3, min_class_samples=10):
    """Never split adjacent windows, video exports, or a capture-day group."""
    if not records:
        raise ValueError('No human-reviewed, observable endpoint samples')
    by_origin = {}
    for r in records:
        for key in (('video', r['video_sha256']), ('clip', r['clip_id'])):
            if key in by_origin and by_origin[key] != r['group']:
                raise ValueError('Same original video/clip crosses capture groups')
            by_origin[key] = r['group']
    classes = sorted({r['label'] for r in records})
    if len(classes) < 2:
        raise ValueError('At least two reviewed classes are required')
    for label in classes:
        values = [r for r in records if r['label'] == label]
        if len(values) < min_class_samples or len({r['group'] for r in values}) < min_groups:
            raise ValueError(f'Insufficient reviewed samples/groups for {label}')
    groups = sorted({r['group'] for r in records})
    return [(g, np.array([i for i, r in enumerate(records) if r['group'] != g]),
             np.array([i for i, r in enumerate(records) if r['group'] == g])) for g in groups]


class FrozenFeatures:
    """Exact pre-FC pooling of the verified NTU60 backbone, without updates."""

    def __init__(self, runtime):
        self.runtime = runtime
        runtime.model.eval()
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)

    def extract(self, window):
        tensor, metadata = prepare_tensor(window)
        with self.runtime.lock, self.runtime.torch.inference_mode():
            backbone, _ = self.runtime.model.extract_feat(
                self.runtime.torch.from_numpy(tensor), stage='backbone')
            # GCNHead averages spatial/time and person axes before its FC layer.
            feature = backbone.mean(dim=(1, 3, 4))[0].cpu().numpy().copy()
        if feature.ndim != 1 or not np.isfinite(feature).all():
            raise ValueError('Invalid frozen ST-GCN++ features')
        return feature, metadata


def train_linear(features, targets, seed=17, epochs=200):
    """Only a new small linear head is trained; no backbone optimizer exists."""
    import torch

    torch.manual_seed(seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    inputs, labels = torch.from_numpy((x - mean) / scale), torch.from_numpy(y)
    head = torch.nn.Linear(x.shape[1], int(y.max()) + 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.01)
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    if (counts == 0).any():
        raise ValueError('Training fold lacks a class')
    weights = torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(inputs), labels, weight=weights)
        loss.backward()
        optimizer.step()
    return {'mean': mean, 'scale': scale,
            'weight': head.weight.detach().numpy(), 'bias': head.bias.detach().numpy()}


def predict_linear(parameters, features):
    x = (features - parameters['mean']) / parameters['scale']
    logits = x @ parameters['weight'].T + parameters['bias']
    logits -= logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    return scores / scores.sum(axis=1, keepdims=True)
