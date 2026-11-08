"""Experimental causal fall-process binary head with reviewed-label gates.

Scores describe the observed trailing window at its current endpoint. They are
not future fall risks or independent evidence from the shared pose detector.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .stgcn_home import predict_linear, train_linear

CLASSES = ('non_fall_activity', 'fall_process')
QUALITY = {'observable', 'occluded', 'unobservable', 'uncertain'}
IDENTITY_FIELDS = ('subject_id', 'camera_setup_id', 'session_id')


def guard_destination(path, runs_root):
    path, root = Path(path).absolute(), Path(runs_root).absolute()
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output must stay inside project runs')
    if '..' in path.parts or '_shared-assets' in path.parts:
        raise ValueError('Shared assets and traversal are protected')
    for component in [path, *path.parents]:
        if component == root.parent:
            break
        if component.is_symlink():
            raise ValueError('Never write through existing historical symlinks')


def _confirmed(item, clips):
    if item.get('review_status') != 'human_confirmed':
        return False
    clip = clips.get(item.get('clip_id'))
    if clip is None or item.get('video_sha256') != clip['video']['sha256']:
        raise ValueError('Human review must bind to the original video SHA-256')
    if not isinstance(item.get('reviewed_by'), str) or not item['reviewed_by'].strip() \
            or not isinstance(item.get('reviewed_at'), str) or not item['reviewed_at'].strip():
        raise ValueError('Human reviewer identity and review time are required')
    return True


def _number(value, name, maximum, allow_none=False):
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f'Invalid {name}')
    return float(value)


def reviewed_regions(reviews, manifest):
    """Explicit endpoint intervals; never expand an event to its recovery end."""
    clips = {c['id']: c for c in manifest['clips']}
    regions, accepted_events, seen = [], [], set()
    for event in reviews.get('events', []):
        if not _confirmed(event, clips):
            continue
        duration = clips[event['clip_id']]['duration_s']
        kind = event.get('event_kind')
        if kind not in {'fall', 'intentional_lying', 'uncertain'}:
            raise ValueError('Event kind must distinguish fall, intentional_lying and uncertain')
        key = event['clip_id'], event.get('event_id')
        if not key[1] or key in seen:
            raise ValueError('An event needs a unique event_id within its original video')
        seen.add(key)
        quality = event.get('quality') or {}
        if any(quality.get(phase) not in QUALITY for phase in ['process', 'lying', 'recovery']):
            raise ValueError('Event process/lying/recovery quality must be separate')
        onset = _number(event.get('onset_s'), 'onset_s', duration)
        landing = _number(event.get('landing_s'), 'landing_s', duration)
        recovery = _number(event.get('recovery_start_s'), 'recovery_start_s', duration, True)
        end = _number(event.get('recovery_end_s'), 'recovery_end_s', duration, True)
        if not onset < landing or (recovery is not None and recovery < landing) \
                or (end is not None and (recovery is None or end < recovery)):
            raise ValueError('Event phase times must be ordered; censored recovery may be null')
        uncertainty = event.get('uncertainty_s') or {}
        for boundary in ['onset', 'landing', 'recovery_start', 'recovery_end']:
            _number(uncertainty.get(boundary), boundary + ' uncertainty', duration)
        accepted_events.append(dict(event))
        # An intentional descent is not automatically an observable negative.
        # It requires its own explicit negative interval review below.
        if kind != 'fall' or quality['process'] != 'observable':
            continue
        start, stop = onset + uncertainty['onset'], landing - uncertainty['landing']
        if start < stop:
            regions.append(dict(clip_id=event['clip_id'], start_s=start, end_s=stop,
                                label='fall_process', event_id=event['event_id'],
                                source='human_event_process', demonstration=bool(event.get('demonstration', True))))
    for negative in reviews.get('non_fall_intervals', []):
        if not _confirmed(negative, clips):
            continue
        if negative.get('label') != 'non_fall_activity':
            raise ValueError('Negative intervals must explicitly be non_fall_activity')
        quality = negative.get('quality')
        if quality not in QUALITY:
            raise ValueError('Negative observation quality must be separate from class')
        duration = clips[negative['clip_id']]['duration_s']
        start = _number(negative.get('start_s'), 'start_s', duration)
        stop = _number(negative.get('end_s'), 'end_s', duration)
        uncertainty = _number(negative.get('boundary_uncertainty_s'), 'boundary uncertainty', duration)
        if start >= stop or not negative.get('activity_note'):
            raise ValueError('Negative intervals require ordered boundaries and an activity note')
        if quality == 'observable' and start + uncertainty < stop - uncertainty:
            regions.append(dict(clip_id=negative['clip_id'], start_s=start + uncertainty,
                                end_s=stop - uncertainty, label='non_fall_activity', event_id=None,
                                source='human_explicit_negative', activity_note=negative['activity_note']))
    # Overlap is ambiguous even if both labels happen to agree: prevent duplicate
    # event attribution or the same endpoint counting twice.
    for clip in clips:
        ordered = sorted((r for r in regions if r['clip_id'] == clip), key=lambda r: r['start_s'])
        for left, right in zip(ordered, ordered[1:]):
            if left['end_s'] > right['start_s']:
                raise ValueError('Reviewed training regions overlap')
    return regions, accepted_events


def endpoint_region(regions, record):
    if record['window_end_s'] != record['source_time_s']:
        raise ValueError('Feature endpoint is not the current source time')
    if not record['window'].get('causal') or record['window'].get('temporal_wrap'):
        raise ValueError('Only causal trailing features are permitted')
    t = record['video_time_s']
    values = [r for r in regions if r['clip_id'] == record['clip_id'] and r['start_s'] <= t < r['end_s']]
    if len(values) > 1:
        raise ValueError('Ambiguous endpoint label')
    return values[0] if values else None


def reviewed_identities(reviews, manifest):
    clips = {c['id']: c for c in manifest['clips']}
    identities = {}
    for identity in reviews.get('clip_identities', []):
        if not _confirmed(identity, clips):
            continue
        if identity['clip_id'] in identities:
            raise ValueError('Duplicate identity review for an original video')
        for key in IDENTITY_FIELDS:
            value = identity.get(key)
            if not isinstance(value, str) or not value.strip() \
                    or value.lower().strip() in {'unknown', 'uncertain', 'none', 'null'}:
                raise ValueError('Unknown subject/camera/session cannot establish independent groups')
        identities[identity['clip_id']] = {key: identity[key] for key in IDENTITY_FIELDS}
    return identities


def leakage_groups(records, strict=False):
    """Development keeps sessions/content whole; strict adds person and view."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, record in enumerate(records):
        fields = (*IDENTITY_FIELDS, 'video_sha256', 'clip_id') if strict else ('group', 'video_sha256', 'clip_id')
        for key in fields:
            value = record.get(key)
            if value is None or str(value).strip().lower() in {'', 'unknown', 'uncertain', 'none', 'null'}:
                raise ValueError('Unknown grouping identity; cannot claim person/view/session separation')
            pair = key, str(value)
            if pair in first:
                parent[find(i)] = find(first[pair])
            else:
                first[pair] = i
    roots = {root: number for number, root in enumerate(sorted({find(i) for i in range(len(records))}))}
    return np.array([roots[find(i)] for i in range(len(records))], dtype=int)


def grouped_folds(records, min_groups=3, min_samples_per_class=6, strict=False):
    if not records:
        raise ValueError('No human-reviewed observable fall-process/non-fall endpoint samples')
    if {r['label'] for r in records} != set(CLASSES):
        raise ValueError('Both reviewed classes are required')
    groups = leakage_groups(records, strict=strict)
    for label in CLASSES:
        selected = [i for i, r in enumerate(records) if r['label'] == label]
        if len(selected) < min_samples_per_class or len(set(groups[selected])) < min_groups:
            raise ValueError(f'Insufficient independent grouping components or samples for {label}')
    return [(int(group), np.flatnonzero(groups != group), np.flatnonzero(groups == group))
            for group in sorted(set(groups))]


def binary_metrics(targets, predictions):
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (targets, predictions), 1)
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp, truth, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        per_class[label] = dict(support=truth, precision=tp / predicted if predicted else None,
                                recall=tp / truth if truth else None,
                                f1=2 * tp / (truth + predicted) if truth + predicted else None)
    return {'classes': CLASSES, 'confusion_matrix': matrix.tolist(), 'per_class': per_class}


def train_fall_head(features, targets, seed=17):
    if set(np.asarray(targets).tolist()) != {0, 1}:
        raise ValueError('Each training fold needs both binary classes')
    return train_linear(features, targets, seed=seed, epochs=200)


def predict_fall_head(parameters, features):
    scores = predict_linear(parameters, features)
    if scores.shape[1] != 2:
        raise ValueError('Expected a two-class fall-process head')
    return scores
