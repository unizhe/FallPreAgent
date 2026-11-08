"""Fall-process binary-head audit and bounded offline development training.

Family reviews and official-frame ordinal proxy labels use separate entrypoints,
provenance, reports and weights. Neither route changes realtime decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.stgcn_fall_head import (  # noqa: E402
    CLASSES,
    binary_metrics,
    endpoint_region,
    grouped_folds,
    guard_destination,
    predict_fall_head,
    reviewed_regions,
    train_fall_head,
)
from fallpre.weekly.stgcn_home import sha256  # noqa: E402

OLD = ROOT / 'runs/stgcn-fusion-20261106/data'
DEFAULT = ROOT / 'runs/stgcn-fall-20261107/data'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def reference(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha256(path)}


def verify_reference(item):
    if sha256(item['path']) != item['sha256']:
        raise ValueError('Referenced input content changed: ' + item['path'])
    return Path(item['path'])


def audit(output):
    guard_destination(output, ROOT / 'runs')
    output.mkdir(parents=True, exist_ok=False)
    original = read(OLD / 'manifest.json')
    annotations = read(OLD / 'labels.provisional.json')
    features = read(OLD / 'features/provenance.json')
    for path, expected in [(OLD / 'manifest.json', features['manifest_sha256']),
                           (OLD / 'features/records.json', features['records_sha256']),
                           (OLD / 'features/frozen-features.npz', features['feature_sha256'])]:
        if sha256(path) != expected:
            raise ValueError('Previously frozen input changed')
    clips = [{key: c[key] for key in ['id', 'title', 'duration_s', 'video', 'group',
                                     'subject_id', 'camera_id', 'capture_started_at']}
             for c in original['clips']]
    events, negatives, sparse = [], [], []
    safe_activities = {'standing', 'seated', 'sitting_down', 'standing_up', 'walking_turning',
                       'bending', 'straightening', 'squatting_down', 'squatting', 'squat_rising'}
    for c in annotations['clips']:
        video_hash = next(x['video']['sha256'] for x in clips if x['id'] == c['id'])
        for event in c['floor_events']:
            events.append({'clip_id': c['id'], 'event_id': event['id'], 'video_sha256': video_hash,
                'scene': event.get('scene'), 'onset_s': event.get('motion_start_s', event['window'][0]),
                'landing_s': event.get('low_posture_by_s'), 'recovery_start_s': event.get('recovery_start_s'),
                'recovery_end_s': None, 'original_event_window_s': event['window'],
                'event_kind': 'uncertain', 'demonstration': True,
                'quality': dict(process='uncertain', lying='uncertain', recovery='uncertain'),
                'uncertainty_s': dict.fromkeys(['onset', 'landing', 'recovery_start', 'recovery_end'],
                                             event.get('boundary_uncertainty_s', .2)),
                'review_status': 'ai_provisional_pending_human', 'reviewed_by': None, 'reviewed_at': None,
                'note': event.get('annotation_note', '倒地演示；需确认过程/落地及主动躺下或模拟跌倒的区别'),
                'training_eligible': False})
        for segment in c['segments']:
            label = segment.get('proposed_home_label') or segment['label']
            if label not in safe_activities:
                continue
            negatives.append({'clip_id': c['id'], 'video_sha256': video_hash,
                'start_s': segment['start_s'], 'end_s': segment['end_s'],
                'label': 'non_fall_activity', 'activity_note': label,
                'boundary_uncertainty_s': segment['boundary_uncertainty_s'],
                'quality': 'uncertain', 'review_status': 'ai_provisional_pending_human',
                'reviewed_by': None, 'reviewed_at': None, 'training_eligible': False})
        sparse.extend({'clip_id': c['id'], **point, 'training_eligible': False,
                       'reason': 'Point check cannot be expanded to a negative interval'} for point in c['point_checks'])
    manifest = {'schema_version': 1, 'date': '2026-11-07', 'classes': CLASSES,
                'clips': clips, 'feature_archive': reference(OLD / 'features/frozen-features.npz'),
                'feature_records': reference(OLD / 'features/records.json'),
                'feature_provenance': reference(OLD / 'features/provenance.json'),
                'original_manifest': reference(OLD / 'manifest.json'),
                'original_labels': reference(OLD / 'labels.provisional.json'),
                'human_confirmed_family_events': 0, 'human_confirmed_negative_intervals': 0,
                'feature_shape': [features['features'], features['feature_dim']],
                'split': 'development_only; capture-session/day grouped; same person/view unknown',
                'independent_test': [], 'copied_large_files': False}
    write(output / 'manifest.json', manifest)
    write(output / 'review-candidates.json', {'events': events, 'non_fall_intervals': negatives,
                                             'sparse_points_excluded': sparse})
    write(output / 'reviews.template.json', {'schema_version': 1, 'events': [], 'non_fall_intervals': [],
        'clip_identities': [], 'instructions': 'Copy candidate entries to a NEW reviewed file after actual human review. '
        'Set review_status=human_confirmed with reviewed_by/reviewed_at. Correct phase bounds/uncertainty and '
        'event_kind (fall, intentional_lying, uncertain), keep process/lying/recovery quality separate. '
        'A full event window, lying or recovery never automatically supplies fall_process labels.'})
    write(output / 'training-gate.json', {'family_status': 'blocked_pending_human_review',
        'family_events': len(events), 'negative_interval_candidates': len(negatives),
        'confirmed_family_training_samples': 0, 'weights_updated': False,
        'public_proxy_status': 'Separate official-depth-frame to RGB-ordinal exploratory route; not family ground truth',
        'required_before_live': 'Same-core event replay, reviewed event metrics, natural monitoring and sealed new sessions'})
    print(json.dumps({'events': len(events), 'negative_candidates': len(negatives), 'human_confirmed': 0}))


def load_family(output, review_path):
    manifest = read(output / 'manifest.json')
    reviews = read(review_path)
    regions, events = reviewed_regions(reviews, manifest)
    records = read(verify_reference(manifest['feature_records']))
    x = np.load(verify_reference(manifest['feature_archive']))['features']
    selected = []
    for record in records:
        region = endpoint_region(regions, record)
        if region:
            selected.append(dict(record, label=region['label'], event_id=region['event_id']))
    grouped_folds(selected)  # An empty/one-class reviewed set fails before any weight directory is created.
    return x[[r['feature_row'] for r in selected]], selected, {
        'source': 'human_reviewed_family', 'reviews': reference(review_path), 'events': events,
        'feature_provenance': read(verify_reference(manifest['feature_provenance'])),
        'independent_generalization': False, 'split_scope': manifest['split']}


def fit(features, records, provenance, destination, variant='st_only'):
    import torch

    torch.set_num_threads(2)
    folds = grouped_folds(records)
    guard_destination(destination, ROOT / 'runs')
    destination.mkdir(parents=True, exist_ok=False)
    y = np.array([CLASSES.index(r['label']) for r in records], dtype=int)
    seeds, reports, oof = [17, 29, 43], [], []
    for seed in seeds:
        predictions = np.full(len(y), -1, dtype=int)
        scores = np.full((len(y), 2), np.nan, dtype=float)
        for group, train, val in folds:
            parameters = train_fall_head(features[train], y[train], seed)
            scores[val] = predict_fall_head(parameters, features[val])
            predictions[val] = scores[val].argmax(1)
            reports.append({'seed': seed, 'held_out_group': group, 'training_samples': len(train),
                            'validation_samples': len(val), **binary_metrics(y[val], predictions[val])})
        oof.append({'seed': seed, 'metrics': binary_metrics(y, predictions),
                    'predictions': [dict(record, predicted_label=CLASSES[int(predictions[i])],
                                         fall_process_score=float(scores[i, 1])) for i, record in enumerate(records)]})
    parameters = train_fall_head(features, y, seeds[0])
    np.savez_compressed(destination / 'fall-process-head.npz', **parameters, classes=np.array(CLASSES))
    write(destination / 'report.json', {'schema_version': 1, 'variant': variant, 'classes': CLASSES,
        'samples': len(records), 'class_counts': dict(Counter(r['label'] for r in records)),
        'positive_source_clips': sorted({r['clip_id'] for r in records if r['label'] == 'fall_process'}),
        'folds': len(folds), 'seeds': seeds, 'fold_metrics': reports,
        'pooled_oof_metrics': [{'seed': value['seed'], **value['metrics']} for value in oof],
        'pretrained_falling_score_ge_0_5': binary_metrics(
            y, np.array([int(r.get('pretrained_falling_score', 0.) >= .5) for r in records]))
            if all('pretrained_falling_score' in r for r in records) else None,
        'provenance': provenance, 'backbone_updated': False, 'head_updated': True,
        'hyperparameters': dict(epochs=200, lr=.01, weight_decay=.01, optimizer='AdamW',
                               class_weights='inverse training-fold frequency',
                               normalization='training fold only', seed_selection='fixed 17, no best-seed selection'),
        'weights': reference(destination / 'fall-process-head.npz'),
        'training_code': reference(__file__), 'head_code': reference(ROOT / 'src/fallpre/weekly/stgcn_fall_head.py'),
        'deployment_eligible': False,
        'limits': ['Window metrics are not event recall or false alarms per hour.',
                   'Fold-shared people/views are unknown for development CV; no independent migration claim.',
                   'No present or future fall probability calibration.',
                   'No primary action or alarm modification; require causal event-level fusion replay.']})
    write(destination / 'oof-predictions.json', oof)
    print(json.dumps({'output': str(destination), 'samples': len(records), 'folds': len(folds)}))


def official_proxy(source, destination, variant):
    provenance = read(source / 'provenance.json')
    if provenance.get('weak_label_alignment') is not True:
        raise ValueError('Public ordinal-label proxy reliability must be explicit')
    for name, expected in provenance['files_sha256'].items():
        if sha256(source / name) != expected:
            raise ValueError('Public frozen artifact hash mismatch: ' + name)
    records = read(source / 'records.json')
    archive = np.load(source / 'frozen-features.npz')
    # Never disguise ordinally transferred depth labels as human RGB annotation.
    provenance = dict(provenance, source='official_depth_state_to_rgb_ordinal_proxy',
                      label_reliability='weak_label_alignment', human_verified_rgb_alignment=False,
                      independent_generalization=False,
                      artifact_references={name: reference(source / name) for name in
                                           ['provenance.json', 'records.json', 'frozen-features.npz']})
    eligible = [r for r in records if r.get('label') in CLASSES
                and r.get('eligible_for_training', True)]
    for record in eligible:
        if record['window_end_s'] != record['source_time_s'] or not record['window'].get('causal'):
            raise ValueError('Noncausal public feature endpoint')
        if record['label'] == 'fall_process' and not (
                record['official_state'] == 0 and record['original_sequence'].startswith('fall-')):
            raise ValueError('Only official fall-sequence transition proxies can be positive')
        if record['label'] == 'non_fall_activity' and record['official_state'] != -1:
            raise ValueError('Sustained lying/ambiguous ADL cannot automatically be negative')
    if variant in {'st_geometry', 'st_common_geometry'}:
        eligible = [r for r in eligible if archive['geometry_available'][r['feature_row']]]
    indices = [r['feature_row'] for r in eligible]
    features = archive['features'][indices]
    if variant == 'st_geometry':
        features = np.concatenate([features, archive['geometry'][indices]], axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Missing geometry/quality cannot be converted into a negative class')
    provenance['excluded_records'] = len(records) - len(eligible)
    provenance['variant'] = variant
    fit(features, eligible, provenance, destination, variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'gate', 'train', 'train-official-proxy'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--variant', choices=['st_only', 'st_geometry', 'st_common_geometry'], default='st_only')
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args.output)
    elif args.command in {'gate', 'train'}:
        if args.reviews is None:
            parser.error('--reviews is required')
        x, rows, provenance = load_family(args.output, args.reviews)
        if args.command == 'train':
            if args.destination is None:
                parser.error('--destination must be a new directory')
            fit(x, rows, provenance, args.destination)
    else:
        if args.source is None or args.destination is None:
            parser.error('--source and a new --destination are required')
        official_proxy(args.source, args.destination, args.variant)


if __name__ == '__main__':
    main()
