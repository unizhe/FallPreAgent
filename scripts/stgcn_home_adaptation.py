"""Audit home labels, cache causal frozen features, and train a reviewed head.

Never writes historical paths, links, source video, or pretrained weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.stgcn_home import (  # noqa: E402
    HOME_CLASSES,
    FrozenFeatures,
    endpoint_label,
    grouped_folds,
    predict_linear,
    proposed_label,
    sha256,
    train_linear,
    validate_reviews,
)
from fallpre.weekly.stgcn_temporal import STGCNRuntime, TemporalConfig, TemporalPoseBuffer  # noqa: E402

SOURCE = ROOT / 'runs/stgcn-integration-20261029'
DEFAULT_OUTPUT = ROOT / 'runs/stgcn-fusion-20261106/data'


def ensure_artifact_destination(path, runs_root=ROOT / 'runs'):
    """Refuse shared assets, traversal and writes through any historical link."""
    path, runs_root = Path(path).absolute(), Path(runs_root).absolute()
    if not path.is_relative_to(runs_root) or not path.resolve().is_relative_to(runs_root.resolve()):
        raise ValueError('Artifact destination must stay inside project runs')
    if '..' in path.parts or '_shared-assets' in path.parts:
        raise ValueError('Shared assets and path traversal are protected')
    for component in [path, *path.parents]:
        if component == runs_root.parent:
            break
        if component.is_symlink():
            raise ValueError('Never write through an existing historical symlink')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def artifact(path):
    path = Path(path)
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'is_symlink': path.is_symlink(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def audit(output):
    ensure_artifact_destination(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Audit needs a new empty directory; historical outputs are read-only')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Audit is frozen; use a new output directory')
    clips, labels, queue = [], [], []
    for c in read(SOURCE / 'clips.json'):
        annotations = read(SOURCE / c['id'] / 'annotations.json')
        origin = read(annotations['source']) if annotations.get('source') else {}
        diagnostic = read(c['diagnostic'])
        started = diagnostic.get('started_at')
        day = datetime.fromisoformat(started.replace('Z', '+00:00')).astimezone(
            ZoneInfo('Asia/Shanghai')).date().isoformat()
        item = {k: c[k] for k in ('id', 'title', 'duration_s', 'observations')}
        item.update(video=artifact(c['video']), diagnostic=artifact(c['diagnostic']),
                    perception=artifact(c['perception']), baseline=artifact(c['baseline_output']),
                    annotations=artifact(SOURCE / c['id'] / 'annotations.json'),
                    original_annotations=artifact(annotations['source']) if annotations.get('source') else None,
                    capture_started_at=started, group='capture-day-' + day,
                    subject_id=None, camera_id=None, split='development_only',
                    grouping_note='All recordings on a local capture day stay together; subject identity unverified')
        clips.append(item)
        uncertainty = origin.get('boundary_uncertainty_s', .2)
        intervals = []
        for s in annotations['segments']:
            seg = dict(s, boundary_uncertainty_s=s.get('boundary_uncertainty_s', uncertainty),
                       review_status='ai_prelabel_pending_human_review', proposed_home_label=proposed_label(s))
            intervals.append(seg)
            queue.append({'clip_id': c['id'], 'segment_id': s.get('id'),
                          'start_s': s['start_s'], 'end_s': s['end_s'],
                          'proposed_label': seg['proposed_home_label'],
                          'original_label': s['label'], 'subaction': s.get('subaction'),
                          'priority': 1 if 'squat' in str(s) or s['label'] in {'sitting_down', 'standing_up'} else 2,
                          'boundary_uncertainty_s': seg['boundary_uncertainty_s'],
                          'review_status': seg['review_status'], 'video_path': c['video'],
                          'video_sha256': item['video']['sha256'],
                          'question': '核对动作阶段、开始/结束与可见性；模糊区间保留 uncertain，不照抄模型预测'})
        points = origin.get('clips', {}).get(c['id'][-2:], {}).get('point_checks', [])
        labels.append({'id': c['id'], 'source': annotations.get('source'),
                       'source_sha256': item['original_annotations']['sha256'] if item['original_annotations'] else None,
                       'status': 'ai_provisional', 'boundary_uncertainty_s': uncertainty,
                       'segments': intervals, 'point_checks': points, 'floor_events': annotations['floor_events']})
        if not intervals:
            queue.append({'clip_id': c['id'], 'start_s': 0, 'end_s': c['duration_s'],
                          'priority': 1, 'review_status': 'needs_phase_intervals',
                          'video_path': c['video'], 'video_sha256': item['video']['sha256'],
                          'question': '只有抽样点/倒地事件窗；需要完整阶段边界，不能扩展点标签为整段真值'})
    counts = Counter(s['label'] for c in labels for s in c['segments'])
    manifest = {'schema_version': 1, 'created_date': '2026-11-06', 'timezone': 'Asia/Shanghai',
                'clips': clips, 'human_confirmed_segments': 0, 'provisional_segment_counts': dict(counts),
                'human_confirmation_rule': 'Only explicit reviewer provenance bound to video SHA-256; no such records found',
                'split_policy': 'Leave-one-capture-day-out development CV. Same source hash and all exports stay together.',
                'independent_test_clips': [], 'independent_test_status': 'Not received; ten old clips already used in development'}
    priority = [
        ('clip04', 56.5, 60.01, '第四次下蹲→蹲姿→蹲起；ST 57.91–58.83 秒误作跌倒候选'),
        ('new', 40.85, 46.9, '正面深蹲；ST 42.39–43.40 秒误作跌倒候选'),
        ('new', 47.6, 50.85, '侧面深蹲；ST 48.77–49.77 秒误作跌倒候选'),
        ('natural', 36.1, 38.5, 'squat 宽标签混含下降/保持/起身，必须细分'),
        ('clip02', 26.9, 27.8, '从床边起身；27.63 秒主系统未知，ST起身分数仅0.19'),
        ('clip01', 31.3, 34.8, '坐下→坐稳→起身；需核对32.14和34.73秒阶段边界'),
        ('clip04', 4.6, 7.4, '弯腰与恢复直立的结束边界'),
        ('old', 31.4, 42.6, '宽泛 floor 标签需细分；受控低位活动不等同自然跌倒'),
        ('fall', 39.7, 44., '低位到侧卧与恢复；区分事件和持续倒卧'),
    ]
    first = [dict(clip_id=c, start_s=s, end_s=e, reason=why, review_status='pending_human_review')
             for c, s, e, why in priority]
    write(output / 'manifest.json', manifest)
    write(output / 'labels.provisional.json', {'schema_version': 1, 'clips': labels})
    write(output / 'review-queue.json', {'first_batch': first, 'all_intervals': queue})
    write(output / 'reviews.template.json', {'schema_version': 1, 'segments': [],
        'instructions': 'Add only manually confirmed intervals. Required fields: clip_id, video_sha256, '
                        'start_s, end_s, label, boundary_uncertainty_s, observation_quality, '
                        'review_status=human_confirmed, reviewed_by, reviewed_at.',
        'classes': HOME_CLASSES, 'observation_quality': ['observable', 'occluded', 'unobservable', 'uncertain']})
    write(output / 'training-gate.json', {'status': 'blocked_pending_review', 'weights_trained': False,
        'human_confirmed_segments': 0, 'reason': 'All ten clips have AI-only intervals, sparse points or event windows',
        'minimum': 'Two classes, >=10 observable endpoints and >=3 capture-day groups per included class',
        'scope': 'C unavailable; D deferred. No training on primary predictions or provisional labels.'})
    print(json.dumps({'clips': len(clips), 'segments': sum(counts.values()), 'counts': counts,
                      'human_confirmed_segments': 0}, ensure_ascii=False))


def extract(output):
    ensure_artifact_destination(output)
    manifest = read(output / 'manifest.json')
    target = output / 'features'
    target.mkdir(exist_ok=False)
    runtime = STGCNRuntime()
    extractor = FrozenFeatures(runtime)
    config = TemporalConfig()
    records, features, quality = [], [], Counter()
    for c in manifest['clips']:
        if sha256(c['baseline']['path']) != c['baseline']['sha256']:
            raise ValueError('Baseline changed since audit')
        buffer = TemporalPoseBuffer(config)
        for line in Path(c['baseline']['path']).read_text().splitlines():
            row = json.loads(line)
            result, t = row['result'], row['source_time_s']
            action = result['action']
            selection = action.get('selection') or {}
            index = action.get('target_index')
            persons = result.get('persons') or []
            person = persons[index] if isinstance(index, int) and 0 <= index < len(persons) else None
            status = buffer.observe(person, t, result['width'], result['height'],
                                    selection.get('target_epoch'), result.get('quality', {}).get('camera_stable') is True)
            quality[status['status']] += 1
            window = buffer.prepare()
            if window is None:
                continue
            feature, metadata = extractor.extract(window)
            features.append(feature)
            records.append({'clip_id': c['id'], 'group': c['group'], 'index': row['index'],
                            'video_sha256': c['video']['sha256'], 'video_time_s': row['video_time_s'],
                            'source_time_s': t, 'window_start_s': float(window.source_times[0]),
                            'window_end_s': float(window.source_times[-1]),
                            'generation': window.generation, 'target_epoch': window.target_epoch,
                            'window': metadata, 'feature_row': len(records)})
        print(json.dumps({'clip': c['id'], 'total_frozen_features': len(records)}), flush=True)
    np.savez_compressed(target / 'frozen-features.npz', features=np.stack(features))
    write(target / 'records.json', records)
    write(target / 'provenance.json', {'model': runtime.metadata, 'config': vars(config),
        'weights_updated': False, 'backbone_requires_grad': False, 'feature_dim': len(features[0]),
        'pooling': 'GCNHead pre-FC mean over person, time and joint axes',
        'observations': sum(quality.values()), 'quality_counts': dict(quality),
        'features': len(records), 'feature_sha256': sha256(target / 'frozen-features.npz'),
        'records_sha256': sha256(target / 'records.json'), 'manifest_sha256': sha256(output / 'manifest.json'),
        'code_sha256': sha256(ROOT / 'src/fallpre/weekly/stgcn_home.py'),
        'adapter_sha256': sha256(ROOT / 'src/fallpre/weekly/stgcn_temporal.py'),
        'scope': 'Causal same production 2.5s window; frozen features only; not a trained candidate'})


def train(output, reviews_path, destination):
    ensure_artifact_destination(destination)
    manifest = read(output / 'manifest.json')
    reviewed = validate_reviews(read(reviews_path), manifest)
    provenance = read(output / 'features/provenance.json')
    for relative, key in [('manifest.json', 'manifest_sha256'),
                          ('features/records.json', 'records_sha256'),
                          ('features/frozen-features.npz', 'feature_sha256')]:
        if sha256(output / relative) != provenance[key]:
            raise ValueError(f'Frozen feature provenance mismatch: {relative}')
    records = read(output / 'features/records.json')
    selected = [dict(r, label=label) for r in records
                if (label := endpoint_label(reviewed, r['clip_id'], r['video_time_s'])) is not None]
    folds = grouped_folds(selected)
    destination.mkdir(parents=True, exist_ok=False)
    features = np.load(output / 'features/frozen-features.npz')['features']
    x = features[[r['feature_row'] for r in selected]]
    classes = sorted({r['label'] for r in selected})
    y = np.array([classes.index(r['label']) for r in selected])
    seeds = [17, 29, 43]
    evaluations = []
    for group, training, validation in folds:
        for seed in seeds:
            parameters = train_linear(x[training], y[training], seed)
            prediction = predict_linear(parameters, x[validation]).argmax(1)
            confusion = np.zeros((len(classes), len(classes)), dtype=int)
            np.add.at(confusion, (y[validation], prediction), 1)
            per_class = {}
            for i, label in enumerate(classes):
                tp = int(confusion[i, i])
                true_count, predicted_count = int(confusion[i].sum()), int(confusion[:, i].sum())
                per_class[label] = {'support': true_count,
                                    'precision': tp / predicted_count if predicted_count else 0.,
                                    'recall': tp / true_count if true_count else None,
                                    'f1': 2 * tp / (true_count + predicted_count)
                                    if true_count + predicted_count else None}
            evaluations.append({'held_out_group': group, 'seed': seed, 'classes': classes,
                                'confusion_matrix': confusion.tolist(), 'samples': len(validation),
                                'accuracy': float((prediction == y[validation]).mean()),
                                'per_class': per_class,
                                'predictions': [dict(selected[int(j)], predicted_label=classes[int(prediction[k])])
                                                for k, j in enumerate(validation)]})
    # Predeclared seed; no best-seed or best-epoch selection using validation.
    parameters = train_linear(x, y, seeds[0])
    np.savez_compressed(destination / 'home-linear-head.npz', **parameters, classes=np.array(classes))
    write(destination / 'report.json', {'scope': 'Development capture-day CV only', 'classes': classes,
        'evaluations': evaluations, 'seeds': seeds, 'epochs': 200, 'learning_rate': .01,
        'weight_decay': .01, 'selection': 'fixed seed 17; fixed 200 epochs; no validation tuning',
        'samples': len(selected), 'backbone_trained': False, 'head_trained': True,
        'feature_provenance': read(output / 'features/provenance.json'),
        'reviews_sha256': sha256(reviews_path), 'weights_sha256': sha256(destination / 'home-linear-head.npz'),
        'primary_admission': False, 'reason': 'Needs same-fusion B/C replay and regression gates before live use'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'extract', 'train'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--reviews', type=Path)
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    if args.command == 'audit':
        audit(args.output)
    elif args.command == 'extract':
        extract(args.output)
    else:
        if not args.reviews or not args.destination:
            p.error('train requires --reviews and a new --destination')
        train(args.output, args.reviews, args.destination)


if __name__ == '__main__':
    main()
