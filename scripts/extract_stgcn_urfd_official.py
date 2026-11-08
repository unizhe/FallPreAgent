"""Extract causal ST features from existing URFD caches with official proxy labels.

No download, video copying, detector retraining or event-policy change. Official
cam0 depth-frame states are joined to RGB preview frame numbers; this migration
is explicit weak alignment, not human-confirmed RGB fall or accident truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.evaluate_human import load_official_states  # noqa: E402
from fallpre.weekly.stgcn_fall_evidence import GEOMETRY_FEATURE_NAMES, fall_geometry_features  # noqa: E402
from fallpre.weekly.stgcn_home import FrozenFeatures, sha256  # noqa: E402
from fallpre.weekly.stgcn_temporal import STGCNRuntime, TemporalConfig, TemporalPoseBuffer  # noqa: E402
from fallpre.weekly.webcam_target import StableSubject  # noqa: E402

DEFAULT_OUTPUT = ROOT / 'runs/stgcn-fall-20261107/official-dev'
DEFAULT_MANIFEST = ROOT / 'configs/action-v3-dataset.json'
DEFAULT_LABELS = ROOT / 'runs/weekly-20261015/labels'


def official_proxy_label(sequence, source_frame_id, states):
    """Only observed official states are mapped; ambiguous ADL sessions excluded."""
    state = states.get((sequence, source_frame_id))
    if sequence.startswith('adl-') and any(s == 0 for (clip, _), s in states.items() if clip == sequence):
        return None, state, 'adl_has_ambiguous_transition_entire_session_excluded'
    if state is None:
        return None, None, 'official_state_missing'
    if state == 0:
        return ('fall_process', state, 'official_simulated_fall_transition_proxy') if sequence.startswith('fall-') else (
            None, state, 'unresolved_transition_semantics')
    if state == 1:
        return None, state, 'sustained_lying_excluded_from_non_fall_activity'
    if state == -1:
        return 'non_fall_activity', state, 'official_nonlying_activity_proxy'
    raise ValueError('Unexpected official state')


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare_destination(path):
    path = path.absolute()
    runs = ROOT / 'runs'
    if (not path.is_relative_to(runs) or not path.resolve().is_relative_to(runs.resolve())
            or '..' in path.parts or '_shared-assets' in path.parts):
        raise ValueError('Output must be a new directory under project runs, outside shared assets')
    if any(component.is_symlink() for component in [path, *path.parents] if component.is_relative_to(runs)):
        raise ValueError('Output cannot traverse historical symlinks')
    path.mkdir(parents=True, exist_ok=False)
    return path


def extract(output=DEFAULT_OUTPUT, manifest_path=DEFAULT_MANIFEST, labels_dir=DEFAULT_LABELS):
    states, official_sources = load_official_states(labels_dir, download=False)
    entries = [e for e in json.loads(manifest_path.read_text())['entries'] if e['clip_id'].startswith('urfd-')]
    if len({e['clip_id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate source clips')
    output = prepare_destination(output)
    sources, records, features, geometry_vectors, geometry_available, scores = [], [], [], [], [], []
    stats, quality_counts = Counter(), Counter()
    runtime = STGCNRuntime()
    extractor = FrozenFeatures(runtime)
    config = TemporalConfig()
    with (output / 'coverage.jsonl').open('x') as coverage:
        for entry in entries:
            clip_id = entry['clip_id']
            sequence = clip_id.removeprefix('urfd-')
            video = ROOT / entry['video_path']
            if sha256(video) != entry['video_sha256']:
                raise ValueError(f'Video content differs from frozen manifest: {clip_id}')
            choices = [ROOT / 'runs' / version / 'poses' / clip_id / 'observations.jsonl'
                       for version in ('action-v3', 'action-v2')]
            cache = next((p for p in choices if p.is_file()), None)
            if cache is None:
                raise FileNotFoundError(f'No existing skeleton cache: {clip_id}')
            rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
            meta = rows.pop(0)
            if (meta.get('kind') != 'meta' or meta.get('source_sha256') != entry['video_sha256']
                    or len(rows) != entry['frame_count']):
                raise ValueError(f'Cache provenance/frame count mismatch: {clip_id}')
            cap = cv2.VideoCapture(str(video))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            decoded_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if decoded_count != len(rows) or abs(fps - entry['fps']) > 1e-6 or (width, height) != (320, 240):
                raise ValueError(f'Unexpected preview dimensions/clock/frame count: {clip_id}')
            group = entry.get('source_group_id', clip_id)
            source = dict(clip_id=clip_id, original_sequence=sequence, group=group,
                          video_path=str(video), video_sha256=entry['video_sha256'],
                          pose_cache_path=str(cache), pose_cache_sha256=sha256(cache),
                          detector_metadata=meta, image_size=[width, height], source_frames=len(rows),
                          fps=entry['fps'], prior_split=entry.get('split'),
                          prior_usage='previously trained/developed/evaluated; development_only',
                          official_state_0_frames=sum(s == 0 for (c, _), s in states.items() if c == sequence),
                          subject_id=None, camera_id='URFD-cam0', observed_counts={})
            sources.append(source)
            selector, buffer = StableSubject(), TemporalPoseBuffer(config)
            clip_counts = Counter()
            for expected_frame_id, row in enumerate(rows, 1):
                frame_id, t = row['source_frame_id'], row['t']
                if (frame_id != expected_frame_id or row['clip_id'] != clip_id
                        or abs(t - (frame_id - 1) / entry['fps']) > 1e-6):
                    raise ValueError(f'Source frame order/nominal clock mismatch: {clip_id}')
                people = [dict(box=p['box'], confidence=p['det'], keypoints=p['kpts']) for p in row['instances']]
                selected = selector.update(people, t, width, height)
                index = selected['target_index']
                person = people[index] if index is not None else None
                status = buffer.observe(person, t, width, height, selected['target_epoch'], camera_stable=True)
                window = buffer.prepare()
                label, official_state, label_reason = official_proxy_label(sequence, frame_id, states)
                eligible = window is not None and label is not None
                stats['source_frames'] += 1
                stats['official_fall_process_source_frames'] += int(sequence.startswith('fall-') and official_state == 0)
                quality_counts[status['status']] += 1
                clip_counts['source_frames'] += 1
                clip_counts['prepared_windows'] += int(window is not None)
                coverage.write(json.dumps(dict(
                    clip_id=clip_id, group=group, source_frame_id=frame_id, source_time_s=t,
                    official_state=official_state, proxy_label=label, label_reason=label_reason,
                    target_index=index, target_epoch=selected['target_epoch'], selection_state=selected['state'],
                    temporal_quality=status['status'], prepared=window is not None, eligible=eligible,
                    exclusion_reason=label_reason if label is None else status['status'] if window is None else None),
                    ensure_ascii=False, allow_nan=False) + '\n')
                if not eligible:
                    if window is not None:
                        stats['label_excluded_windows'] += 1
                    continue
                feature, metadata = extractor.extract(window)
                original = runtime.predict(window)
                geometry = fall_geometry_features(window.keypoints, window.source_times, width, height)
                available = geometry['vector'] is not None
                record = dict(
                    clip_id=clip_id, original_sequence=sequence, group=group,
                    source_frame_id=frame_id, source_time_s=t, video_time_s=t,
                    feature_row=len(records), label=label, official_state=official_state,
                    label_authority='official_depth_state_migrated_to_rgb_preview_proxy',
                    label_reason=label_reason, human_confirmed_rgb=False,
                    video_sha256=entry['video_sha256'], pose_cache_sha256=source['pose_cache_sha256'],
                    window_start_s=metadata['start_s'], window_end_s=metadata['end_s'], window=metadata,
                    target_epoch=window.target_epoch, generation=window.generation,
                    geometry_available=available, geometry_quality=geometry['quality'],
                    geometry_code=geometry['code'], pretrained_falling_score=original['all_scores'][42],
                    pretrained_top_class_index=original['top3'][0]['index'],
                    pretrained_inference_ms=original['inference_ms'])
                records.append(record)
                features.append(feature)
                scores.append(original['all_scores'])
                geometry_available.append(available)
                geometry_vectors.append(geometry['vector'] if available else [float('nan')] * len(GEOMETRY_FEATURE_NAMES))
                clip_counts[label] += 1
                stats[label] += 1
                if available:
                    clip_counts['geometry_' + label] += 1
                    stats['geometry_' + label] += 1
            source['observed_counts'] = dict(clip_counts)
            print(json.dumps(dict(clip_id=clip_id, **clip_counts)), flush=True)
    if not records:
        raise ValueError('No eligible official proxy windows; no head can be trained')
    np.savez_compressed(output / 'frozen-features.npz', features=np.stack(features),
                        geometry=np.asarray(geometry_vectors, dtype=np.float32),
                        geometry_available=np.asarray(geometry_available, dtype=bool),
                        original_scores=np.asarray(scores, dtype=np.float32))
    write(output / 'records.json', records)
    write(output / 'manifest.json', dict(entries=sources, source_manifest=str(manifest_path),
                                        source_manifest_sha256=sha256(manifest_path),
                                        official_sources=official_sources))
    code_paths = [Path(__file__), ROOT / 'src/fallpre/weekly/stgcn_home.py',
                  ROOT / 'src/fallpre/weekly/stgcn_temporal.py', ROOT / 'src/fallpre/weekly/stgcn_fall_evidence.py',
                  ROOT / 'src/fallpre/weekly/webcam_target.py', ROOT / 'src/fallpre/weekly/evaluate_human.py']
    write(output / 'provenance.json', dict(
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), model=runtime.metadata,
        temporal_config=vars(config), weights_updated=False, head_trained=False,
        feature_dim=len(features[0]), geometry_feature_names=list(GEOMETRY_FEATURE_NAMES),
        quality_counts=dict(quality_counts), counts=dict(stats),
        positive_source_events=sum(s['observed_counts'].get('fall_process', 0) > 0 for s in sources),
        total_simulated_fall_events=sum(s['original_sequence'].startswith('fall-') for s in sources),
        labels=['fall_process', 'non_fall_activity'], official_sources=official_sources,
        weak_label_alignment=True,
        label_alignment='official cam0 DEPTH sequence/frame state -> equal RGB PREVIEW decoded 1-based frame; '
                        'sequence/frame count/order checked, physical RGB-depth registration not independently verified',
        clock='existing preview nominal 30 fps (frame-1)/30; not original acquisition synchronization CSV',
        camera_stability='declared fixed-camera URFD; no new RGB background quality check in this cache extraction',
        label_definition='fall-* state0 positive simulated fall-process proxy; state-1 non_fall_activity proxy. '
                         'State1 sustained lying excluded rather than declared normal activity. Entire ADL sequences '
                         'with state0 excluded, and missing official frames excluded.',
        preprocessing='same production trailing TemporalPoseBuffer and prepare_tensor; no future frames',
        grouping='original URFD sequence; only one preview version per event. All old data development_only; '
                 'unknown person grouping prevents independent-person claims',
        limitations=['Historical YOLO11n pose caches, not current YOLO26 detections',
                     'Low-resolution 320x240 preview RGB, not original 640x480 RGB',
                     'Model readiness/identity/quality losses retained in coverage.jsonl',
                     'No elderly nursing-home data or independent test; no alarm policy changes'],
        source_manifest_sha256=sha256(manifest_path), code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        files_sha256={p.name: sha256(p) for p in [output / 'records.json', output / 'frozen-features.npz',
                                                output / 'manifest.json', output / 'coverage.jsonl']}))
    print(json.dumps(dict(output=str(output), **stats)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--labels-dir', type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    extract(args.output, args.manifest, args.labels_dir)
