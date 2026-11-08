"""Matched before/after live replay of a local recording; no label-driven inference.

Collect fresh perception once, then run isolated source snapshots on identical
pixels, detections and source clocks. The prelabels are used only by scoring.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'runs/personal-fix-20261023'
CAPTURE = ROOT / 'runs/personal-capture-20261023'
LABELS = ('standing', 'sitting_down', 'seated', 'standing_up')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_service(code_root):
    sys.path.insert(0, str(ROOT / 'third_party/ultralytics'))
    sys.path.insert(0, str(code_root / 'src'))
    spec = importlib.util.spec_from_file_location('personal_replay_service', code_root / 'scripts/webcam_pose_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import fallpre.weekly.webcam_actions as action_module
    assert Path(action_module.__file__).resolve().is_relative_to(code_root.resolve()), 'Replay source isolation failed'
    # Read identical local, hash-locked weights/config, regardless of code snapshot.
    module.ROOT = ROOT
    import fallpre.weekly.webcam_rgb_model as rgb
    rgb.ROOT = ROOT
    rgb.MODEL_CONFIG = ROOT / 'configs/rgb-fall-model.json'
    return module


def source_hashes(code_root):
    return {str(p.relative_to(code_root)): digest(p)
            for directory in ('src/fallpre/weekly', 'scripts')
            for p in sorted((code_root / directory).glob('*.py'))}


def collect(out, diagnostic=None, video=None):
    import cv2
    service = load_service(ROOT)
    diagnostic = Path(diagnostic) if diagnostic else next((CAPTURE / 'source').glob('*73738905.json'))
    video = Path(video) if video else next((CAPTURE / 'source').glob('*73738905.webm'))
    source = json.loads(diagnostic.read_text())
    samples = source['samples']
    engine = service.PoseEngine('cpu', 640, source['model']['id'], source['fall_mode'])
    assert engine.model_metadata[source['model']['id']]['sha256'] == source['model']['sha256']
    image_dir = out / 'frames'
    image_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    ok, next_frame = cap.read()
    next_pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    previous, previous_pts, frame_id = None, None, -1
    rows = []
    with (out / 'perception.jsonl').open('w') as target:
        for i, sample in enumerate(samples):
            t = sample['client_timing']['recording_time_s']
            while ok and next_pts <= t + 1e-9:
                previous, previous_pts = next_frame, next_pts
                frame_id += 1
                ok, next_frame = cap.read()
                next_pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if ok else float('inf')
            if previous is None:
                raise ValueError('No past video frame at observation time')
            h, w = previous.shape[:2]
            ratio = min(1, 960/w, 720/h)
            resized = cv2.resize(previous, (round(w*ratio), round(h*ratio)))
            encoded_ok, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            assert encoded_ok
            filename = f'frames/{i:04d}.jpg'
            (out / filename).write_bytes(encoded.tobytes())
            image = service.decode_frame(encoded.tobytes())
            result = engine.infer(image)
            rgb = engine.rgb_backend.infer(image) if engine.rgb_backend is not None else None
            row = dict(index=i, source_time_s=sample['t'], video_time_s=t,
                       frame_pts_s=previous_pts, frame_id=frame_id, image=filename,
                       pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                       result=result, rgb=rgb)
            target.write(json.dumps(row, allow_nan=False) + '\n')
            target.flush()
            rows.append(row)
            if (i+1) % 50 == 0:
                print(f'Fresh perception {i+1}/{len(samples)}', flush=True)
    cap.release()
    assert len(rows) == len(samples)
    assert len({r['frame_id'] for r in rows}) == len(rows)
    manifest = dict(video=str(video), video_sha256=digest(video),
                    diagnostic=str(diagnostic), diagnostic_sha256=digest(diagnostic),
                    model=source['model'], fall_mode=source['fall_mode'],
                    fall_model=source['fall_model'], samples=len(rows),
                    cache_sha256=digest(out / 'perception.jsonl'),
                    selection='Latest real decoded PTS <= original recording-relative request time; no future frame or interpolation.',
                    timebase='Original diagnostic source time for both state machines; recording-relative time for scoring.',
                    limitations=[
                        'WebM decode then 960x540 JPEG85 is not byte-identical to original webcam JPEG requests.',
                        'Original recording event alignment is approximate; visual boundaries +/-0.2 s.',
                        'Matched replay isolates code changes. Live camera scheduling, transport and display latency are not simulated.',
                        'Single-person personal development clip, AI prelabels pending human review; not an independent accuracy benchmark.',
                    ])
    write_json(out / 'perception-manifest.json', manifest)
    print('Perception complete', flush=True)


def replay(out, variant, code_root):
    code_root = code_root.resolve()
    sources = source_hashes(code_root)
    service = load_service(code_root)
    manifest = json.loads((out / 'perception-manifest.json').read_text())
    assert digest(out / 'perception.jsonl') == manifest['cache_sha256']
    # The selected camera recording used RGB mode. Production infer_live also
    # disables pose recovery in RGB mode; pose-only recovery has separate tests.
    assert manifest['fall_mode'] == 'rgb'
    model = service.CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')
    session = service.LiveSession(model, manifest['model']['id'], 'camera', manifest['fall_mode'])
    rows = [json.loads(line) for line in (out / 'perception.jsonl').read_text().splitlines()]
    with (out / f'{variant}.jsonl').open('w') as target:
        for row in rows:
            assert row['result']['model']['sha256'] == manifest['model']['sha256']
            assert row['rgb']['model']['sha256'] == manifest['fall_model']['sha256']
            for key in ('id', 'sha256'):
                assert row['rgb']['model'].get('person_model', {}).get(key) == manifest['fall_model'].get('person_model', {}).get(key)
            image = service.decode_frame((out / row['image']).read_bytes())
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['pixel_sha256']
            result = copy.deepcopy(row['result'])
            start = time.perf_counter()
            session.update(image, result, row['source_time_s'], rgb_result=copy.deepcopy(row['rgb']))
            result['postprocess_ms'] = (time.perf_counter() - start)*1000
            target.write(json.dumps(dict(index=row['index'], video_time_s=row['video_time_s'],
                                         source_time_s=row['source_time_s'], result=result), allow_nan=False) + '\n')
    assert sources == source_hashes(code_root), 'Source changed during replay; rerun after edits stop'
    write_json(out / f'{variant}-provenance.json', dict(code_root=str(code_root), sources=sources,
               perception_sha256=manifest['cache_sha256'], model_sha256=digest(ROOT / 'runs/action-v7/training/model.json')))
    print(f'{variant}: {len(rows)} complete', flush=True)


def class_stats(rows, segments):
    def at(t):
        return next((s for s in segments if s['start_s'] <= t < s['end_s']), None)
    result = {}
    for label in (*LABELS, 'four_class', 'other', 'all'):
        selected = [(r, at(r['video_time_s'])) for r in rows]
        selected = [(r, s) for r, s in selected if s and (label == 'all' or
                    (label == 'four_class' and s['label'] in LABELS) or s['label'] == label)]
        counts = Counter(r['result']['action']['label'] for r, _ in selected)
        correct = sum(r['result']['action']['label'] == s['label'] for r, s in selected)
        n = len(selected)
        eligible = label in (*LABELS, 'four_class')
        result[label] = dict(n=n, unknown=counts['unknown'], unknown_rate=counts['unknown']/n if n else None,
                             provisional_match=correct if eligible else None,
                             wrong_known=n-counts['unknown']-correct if eligible else None,
                             known_outputs=n-counts['unknown'],
                             outputs=dict(counts))
    return result


def segment_latency(rows, segments):
    """Onset to first correct observation INSIDE segment, misses are never zero.

    Also report first two consecutive correct samples and stale previous-state
    tail. No late output is credited to a missed action interval.
    """
    out = []
    for i, s in enumerate(segments):
        if s['label'] not in LABELS:
            continue
        selected = [r for r in rows if s['start_s'] <= r['video_time_s'] < s['end_s']]
        matches = [r for r in selected if r['result']['action']['label'] == s['label']]
        stable = next((b['video_time_s'] for a, b in zip(selected, selected[1:])
                       if a['result']['action']['label'] == b['result']['action']['label'] == s['label']
                       and b['video_time_s']-a['video_time_s'] <= .35), None)
        previous = segments[i-1]['label'] if i else None
        stale = [r['video_time_s'] for r in selected if r['result']['action']['label'] == previous]
        out.append(dict(id=s['id'], label=s['label'], start_s=s['start_s'], end_s=s['end_s'],
                        first_correct_latency_s=matches[0]['video_time_s']-s['start_s'] if matches else None,
                        two_consecutive_latency_s=stable-s['start_s'] if stable is not None else None,
                        missed=not matches, last_previous_label_offset_s=max(stale)-s['start_s'] if stale else None))
    return out


def transition_audit(before, after, segments):
    """Expose new rise decisions, lost correct states and raw-model isolation."""
    added, lost, raw_differences = [], [], []
    for a, b in zip(before, after, strict=True):
        assert a['video_time_s'] == b['video_time_s']
        t = a['video_time_s']
        segment = next((s for s in segments if s['start_s'] <= t < s['end_s']), None)
        aa, bb = a['result']['action'], b['result']['action']
        if (aa.get('raw_label'), aa.get('phase')) != (bb.get('raw_label'), bb.get('phase')):
            raw_differences.append(t)
        if segment is None:
            continue
        row = dict(time_s=t, segment_id=segment['id'], prelabel=segment['label'],
                   before=aa['label'], after=bb['label'],
                   distance_to_boundary_s=min(t-segment['start_s'], segment['end_s']-t))
        if aa['label'] != 'standing_up' and bb['label'] == 'standing_up':
            added.append(row)
        if segment['label'] in LABELS and aa['label'] == segment['label'] and bb['label'] != segment['label']:
            lost.append(row)
    return dict(added_rise=added, added_rise_by_prelabel=dict(Counter(x['prelabel'] for x in added)),
                lost_prelabel_matches=lost, raw_prediction_difference_times=raw_differences)


def score(out):
    annotations = json.loads((CAPTURE / 'annotations.json').read_text())
    segments = annotations['segments']
    data = {v: [json.loads(line) for line in (out / f'{v}.jsonl').read_text().splitlines()]
            for v in ('before', 'after')}
    assert len(data['before']) == len(data['after']) == 452
    stats = {v: class_stats(rows, segments) for v, rows in data.items()}
    sensitivity = {}
    for v, rows in data.items():
        interior = [r for r in rows if any(s['start_s']+.2 <= r['video_time_s'] < s['end_s']-.2
                                          for s in segments)]
        shifts = {str(shift): class_stats([dict(r, video_time_s=r['video_time_s']+shift) for r in rows], segments)
                  for shift in (-.1, 0., .1)}
        sensitivity[v] = dict(interior_margin_s=.2, interior=class_stats(interior, segments),
                              alignment_shifts=shifts)
    delays = {v: segment_latency(rows, segments) for v, rows in data.items()}
    delay_summary = {}
    for label in LABELS:
        group = {}
        for v in data:
            records = [s for s in delays[v] if s['label'] == label]
            values = [s['first_correct_latency_s'] for s in records if not s['missed']]
            stable = [s['two_consecutive_latency_s'] for s in records if s['two_consecutive_latency_s'] is not None]
            group[v] = dict(segments=len(records), detected=len(values), missed=len(records)-len(values),
                            median_latency_s=statistics.median(values) if values else None,
                            two_consecutive_detected=len(stable),
                            median_two_consecutive_latency_s=statistics.median(stable) if stable else None)
        paired = [(a['first_correct_latency_s'], b['first_correct_latency_s'])
                  for a, b in zip(delays['before'], delays['after'])
                  if a['label'] == label and not a['missed'] and not b['missed']]
        group['paired'] = dict(n=len(paired), before_median_s=statistics.median(x[0] for x in paired) if paired else None,
                               after_median_s=statistics.median(x[1] for x in paired) if paired else None)
        delay_summary[label] = group
    fall_keys = ('fall', 'pose_fall', 'fall_observation', 'pose_fall_observation', 'quality')
    fall_diff = []
    for a, b in zip(data['before'], data['after']):
        assert a['source_time_s'] == b['source_time_s'] and a['video_time_s'] == b['video_time_s']
        changed = [k for k in fall_keys if a['result'].get(k) != b['result'].get(k)]
        if changed:
            fall_diff.append(dict(time_s=a['video_time_s'], keys=changed))
    fall_counts = {v: dict(Counter(r['result']['fall']['state'] for r in rows)) for v, rows in data.items()}
    summary = dict(stats=stats, latency=delay_summary, segment_latency=delays, sensitivity=sensitivity,
                    transition_audit=transition_audit(data['before'], data['after'], segments),
                    fall=dict(difference_frames=len(fall_diff), differences=fall_diff, states=fall_counts),
                    annotations_sha256=digest(CAPTURE / 'annotations.json'),
                    latency_definition='Source time from visual segment start to first correct observation inside segment; missed intervals excluded from medians and counted separately. Does not include camera/network/rendering delay. Two consecutive evidence reported separately.',
                    boundary_uncertainty_s=.2,
                    limitation='Same development video and provisional AI labels; unknown is not error rate. Report wrong known states alongside unknown.')
    write_json(out / 'comparison.json', summary)
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['video_time_s', 'prelabel', 'before', 'after', 'before_raw', 'after_raw', 'before_reason', 'after_reason']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for a, b in zip(data['before'], data['after']):
            s = next(s for s in segments if s['start_s'] <= a['video_time_s'] < s['end_s'])
            aa, bb = a['result']['action'], b['result']['action']
            writer.writerow(dict(video_time_s=a['video_time_s'], prelabel=s['label'], before=aa['label'], after=bb['label'],
                                 before_raw=aa['raw_label'], after_raw=bb['raw_label'],
                                 before_reason=aa['decision_code'], after_reason=bb['decision_code']))
    print(json.dumps({k: v for k, v in summary.items() if k not in {'segment_latency', 'sensitivity'}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'replay', 'score'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--variant', default='after')
    parser.add_argument('--code-root', type=Path, default=ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'collect':
        collect(args.out)
    elif args.command == 'replay':
        replay(args.out, args.variant, args.code_root)
    else:
        score(args.out)


if __name__ == '__main__':
    main()
