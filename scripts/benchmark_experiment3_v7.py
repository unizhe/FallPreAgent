"""Frozen V7 local CPU file-replay timing; no training, rendering or notifications.

Only URFD fall01 original is replayed. Source CSV times schedule frame arrivals,
then one FIFO worker processes every frame. This tests a controlled local file
pipeline, not physical sensor-to-alert latency or a deployment's capture policy.
"""
import argparse
import bisect
import csv
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'third_party/ultralytics')]

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.human import sha256
from fallpre.weekly.pose_recovery import RTMOBackend
from fallpre.weekly.pose_refinement import (
    RefinedPoseRecovery, RTMPoseBackend, predict_with_pose_quality,
)
from fallpre.weekly.verified_tracking import VerifiedPersonTracker

CLIP = 'urfd-fall-01-original'
PREDICT_ARGS = dict(imgsz=640, conf=.1, iou=.45, device='cpu', verbose=False)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), median_s=float(np.median(a)),
                p95_s=float(np.percentile(a, 95)), max_s=float(np.max(a)),
                mean_s=float(np.mean(a)))


def pose_instances(result):
    return [dict(box=b, det=s, kpts=k) for b, s, k in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
        result.keypoints.data.cpu().tolist())]


def detector_instances(result):
    return [dict(class_name=result.names[int(k)], conf=float(s), box=b)
            for b, s, k in zip(result.boxes.xyxy.tolist(),
                               result.boxes.conf.tolist(), result.boxes.cls.tolist())]


def decode(archive, name):
    image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not decode {name}')
    return image


def run_group(group, output, times, names, input_path, cached, pinned):
    load_begin = time.perf_counter()
    model = CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')
    pose = YOLO(str(ROOT / 'models/yolo11n-pose.pt'))
    detector = YOLO(str(ROOT / 'models/yolo11n.pt')) if group != 'A' else None
    # These original recovery adapters load ONNX lazily. Preload their frozen
    # runtimes here, excluding model construction from warm-start timing.
    recovery = (RefinedPoseRecovery(backend=RTMOBackend(), refiner=RTMPoseBackend(),
        allow_partial_observation=True, allow_detector_proposals=True,
        resolve_native_ambiguity=True) if group == 'C' else None)
    load_wall_s = time.perf_counter() - load_begin
    with zipfile.ZipFile(input_path) as archive:
        first_image = decode(archive, names[0])
        warm_begin = time.perf_counter()
        pose.predict(first_image, **PREDICT_ARGS)
        if detector is not None:
            detector.predict(first_image, **PREDICT_ARGS)
        warm_wall_s = time.perf_counter() - warm_begin
        # Neither warm-up nor cached results are passed into these fresh states.
        tracker = VerifiedPersonTracker(30) if detector is not None else PersonTracker(30)
        online = OnlineCompletion(model)
        rows, divergences = [], []
        start_utc = datetime.now(timezone.utc).isoformat()
        origin = time.perf_counter()
        print(json.dumps(dict(group=group, status='started', model_load_s=load_wall_s,
            warmup_s=warm_wall_s, frames=len(times)), ensure_ascii=False), flush=True)
        with (output / f'{group}.frames.jsonl').open('x', buffering=1) as log:
            for i, (source_time, name) in enumerate(zip(times, names)):
                scheduled_arrival = source_time - times[0]
                ahead = origin + scheduled_arrival - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
                frame_start = time.perf_counter()
                image = decode(archive, name)
                decoded_at = time.perf_counter()
                result = pose.predict(image, **PREDICT_ARGS)[0]
                observation = dict(source_frame_id=i+1, t=source_time,
                                   instances=pose_instances(result))
                pose_done = time.perf_counter()
                if detector is not None:
                    observation['detector_instances'] = detector_instances(
                        detector.predict(image, **PREDICT_ARGS)[0])
                detector_done = time.perf_counter()
                recovery_audit = None
                if recovery is not None:
                    observation, recovery_audit = recovery.augment(image, observation, tracker)
                recovery_done = time.perf_counter()
                tracked = tracker.update(observation)
                tracking_done = time.perf_counter()
                prediction = predict_with_pose_quality(online, tracked)
                action_done = time.perf_counter()
                output_offset = action_done - origin
                start_offset = frame_start - origin
                # Only diagnostic accounting consults future arrival timestamps;
                # inference above receives current and past images/observations.
                row = dict(group=group, original_frame_id=i+1, source_time_s=source_time,
                    scheduled_arrival_wall_offset_s=scheduled_arrival,
                    service_start_wall_offset_s=start_offset,
                    action_output_wall_offset_s=output_offset,
                    queue_wait_s=start_offset-scheduled_arrival,
                    service_wall_s=action_done-frame_start,
                    local_arrival_to_output_s=output_offset-scheduled_arrival,
                    stages_s=dict(decode=decoded_at-frame_start, pose=pose_done-decoded_at,
                        detector=detector_done-pose_done, recovery=recovery_done-detector_done,
                        tracking=tracking_done-recovery_done, action=action_done-tracking_done),
                    pending_arrived_frames_at_start=max(0, bisect.bisect_right(times, start_offset+times[0])-i-1),
                    pending_arrived_frames_at_output=max(0, bisect.bisect_right(times, output_offset+times[0])-i-1),
                    prediction=prediction, target_id=tracked['target_id'],
                    target_observed=bool(tracked['instances']),
                    tracking_reason=tracked['tracking_reason'],
                    recovery_executed=bool(recovery_audit and recovery_audit['executed']),
                    recovery_accepted=bool(recovery_audit and recovery_audit['accepted']),
                    recovery_inference_passes=recovery_audit['inference_passes'] if recovery_audit else 0,
                    refinement_inference_passes=(recovery_audit.get('refinement', {}).get('inference_passes', 0)
                        if recovery_audit else 0))
                expected_row = cached[i]
                expected = expected_row.get('prediction') or expected_row['after']['prediction']
                diffs = {key: dict(cached=expected.get(key), measured=prediction.get(key))
                         for key in ['label', 'phase']
                         if expected.get(key) != prediction.get(key)}
                if diffs:
                    divergences.append(dict(original_frame_id=i+1, changes=diffs))
                row['cached_label_phase_equal'] = not diffs
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                if (i+1) % 40 == 0:
                    print(json.dumps(dict(group=group, frames=i+1, elapsed_s=round(output_offset, 3),
                        latest_latency_s=round(row['local_arrival_to_output_s'], 3),
                        cached_divergences=len(divergences)), ensure_ascii=False), flush=True)
    first = next((r for r in rows if r['prediction']['label'] == 'falling'), None)
    trigger = None
    if first:
        stamp = first['action_output_wall_offset_s']
        trigger = dict(source_frame=first['original_frame_id'], source_time_s=first['source_time_s'],
            action_output_wall_offset_s=stamp, queue_wait_s=first['queue_wait_s'],
            service_wall_s=first['service_wall_s'],
            local_arrival_to_output_s=first['local_arrival_to_output_s'],
            controlled_replay_margin_to_contact_s=[round(v-times[0]-stamp, 9) for v in [3.336, 3.537]],
            controlled_replay_margin_to_stable_onset_s=[round(v-times[0]-stamp, 9) for v in [4.371, 4.571]])
    summary = dict(group=group, frames=len(rows), dropped_frames=0,
        start_utc=start_utc, model_load_s=load_wall_s, warmup_s=warm_wall_s,
        warmup=dict(pose_calls=1, detector_calls=int(detector is not None),
                    recovery_runtime_loaded_before_timing=True if recovery else None,
                    recovery_inference_warmup=False, tracker_updates=0),
        total_action_output_wall_s=rows[-1]['action_output_wall_offset_s'],
        arrival_to_output=distribution([r['local_arrival_to_output_s'] for r in rows]),
        service=distribution([r['service_wall_s'] for r in rows]),
        queue_wait=distribution([r['queue_wait_s'] for r in rows]),
        stages={name: distribution([r['stages_s'][name] for r in rows])
                for name in ['decode', 'pose', 'detector', 'recovery', 'tracking', 'action']},
        max_pending_arrived_frames=max(r['pending_arrived_frames_at_output'] for r in rows),
        raw_falling_trigger=trigger, cached_label_phase_divergence_count=len(divergences),
        cached_label_phase_divergences=divergences,
        recovery_executed_frames=sum(r['recovery_executed'] for r in rows),
        runtime=dict(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                     opencv_threads=cv2.getNumThreads()),
        hashes_unchanged=all(sha256(Path(path)) == value for path, value in pinned.items()))
    assert summary['hashes_unchanged']
    dump(output / f'{group}.summary.json', summary)
    print(json.dumps(dict(group=group, status='complete', first_falling=trigger,
                         cached_divergences=len(divergences))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'runs/abc-v7-20261020/experiment3-20261021/benchmark')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['protocol.json', 'summary.json', 'A.frames.jsonl', 'B.frames.jsonl', 'C.frames.jsonl']):
        raise FileExistsError('Existing benchmark result; choose a fresh output or archive an interrupted attempt explicitly.')
    protocol_path = ROOT / 'runs/abc-v7-20261020/evaluation-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    original_models = json.loads((ROOT / 'runs/evidence-recovery-20261020/protocol.json').read_text())['frozen_models']
    pinned = dict(original_models)
    assert pinned[str(ROOT / 'runs/action-v7/training/model.json')] == protocol['model_sha256']
    source = protocol['source_inputs'][CLIP]
    input_path = Path(source['file'])
    pinned[str(input_path)] = source['sha256']
    csv_path = ROOT / 'runs/dataset-survey-20261019/urfd-fall-01-data.csv'
    times = [float(row[1])/1000 for row in csv.reader(csv_path.open())]
    assert len(times) == 160 and times[0] == 0 and all(b > a for a, b in zip(times, times[1:]))
    with zipfile.ZipFile(input_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.png'))
    assert len(names) == len(times)
    cached = {}
    for group in ['A', 'B', 'C']:
        item = protocol['source_logs'][f'{group}/{CLIP}']
        pinned[item['file']] = item['sha256']
        cached[group] = [json.loads(line) for line in Path(item['file']).read_text().splitlines()]
        assert len(cached[group]) == 160
        assert all(r['original_frame_id'] == i+1 and r['source_time_s'] == times[i]
                   for i, r in enumerate(cached[group]))
    code = [Path(__file__), ROOT / 'scripts/replay_public_samples.py',
        *(ROOT / 'src/fallpre/weekly' / name for name in [
            'action_v2.py', 'action_v7.py', 'verified_tracking.py', 'pose_recovery.py', 'pose_refinement.py'])]
    pinned.update({str(path): sha256(path) for path in [csv_path, protocol_path, *code]})
    assert all(sha256(Path(path)) == digest for path, digest in pinned.items())
    spec = dict(schema='fallpre.experiment3.local_pipeline_timing.v1',
        created_utc=datetime.now(timezone.utc).isoformat(), clip_id=CLIP, frames=160,
        frozen_hashes=pinned, training=False, detector=PREDICT_ARGS,
        groups=protocol['groups'], execution_order=['A', 'B', 'C'], repeats_per_group=1,
        timing_policy='Source CSV arrival schedule; serial FIFO, no frame drop; monotonic wall clock; warm start.',
        measurement_boundary='From scheduled local frame arrival through ZIP read, image decode, pose, object detector if used, recovery if used, tracker and V7 action output.',
        excluded='Model construction and pose/detector first-image warm-up; visualization, notification dispatch, actuator delay, sensor capture and network transport.',
        bookkeeping='JSON logging and cache-comparison occur after each action output; their cost is outside per-frame service, but can contribute to subsequent FIFO queue waiting.',
        trigger_rule='First current/past-only V7 prediction label falling; no debounce, threshold tuning, or future smoothing.',
        contact_interval_source_s=[3.336, 3.537], stable_onset_interval_source_s=[4.371, 4.571],
        margin_formula='human landmark source interval - (CSV first time + measured trigger wall offset); positive is earlier than the boundary in this controlled replay.',
        caveats=['A single sequential run per mode on one host is not a statistical latency estimate.',
                 'This is a local file pipeline test, not physical sensor-to-alert or actuator latency.',
                 'Future timestamps are used only to schedule/describe arrivals; inference receives no future image or label.',
                 'No model or tracking state is updated by warm-up; ONNX recovery runtimes are eager loaded, but recovery kernels are not warmed.',
                 'OS cache, temperature and other host workloads are uncontrolled; A/B/C timing differences are descriptive.'],
        host=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                  logical_cpu_count=os.cpu_count(), python=platform.python_version(),
                  torch=torch.__version__, opencv=cv2.__version__))
    dump(output / 'protocol.json', spec)
    summaries = [run_group(group, output, times, names, input_path, cached[group], pinned)
                 for group in ['A', 'B', 'C']]
    dump(output / 'summary.json', dict(schema=spec['schema'], protocol_file='protocol.json',
        complete=True, groups=summaries, models_and_inputs_unchanged=True,
        total_measured_frames=sum(s['frames'] for s in summaries)))
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['group', 'frames', 'latency_median_s',
            'latency_p95_s', 'latency_max_s', 'service_median_s', 'service_p95_s',
            'first_falling_frame', 'first_falling_output_wall_s', 'contact_margin_lo_s',
            'contact_margin_hi_s', 'cached_label_phase_divergences'])
        writer.writeheader()
        for item in summaries:
            trigger = item['raw_falling_trigger'] or {}
            margin = trigger.get('controlled_replay_margin_to_contact_s', [None, None])
            writer.writerow(dict(group=item['group'], frames=item['frames'],
                latency_median_s=item['arrival_to_output']['median_s'],
                latency_p95_s=item['arrival_to_output']['p95_s'], latency_max_s=item['arrival_to_output']['max_s'],
                service_median_s=item['service']['median_s'], service_p95_s=item['service']['p95_s'],
                first_falling_frame=trigger.get('source_frame'),
                first_falling_output_wall_s=trigger.get('action_output_wall_offset_s'),
                contact_margin_lo_s=margin[0], contact_margin_hi_s=margin[1],
                cached_label_phase_divergences=item['cached_label_phase_divergence_count']))


if __name__ == '__main__':
    main()
