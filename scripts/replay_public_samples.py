"""Frozen-model public RGB audit; no training, notifications or model changes."""
import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'third_party/ultralytics')]
import cv2
import numpy as np
from fallpre.weekly.action_v2 import PersonTracker
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from fallpre.weekly.pose_recovery import PoseRecovery, RTMO_PATH, RTMO_SHA256
from fallpre.weekly.pose_refinement import RefinedPoseRecovery, RTMPOSE_PATH, RTMPOSE_SHA256, predict_with_pose_quality
from fallpre.weekly.five_actions import label_at
from fallpre.weekly.human import LINKS, open_video_writer, sha256
from ultralytics import YOLO


def frames(entry):
    if entry.get('zip_path'):
        with zipfile.ZipFile(entry['zip_path']) as z:
            for name in sorted(n for n in z.namelist() if n.endswith('.png')):
                yield cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(entry['video_path'])
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield img
        finally:
            cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tracking', choices=['verified', 'legacy'], default='verified',
                   help='Verified admission/recovery by default; legacy reproduces the prior audit.')
    p.add_argument('--recovery', choices=['evidence', 'refined', 'refined-strict', 'rtmo', 'off'], default=None,
                   help='Current-image recovery; default evidence for verified, off for legacy.')
    p.add_argument('--clip', action='append', help='Optional clip ID; repeat to select several samples.')
    args = p.parse_args()
    args.recovery = args.recovery or ('evidence' if args.tracking == 'verified' else 'off')
    if args.recovery != 'off' and args.tracking != 'verified':
        p.error('Pose recovery requires verified target admission')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    modelpath = ROOT / 'runs/action-v7/training/model.json'
    weights = ROOT / 'models/yolo11n-pose.pt'
    pinned = {str(p): sha256(p) for p in [modelpath, weights]}
    detector_weights = ROOT / 'models/yolo11n.pt'
    if args.tracking == 'verified':
        lock = json.loads((ROOT / 'models.lock.json').read_text())
        assert sha256(detector_weights) == lock[detector_weights.name]['sha256']
        pinned[str(detector_weights)] = sha256(detector_weights)
    model = CompletionModel.load(modelpath)
    yolo = YOLO(str(weights))
    detector = YOLO(str(detector_weights)) if args.tracking == 'verified' else None
    recovery = (RefinedPoseRecovery(allow_partial_observation=args.recovery!='refined-strict',
                allow_detector_proposals=args.recovery=='evidence',resolve_native_ambiguity=args.recovery=='evidence') if args.recovery in ['evidence','refined','refined-strict'] else
                PoseRecovery() if args.recovery == 'rtmo' else None)
    if recovery:
        assert sha256(RTMO_PATH) == RTMO_SHA256
        pinned[str(RTMO_PATH)] = RTMO_SHA256
    if args.recovery in ['evidence','refined','refined-strict']:
        assert sha256(RTMPOSE_PATH) == RTMPOSE_SHA256
        pinned[str(RTMPOSE_PATH)] = RTMPOSE_SHA256
    source = ROOT / 'runs/dataset-survey-20261019'
    manifest = json.loads((ROOT / 'configs/action-v7-dataset.json').read_text())
    ids = ['utk-s01_e01-transition', 'utk-s10_e01-transition',
           'urfd-adl-04', 'urfd-adl-07', 'urfd-fall-01', 'urfd-fall-02']
    entries = [dict(next(e for e in manifest['entries'] if e['clip_id'] == cid)) for cid in ids]
    times = [float(r[1]) / 1000 for r in csv.reader((source / 'urfd-fall-01-data.csv').open())]
    entries += [dict(clip_id='urfd-fall-01-original', zip_path=str(source / 'urfd-fall-01-rgb.zip'),
                     frame_count=160, fps=30, source_times_s=times, segments=[], split='same-event-as-train'),
                dict(clip_id='multicam-chute01-cam8', video_path=str(source / 'mc-chute01-cam8.avi'),
                     frame_count=1559, fps=30, frame_only=True, segments=[], split='frame-audit')]
    if args.clip:
        unknown = set(args.clip) - {e['clip_id'] for e in entries}
        if unknown: raise ValueError(f'Unknown clip IDs: {sorted(unknown)}')
        entries = [e for e in entries if e['clip_id'] in args.clip]
    for e in entries:
        key = 'zip_path' if e.get('zip_path') else 'video_path'
        e[key] = str((ROOT / e[key]).resolve())
        e['input_sha256'] = sha256(Path(e[key]))
        if e.get('video_sha256'):
            assert e['input_sha256'] == e['video_sha256']
    (out / 'manifest.json').write_text(json.dumps(dict(entries=entries, frozen_models=pinned,
        detector=dict(imgsz=640, conf=.1, iou=.45, device='cpu'), tracking=args.tracking,
        recovery=args.recovery, recovery_config=recovery.configuration() if recovery else None,
        multicam_clock='Frame-only: synthetic index/30 used ONLY for tracker bookkeeping; no action model. Preview 30 FPS is illustrative.'), indent=2))
    summaries = []
    for e in entries:
        dest = out / e['clip_id']
        dest.mkdir()
        mc = e.get('frame_only', False)
        tracker = (VerifiedPersonTracker(e['fps'], frame_only=mc) if detector is not None
                   else PersonTracker(30 if mc else e['fps']))
        online = None if mc else OnlineCompletion(model)
        writer, codec = open_video_writer(dest / 'overlay.mp4', 30, (1280, 720))
        counts, phases, tids = Counter(), Counter(), Counter()
        stats = dict(no_detection=0, no_target=0, low_ankle=0, low_hip=0)
        previous = None
        playback_index = 0
        rows = []
        with (dest / 'frames.jsonl').open('w') as log:
            for i, img in enumerate(frames(e)):
                assert img is not None
                frame_id = e.get('source_original_frame_ids', list(range(1, e['frame_count'] + 1)))[i]
                t = e.get('source_times_s', [j / e['fps'] for j in range(e['frame_count'])])[i]
                r = yolo.predict(img, imgsz=640, conf=.1, iou=.45, device='cpu', verbose=False)[0]
                instances = [dict(box=b, det=s, kpts=k) for b, s, k in zip(
                    r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.keypoints.data.cpu().tolist())]
                native_instances = instances
                observation = dict(source_frame_id=frame_id, t=t, instances=instances)
                if detector is not None:
                    obj = detector.predict(img, imgsz=640, conf=.1, iou=.45, device='cpu', verbose=False)[0]
                    observation['detector_instances'] = [dict(class_name=obj.names[int(k)], conf=float(s), box=b)
                        for b, s, k in zip(obj.boxes.xyxy.tolist(), obj.boxes.conf.tolist(), obj.boxes.cls.tolist())]
                recovery_audit = None
                if recovery:
                    observation, recovery_audit = recovery.augment(img, observation, tracker)
                    instances = observation['instances']
                tr = tracker.update(observation)
                prediction = predict_with_pose_quality(online,tr)
                tracks = (tr['detection_track_ids'] if detector is not None else
                          {int(x.idx): int(x.track_id) for x in tracker.tracker.tracked_stracks if x.is_activated})
                accepted = ({a['index'] for a in tr['admission'] if a['accepted']} if detector is not None
                            else set(range(len(instances))))
                ref = label_at(e, t) if not mc else 'unannotated'
                if e['clip_id'] == 'urfd-fall-01-original':
                    ref = 'nonlying' if frame_id < 83 else ('falling' if frame_id < 113 else 'lying')
                if mc:
                    ref = next((s for a, b, s in [(874, 1011, 'code1_walk_or_stand'),
                        (1012, 1079, 'code6_moving_up_NOT_chair_rise'), (1080, 1108, 'code2_falling'),
                        (1109, 1285, 'code3_lying')] if a <= frame_id <= b), 'unannotated')
                pts = np.array(tr['instances'][0]['kpts']) if tr['instances'] else None
                row = dict(decoded_frame_1based=i+1, original_frame_id=frame_id,
                    source_time_s=None if mc else t, preview_time_s=i/30 if mc else t,
                    reference=ref, instances=instances, detection_track_ids=tracks,
                    native_pose_instances=native_instances, recovery=recovery_audit,
                    target_id=tr['target_id'], target_observed=bool(tr['instances']),
                    tracking_reason=tr['tracking_reason'], prediction=prediction,
                    tracking_mode=args.tracking, admission=tr.get('admission'),
                    detector_instances=observation.get('detector_instances'), target_index=tr['target_index'],
                    reliable_joints=int((pts[:, 2] >= .2).sum()) if pts is not None else 0,
                    ankles_reliable=bool(pts is not None and (pts[15:17, 2] >= .2).all()),
                    hips_reliable=bool(pts is not None and (pts[11:13, 2] >= .2).all()))
                log.write(json.dumps(row) + '\n')
                rows.append(row)
                counts[prediction['label']] += 1
                phases[prediction['phase']] += 1
                if tr['instances']: tids[tr['target_id']] += 1
                stats['no_detection'] += not instances
                stats['no_target'] += not tr['instances']
                stats['low_ankle'] += not row['ankles_reliable']
                stats['low_hip'] += not row['hips_reliable']
                for j, person in enumerate(instances):
                    color = (80, 220, 140) if j == tr['target_index'] else (0, 180, 255)
                    if j not in accepted: color = (80, 80, 240)
                    b = list(map(int, person['box']))
                    cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), color, 2)
                    cv2.putText(img, f"{person.get('pose_backend', 'YOLO')} BT {tracks.get(j, '?')} {person['det']:.2f}" if j in accepted else 'REJECTED', (b[0], max(14, b[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                    if j not in accepted: continue
                    k = person['kpts']
                    for a, b in LINKS:
                        if min(k[a][2], k[b][2]) >= .2:
                            cv2.line(img, tuple(map(int,k[a][:2])), tuple(map(int,k[b][:2])), color, 2)
                    for x, y, conf in k:
                        if conf >= .2: cv2.circle(img, (int(x),int(y)), 3, color, -1)
                canvas = np.full((720,1280,3), 25, np.uint8)
                scale = min(720/img.shape[1], 510/img.shape[0])
                small = cv2.resize(img, (round(img.shape[1]*scale),round(img.shape[0]*scale)))
                canvas[70:70+small.shape[0],:small.shape[1]] = small
                lines = [e['clip_id'], f"Original frame {frame_id} / decoded {i+1}",
                    'FRAME ONLY; physical FPS UNKNOWN' if mc else f'Source clock {t:.3f}s',
                    f"Detections {len(instances)} | target {tr['target_id']}",
                    f"Observed {row['target_observed']} | joints {row['reliable_joints']}/17",
                    f"Ankles reliable: {row['ankles_reliable']}", '',
                    f"V7 action: {prediction['label']}", f"V7 phase: {prediction['phase']}",
                    f"V3 action: {prediction.get('v3_label','disabled')}", '',
                    'Reference (not model input):', ref,
                    f"Track: {tr['tracking_reason']}"]
                for j, line in enumerate(lines):
                    cv2.putText(canvas, line, (750, 55+j*33), cv2.FONT_HERSHEY_SIMPLEX, .49,
                                (90,220,150) if j in [7,8] else (235,235,235), 1, cv2.LINE_AA)
                cv2.putText(canvas, e['clip_id'], (15,35), cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),1)
                cv2.putText(canvas, '30 FPS illustrative playback ONLY; action inference disabled' if mc else
                    f'Frozen Pose + ByteTrack + V7 | tracking: {args.tracking} | current/past only',
                    (15,625), cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
                cv2.putText(canvas, 'Green: target | Orange: eligible | Red: rejected | BT: raw ByteTrack ID',
                    (15,665),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
                # Repeat past observations on a 30 Hz display timeline; never interpolate poses.
                target_index = i if mc else round(t*30)
                while playback_index < target_index:
                    writer.write(previous if previous is not None else canvas)
                    playback_index += 1
                writer.write(canvas)
                playback_index += 1
                previous = canvas
                if i % 10 == 0 or (mc and 1000 <= frame_id <= 1300 and i % 3 == 0):
                    shots = dest / 'review'; shots.mkdir(exist_ok=True)
                    cv2.imwrite(str(shots/f'{frame_id:06d}.jpg'),canvas)
                if (i+1) % 250 == 0: print(e['clip_id'], i+1, flush=True)
        writer.release()
        assert len(rows) == e['frame_count'], (e['clip_id'], len(rows))
        summary = dict(clip=e['clip_id'], frames=len(rows), video_frames=playback_index, codec=codec,
            counts=dict(counts), phases=dict(phases), observed_target_ids=dict(tids), **stats)
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    assert pinned == {p:sha256(Path(p)) for p in pinned}
    (out/'summary.json').write_text(json.dumps(dict(model_hashes_unchanged=True, clips=summaries),indent=2))


if __name__ == '__main__':
    main()
