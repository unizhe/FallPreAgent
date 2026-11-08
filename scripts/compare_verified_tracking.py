"""Compare immutable pose observations with verified tracking; no training.

Reuses the previous run's exact YOLO-Pose observations and freshly runs the
frozen COCO object detector on every source frame. References are display only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/ultralytics')]
import cv2
import numpy as np
from replay_public_samples import frames
from fallpre.weekly.action_v7 import CompletionModel, OnlineCompletion
from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from fallpre.weekly.human import LINKS,open_video_writer,sha256
from ultralytics import YOLO


def panel(img, poses, chosen, accepted, ids, title, label, phase, reason, target):
    img=img.copy()
    for i,p in enumerate(poses):
        color=(90,220,120) if i==chosen else (0,185,245) if i in accepted else (80,80,240)
        b=list(map(int,p['box']));cv2.rectangle(img,tuple(b[:2]),tuple(b[2:]),color,2)
        text=f"BT {ids.get(i,'?')} {p['det']:.2f}" if i in accepted else 'REJECTED pose'
        cv2.putText(img,text,(b[0],max(14,b[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        if i in accepted:
            k=p['kpts']
            for a,b in LINKS:
                if min(k[a][2],k[b][2])>=.2:
                    cv2.line(img,tuple(map(int,k[a][:2])),tuple(map(int,k[b][:2])),color,2)
    canvas=np.full((660,640,3),24,np.uint8)
    scale=min(640/img.shape[1],450/img.shape[0])
    img=cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale)))
    canvas[40:40+img.shape[0],:img.shape[1]]=img
    cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.68,(240,240,240),2)
    for n,line in enumerate([f'Target {target} | observed {chosen is not None}',
        f'Action: {label} | phase: {phase}',reason,
        'Green target / orange other / red rejected']):
        cv2.putText(canvas,line,(10,510+n*32),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'runs/public-samples-20261020')
    p.add_argument('--output',type=Path,default=ROOT/'runs/verified-tracking-20261020')
    p.add_argument('--detector-cache',type=Path,
                   help='Reuse a prior comparison detector stream with identical weights and baseline hashes.')
    args=p.parse_args();base=args.baseline.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text())
    weights=ROOT/'models/yolo11n.pt';modelpath=ROOT/'runs/action-v7/training/model.json'
    lock=json.loads((ROOT/'models.lock.json').read_text())
    assert sha256(weights)==lock[weights.name]['sha256']
    hashes=dict(manifest['frozen_models'],**{str(weights):sha256(weights)})
    for path,digest in hashes.items():assert sha256(Path(path))==digest
    config=VerifiedPersonTracker().configuration()
    if args.detector_cache:
        cached=json.loads((args.detector_cache/'protocol.json').read_text())
        assert cached['frozen_models']==hashes
        for e in manifest['entries']:
            assert cached['baseline_pose_logs'][e['clip_id']]==sha256(base/e['clip_id']/'frames.jsonl')
    checkpoint=dict(baseline=str(base),config=config,frozen_models=hashes,
        tracking_code_sha256=sha256(ROOT/'src/fallpre/weekly/verified_tracking.py'),
        note='No training. Pose observations replayed exactly. ' + ('COCO detector cached from previous revision.' if args.detector_cache else 'COCO detector fresh.'),
        detector_cache=str(args.detector_cache) if args.detector_cache else None,
        baseline_pose_logs={e['clip_id']:sha256(base/e['clip_id']/'frames.jsonl') for e in manifest['entries']})
    (out/'protocol.json').write_text(json.dumps(checkpoint,indent=2))
    detector=YOLO(str(weights)) if not args.detector_cache else None;model=CompletionModel.load(modelpath)
    summaries=[]
    # Order does not enter the inference; each clip has fresh state.
    entries=sorted(manifest['entries'],key=lambda e:not e.get('frame_only',False))
    for e in entries:
        cid=e['clip_id'];dest=out/cid;dest.mkdir(exist_ok=False)
        assert sha256(Path(e.get('zip_path',e.get('video_path'))))==e['input_sha256']
        old=[json.loads(l) for l in (base/cid/'frames.jsonl').read_text().splitlines()]
        cache_rows=([json.loads(l) for l in (args.detector_cache/cid/'frames.jsonl').read_text().splitlines()]
                    if args.detector_cache else None)
        if cache_rows is not None: assert len(cache_rows)==len(old)
        mc=e.get('frame_only',False);tracker=VerifiedPersonTracker(e['fps'],frame_only=mc)
        online=None if mc else OnlineCompletion(model)
        writer,codec=open_video_writer(dest/'comparison.mp4',30,(1280,720))
        previous=None;play_index=0;count=0
        before=Counter();after=Counter();reasons=Counter();epochs=Counter()
        stats=dict(no_target_before=0,no_target_after=0,rejected_pose_candidates=0,
                   changed_action_frames=0,changed_phase_frames=0,continuity_only_frames=0)
        with (dest/'frames.jsonl').open('x') as log:
            for i,img in enumerate(frames(e)):
                row=old[i];poses=row['instances'];fid=row['original_frame_id']
                if cache_rows is not None:
                    assert cache_rows[i]['original_frame_id']==fid
                    assert cache_rows[i]['pose_instances']==poses
                    objects=cache_rows[i]['detector_instances']
                else:
                    r=detector.predict(img,imgsz=640,conf=.1,iou=.45,device='cpu',verbose=False)[0]
                    objects=[dict(class_name=r.names[int(k)],conf=float(s),box=b) for b,s,k in
                             zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist())]
                t=i/30 if mc else row['source_time_s']
                tr=tracker.update(dict(t=t,source_frame_id=fid,instances=poses,detector_instances=objects))
                pred=online.update(tr) if online else dict(label='disabled_frame_only',phase='disabled')
                oldids={int(k):v for k,v in row['detection_track_ids'].items()}
                oldchosen=next((j for j,tid in oldids.items() if tid==row['target_id']),None) if row['target_observed'] else None
                accepted={a['index'] for a in tr['admission'] if a['accepted']}
                record=dict(original_frame_id=fid,decoded_frame_1based=i+1,source_time_s=None if mc else t,
                    preview_time_s=i/30 if mc else t,reference=row['reference'],
                    detector_instances=objects,pose_instances=poses,
                    before=dict(target_id=row['target_id'],target_index=oldchosen,observed=row['target_observed'],
                                prediction=row['prediction']),
                    after=dict(target_id=tr['target_id'],target_epoch=tr['target_epoch'],target_index=tr['target_index'],
                        observed=bool(tr['instances']),raw_track_ids=tr['detection_track_ids'],
                        prediction=pred,reason=tr['tracking_reason'],admission=tr['admission']))
                log.write(json.dumps(record)+'\n')
                before[row['prediction']['label']]+=1;after[pred['label']]+=1;reasons[tr['tracking_reason']]+=1
                if tr['instances']:epochs[tr['target_epoch']]+=1
                stats['no_target_before']+=not row['target_observed'];stats['no_target_after']+=not tr['instances']
                stats['rejected_pose_candidates']+=len(poses)-len(accepted)
                stats['changed_action_frames']+=row['prediction']['label']!=pred['label']
                stats['changed_phase_frames']+=row['prediction']['phase']!=pred['phase']
                stats['continuity_only_frames']+=tr['tracking_reason']=='pose_continuity_only'
                canvas=np.full((720,1280,3),20,np.uint8)
                title=f'{cid} | original frame {fid}' + (' | FRAME ONLY' if mc else f' | source {t:.3f}s')
                cv2.putText(canvas,title,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
                canvas[50:710,:640]=panel(img,poses,oldchosen,set(range(len(poses))),oldids,
                    'BEFORE: pose + ByteTrack',row['prediction']['label'],row['prediction']['phase'],
                    row['tracking_reason'],row['target_id'])
                canvas[50:710,640:]=panel(img,poses,tr['target_index'],accepted,tr['detection_track_ids'],
                    'AFTER: verified admission + continuity',pred['label'],pred['phase'],
                    tr['tracking_reason'],tr['target_id'])
                goal=i if mc else round(t*30)
                while play_index<goal:
                    writer.write(previous if previous is not None else canvas);play_index+=1
                writer.write(canvas);play_index+=1;previous=canvas
                if i%50==0 or mc and fid in [851,1012,1080,1107,1109,1135,1252,1285,1386]:
                    shots=dest/'review';shots.mkdir(exist_ok=True);cv2.imwrite(str(shots/f'{fid:06d}.jpg'),canvas)
                count+=1
                if count%300==0:print(cid,count,flush=True)
        writer.release();assert count==len(old)==e['frame_count']
        summary=dict(clip=cid,frames=count,video_frames=play_index,codec=codec,before=dict(before),after=dict(after),
                     reasons=dict(reasons),observed_epochs=dict(epochs),**stats)
        (dest/'summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
        print(json.dumps(summary),flush=True)
    for path,digest in hashes.items():assert sha256(Path(path))==digest
    assert checkpoint['tracking_code_sha256']==sha256(ROOT/'src/fallpre/weekly/verified_tracking.py')
    (out/'summary.json').write_text(json.dumps(dict(models_unchanged=True,clips=summaries),indent=2))


if __name__=='__main__':main()
