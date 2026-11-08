"""Partial target presence versus strict pose completeness gate."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/ultralytics')]
import cv2
import numpy as np
from replay_public_samples import frames
from compare_verified_tracking import panel
from fallpre.weekly.action_v7 import CompletionModel,OnlineCompletion
from fallpre.weekly.verified_tracking import VerifiedPersonTracker
from fallpre.weekly.pose_recovery import RTMO_PATH,RTMO_SHA256
from fallpre.weekly.pose_refinement import RefinedPoseRecovery,RTMPOSE_PATH,RTMPOSE_SHA256,predict_with_pose_quality
from fallpre.weekly.human import open_video_writer,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'runs/pose-refinement-20261020')
    p.add_argument('--output',type=Path,default=ROOT/'runs/partial-observation-20261020')
    p.add_argument('--clip',action='append')
    args=p.parse_args();base=args.baseline.resolve();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((ROOT/'runs/public-samples-20261020/manifest.json').read_text())
    prior=json.loads((base/'protocol.json').read_text());hashes=dict(prior['frozen_models'])
    hashes[str(RTMO_PATH)]=RTMO_SHA256
    hashes[str(RTMPOSE_PATH)]=RTMPOSE_SHA256
    for path,digest in hashes.items():assert sha256(Path(path))==digest
    model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json');recovery=RefinedPoseRecovery()
    pinned_code={str(p):sha256(p) for p in [ROOT/'src/fallpre/weekly/pose_refinement.py',ROOT/'src/fallpre/weekly/pose_recovery.py',ROOT/'src/fallpre/weekly/verified_tracking.py']}
    protocol=dict(baseline=str(base),frozen_models=hashes,recovery_config=recovery.configuration(),
                  code_hashes=pinned_code,tracking_config=VerifiedPersonTracker().configuration(),
                  input_pose_mode='exact cached native YOLO-Pose and Detect; fresh current-image RTMO and RTMPose when eligible',
                  baseline_logs={e['clip_id']:sha256(base/e['clip_id']/'frames.jsonl') for e in manifest['entries']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summaries=[]
    for e in sorted(manifest['entries'],key=lambda e:not e.get('frame_only',False)):
        cid=e['clip_id']
        if args.clip and cid not in args.clip:continue
        dest=out/cid;dest.mkdir(exist_ok=False)
        assert sha256(Path(e.get('zip_path',e.get('video_path'))))==e['input_sha256']
        old=[json.loads(l) for l in (base/cid/'frames.jsonl').read_text().splitlines()]
        mc=e.get('frame_only',False);tracker=VerifiedPersonTracker(e['fps'],frame_only=mc)
        online=None if mc else OnlineCompletion(model)
        writer,codec=open_video_writer(dest/'comparison.mp4',30,(1280,720))
        oldcounts,newcounts,reasons=Counter(),Counter(),Counter()
        stats=dict(missing_before=0,missing_after=0,recovery_attempt_frames=0,recovery_added_frames=0,
                   recovery_selected_frames=0,changed_actions=0,changed_phases=0,total_rtmo_passes=0,refined_frames=0,refinement_attempts=0,total_rtmpose_passes=0,presence_only_frames=0)
        timings=[];previous=None;play_index=0;count=0
        with (dest/'frames.jsonl').open('x') as f:
            for i,img in enumerate(frames(e)):
                oldrow=old[i];bef=oldrow['after'];fid=oldrow['original_frame_id']
                t=i/30 if mc else oldrow['source_time_s']
                row=dict(t=t,source_frame_id=fid,instances=oldrow['native_pose_instances'],detector_instances=oldrow['detector_instances'])
                start=time.perf_counter();aug,audit=recovery.augment(img,row,tracker);elapsed=time.perf_counter()-start
                tr=tracker.update(aug);pred=predict_with_pose_quality(online,tr)
                chosen=tr['target_index'];selected=bool(chosen is not None and aug['instances'][chosen].get('recovered'))
                after=dict(target_index=chosen,target_id=tr['target_id'],target_epoch=tr['target_epoch'],
                    observed=bool(tr['instances']),reason=tr['tracking_reason'],raw_track_ids=tr['detection_track_ids'],
                    admission=tr['admission'],prediction=pred,recovered_selected=selected,refined=bool(chosen is not None and aug['instances'][chosen].get('refined')),presence_only=bool(chosen is not None and aug['instances'][chosen].get('action_eligible') is False))
                record=dict(original_frame_id=fid,decoded_frame_1based=i+1,source_time_s=None if mc else t,
                    preview_time_s=oldrow['preview_time_s'],reference=oldrow['reference'],
                    before=bef,after=after,pose_instances=aug['instances'],native_pose_instances=row['instances'],
                    detector_instances=row['detector_instances'],recovery=dict(audit,wall_s=elapsed))
                f.write(json.dumps(record)+'\n')
                oldcounts[bef['prediction']['label']]+=1;newcounts[pred['label']]+=1;reasons[audit['reason']]+=1
                stats['presence_only_frames']+=after['presence_only']
                stats['refined_frames']+=after['refined']
                stats['refinement_attempts']+=audit['refinement']['executed']
                stats['total_rtmpose_passes']+=audit['refinement']['inference_passes']
                stats['missing_before']+=not bef['observed'];stats['missing_after']+=not after['observed']
                stats['recovery_attempt_frames']+=audit['executed'];stats['recovery_added_frames']+=audit['accepted']
                stats['recovery_selected_frames']+=selected;stats['total_rtmo_passes']+=audit['inference_passes']
                stats['changed_actions']+=bef['prediction']['label']!=pred['label']
                stats['changed_phases']+=bef['prediction']['phase']!=pred['phase']
                if audit['executed']:timings.append(elapsed)
                canvas=np.full((720,1280,3),20,np.uint8)
                title=f'{cid} | original frame {fid}' + (' | FRAME ONLY' if mc else f' | source {t:.3f}s')
                cv2.putText(canvas,title,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
                canvas[50:710,:640]=panel(img,oldrow['pose_instances'],bef['target_index'],
                    {a['index'] for a in bef['admission'] if a['accepted']},
                    {int(k):v for k,v in bef['raw_track_ids'].items()},'BEFORE: strict pose gate',
                    bef['prediction']['label'],bef['prediction']['phase'],bef['reason'],bef['target_id'])
                canvas[50:710,640:]=panel(img,aug['instances'],chosen,
                    {a['index'] for a in tr['admission'] if a['accepted']},tr['detection_track_ids'],
                    'AFTER: partial person evidence',pred['label'],pred['phase'],
                    ('PRESENCE ONLY / action withheld' if after['presence_only'] else 'PARTIAL / RTMPose consensus' if after['refined'] else 'RTMO '+audit.get('selected_view','')) if selected else 'YOLO / '+audit['reason'],tr['target_id'])
                goal=i if mc else round(t*30)
                while play_index<goal:writer.write(previous if previous is not None else canvas);play_index+=1
                writer.write(canvas);play_index+=1;previous=canvas
                if i%50==0 or audit['executed']:
                    review=dest/'review';review.mkdir(exist_ok=True);cv2.imwrite(str(review/f'{fid:06d}.jpg'),canvas)
                count+=1
                if count%300==0:print(cid,count,flush=True)
        writer.release();assert count==len(old)==e['frame_count']
        s=dict(clip=cid,frames=count,video_frames=play_index,codec=codec,before=dict(oldcounts),after=dict(newcounts),
               recovery_reasons=dict(reasons),timing_median_s=float(np.median(timings)) if timings else None,
               timing_max_s=max(timings) if timings else None,**stats)
        (dest/'summary.json').write_text(json.dumps(s,indent=2));summaries.append(s);print(json.dumps(s),flush=True)
    for path,digest in {**hashes,**pinned_code}.items():assert sha256(Path(path))==digest
    (out/'summary.json').write_text(json.dumps(dict(models_unchanged=True,clips=summaries),indent=2))


if __name__=='__main__':main()
