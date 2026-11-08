"""Public cached regression plus explicitly synthetic extra-detection stress."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fallpre.weekly.webcam_actions import LiveActions, SinglePersonActions
from fallpre.weekly.webcam_target import candidate
from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.human import POSE_SHA256
OUT=ROOT/'runs/webcam-target-20261021'
model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json')
entries={e['clip_id']:e for e in json.load(open(ROOT/'configs/action-v7-dataset.json'))['entries']}
clips=['utk-s01_e01-transition','utk-s10_e01-transition','utk-s02_e01-transition','utk-s03_e01-transition','urfd-adl-07','urfd-adl-04','urfd-fall-01','tum-sitting-independent']
multicam=[json.loads(l) for l in (ROOT/'runs/public-samples-20261020/multicam-chute01-cam8/frames.jsonl').read_text().splitlines()]
chair=multicam[0]['instances'][1]
summary={}
for clip in clips:
 e=entries[clip]
 paths=[ROOT/f'runs/action-v{v}/poses/{clip}/observations.jsonl' for v in [3,2]]
 if clip=='tum-sitting-independent':paths.insert(0,ROOT/'runs/action-v4/demo-seated/fresh-poses/tum-sitting-independent/observations.jsonl')
 path=next(p for p in paths if p.exists());meta,*rows=[json.loads(l) for l in path.read_text().splitlines()]
 assert meta['weights_sha256']==POSE_SHA256 and meta['source_sha256']==e['video_sha256']
 cap=cv2.VideoCapture(str(ROOT/e['video_path']));w,h=[int(cap.get(k)) for k in [3,4]];cap.release()
 modes=['original','extra_candidate'] if clip in ['utk-s01_e01-transition','utk-s10_e01-transition','utk-s03_e01-transition'] else ['original']
 for mode in modes:
  old,new=SinglePersonActions(model),LiveActions(model);result=[]
  for row in rows:
   persons=[dict(box=p['box'],confidence=p['det'],keypoints=p['kpts']) for p in row['instances'] if p['det']>=.25]
   if mode=='extra_candidate':
    # Actual chair-generated pose from MultiCam, injected only into observations.
    # Input RGB is not changed; this is not an additional real-world video.
    distractor=dict(box=chair['box'],confidence=chair['det'],keypoints=chair['kpts'])
    persons=copy.deepcopy(persons+[distractor])
   before=old.update(persons,row['t'],w,h);after=new.update(persons,row['t'],w,h)
   assert (before['label'],before['raw_label'],before['phase'])==(after['previous_label'],after['previous_raw_label'],after['previous_phase'])
   if mode=='extra_candidate' and after['target_index'] is not None:assert after['target_index']!=len(persons)-1
   result.append(dict(t=row['t'],source_frame_id=row['source_frame_id'],persons=persons,before=before,after=after))
  ref=e.get('state_segments',[])
  if clip=='utk-s10_e01-transition':ref=[dict(start_s=122/30,end_s=223/30,state='seated')]
  if clip=='urfd-adl-07':ref=[dict(start_s=4.4,end_s=6,state='seated')]
  seated=[r for r in result if any(s['start_s']<=r['t']<s['end_s'] and s.get('state')=='seated' for s in ref)]
  assert not any(r['after']['label']=='standing_up' and r['before']['label']!='standing_up' for r in seated)
  item=dict(frames=len(result),before=dict(Counter(r['before']['label'] for r in result)),after=dict(Counter(r['after']['label'] for r in result)),
    selected=sum(r['after']['target_index'] is not None for r in result),selection_states=dict(Counter(r['after']['selection']['state'] for r in result)),
    epochs=sorted(set(r['after']['selection']['target_epoch'] for r in result)),
    seated_before=dict(Counter(r['before']['label'] for r in seated)),seated_after=dict(Counter(r['after']['label'] for r in seated)),
    added_seated_false_rise=0,synthetic_extra_detection=mode!='original')
  key=clip+'-'+mode;summary[key]=item;(OUT/(key+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(key,item,flush=True)
# MultiCam remains frame-level: no fabricated timestamps or temporal tracking.
checks=[]
for row in multicam:
 persons=[dict(box=p['box'],confidence=p['det'],keypoints=p['kpts']) for p in row['instances'] if p['det']>=.25]
 checks.append(dict(frame=row['original_frame_id'],candidates=[dict(box=p['box'],confidence=p['confidence'],eligible=(c:=candidate(p,640,480)) is not None,acquisition_ready=bool(c and c['strong'])) for p in persons]))
(OUT/'multicam-frame-quality.json').write_text(json.dumps(checks,indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
