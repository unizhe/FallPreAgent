import json,sys,time,zipfile,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'third_party/ultralytics'),str(root/'scripts')]
from benchmark_experiment3_v7 import decode,PREDICT_ARGS
from ultralytics import YOLO
import torch
pose=YOLO(str(root/'models/yolo11n-pose.pt')); detector=YOLO(str(root/'models/yolo11n.pt'))
with zipfile.ZipFile(root/'runs/dataset-survey-20261019/urfd-fall-01-rgb.zip') as z: img=decode(z,sorted(n for n in z.namelist() if n.endswith('.png'))[0])
for model in [pose,detector]:model.predict(img,**PREDICT_ARGS)
blocks=[]
for n in [8,2,2,8]:
 torch.set_num_threads(n)
 for model in [pose,detector]:model.predict(img,**PREDICT_ARGS)
 samples=[]
 for i in range(10):
  t=time.perf_counter();pose.predict(img,**PREDICT_ARGS);p=time.perf_counter();detector.predict(img,**PREDICT_ARGS);d=time.perf_counter()
  assert torch.get_num_threads()==n
  samples.append({'pose_s':p-t,'detector_s':d-p,'pair_s':d-t})
 blocks.append({'threads':n,'samples':samples,'median_pair_s':statistics.median(r['pair_s'] for r in samples)})
 print(n,blocks[-1]['median_pair_s'],flush=True)
out={'source_frame':1,'purpose':'Neutral opening-frame runtime probe; no event-boundary selection or threshold tuning','device':'cpu','order':[8,2,2,8],'warmup_per_block':1,'samples_per_block':10,'blocks':blocks,'median_pair_by_threads':{str(n):statistics.median(r['pair_s'] for b in blocks if b['threads']==n for r in b['samples']) for n in [8,2]}}
(root/'runs/abc-v7-20261020/latency-optimization-20261021/thread-probe.json').write_text(json.dumps(out,indent=2)+'\n')
