"""Paired fixed-V7 file-replay experiment: FIFO versus latest-arrived frame.

All model weights, evidence checks, thresholds and source clocks remain frozen.
The new behavior is opt-in scheduling, matching the existing live capture's
single pending slot. This is not a deployment or a sensor/actuator measurement.
"""
import argparse
import bisect
import csv
import json
import platform
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/ultralytics')]
from benchmark_experiment3_v7 import (CompletionModel,OnlineCompletion,PersonTracker,
 VerifiedPersonTracker,RefinedPoseRecovery,RTMOBackend,RTMPoseBackend,YOLO,
 PREDICT_ARGS,decode,pose_instances,detector_instances,predict_with_pose_quality,
 distribution,sha256,torch,cv2)
from fallpre.weekly.replay_scheduling import ReplaySchedule
from evaluate_abc_v7 import reference_at,sample_weights,LABELS

BASE=ROOT/'runs/abc-v7-20261020'
OUT=BASE/'latency-optimization-20261021'
CLIP='urfd-fall-01-original'


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def classify_output(stamp,bounds):
 if stamp is None:return 'miss_no_falling_output'
 if stamp<bounds[0]:return 'before_contact_certain'
 if stamp<=bounds[1]:return 'within_contact_uncertainty'
 return 'after_contact_certain'


def measure(mode,policy,repeat,out,times,names,input_path,cached,ann,pinned):
 dest=out/'trials'/f'{mode}-{policy}-r{repeat}'
 dest.mkdir(parents=True,exist_ok=False)
 model=CompletionModel.load(ROOT/'runs/action-v7/training/model.json')
 pose=YOLO(str(ROOT/'models/yolo11n-pose.pt'))
 detector=YOLO(str(ROOT/'models/yolo11n.pt')) if mode!='A' else None
 recovery=(RefinedPoseRecovery(backend=RTMOBackend(),refiner=RTMPoseBackend(),
  allow_partial_observation=True,allow_detector_proposals=True,resolve_native_ambiguity=True)
  if mode=='C' else None)
 with zipfile.ZipFile(input_path) as archive:
  image=decode(archive,names[0])
  warm=time.perf_counter()
  pose.predict(image,**PREDICT_ARGS)
  if detector:detector.predict(image,**PREDICT_ARGS)
  warm_wall=time.perf_counter()-warm
  assert torch.get_num_threads()==8,'Runtime changed relative to historical CPU baseline'
  tracker=VerifiedPersonTracker(30) if mode!='A' else PersonTracker(30)
  online=OnlineCompletion(model)
  schedule=ReplaySchedule(times,policy=policy)
  processed=[];skipped=[];previous_time=None
  origin=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
  print(json.dumps({'mode':mode,'policy':policy,'repeat':repeat,'status':'started'}),flush=True)
  with (dest/'frames.jsonl').open('x',buffering=1) as log:
   while not schedule.complete:
    select_wall=time.perf_counter()-origin
    sel=schedule.next(select_wall)
    if sel is None:
     delay=schedule.next_arrival_s-(time.perf_counter()-origin)
     if delay>0:time.sleep(delay)
     continue
    i=sel.index; t=times[i]; fid=i+1
    start=time.perf_counter()
    image=decode(archive,names[i]);decoded=time.perf_counter()
    predpose=pose.predict(image,**PREDICT_ARGS)[0]
    native=pose_instances(predpose)
    observation={'t':t,'source_frame_id':fid,'instances':native}
    pose_done=time.perf_counter()
    if detector:
     observation['detector_instances']=detector_instances(detector.predict(image,**PREDICT_ARGS)[0])
    detector_done=time.perf_counter()
    recovery_audit=None
    if recovery:observation,recovery_audit=recovery.augment(image,observation,tracker)
    recovered=time.perf_counter()
    tracked=tracker.update(observation);tracked_at=time.perf_counter()
    prediction=predict_with_pose_quality(online,tracked);finished=time.perf_counter()
    output_wall=finished-origin
    arrived_waiting=max(0,bisect.bisect_right(times,output_wall)-schedule.cursor)
    old=cached[mode][i]['prediction'] if mode=='A' else cached[mode][i]['after']['prediction']
    source_gap=None if previous_time is None else t-previous_time
    record={'mode':mode,'policy':policy,'repeat':repeat,'source_frame_id':fid,'source_time_s':t,
      'selection_wall_s':select_wall,'service_start_wall_s':start-origin,'output_wall_s':output_wall,
      'queue_wait_s':start-origin-t,'service_s':finished-start,'output_age_s':output_wall-t,
      'stages_s':{'decode':decoded-start,'pose':pose_done-decoded,'detector':detector_done-pose_done,
         'recovery':recovered-detector_done,'tracking':tracked_at-recovered,'action':finished-tracked_at},
      'dropped_before_this_frame':[j+1 for j in sel.dropped_indices],
      'arrived_unprocessed_before_selection':sel.arrived_pending_before,
      'pending_slots_at_output':min(1,arrived_waiting) if policy=='latest' else arrived_waiting,
      'arrived_not_yet_accounted_at_output':arrived_waiting,
      'source_gap_s':source_gap,'gap_exceeds_action_reset':source_gap is not None and source_gap>.35,
      'gap_exceeds_tracker_reset':source_gap is not None and source_gap>.5,
      'observed':bool(tracked['instances']),'target_id':tracked['target_id'],
      'target_epoch':tracked.get('target_epoch'),'target_index':tracked['target_index'],
      'tracking_reason':tracked['tracking_reason'],'prediction':prediction,
      'native_pose_instances':native,'pose_instances':observation['instances'],
      'detector_instances':observation.get('detector_instances'),'recovery':recovery_audit,
      'same_source_full_fifo_label_equal':prediction['label']==old['label'],
      'same_source_full_fifo_phase_equal':prediction['phase']==old['phase'],
      'same_source_full_fifo_prediction_equal':prediction==old}
    # No skipped observation is synthesized or used by tracker/model; this
    # bookkeeping takes place only after the selected frame's output exists.
    skipped.extend(record['dropped_before_this_frame']);processed.append(record)
    log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    previous_time=t
    assert torch.get_num_threads()==8
    if len(processed)%40==0:print(json.dumps({'trial':dest.name,'processed':len(processed),'last_frame':fid,'dropped':len(skipped),'elapsed_s':round(output_wall,3)}),flush=True)
 weights,_=sample_weights(times)
 ids={r['source_frame_id'] for r in processed};dropids=set(skipped)
 assert len(ids)+len(dropids)==len(times) and not ids&dropids and ids|dropids==set(range(1,len(times)+1))
 first=next((r for r in processed if r['prediction']['label']=='falling'),None)
 boundary=ann['event_landmarks']['contact']['time_interval_s']
 trigger=None
 if first:
  trigger={'frame':first['source_frame_id'],'source_time_s':first['source_time_s'],
    'output_wall_s':first['output_wall_s'],'source_margin_s':[v-first['source_time_s'] for v in boundary],
    'contact_margin_s':[v-first['output_wall_s'] for v in boundary],
    'queue_wait_s':first['queue_wait_s'],'service_s':first['service_s'],'output_age_s':first['output_age_s']}
 all_refs={i+1:reference_at(ann['action_segments'],i+1) for i in range(len(times))}
 all_refs={i:(v if v in LABELS[:-1] else None) for i,v in all_refs.items()}
 official_fall=next(s for s in ann['action_segments'] if s['label']=='falling')
 event_ids=set(range(official_fall['start_frame'],official_fall['end_frame']+1))
 event_hits=[r for r in processed if r['source_frame_id'] in event_ids and r['prediction']['label']=='falling']
 false_falling=[r['source_frame_id'] for r in processed if all_refs[r['source_frame_id']] not in [None,'falling'] and r['prediction']['label']=='falling']
 unscored_falling=[r['source_frame_id'] for r in processed if all_refs[r['source_frame_id']] is None and r['prediction']['label']=='falling']
 known=[r for r in processed if all_refs[r['source_frame_id']] is not None]
 matched=[r for r in known if all_refs[r['source_frame_id']]==r['prediction']['label']]
 unknown=[r for r in processed if r['prediction']['label']=='unknown']
 epochs=[r['target_epoch'] for r in processed if r['observed']]
 summary={'trial':dest.name,'mode':mode,'policy':policy,'repeat':repeat,'started_utc':started,
  'warmup_s':warm_wall,'source_observations':len(times),'processed_frames':len(processed),
  'dropped_frames':len(skipped),'dropped_source_frame_ids':skipped,'processed_fraction':len(processed)/len(times),
  'last_output_wall_s':processed[-1]['output_wall_s'],'source_span_s':times[-1]-times[0],
  'service':distribution([r['service_s'] for r in processed]),
  'queue_wait':distribution([r['queue_wait_s'] for r in processed]),
  'output_age':distribution([r['output_age_s'] for r in processed]),
  'stages':{k:distribution([r['stages_s'][k] for r in processed]) for k in processed[0]['stages_s']},
  'max_pending_slots':max(r['pending_slots_at_output'] for r in processed),
  'max_source_gap_s':max((r['source_gap_s'] or 0 for r in processed),default=0),
  'source_gaps_gt_0_35_s':sum(r['gap_exceeds_action_reset'] for r in processed),
  'source_gaps_gt_0_5_s':sum(r['gap_exceeds_tracker_reset'] for r in processed),
  'outputs_age_gt_0_35_s':sum(r['output_age_s']>.35 for r in processed),
  'outputs_age_gt_0_5_s':sum(r['output_age_s']>.5 for r in processed),
  'first_falling':trigger,'contact_result':classify_output(first['output_wall_s'] if first else None,boundary),
  'official_fall_event':{'start_frame':official_fall['start_frame'],'end_frame':official_fall['end_frame'],
    'total_reference_frames':len(event_ids),'processed_reference_frames':len(ids&event_ids),
    'dropped_reference_frames':len(dropids&event_ids),'falling_output_frames':[r['source_frame_id'] for r in event_hits],
    'hit':bool(event_hits),'reference_status':official_fall.get('status'),'human_reviewed':official_fall.get('human_reviewed',False)},
  'unknown_processed_frames':len(unknown),'unobserved_processed_frames':sum(not r['observed'] for r in processed),
  'epoch_transitions_among_observed':sum(a!=b for a,b in zip(epochs,epochs[1:])),
  'same_source_fifo_label_changes':[r['source_frame_id'] for r in processed if not r['same_source_full_fifo_label_equal']],
  'same_source_fifo_phase_changes':[r['source_frame_id'] for r in processed if not r['same_source_full_fifo_phase_equal']],
  'full_prediction_equality_count':sum(r['same_source_full_fifo_prediction_equal'] for r in processed),
  'processed_label_counts':dict(Counter(r['prediction']['label'] for r in processed)),
  'provisional_labeled_false_falling_frames':false_falling,'unscored_falling_frames':unscored_falling,
  'coverage':{'total_source_reference_samples':len(times),
    'processed_labeled_samples':len(known),'matched_processed_labeled_samples':len(matched),
    'all_labeled_source_samples':sum(v is not None for v in all_refs.values()),
    'dropped_labeled_samples':sum(all_refs[i] is not None for i in dropids),
    'matched_source_weight_s':sum(weights[r['source_frame_id']-1] for r in matched),
    'processed_source_weight_s':sum(weights[i-1] for i in ids),
    'dropped_source_weight_s':sum(weights[i-1] for i in dropids),
    'note':'Original 160-observation denominators retained; dropped frames are unavailable, never correct or interpolated. Action references remain provisional.'},
  'recovery_executed_frames':[r['source_frame_id'] for r in processed if r['recovery'] and r['recovery']['executed']],
  'runtime':{'torch_threads':torch.get_num_threads(),'opencv_threads':cv2.getNumThreads()},
  'immutable_hashes_match':all(sha256(Path(p))==h for p,h in pinned.items())}
 assert summary['immutable_hashes_match']
 write(dest/'summary.json',summary)
 print(json.dumps({'trial':dest.name,'processed':len(processed),'dropped':len(skipped),'first_falling':trigger,'contact_result':summary['contact_result']}),flush=True)
 return summary


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
 args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
 assert not (out/'protocol.json').exists(),'Use a new output directory to avoid overwriting an experiment'
 frozen=json.loads((BASE/'evaluation-protocol.json').read_text())
 historic=json.loads((BASE/'experiment3-20261021/benchmark/protocol.json').read_text())
 pinned=dict(historic['frozen_hashes'])
 annotation_path=BASE/'annotations/annotations.reviewed.json'
 annotation=json.loads(annotation_path.read_text());ann=next(e for e in annotation['entries'] if e['clip_id']==CLIP)
 pinned[str(annotation_path)]=sha256(annotation_path)
 pinned[str(Path(__file__))]=sha256(Path(__file__))
 for path in [out/'thread-probe.json',out/'diagnosis.json']:
  if path.exists():pinned[str(path)]=sha256(path)
 sched=ROOT/'src/fallpre/weekly/replay_scheduling.py';pinned[str(sched)]=sha256(sched)
 assert all(sha256(Path(p))==h for p,h in pinned.items())
 times=ann['clock']['source_times_s'];assert len(times)==160 and times[0]==0
 source=Path(frozen['source_inputs'][CLIP]['file'])
 with zipfile.ZipFile(source) as archive:names=sorted(n for n in archive.namelist() if n.endswith('.png'))
 cached={m:[json.loads(l) for l in Path(frozen['source_logs'][f'{m}/{CLIP}']['file']).read_text().splitlines()] for m in 'ABC'}
 for m,rs in cached.items():assert all(r['original_frame_id']==i+1 and r['source_time_s']==times[i] for i,r in enumerate(rs))
 order=[]
 for repeat in range(1,4):
  for gi,mode in enumerate('ABC'):
   for policy in (['fifo','latest'] if (repeat+gi)%2 else ['latest','fifo']):order.append({'mode':mode,'policy':policy,'repeat':repeat})
 spec={'schema':'fallpre.v7.queue_policies.v1','created_utc':datetime.now(timezone.utc).isoformat(),
  'clip_id':CLIP,'source_frames':160,'training':False,'model_sha256':frozen['model_sha256'],
  'change':'Only scheduling: FIFO all pending frames versus newest already-arrived source frame; max1 pending slot.',
  'runtime_unchanged':{'device':'cpu','torch_threads':8,'predict':PREDICT_ARGS},
  'rejected_runtime_probe':'First-frame paired pose+detect probe found2threads slower than8; no thread change adopted.',
  'group_meanings':frozen['groups'],'repeats_per_cell':3,'order':order,'frozen_hashes':pinned,
  'trigger':'First V7 label falling from start of complete source stream, no debounce or annotation gate.',
  'contact_interval_s':ann['event_landmarks']['contact']['time_interval_s'],
  'success':'first output wall offset<3.336s; 3.336..3.537 ambiguous; >3.537 late; absent trigger miss.',
  'measurement':'Monotonic local scheduled input arrival through file read/decode and complete processing to V7 output; excludes model load, first-image pose/detect warmup, physical capture, networking, delivery.',
  'latest_semantics':'Only already-arrived compressed source-frame references are selected; discarded frames are not decoded or used by models; sourceclock/IDs retained. This emulates existing live pending-slot replacement, not camera driver/capture timing.',
  'sampling':'New images, tracking, evidence recovery, and V7 recomputed on selected inputs; missing inputs never interpolated or fed as empty. Skips can change tracking histories and labels.',
  'warmup':'Same first-image pose/detector warm-up for each run, no state updates; C ONNX runtimes loaded but no recovery inference warm-up.',
  'quality':'Keep original160 denominators and officialfall83–112 reference; log skipids, source gaps, unknown, firstdetection, changes vs original fullFIFO. References unchanged and mostlyprovisional.',
  'host':{'platform':platform.platform(),'torch':torch.__version__},
  'interpretation':'One dev event with train overlap, three paired repeats for timing variability, no population/generalization claim; noactualnotification.'}
 write(out/'protocol.json',spec)
 (out/'annotations.reviewed.snapshot.json').write_bytes(annotation_path.read_bytes())
 summaries=[]
 for task in order:
  summaries.append(measure(**task,out=out,times=times,names=names,input_path=source,cached=cached,ann=ann,pinned=pinned))
  write(out/'progress.json',{'completed':len(summaries),'total':len(order),'trials':summaries})
 write(out/'results.json',{'schema':spec['schema'],'complete':True,'trials':summaries,
   'historical_benchmark':'../experiment3-20261021/benchmark/summary.json','all_frozen_hashes_unchanged':all(sha256(Path(p))==h for p,h in pinned.items())})

if __name__=='__main__':main()
