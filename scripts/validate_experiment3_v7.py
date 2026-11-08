"""Cross-artifact integrity and timing audit; no neural inference."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/abc-v7-20261020'
OUT=BASE/'experiment3-20261021'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l]

def main():
 checks={}
 frozen=read(BASE/'evaluation-protocol.json')
 current=BASE/'annotations/annotations.reviewed.json'
 snap=OUT/'annotations.reviewed.snapshot.json'
 annotation=read(snap)
 checks['snapshot_equals_current_review']=sha(current)==sha(snap)
 checks['original_annotation_immutable']=sha(BASE/'annotations/annotations.json')==frozen['annotations_sha256']
 checks['24_source_logs_immutable']=all(sha(Path(v['file']))==v['sha256'] for v in frozen['source_logs'].values())
 checks['same_v7_immutable']=sha(ROOT/'runs/action-v7/training/model.json')==frozen['model_sha256']
 session=read(BASE/'human-review/review-session.json')
 checks['four_decisions_two_proxy_outcomes']=session['completed_cases']==4 and len(session['proxy_only_cases'])==2
 checks['reviewed_hash_matches_session']=sha(current)==session['annotation_sha256_current']
 original=next(e for e in annotation['entries'] if e['clip_id']=='urfd-fall-01-original')
 bp=read(OUT/'benchmark/protocol.json')
 checks['benchmark_contact_matches_reviewed_snapshot']=bp['contact_interval_source_s']==original['event_landmarks']['contact']['time_interval_s']
 checks['benchmark_stable_matches_reviewed_snapshot']=bp['stable_onset_interval_source_s']==original['event_landmarks']['stable_lying']['time_interval_s']
 for e in annotation['entries']:
  if e['clip_id'].startswith('utk-'):
   seat=e['event_landmarks']['seat_off']
   checks[e['clip_id']+'_physical_boundary_unknown']=all(seat[k] is None for k in ['frame','time_s','frame_interval','time_interval_s']) and not seat['human_reviewed']
 bm=read(OUT/'benchmark/summary.json')
 for g in bm['groups']:
  mode=g['group'];actual=rows(OUT/f'benchmark/{mode}.frames.jsonl')
  cached=rows(Path(frozen['source_logs'][f'{mode}/urfd-fall-01-original']['file']))
  checks[mode+'_complete_prediction_dict_matches_cache']=len(actual)==len(cached)==160 and all(a['prediction']==(c['prediction'] if mode=='A' else c['after']['prediction']) for a,c in zip(actual,cached))
  checks[mode+'_local_fifo_identity']=all(abs(r['queue_wait_s']+r['service_wall_s']-r['local_arrival_to_output_s'])<1e-8 for r in actual)
  first=next(r for r in actual if r['prediction']['label']=='falling')
  checks[mode+'_first_falling87']=first['original_frame_id']==87
  checks[mode+'_output_after_contact_upper']=first['action_output_wall_offset_s']>bp['contact_interval_source_s'][1]
 checks['benchmark_validation29_pass']=read(OUT/'benchmark/validation.json')['passed']
 checks['cache_validation_pass']=read(OUT/'cache-validation.json')['status']=='passed'
 suite=ET.parse(OUT/'timing-tests.xml').getroot()
 checks['eight_timing_tests_pass']=sum(int(x.get('tests',0)) for x in suite.iter('testsuite'))==8 and all(int(x.get('failures',0))==int(x.get('errors',0))==0 for x in suite.iter('testsuite'))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in {'src','href'} and v and not v.startswith(('http','#','data:')):self.refs.append(v)
 page=Links();page.feed((OUT/'index.html').read_text())
 missing=[x for x in page.refs if x.split('#')[0]!='validation.json' and not (OUT/x.split('#')[0]).exists()]
 checks['all_report_local_links_exist']=not missing
 evidence={str(p.relative_to(ROOT)):sha(p) for p in [snap,OUT/'timing.json',OUT/'protocol.json',OUT/'benchmark/protocol.json',OUT/'benchmark/summary.json',OUT/'cache-validation.json',OUT/'timing-tests.xml',OUT/'index.html',OUT/'source-timeline.png',OUT/'queue-latency.png',ROOT/'scripts/evaluate_experiment3_v7.py',ROOT/'scripts/report_experiment3_v7.py',ROOT/'scripts/plot_experiment3_v7.py',Path(__file__)]}
 result={'schema':'fallpre.experiment3.integrated_validation.v1','passed':all(checks.values()),'checks':checks,'count':len(checks),'missing_links':missing,'annotation_snapshot_sha256':sha(snap),'evidence_hashes':evidence,'scope':'7449 existing log observations audited; 480 new visual inference frames only to fill missing full-pipeline timing. Complete prediction dictionaries match for those480; no claim of full trajectory-field equality. Prefix tests cover the scanner, while original full causal replay evidence remains in experiment-one audit.','delivery_latency_measured':False,'physical_sensor_latency_measured':False}
 (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 assert result['passed'],[k for k,v in checks.items() if not v]
 print(json.dumps({'passed':result['passed'],'checks':len(checks),'prediction_records_verified':480,'snapshot':sha(snap)}))

if __name__=='__main__':main()
