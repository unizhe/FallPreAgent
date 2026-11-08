"""Fall-centric shadow replay; cached perception/ST, identical primary events.

Cached inference duration only simulates ST availability, never actual alarm
output latency. Clinical/independent claims require reviewed exposure and events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.fall_event_metrics import evaluate_fall_events  # noqa: E402
from fallpre.weekly.stgcn_fall_evidence import FallTemporalEvidence  # noqa: E402
from fallpre.weekly.stgcn_temporal import TemporalPoseBuffer  # noqa: E402

OLD = ROOT / 'runs/stgcn-integration-20261029'
LABELS = ROOT / 'runs/stgcn-fusion-20261106/data/labels.provisional.json'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ground_truth(clip, recording_end):
    events = [dict(id=e['id'], motion_start_s=e.get('motion_start_s', e['window'][0]),
                   match_end_s=min(recording_end, e.get('recovery_start_s', e['window'][1])),
                   original_match_end_s=e.get('recovery_start_s', e['window'][1]),
                   right_censored=e.get('recovery_start_s', e['window'][1]) > recording_end,
                   review_status='ai_provisional', event_kind='demonstration_needs_review')
              for e in clip['floor_events']]
    normal = []
    for s in clip['segments']:
        if s['label'] in {'setup', 'other', 'unknown'} or s.get('context') == 'floor' or s.get('confidence') == 'low':
            continue
        u = s.get('boundary_uncertainty_s', clip['boundary_uncertainty_s'])
        start, end = s['start_s']+u, min(recording_end, s['end_s']-u)
        # Never infer normality from a model output or widen a sparse point label.
        if start < end and not any(start < e['match_end_s'] and end > e['motion_start_s'] for e in events):
            normal.append(dict(start_s=start, end_s=end, review_status='ai_provisional'))
    return events, normal


def run(out):
    target = out/'B'
    target.mkdir(parents=True, exist_ok=False)
    clips = read(OLD/'clips.json')
    labels = {c['id']: c for c in read(LABELS)['clips']}
    reports, provenance, counts = {}, [], Counter()
    for c in clips:
        cid = c['id']
        assert sha(c['baseline_output']) == c['baseline_output_sha256']
        assert sha(c['perception']) == c['perception_sha256']
        prediction_path = OLD/'protocol-production'/cid/'predictions.jsonl'
        sources, predictions = rows(c['baseline_output']), rows(prediction_path)
        branch, buffer = FallTemporalEvidence(), TemporalPoseBuffer()
        pending = latest = None
        primary_rows, temporal_rows, timeline = [], [], []
        for source, cached in zip(sources, predictions, strict=True):
            assert source['index'] == cached['index'] and source['source_time_s'] == cached['source_time_s']
            t, vt, result = source['source_time_s'], source['video_time_s'], source['result']
            before = json.dumps(result, sort_keys=True)
            action = result['action']
            index = action.get('target_index')
            person = result['persons'][index] if isinstance(index, int) and 0 <= index < len(result['persons']) else None
            state = buffer.observe(person, t, result['width'], result['height'],
                                   action.get('selection', {}).get('target_epoch'),
                                   camera_stable=result['quality'].get('camera_stable') is True)
            if pending and pending[0] <= t:
                latest = pending[1] if buffer.is_result_current(pending[1]) else None
                pending = None
            if state['status'] != 'ready':
                latest = None
            if cached['prediction']:
                assert pending is None, 'Needs full asynchronous scheduler replay'
                pending = (t+cached['prediction']['inference_ms']/1000, cached['prediction'])
            temporal = dict(state, top3=[])
            if latest and buffer.is_result_current(latest):
                temporal.update(latest, result_age_s=t-latest['window_end_s'])
            else:
                temporal['status'] = 'pending' if state['status'] == 'ready' else state['status']
            evidence = branch.update(result, t, temporal)
            assert before == json.dumps(result, sort_keys=True)
            assert evidence['changes_alarm'] is False
            counts[evidence.get('code', evidence.get('status'))] += 1
            f = result['fall']
            pr = dict(time_s=vt, observable=f.get('observed') is True and result['quality'].get('camera_stable') is True,
                      fall={k:f.get(k) for k in ('state','event_id','new_event','current_event_id','lifecycle')})
            primary_rows.append(pr)
            tq = evidence.get('observation_quality', {})
            tr = dict(pr, observable=tq.get('status') == 'observable'
                      and evidence.get('model_evidence', {}).get('status') == 'observed')
            temporal_rows.append(tr)
            timeline.append(dict(time_s=vt, source_time_s=t, action=action['label'], fall=pr['fall'],
                                 primary_observable=pr['observable'], temporal_evidence=evidence))
        recording_end = max(c['duration_s'], primary_rows[-1]['time_s'])
        events, normal = ground_truth(labels[cid], recording_end)
        report = evaluate_fall_events(primary_rows, events, normal, camera_id='unknown-camera',
                                     recording_start_s=0., recording_end_s=recording_end)
        tqreport = evaluate_fall_events(temporal_rows, events, normal, camera_id='unknown-camera',
                                       recording_start_s=0., recording_end_s=recording_end)
        opportunities = []
        for event in events:
            window = [r for r in timeline if event['motion_start_s'] <= r['time_s'] < event['match_end_s']]
            supported = [r for r in window if r['temporal_evidence'].get('support') is True]
            alarms = [r for r in window if r['fall'].get('new_event')]
            first_support = supported[0]['time_s'] if supported else None
            first_alarm = alarms[0]['time_s'] if alarms else None
            opportunities.append(dict(id=event['id'], first_support_s=first_support,
                first_primary_alarm_source_s=first_alarm,
                support_lead_vs_primary_source_s=(first_alarm-first_support
                    if first_alarm is not None and first_support is not None else None),
                interpretation='Descriptive evidence timing only; no earlier alarm was generated or validated'))
        reports[cid] = dict(title=c['title'], observations=len(timeline), primary_metrics=report,
                            recording_clock_extension_s=recording_end-c['duration_s'],
                            event_label_censoring=[e for e in events if e['right_censored']],
                            temporal_coverage=tqreport.get('coverage'),
                            evidence_opportunities=opportunities,
                            evidence_codes=dict(Counter(r['temporal_evidence'].get('code') for r in timeline)),
                            primary_events_unchanged=True, candidate_alarm_metrics_equal_A=True)
        (target/f'{cid}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in timeline))
        provenance.append(dict(clip=cid, baseline=c['baseline_output'], baseline_sha256=sha(c['baseline_output']),
                               temporal_prediction_sha256=sha(prediction_path)))
        print(cid, len(timeline), reports[cid]['evidence_codes'], flush=True)
    metrics = dict(objective='fall event detection', mode='shadow', primary_promoted=False,
        observations=sum(c['observations'] for c in clips), evidence_codes=dict(counts), clips=reports,
        provenance=provenance, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__), ROOT/'src/fallpre/weekly/stgcn_fall_evidence.py',ROOT/'src/fallpre/weekly/fall_event_metrics.py']},
        label_sha256=sha(LABELS), verified_false_alerts_per_camera_hour=None,
        output_alarm_delay_s=None, trained_weights=False,
        limitations=['No reviewed nursing-home monitoring exposure, true fall-process labels, or independent person/view videos.',
            'All ten old videos are development material. Demonstration events remain provisional.',
            'ST completion is simulated using historical model duration. Actual alarm output latency is unknown.',
            'Support/conflict observations are not alarms; no candidate alarm is invented from a score.',
            'Action metrics are regression diagnostics only; they are not the promotion objective.'])
    dump(target/'metrics.json', metrics)
    dump(out/'clips.json',[dict(id=c['id'],title=c['title']) for c in clips])
    media = out/'media'
    media.mkdir(exist_ok=True)
    for c in clips:
        dest=media/f'{c["id"]}.mp4'
        dest.symlink_to(os.path.relpath(OLD/c['id']/'baseline.mp4',media))
    build_page(out)


def build_page(out):
    (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跌倒事件时序证据审计</title><style>body{max-width:1200px;margin:24px auto;padding:20px;background:#112027;color:#e0ede7;font:16px system-ui}a{color:#9ed6bf}video{width:100%;max-height:65vh}pre{white-space:pre-wrap;font:14px system-ui}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}select{font:inherit;padding:10px}.card{padding:18px;background:#21343b;border-radius:8px}</style><h1>跌倒事件：A报警＋B时序证据</h1><p>养老院事件识别为目标；十段旧开发片尚不能验证养老院误报负担或迁移效果。B为旁路，原报警保持。</p><p><a href="B/metrics.json">事件指标与观测覆盖明细</a> · <a href="http://127.0.0.1:8810/">新实时证据旁路</a> · <a href="http://127.0.0.1:8780/">原实时基线</a></p><p>正式每小时误报率及报警输出延迟：未具备可靠标注/实际输出时点，留空。旧14段倒地演示只作暂定回归，不能等同养老院真实跌倒。</p><select id="clip"></select><video id="video" controls></video><div class="grid"><div class="card"><h2>A：原报警与事件管理</h2><pre id="a"></pre></div><div class="card"><h2>B：独立时序证据</h2><pre id="b"></pre></div></div><p>高分类分数不等于跌倒风险；骨架不足与没有跌倒分开。原视频标注框是A的结果，B不生成报警。</p><script>const $=id=>document.getElementById(id);let rows=[],g=0;async function load(){const n=++g,c=$('clip').value,t=await fetch('B/'+c+'.jsonl').then(x=>x.text());if(n!==g)return;rows=t.trim().split('\\n').map(JSON.parse);$('video').src='media/'+c+'.mp4'}function paint(){let r=null,t=$('video').currentTime;for(const x of rows){if(x.time_s>t)break;r=x}if(!r||t-r.time_s>.45){$('a').textContent=$('b').textContent='当前无返回观测';return}$('a').textContent=JSON.stringify({time_s:r.time_s,fall:r.fall,observable:r.primary_observable},null,2);$('b').textContent=JSON.stringify(r.temporal_evidence,null,2)}fetch('clips.json').then(x=>x.json()).then(cs=>{for(const c of cs){const o=document.createElement('option');o.value=c.id;o.textContent=c.title;$('clip').append(o)}load()});$('clip').onchange=load;$('video').ontimeupdate=paint;</script></html>''')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/stgcn-fall-20261107')
    run(parser.parse_args().output.resolve())
