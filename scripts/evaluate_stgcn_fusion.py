"""A/B regression on immutable development traces; no detector rerun or label tuning.

The same ConstrainedSTGCNFusion runs here and in webcam_pose_demo. Temporal
outputs become available at the next observation after their recorded inference
duration. This is a source-time simulation, not a capture/HTTP latency benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.stgcn_fusion import ConstrainedSTGCNFusion, FusionConfig  # noqa: E402
from fallpre.weekly.stgcn_temporal import TemporalConfig, TemporalPoseBuffer  # noqa: E402

OLD = ROOT / 'runs/stgcn-integration-20261029'
DEFAULT = ROOT / 'runs/stgcn-fusion-20261106'
CORE = {'standing', 'seated', 'sitting_down', 'standing_up', 'bending', 'squatting'}
TRANSITIONS = {'sitting_down', 'standing_up', 'squatting_down', 'squat_rising', 'straightening'}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latency(values):
    return {'n': len(values), 'mean_s': statistics.mean(values) if values else None,
            'p95_s': sorted(values)[min(len(values)-1, int(.95*len(values)))] if values else None}


def classify_metrics(samples, key):
    matrix = defaultdict(Counter)
    for r in samples:
        if r['expected'] is not None:
            matrix[r['expected']][r[key]] += 1
    labels = sorted(set(matrix) | {p for c in matrix.values() for p in c})
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(c[label] for c in matrix.values())
        per_class[label] = dict(support=support, predicted=predicted,
                               precision=tp/predicted if predicted else None,
                               recall=tp/support if support else None)
    return {'scored_samples': sum(sum(c.values()) for c in matrix.values()),
            'confusion_matrix': {k: dict(v) for k, v in matrix.items()}, 'per_class': per_class,
            'unknown_samples_all': sum(r[key] == 'unknown' for r in samples),
            'unknown_rate_all': sum(r[key] == 'unknown' for r in samples)/len(samples),
            'stable_unknown_s': {label: sum(r['scored_duration_s'] for r in samples
                                         if r['expected'] == label and r[key] == 'unknown')
                                 for label in ('standing', 'seated')}}


def transition_metrics(samples, segments):
    checks = []
    for s in segments:
        label = s.get('proposed_home_label') or s['label']
        if label not in TRANSITIONS:
            continue
        in_segment = [r for r in samples if s['start_s'] <= r['video_time_s'] < s['end_s']]
        check = dict(id=s.get('id'), label=label, start_s=s['start_s'], end_s=s['end_s'],
                     boundary_uncertainty_s=s.get('boundary_uncertainty_s', .3))
        for key in ('A', 'B'):
            hits = [r for r in in_segment if r[key] == label]
            check[key] = dict(hit=bool(hits), first_delay_s=hits[0]['video_time_s']-s['start_s'] if hits else None)
        checks.append(check)
    return checks


def event_metrics(samples, events):
    independent = []
    seen = set()
    repeated_emissions = []
    for r in samples:
        f = r['fall']
        if f.get('new_event') and f.get('event_id') in seen:
            repeated_emissions.append(dict(event_id=f['event_id'], time_s=r['video_time_s']))
        if f.get('new_event') and f.get('event_id') is not None and f['event_id'] not in seen:
            independent.append(dict(event_id=f['event_id'], time_s=r['video_time_s']))
            seen.add(f['event_id'])
    checks, assigned = [], set()
    ordered = sorted(events, key=lambda e: e.get('motion_start_s', e['window'][0]))
    for i, e in enumerate(ordered):
        start = e.get('motion_start_s', e['window'][0])
        end = e.get('recovery_start_s', e['window'][1])
        hits = [r for r in independent if start <= r['time_s'] < end and r['event_id'] not in assigned]
        assigned.update(r['event_id'] for r in hits)
        recovery_limit = (ordered[i+1].get('motion_start_s', ordered[i+1]['window'][0])
                          if i+1 < len(ordered) else samples[-1]['video_time_s'])
        normal = next((r['video_time_s'] for r in samples
                       if end <= r['video_time_s'] <= recovery_limit and r['fall'].get('state') == 'normal'), None)
        checks.append(dict(id=e['id'], start_s=start, recovery_start_s=end, hit=bool(hits),
                           first_confirmation_delay_s=hits[0]['time_s']-start if hits else None,
                           duplicate_new_events=max(0, len(hits)-1),
                           recovery_to_normal_delay_s=normal-end if normal is not None else None,
                           recovery_observed=normal is not None, recovery_censored_at_s=recovery_limit))
    normal_events = [e for e in independent if any(r['video_time_s'] == e['time_s'] and
                    r['expected'] in CORE | TRANSITIONS | {'walking', 'walking_turning', 'turning'}
                    and r.get('context') != 'floor' for r in samples)
                    and e['event_id'] not in assigned]
    return dict(independent_new_red_events=independent, events=checks,
                duplicate_emissions_same_event_id=repeated_emissions,
                normal_activity_false_events=normal_events,
                red_samples=sum(r['fall'].get('state') == 'confirmed' for r in samples),
                red_continuation_samples=sum(r['fall'].get('state') == 'confirmed' and
                                                 not r['fall'].get('new_event') for r in samples),
                unscored_new_events=[e for e in independent if e['event_id'] not in assigned
                                    and e not in normal_events])


def evaluate(out):
    destination = out / 'B'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'metrics.json').exists():
        raise FileExistsError('Use a new output directory; previous experiments are immutable')
    # Prespecified before reading predictions; no threshold selection on these clips.
    dump(destination / 'protocol.json', dict(fusion=asdict(FusionConfig()), temporal=asdict(TemporalConfig()),
        all_videos_development=True, human_labels_required_for_promotion=True,
        promotion=dict(strict_core_improvement=True, no_core_class_precision_or_recall_regression=True,
                       no_fall_event_or_recovery_regression=True, independent_new_video_validation=True,
                       realtime_http_validation=True),
        timing='past-only windows; publish after recorded inference duration, at next observation',
        metrics='same full observations; interval interiors use original boundary uncertainty; missing stays in denominator',
        not_measured='live capture/display latency; actual risk; independent generalization'))
    inventory = {c['id']: c for c in read(out / 'data/labels.provisional.json')['clips']}
    clips = read(OLD / 'clips.json')
    all_samples, summaries, provenance = [], {}, []
    for c in clips:
        cid = c['id']
        source = Path(c['baseline_output'])
        assert sha(source) == c['baseline_output_sha256'], f'{cid}: baseline changed'
        assert sha(c['perception']) == c['perception_sha256'], f'{cid}: skeleton input changed'
        pred_path = OLD / 'protocol-production' / cid / 'predictions.jsonl'
        baseline, predictions = rows(source), rows(pred_path)
        annotations = inventory[cid]
        segments = annotations['segments']
        fusion, buffer = ConstrainedSTGCNFusion(), TemporalPoseBuffer()
        pending = latest = None
        samples, detail = [], []
        for i, (r, p) in enumerate(zip(baseline, predictions, strict=True)):
            assert r['index'] == p['index'] and r['source_time_s'] == p['source_time_s']
            t, vt, result = r['source_time_s'], r['video_time_s'], r['result']
            original = json.dumps(result, sort_keys=True)
            action = result['action']
            index = action.get('target_index')
            person = result['persons'][index] if isinstance(index, int) and 0 <= index < len(result['persons']) else None
            state = buffer.observe(person, t, result['width'], result['height'],
                                   action.get('selection', {}).get('target_epoch'),
                                   camera_stable=result.get('quality', {}).get('camera_stable') is True)
            if pending and pending[0] <= t:
                latest = pending[1] if buffer.is_result_current(pending[1]) else None
                pending = None
            if state['status'] != 'ready':
                latest = None
            if p['prediction'] is not None:
                assert pending is None, 'Recorded inference exceeds schedule; requires full async replay'
                pending = (t + p['prediction']['inference_ms']/1000, p['prediction'])
            temporal = dict(state, top3=[], changes_primary_decision=False)
            if latest and buffer.is_result_current(latest):
                temporal.update(latest, result_age_s=t-latest['window_end_s'])
            else:
                temporal['status'] = 'pending' if state['status'] == 'ready' else state['status']
            fused = fusion.update(result, t, temporal)
            assert original == json.dumps(result, sort_keys=True), 'Fusion modified original output'
            assert fused['changes_primary_decision'] is False and fused['changes_alarm'] is False
            s = next((s for s in segments if s['start_s'] <= vt < s['end_s']), None)
            u = s.get('boundary_uncertainty_s', annotations['boundary_uncertainty_s']) if s else 0
            interior = bool(s and s['start_s'] + u <= vt < s['end_s'] - u and
                            s.get('confidence') != 'low' and s.get('context') != 'floor')
            truth = (s.get('proposed_home_label') or s['label']) if interior else None
            if truth in {'setup', 'other', 'unobservable'}:
                truth = None
            dt = max(0., min(baseline[i+1]['video_time_s'] if i+1 < len(baseline) else c['duration_s'],
                             s['end_s'] - u if s else vt) - vt)
            row = dict(clip=cid, index=r['index'], video_time_s=vt, source_time_s=t,
                       A=action['label'], B=fused['label'], expected=truth,
                       provisional_label=s['label'] if s else None, context=s.get('context') if s else None,
                       scored_duration_s=dt if truth is not None and dt <= .45 else 0.,
                       fall={k: result['fall'].get(k) for k in ('state', 'event_id', 'new_event', 'lifecycle')},
                       temporal_ready=temporal['status'] == 'ready', fused_action=fused,
                       top3=temporal.get('top3', []), temporal_age_s=temporal.get('result_age_s'))
            samples.append(row)
            detail.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
        (destination / f'{cid}.jsonl').write_text('\n'.join(detail) + '\n')
        event = event_metrics(samples, annotations['floor_events'])
        transitions = transition_metrics(samples, segments)
        summaries[cid] = dict(A=classify_metrics(samples, 'A'), B=classify_metrics(samples, 'B'),
            observations=len(samples), differences=[r for r in samples if r['A'] != r['B']],
            decisions=dict(Counter(r['fused_action'].get('code') for r in samples)),
            conflicts=[dict(time_s=r['video_time_s'], conflicts=r['fused_action'].get('conflicts'),
                            expected=r['expected']) for r in samples if r['fused_action'].get('conflicts')],
            transitions=transitions, fall=event)
        provenance.append(dict(clip=cid, baseline=str(source), baseline_sha256=sha(source),
                               temporal_predictions=str(pred_path), temporal_sha256=sha(pred_path)))
        all_samples.extend(samples)
        print(cid, len(samples), 'changes', len(summaries[cid]['differences']), flush=True)
    transitions = [dict(clip=c, **e) for c, v in summaries.items() for e in v['transitions']]
    events = [dict(clip=c, **e) for c, v in summaries.items() for e in v['fall']['events']]
    summary = dict(observations=len(all_samples), clips=len(clips),
                   A=classify_metrics(all_samples, 'A'), B=classify_metrics(all_samples, 'B'),
                   changed_samples=sum(r['A'] != r['B'] for r in all_samples),
                   temporal_ready_samples=sum(r['temporal_ready'] for r in all_samples),
                   conflicts=sum(bool(r['fused_action'].get('conflicts')) for r in all_samples),
                   fall_events=len(events), fall_event_hits=sum(e['hit'] for e in events),
                   fall_misses=sum(not e['hit'] for e in events),
                   fall_first_confirmation_delay=latency([e['first_confirmation_delay_s'] for e in events if e['hit']]),
                   event_duplicates=sum(e['duplicate_new_events'] for e in events),
                   duplicate_emissions_same_event_id=sum(len(v['fall']['duplicate_emissions_same_event_id'])
                                                         for v in summaries.values()),
                   fall_outputs_equal=True, primary_inputs_unchanged=True)
    for key in ('A', 'B'):
        subset = [e for e in transitions if e['label'] in {'sitting_down', 'standing_up'}]
        summary[key]['sit_stand_segments'] = len(subset)
        summary[key]['sit_stand_hits'] = sum(e[key]['hit'] for e in subset)
        summary[key]['sit_stand_misses'] = sum(not e[key]['hit'] for e in subset)
        summary[key]['sit_stand_first_delay'] = latency([e[key]['first_delay_s'] for e in subset if e[key]['hit']])
    report = dict(summary=summary, clips=summaries, provenance=provenance,
                  labels_sha256=sha(out / 'data/labels.provisional.json'),
                  fusion_source_sha256=sha(ROOT / 'src/fallpre/weekly/stgcn_fusion.py'),
                  evaluation_source_sha256=sha(__file__),
                  promotion=dict(approved=False, reason='No human-reviewed labels or sealed independent test; remain shadow'),
                  limitations=['All ten clips are development data; interval metrics are AI-label agreement, not accuracy.',
                    'Point-only clips have no invented dense labels; floor events are provisional demonstrations.',
                    'This reruns fusion on cached main and pretrained ST outputs; no YOLO or ST inference rerun.',
                    'Measured inference completion gating approximates asynchronous availability; HTTP measured separately.',
                    'A and B share exact input observations including failures. Alarm is never overridden.',
                    'Delay means include hits only; misses and per-event rows are reported alongside.',
                    'Segment hits include provisional boundary regions for historical comparison; sample metrics exclude uncertain boundaries.',
                    'Red continuation counts all retained red frames, not necessarily an abnormal old-event tail.'])
    dump(destination / 'metrics.json', report)
    build_page(out, clips, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_page(out, clips, summary):
    media = out / 'media'
    media.mkdir(exist_ok=True)
    for c in clips:
        path = media / f'{c["id"]}.mp4'
        if not path.exists():
            path.symlink_to(os.path.relpath(OLD / c['id'] / 'baseline.mp4', media))
    dump(out / 'clips.json', [{'id': c['id'], 'title': c['title']} for c in clips])
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO＋ST-GCN++ 有约束融合 A/B</title>
<style>body{font:16px system-ui;background:#101c22;color:#e1edeb;max-width:1280px;margin:auto;padding:24px}a{color:#9bdec4}select,button{font:inherit;padding:10px;background:#263b43;color:white;border:1px solid #60787b;border-radius:6px}video{width:100%;max-height:60vh;background:#000}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px system-ui}.card{background:#1c2d34;padding:16px;border-radius:8px}td,th{text-align:left;padding:8px;border-bottom:1px solid #53676a}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>YOLO＋ST-GCN++：有约束融合候选</h1><p>十段开发录像 · AI 暂定标签 · B 保持旁路 · 未证明独立泛化</p>
<p id="summary"></p><p><a href="B/metrics.json">完整指标、逐类混淆与差异</a> · <a href="data/review-queue.json">待人工复核</a> · <a href="http://127.0.0.1:8809/">候选实时页面</a> · <a href="http://127.0.0.1:8780/">保留的实时基线</a></p>
<label>录像 <select id="clip"></select></label><p>画面为保留的 A 主识别标注视频；下面同步显示 A 与 B。B 不改变报警和红框。候选结果在模拟推理返回后才可见；这不是实时端到端测量。</p>
<div class="grid"><div><p>A · 保留的主识别框</p><video id="video" controls preload="metadata"></video></div><div><p>B · 同一主识别框＋旁路候选</p><canvas id="candidate" style="width:100%;background:#000"></canvas></div></div><div class="grid"><section class="card"><h2>A · 当前基线</h2><pre id="a"></pre></section><section class="card"><h2>B · 融合候选</h2><pre id="b"></pre></section></div><h2>优先人工复核</h2><div id="review"></div><p>核对当前动作阶段和边界；不确定时保留不确定。旧预标注不能直接作为训练真值。</p><h2>差异与冲突区间</h2><div id="changes"></div>
<script>const $=id=>document.getElementById(id);let rows=[],generation=0;const names={standing:'站立',seated:'坐姿',standing_up:'从坐姿起身',sitting_down:'坐下',squatting:'蹲姿',bending:'弯腰',unknown:'未知'};
async function load(){const g=++generation,c=$('clip').value,text=await fetch('B/'+c+'.jsonl').then(r=>r.text());if(g!==generation)return;rows=text.trim().split('\\n').map(JSON.parse);$('video').src='media/'+c+'.mp4';$('changes').replaceChildren();let last=-10;for(const r of rows){if((r.A!==r.B||r.fused_action.conflicts?.length)&&r.video_time_s-last>1){last=r.video_time_s;const b=document.createElement('button');b.textContent=r.video_time_s.toFixed(2)+'s '+(r.A!==r.B?'候选变化':'冲突保留');b.onclick=()=>{$('video').currentTime=r.video_time_s;paint()};$('changes').append(b)}}if(!$('changes').children.length)$('changes').textContent='本片无候选改写或显式冲突。';paint()}
function paint(){const t=$('video').currentTime;let r=null;for(const x of rows){if(x.video_time_s>t)break;r=x}if(!r||t-r.video_time_s>.45){$('a').textContent=$('b').textContent='此时没有有效返回观测';return}$('a').textContent=(names[r.A]||r.A)+'\\n报警：'+r.fall.state+' · '+(r.fall.event_id||'无事件')+'\\n源录像：'+r.video_time_s.toFixed(2)+'s\\n暂定标签：'+(r.expected||'边界不确定/未标注');$('b').textContent=(names[r.B]||r.B)+'\\n'+r.fused_action.reason+'\\n'+r.fused_action.code+'\\n冲突：'+(r.fused_action.conflicts?.map(x=>x.code).join('、')||'无')+'\\nST：'+r.top3.map(x=>x.label_zh+' '+x.score.toFixed(3)).join(' / ')+'\\n主判断和报警保持 A';}
Promise.all([fetch('clips.json').then(r=>r.json()),fetch('B/metrics.json').then(r=>r.json())]).then(([cs,m])=>{for(const c of cs){const o=document.createElement('option');o.value=c.id;o.textContent=c.title;$('clip').append(o)}const s=m.summary;$('summary').textContent=s.observations+' 次相同观测；起坐片段命中 A '+s.A.sit_stand_hits+'/'+s.A.sit_stand_segments+'，B '+s.B.sit_stand_hits+'/'+s.B.sit_stand_segments+'；候选改变 '+s.changed_samples+' 次；倒地演示 '+s.fall_event_hits+'/'+s.fall_events+'（报警保持）。片段命中不等于逐帧准确率。';load()});$('clip').onchange=load;$('video').ontimeupdate=paint;
fetch('data/review-queue.json').then(r=>r.json()).then(q=>{for(const r of q.first_batch){const b=document.createElement('button');b.textContent=r.clip_id+' '+r.start_s.toFixed(1)+'–'+r.end_s.toFixed(1)+'s';b.title=r.reason;b.onclick=async()=>{$('clip').value=r.clip_id;await load();$('video').addEventListener('loadedmetadata',()=>{$('video').currentTime=r.start_s},{once:true})};$('review').append(b)}});
function frame(){const v=$('video'),c=$('candidate');if(v.readyState>=2){if(c.width!==v.videoWidth){c.width=v.videoWidth;c.height=v.videoHeight}const x=c.getContext('2d');x.drawImage(v,0,0,c.width,c.height);x.fillStyle='rgba(0,0,0,.8)';x.fillRect(0,0,c.width,34);x.fillStyle='#9bdec4';x.font='16px system-ui';x.fillText('B 旁路候选：'+$('b').textContent.split('\\n')[0]+' · 报警保持A',12,23)}requestAnimationFrame(frame)}frame();</script></html>'''
    (out / 'index.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    evaluate(parser.parse_args().output.resolve())
