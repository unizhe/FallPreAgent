"""Replay existing pose observations through frozen before/after webcam adapters."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fallpre.weekly.action_v7 import CompletionModel
from fallpre.weekly.human import POSE_SHA256
from fallpre.weekly.webcam_actions import SinglePersonActions as LiveActions

OUT = ROOT / 'runs/webcam-rise-evidence-20261021'
spec = importlib.util.spec_from_file_location('fallpre.weekly.webcam_before', OUT / 'before/webcam_actions.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
model = CompletionModel.load(ROOT / 'runs/action-v7/training/model.json')
entries = {e['clip_id']: e for e in json.loads((ROOT / 'configs/action-v7-dataset.json').read_text())['entries']}
clips = ['utk-s01_e01-transition', 'utk-s10_e01-transition', 'utk-s02_e01-transition',
         'utk-s03_e01-transition', 'urfd-adl-07', 'urfd-adl-04', 'urfd-fall-01', 'tum-sitting-independent']
summary = {}
for clip in clips:
    entry = entries[clip]
    locations = [ROOT / f'runs/action-v{v}/poses/{clip}/observations.jsonl' for v in [3, 2]]
    if clip == 'tum-sitting-independent':
        locations.insert(0, ROOT / f'runs/action-v4/demo-seated/fresh-poses/{clip}/observations.jsonl')
    path = next(p for p in locations if p.exists())
    meta, *observations = [json.loads(l) for l in path.read_text().splitlines()]
    assert meta['source_sha256'] == entry['video_sha256']
    assert meta['weights_sha256'] == POSE_SHA256
    times = entry.get('source_times_s', [i / entry['fps'] for i in range(len(observations))])
    assert len(times) == len(observations)
    assert all(r['source_frame_id'] == i + 1 and abs(r['t'] - times[i]) < 1e-7
               for i, r in enumerate(observations))
    cap = cv2.VideoCapture(str(ROOT / entry['video_path']))
    width, height = (int(cap.get(k)) for k in (3, 4))
    cap.release()
    assert width and height
    old, new = before.LiveActions(model), LiveActions(model)
    rows = []
    for row in observations:
        persons = [dict(box=p['box'], confidence=p['det'], keypoints=p['kpts'])
                   for p in row['instances'] if p['det'] >= .25]
        a, b = [s.update(persons, row['t'], width, height) for s in (old, new)]
        assert a['raw_label'] == b['raw_label'] and a['phase'] == b['phase']
        assert a['label'] != 'standing_up' or b['label'] == 'standing_up'
        assert (a['label'] == 'sitting_down') == (b['label'] == 'sitting_down')
        assert a['label'] == b['previous_label']
        assert (a['label'] == b['label'] or (a['label'], b['label']) in {('unknown', 'seated'), ('seated', 'unknown'), ('unknown', 'standing_up'), ('seated', 'standing_up')})
        rows.append(dict(t=row['t'], source_frame_id=row['source_frame_id'], before=a, after=b))
    assert len(rows) == entry['frame_count']
    (OUT / f'{clip}-comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    seated = entry.get('state_segments', [])
    # Additional previously-reviewed intervals; explicitly provisional, not official ground truth.
    if clip == 'utk-s10_e01-transition':
        seated = [dict(start_s=122/30, end_s=223/30, state='seated')]
    elif clip == 'urfd-adl-07':
        seated = [dict(start_s=4.4, end_s=6, state='seated')]
    subset = [r for r in rows if any(s['start_s'] <= r['t'] < s['end_s'] and s.get('state') == 'seated' for s in seated)]
    new_seated = [r for r in rows if r['before']['label'] != 'seated' and r['after']['label'] == 'seated']
    official_rise = [r for r in rows if any(s['start_s'] <= r['t'] < s['end_s']
                    and s['label'] == 'standing_up' for s in entry.get('segments', []))]
    assert not any(r['after']['label'] == 'standing_up' for r in subset)
    # Conservative visible-ascent intervals from raw-frame review, not model
    # onset times or precise biomechanical seat-off labels. Broad official
    # intervals are also reported because they include preparation/forward lean.
    ascent_interval = {'utk-s01_e01-transition': (6.2, 6.8),
                       'utk-s10_e01-transition': (8.6, 9.8),
                       'utk-s03_e01-transition': (7.8, 8.6)}.get(clip)
    ascent = [r for r in rows if ascent_interval and ascent_interval[0] <= r['t'] < ascent_interval[1]]
    assert not any(r['after']['label'] == 'seated' for r in ascent)
    summary[clip] = dict(frames=len(rows), input=str(path.relative_to(ROOT)),
                        before=dict(Counter(r['before']['label'] for r in rows)),
                        after=dict(Counter(r['after']['label'] for r in rows)),
                        raw_predictions_unchanged=True, existing_rise_and_sit_down_preserved=True,
                        seated_reference_frames=len(subset), seated_false_rise=0,
                        visible_ascent_before=dict(Counter(r['before']['label'] for r in ascent)),
                        visible_ascent_after=dict(Counter(r['after']['label'] for r in ascent)),
                        seated_reference_before=dict(Counter(r['before']['label'] for r in subset)),
                        seated_reference_after=dict(Counter(r['after']['label'] for r in subset)),
                        official_rise_frames=len(official_rise),
                        official_rise_seated_before=sum(r['before']['label'] == 'seated' for r in official_rise),
                        official_rise_seated_after=sum(r['after']['label'] == 'seated' for r in official_rise),
                        visible_ascent_interval=ascent_interval, visible_ascent_frames=len(ascent),
                        official_rise_before=dict(Counter(r['before']['label'] for r in official_rise)),
                        official_rise_after=dict(Counter(r['after']['label'] for r in official_rise)),
                        added_rise_times=[r['t'] for r in rows if r['before']['label'] != 'standing_up' and r['after']['label'] == 'standing_up'],
                        visible_ascent_seated_after=sum(r['after']['label'] == 'seated' for r in ascent),
                        added_seated_times=[r['t'] for r in new_seated])
    print(clip, summary[clip]['before'], '->', summary[clip]['after'], flush=True)
(OUT / 'comparison-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
