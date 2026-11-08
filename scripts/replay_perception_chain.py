"""Replay full observation JSONL through the perception contract, without video copies.

Cached temporal predictions are used by default. --infer-temporal explicitly loads
ST-GCN++ and experimental heads, releasing each result only after its measured
inference duration on the source timeline. This is an offline timing simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

from fallpre.weekly.perception_chain import PerceptionChain, PerceptionRuntime
from fallpre.weekly.stgcn_temporal import TemporalPoseBuffer


def read_observations(path):
    """Validate the transport contract before creating any replay artifacts."""
    rows, previous = [], None
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            result = row['result']
            if not isinstance(result, dict):
                raise ValueError('result must be an object')
            for key, kind in [('width', int), ('height', int), ('persons', list),
                              ('action', dict), ('quality', dict), ('fall', dict)]:
                if not isinstance(result.get(key), kind) or isinstance(result.get(key), bool):
                    raise ValueError(f'result.{key} must be {kind.__name__}')
            if result['width'] <= 0 or result['height'] <= 0:
                raise ValueError('frame dimensions must be positive')
            if 'camera_stable' not in result['quality'] or not isinstance(result['quality']['camera_stable'], bool):
                raise ValueError('result.quality.camera_stable must be boolean')
            if not isinstance(result['quality'].get('reason'), str):
                raise ValueError('result.quality.reason is required')
            if not isinstance(result['action'].get('selection'), dict):
                raise ValueError('result.action.selection is required')
            if result.get('fall_mode') != 'rgb':
                raise ValueError('PerceptionChain replay requires result.fall_mode="rgb"')
            observation = result.get('pose_fall_observation')
            if (not isinstance(observation, dict) or 'person' not in observation
                    or observation['person'] is not None and not isinstance(observation['person'], dict)
                    or not isinstance(observation.get('selection'), dict)
                    or 'target_epoch' not in observation['selection']):
                raise ValueError('result.pose_fall_observation requires person (or null) and selection.target_epoch')
            rgb = result.get('rgb_raw_candidates')
            if (not isinstance(rgb, dict) or not isinstance(rgb.get('persons'), list)
                    or not isinstance(rgb.get('fallen'), list)):
                raise ValueError('result.rgb_raw_candidates requires persons and fallen lists')
            for optional in ('temporal', 'fall_observation', 'rgb_diagnostics'):
                if result.get(optional) is not None and not isinstance(result[optional], dict):
                    raise ValueError(f'result.{optional} must be an object or null')
            timestamp = row.get('source_time_s')
            if timestamp is None:
                timestamp = result.get('source_time')
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp) or timestamp < 0
                    or previous is not None and timestamp <= previous):
                raise ValueError('source timestamps must be finite, nonnegative and strictly increasing')
            nested = result.get('source_time')
            if nested is not None and nested != timestamp:
                raise ValueError('source_time_s and result.source_time disagree')
            # No NaN/Infinity may enter outputs or be silently treated as evidence.
            json.dumps(row, allow_nan=False)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Input line {number}: {error}') from error
        rows.append((float(timestamp), row))
        previous = timestamp
    if not rows:
        raise ValueError('Input has no observations')
    return rows


class CausalTemporalReplay:
    def __init__(self, runtime):
        self.runtime = runtime
        self.buffer = TemporalPoseBuffer()
        self.pending = self.latest = None

    def update(self, result, timestamp):
        action = result['action']
        selection = action['selection']
        index = action.get('target_index')
        people = result['persons']
        person = (people[index] if isinstance(index, int) and not isinstance(index, bool)
                  and 0 <= index < len(people) and selection.get('state') == 'locked' else None)
        state = self.buffer.observe(person, timestamp, result['width'], result['height'],
                                    selection.get('target_epoch'),
                                    camera_stable=result['quality']['camera_stable'])
        if self.pending and self.pending[0] <= timestamp:
            prediction = self.pending[1]
            self.latest = prediction if self.buffer.is_result_current(prediction) else None
            self.pending = None
        if state['status'] != 'ready':
            self.latest = None
        if self.pending is None:
            window = self.buffer.prepare()
            if window is not None:
                prediction = self.runtime.predict(window)
                duration_ms = prediction['inference_ms']
                if not math.isfinite(duration_ms) or duration_ms < 0:
                    raise ValueError('Runtime returned an invalid inference duration')
                self.pending = (timestamp + duration_ms / 1000, prediction)
        temporal = dict(state, top3=[])
        if self.latest and self.buffer.is_result_current(self.latest):
            temporal.update(self.latest, result_age_s=timestamp - self.latest['window_end_s'])
        else:
            temporal['status'] = 'pending' if state['status'] == 'ready' else state['status']
        return temporal


def replay(input_path, output, stream_id, *, infer_temporal=False, runtime_factory=PerceptionRuntime):
    if output.exists():
        raise ValueError('Output already exists; use a new directory to preserve previous results')
    rows = read_observations(input_path)
    chain = PerceptionChain(stream_id)
    temporal = CausalTemporalReplay(runtime_factory()) if infer_temporal else None
    counts, notifications = Counter(), []
    summary = dict(schema_version=1, status='running', stream_id=stream_id,
                   input_name=input_path.name, input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                   temporal_mode='inference_with_measured_delay' if temporal else 'cached_input',
                   external_action_performed=False, counts={}, notifications=[])
    output.mkdir(parents=True, exist_ok=False)

    def save_summary():
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8')

    save_summary()
    try:
        with (output / 'pipeline.jsonl').open('w', encoding='utf-8') as handle:
            for timestamp, row in rows:
                result = deepcopy(row['result'])
                if temporal:
                    result['temporal'] = temporal.update(result, timestamp)
                pipeline = chain.update(result, timestamp)
                counts['samples'] += 1
                counts['observable'] += bool(pipeline['observation']['observable'])
                counts['action_known'] += bool(pipeline['current_action']['known'])
                counts['temporal_current'] += bool(pipeline['temporal_action']['result_current'])
                counts['experimental_process_available'] += pipeline['fall_process']['status'] == 'experimental'
                if pipeline['notification']['emit']:
                    notifications.append(dict(source_time_s=timestamp, **pipeline['notification']))
                record = dict(source_time_s=timestamp, video_time_s=row.get('video_time_s', row.get('time_s')),
                              baseline_fall=result['fall'], pipeline=pipeline)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        summary['status'] = 'complete'
    except Exception:
        summary['status'] = 'failed'
        raise
    finally:
        summary.update(counts=dict(counts), notifications=notifications)
        save_summary()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-jsonl', type=Path, required=True, help='Full result records, not UI scalar logs')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--stream-id', default='offline-replay', help='Identity namespace for this continuous stream')
    parser.add_argument('--infer-temporal', action='store_true', help='Recompute ST-GCN++ and experimental C/D heads')
    args = parser.parse_args()
    try:
        report = replay(args.input_jsonl, args.output, args.stream_id, infer_temporal=args.infer_temporal)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
