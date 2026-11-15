"""Replay schema v1 perception into one Agent and simulated tools."""
import argparse
import hashlib
import json
from collections import Counter
from contextlib import closing
from pathlib import Path

from fallpre.agent import FallPreventionAgent
from fallpre.care_loop import CareLoop, perception_payload
from fallpre.event_store import EventStore
from fallpre.state_store import StateStore

DEMO = Path(__file__).resolve().parents[1] / 'tests/fixtures/care-agent-demo.jsonl'


def read_payloads(path):
    with path.open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected a JSON object')
                if 'perception' in row:
                    payload = row['perception']  # Live shadow log
                elif 'pipeline' in row:
                    payload = row['pipeline']  # PerceptionChain replay log
                elif 'primary' in row:
                    payload = perception_payload(path.stem, row)  # Historical backend replay
                else:
                    payload = row  # Direct schema v1
                if not isinstance(payload, dict) or payload.get('schema_version') != 1:
                    raise ValueError('Expected schema v1 perception, PerceptionChain output, or nested legacy replay. '
                                     'Older flat shadow logs lack the required observation and event fields.')
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
            yield number, payload


def replay(paths, output, session):
    counts = Counter()
    sources = []
    output.mkdir(parents=True, exist_ok=False)
    with (EventStore(output / 'events.sqlite') as store,
          closing(StateStore(str(output / 'execution.sqlite'))) as execution,
          closing(CareLoop(store, execution)) as loop,
          (output / 'trace.jsonl').open('w', encoding='utf-8') as trace):
        agent = FallPreventionAgent(loop)
        for path in paths:
            with path.open('rb') as source:
                sources.append(dict(file=path.name, sha256=hashlib.file_digest(source, 'sha256').hexdigest()))
            for number, payload in read_payloads(path):
                try:
                    out = agent.observe(session, payload)
                    retry = agent.observe(session, payload)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
                if retry['ingest']['status'] != 'duplicate' or retry['executions']:
                    raise RuntimeError(f'{path}:{number}: repeated input changed execution')
                counts['frames'] += 1
                counts['source_new_alarms'] += bool(payload['notification']['emit'])
                counts['duplicate_retries_checked'] += 1
                for receipt in out['executions']:
                    counts[receipt['action_type']] += 1
                    record = dict(stream=payload['stream_id'], source_time_s=payload['source_time_s'],
                                  result=receipt)
                    trace.write(json.dumps(record, ensure_ascii=False) + '\n')
        if not counts['frames']:
            raise ValueError('Input contains no observations')
        requests = [json.loads(r[0]) for r in store.db.execute('SELECT body FROM care_suggestions ORDER BY rowid')]
        with (output / 'jev-requests.jsonl').open('w', encoding='utf-8') as destination:
            for request in requests:
                destination.write(json.dumps(request, ensure_ascii=False) + '\n')
        summary = dict(counts=counts, sources=sources, events=len(store.list_events()),
                       confirmed_events=sum(e['confirmed'] for e in store.list_events()),
                       pending_notifications=len(store.pending_notifications()),
                       execution_count=len(execution.list_actions()), jev_requests=len(requests),
                       jev_status='NOT_RUN', external_actions=0,
                       evaluation='perception message replay; no new video inference or wall-clock claim',
                       reminder='ordinary sit-to-stand cue simulation; not validated pre-fall warning')
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Replay the bundled synthetic scenario')
    source.add_argument('--input', type=Path, help='JSONL file or directory of JSONL files from one producer session')
    parser.add_argument('--session', default='recorded-replay', help='Producer session identifier')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.demo:
        paths = [DEMO]
    elif args.input.is_dir():
        paths = sorted(args.input.glob('*.jsonl'))
    else:
        paths = [args.input]
    if not paths or any(not path.is_file() for path in paths):
        parser.error('No readable JSONL inputs found')
    try:
        summary = replay(paths, args.output, args.session)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
