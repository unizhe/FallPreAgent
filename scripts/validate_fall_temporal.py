"""Reuse protected primary regression and real JPEG/HTTP paths in a new run."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['replay', 'http'])
    parser.add_argument('--port', type=int, default=8810)
    args = parser.parse_args()
    output = ROOT/'runs/stgcn-fall-20261107/validation'
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('protected_regression',
        ROOT/'runs/stgcn-integration-20261029/verify_integration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = output
    if args.mode == 'replay':
        module.replay()
    else:
        module.http(args.port, ['clip04', 'fall'])
        checks = []
        for name in ['clip04', 'fall']:
            path = output/f'http-{args.port}-{name}.jsonl'
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            for row in entries:
                e = row['result']['fall_temporal_evidence']
                assert not any(e[k] for k in ['changes_alarm', 'changes_event', 'changes_primary_decision'])
            checks.append(dict(clip=name, observations=len(entries), primary_unchanged=True,
                               evidence_present=True, support_observations=sum(
                                   r['result']['fall_temporal_evidence']['support'] for r in entries)))
        (output/'fall-evidence-http.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__ == '__main__':
    main()
