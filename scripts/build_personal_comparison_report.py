"""Build the local video/diagnostic before-and-after review page."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/personal-fix-20261023'
CAPTURE = ROOT / 'runs/personal-capture-20261023'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--title', default='起坐修复前后对照')
    parser.add_argument('--before-revision', default='原始版本')
    parser.add_argument('--after-revision', default='2026-10-23.1')
    args = parser.parse_args()
    out = args.out
    summary = json.loads((out / 'comparison.json').read_text())
    annotations = json.loads((CAPTURE / 'annotations.json').read_text())
    manifest = json.loads((out / 'perception-manifest.json').read_text())
    series = {}
    for variant in ('before', 'after'):
        rows = [json.loads(line) for line in (out / f'{variant}.jsonl').read_text().splitlines()]
        series[variant] = []
        for row in rows:
            action, fall = row['result']['action'], row['result']['fall']
            series[variant].append(dict(t=row['video_time_s'], label=action['label'],
                raw=action['raw_label'], phase=action['phase'], resolved=action.get('resolved_label'),
                resolved_phase=action.get('resolved_phase'), reason=action['reason'],
                code=action['decision_code'], evidence=action.get('reconciliation'),
                rise_evidence=action.get('rise_evidence'), fall=fall['state']))
    data = dict(summary=summary, segments=annotations['segments'], series=series,
                duration=annotations['duration_s'], manifest=manifest,
                comparison=dict(title=args.title, before_revision=args.before_revision,
                                after_revision=args.after_revision))
    template = (ROOT / 'assets/webcam/comparison.html').read_text()
    output = template.replace('__COMPARISON_DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    (out / 'index.html').write_text(output)
    print(f'Wrote {out / "index.html"}')


if __name__ == '__main__':
    main()
