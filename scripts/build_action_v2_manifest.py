"""Verify the frozen annotation manifest; optionally export an identical copy.

The checked-in manifest is the annotation source of truth. Never regenerate labels
or replace a test split after seeing its predictions. Public acquisition is handled
by prepare_action_v2.py and the first-stage setup scripts.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new copy; existing files are never replaced')
    args = parser.parse_args()
    source = ROOT / 'configs/action-v2-dataset.json'
    data = source.read_bytes()
    manifest = json.loads(data)
    frozen = json.loads((ROOT / 'runs/action-v2/training/freeze.json').read_text())
    if hashlib.sha256(data).hexdigest() != frozen['manifest_sha256']:
        raise ValueError('Annotation manifest differs from frozen evaluation')
    seen = set()
    sources = set()
    counts = {}
    for entry in manifest['entries']:
        if entry['clip_id'] in seen or entry['video_sha256'] in sources:
            raise ValueError('Duplicate video across partitions')
        seen.add(entry['clip_id'])
        sources.add(entry['video_sha256'])
        path = ROOT / entry['video_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['video_sha256']:
            raise ValueError(f'Video mismatch: {path}')
        counts[entry['split']] = counts.get(entry['split'], 0) + 1
    if args.output:
        with args.output.open('xb') as handle:
            handle.write(data)
    print(json.dumps({'verified': True, 'split_counts': counts}, indent=2))


if __name__ == '__main__':
    main()
