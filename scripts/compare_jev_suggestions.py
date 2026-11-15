"""Separate advisory job: cannot block perception or execute actions."""
import argparse
import json
from pathlib import Path

from fallpre.adapters.jev_suggestions import assess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--enable-network', action='store_true')
    args = p.parse_args()
    with args.output.open('x') as out:
        for line in args.input.open():
            row = json.loads(line)
            result = assess(row, enabled=args.enable_network)
            out.write(json.dumps(dict(request_id=row['request_id'], evidence_digest=row['evidence_digest'],
                                      baseline=row['state']['baseline'], advisory=result,
                                      historical_only=True), ensure_ascii=False)+'\n')
            out.flush()


if __name__ == '__main__':
    main()
