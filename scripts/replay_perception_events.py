"""Persist PerceptionChain JSONL output and optionally acknowledge local simulations."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fallpre.event_store import EventStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--session', required=True,
                        help='Stable producer session; change when its clock/event counter resets')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--simulate-delivery', action='store_true',
                        help='Record local receipts only; never send external messages')
    args = parser.parse_args()
    counts = Counter()
    with EventStore(args.database) as store:
        for path in args.input:
            with path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    result = store.ingest(args.session, row.get('pipeline', row))
                    counts[result['status']] += 1
        receipts = []
        if args.simulate_delivery:
            receipts = [store.simulate_delivery(item['id']) for item in store.pending_notifications()]
        print(json.dumps(dict(ingestion=dict(counts), events=len(store.list_events()),
                              pending=len(store.pending_notifications()),
                              local_receipts=len(receipts), external_actions=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
