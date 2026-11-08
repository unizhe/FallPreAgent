"""Rebuild the lightweight report and original-media references from frozen B."""
import argparse
import os
import re
from pathlib import Path

from evaluate_stgcn_fusion import OLD, build_page, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runs/stgcn-fusion-20261106')
    out = parser.parse_args().output.resolve()
    clips = read(OLD / 'clips.json')
    build_page(out, clips, read(out / 'B/metrics.json')['summary'])
    for c in clips:
        dest = out / 'media' / f'{c["id"]}.webm'
        if not dest.exists():
            dest.symlink_to(os.path.relpath(OLD / c['id'] / 'raw.webm', dest.parent))
    page = out / 'index.html'
    page.write_text(page.read_text().replace('href="data/review-queue.json"', 'href="data/review.html"'))
    review = out / 'data/review.html'
    if review.is_file():
        review.write_text(re.sub(r'../../stgcn-integration-20261029/([^/]+)/raw.webm',
                                r'../media/\1.webm', review.read_text()))


if __name__ == '__main__':
    main()
