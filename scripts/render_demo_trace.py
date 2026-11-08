"""Render an offline, interactive replay of a real showcase recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(trace: Path, output: Path) -> None:
    data = json.loads(trace.read_text())
    if data.get("format") != "fallpre-showcase-v1" or not data.get("dry_run"):
        raise ValueError("Expected a dry-run showcase recording")
    template = Path(__file__).resolve().parents[1] / "assets/demo-showcase.html"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.read_text().replace("__FALLPRE_DATA__", payload)
    if len(html.encode()) >= 1_000_000:
        raise ValueError("Recording exceeds inline preview size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.trace, args.output)
