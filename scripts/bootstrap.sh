#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/fetch_sources.py
uv sync --python 3.11 --locked
uv run fallpre download-models
