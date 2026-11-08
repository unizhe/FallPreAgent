#!/usr/bin/env python3
"""Workspace entry point, also works when macOS hides editable-install .pth files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/ultralytics")]

if __name__ == "__main__":
    from fallpre.weekly.__main__ import main

    main()
