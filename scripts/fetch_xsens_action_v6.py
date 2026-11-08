"""Download the remaining public source participants for native-motion screening."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch(sid):
    p = ROOT / f"runs/action-v6/inputs/xsens-{sid}.zip"
    if not p.exists():
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--max-time",
                "240",
                "--retry",
                "1",
                "-sS",
                f"https://raw.githubusercontent.com/Assistive-Robotics-Lab/STS_Dataset/main/{sid}.zip",
                "-o",
                str(p),
            ],
            check=True,
        )
    print(sid, p.stat().st_size, flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(fetch, ["b02", "b04", "b05"]))
