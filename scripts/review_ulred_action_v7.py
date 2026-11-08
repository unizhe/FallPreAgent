"""Inspect native AMC traces without consulting action-model predictions."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v7/inputs/ulred"


def read_amc(path):
    rows = []
    frame = None
    for line in path.read_text().splitlines():
        bits = line.split()
        if not bits or bits[0].startswith(":"):
            continue
        if len(bits) == 1 and bits[0].isdigit():
            frame = {}
            rows.append(frame)
        elif frame is not None:
            frame[bits[0]] = list(map(float, bits[1:4]))
    return rows


def main():
    out = ROOT / "runs/action-v7/slow-review"
    out.mkdir(exist_ok=True)
    for p in sorted(BASE.glob("S*/marker-less/raw/*amc")):
        rows = read_amc(p)
        t = np.arange(len(rows)) / 30
        j = np.array([(np.array(r["LeftHip"]) + r["RightHip"]) / 2 for r in rows]) / 1000
        fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        for k in range(3):
            ax[k].plot(t, j[:, k])
            ax[k].set_ylabel(["hip X (m)", "hip Y (m)", "hip Z (m)"][k])
            ax[k].grid()
        ax[-1].set_xlabel("Native source seconds (30 Hz), unfiltered markerless coordinates")
        fig.suptitle(p.name + " | source-only review; normal / fast / slow instructed order")
        fig.tight_layout()
        fig.savefig(out / (p.stem + ".png"))
        plt.close(fig)
        (out / (p.stem + "-trace.json")).write_text(
            json.dumps(dict(fps=30, frames=len(rows), hip_xyz_m=j.tolist()))
        )
        print(p.name, len(rows), flush=True)


if __name__ == "__main__":
    main()
