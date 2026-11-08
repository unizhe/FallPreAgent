"""Screen native XSens pelvic ascent, separating movement from seated waiting."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/action-v6"


def main():
    out = []
    for path in sorted((BASE / "inputs").glob("xsens-b0*.zip")):
        z = zipfile.ZipFile(path)
        for name in z.namelist():
            if "Sit_to_stand/" not in name or not name.endswith("xsens_split.csv"):
                continue
            data = z.read(name)
            rows = list(csv.DictReader(io.StringIO(data.decode())))
            t = np.array([float(r["time"]) for r in rows])
            t -= t[0]
            raw = np.array([float(r["field.poses0.position.z"]) for r in rows])
            p = median_filter(raw, size=21)
            lo = p[:20].mean()
            hi = p[-20:].mean()
            amp = hi - lo
            if amp < 0.15:
                continue
            a = np.flatnonzero(p > lo + amp * 0.15)[0]
            b = np.flatnonzero(p > lo + amp * 0.9)[0]
            ids = np.linspace(a, b, 7).astype(int)
            delta = np.diff(p[ids])
            out.append(
                dict(
                    archive=str(path.relative_to(ROOT)),
                    archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    member=name,
                    member_sha256=hashlib.sha256(data).hexdigest(),
                    start_s=float(t[a]),
                    end_s=float(t[b]),
                    duration_s=float(t[b] - t[a]),
                    amplitude_m=float(amp),
                    bin_delta_m=delta.tolist(),
                    continuous_candidate=bool(t[b] - t[a] >= 1.5 and min(delta) > 0.005),
                )
            )
    (BASE / "inputs/xsens-slow-screen.json").write_text(json.dumps(out, indent=2) + "\n")
    print("trials", len(out))
    print(json.dumps(sorted(out, key=lambda r: -r["duration_s"])[:8], indent=2))


if __name__ == "__main__":
    main()
