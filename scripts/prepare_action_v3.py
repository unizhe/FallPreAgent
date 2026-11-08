"""Download official public data only, with resumable verified HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/action-v3/inputs"
UTK_URL = "https://cvrc.ece.utexas.edu/KinectDatasets/RGB.zip"
UTK_SIZE = 1835213149
WEIZ = "https://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/"


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            path.with_suffix(path.suffix + ".part").write_bytes(data)
            path.with_suffix(path.suffix + ".part").replace(path)
            print("downloaded", path.name, len(data), flush=True)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RemoteZip(io.RawIOBase):
    block = 4 << 20

    def __init__(self):
        self.position = 0
        self.parts = OUT / "utk-ranges"
        self.parts.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else UTK_SIZE + offset
        )
        return self.position

    def part(self, i):
        start = i * self.block
        end = min(UTK_SIZE, start + self.block) - 1
        p = self.parts / f"{i:04}.part"
        if p.exists() and p.stat().st_size == end - start + 1:
            return p
        for attempt in range(4):
            try:
                req = urllib.request.Request(UTK_URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{UTK_SIZE}":
                        raise ValueError("Range mismatch")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise ValueError("Truncated range")
                p.write_bytes(data)
                print("UTK range", i, flush=True)
                return p
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    def read(self, n=-1):
        if n < 0:
            n = UTK_SIZE - self.position
        end = min(UTK_SIZE, self.position + n)
        pieces = []
        while self.position < end:
            idx = self.position // self.block
            offset = self.position % self.block
            data = self.part(idx).read_bytes()
            take = min(end - self.position, len(data) - offset)
            pieces.append(data[offset : offset + take])
            self.position += take
        return b"".join(pieces)


def labels():
    p = fetch("https://cvrc.ece.utexas.edu/KinectDatasets/actionLabel.txt", OUT / "utk/actionLabel.txt")
    result = {}
    key = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("s") and ":" not in line:
            key = line
            result[key] = {}
        elif ":" in line:
            action, values = line.split(":")
            v = values.split()
            if len(v) == 2 and v[0] != "NaN":
                result[key][action] = list(map(int, v))
    return result


def utk(index_only=False):
    ll = labels()
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as z:
        infos = z.infolist()
        (OUT / "utk-index.json").write_text(
            json.dumps(
                [
                    dict(
                        name=f.filename, size=f.file_size, offset=f.header_offset, compressed=f.compress_size
                    )
                    for f in infos
                ],
                indent=2,
            )
        )
        if index_only:
            print([f.filename for f in infos[:15]])
            return
        selected = []
        for f in infos:
            match = re.search(r"(s\d{2}_e\d{2})/[^/]*?(\d+)\.jpg$", f.filename, re.I)
            if not match:
                continue
            group, frame = match[1], int(match[2])
            a = ll[group]
            ranges = [(a["walk"][0] - 30, a["walk"][1] + 30), (a["sitDown"][0] - 30, a["standUp"][1] + 30)]
            if any(lo <= frame <= hi for lo, hi in ranges):
                selected.append((f, group, frame))
        if not selected:
            raise ValueError("No RGB files matched official frame IDs")
        blocks = set()
        for f, _, _ in selected:
            blocks.update(
                range(
                    f.header_offset // remote.block,
                    (f.header_offset + f.compress_size + len(f.filename) + 100) // remote.block + 1,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(remote.part, sorted(blocks)))
        records = []
        for f, g, n in selected:
            data = z.read(f)
            p = OUT / "utk/rgb" / g / f"{n:06}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.read_bytes() != data:
                raise ValueError("Previously acquired JPEG changed")
            p.write_bytes(data)
            records.append(
                dict(
                    member=f.filename,
                    path=str(p.relative_to(ROOT)),
                    source_frame_id=n,
                    group=g,
                    sha256=sha(p),
                )
            )
        manifest = dict(
            url=UTK_URL,
            archive_size=UTK_SIZE,
            labels_sha256=sha(OUT / "utk/actionLabel.txt"),
            zip_crc_verified=True,
            selection="walk start-30 through end+30 and sitDown start-30 through standUp end+30; every recorded JPEG in these intervals",
            records=records,
        )
        (OUT / "utk/acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("UTK RGB", len(records), flush=True)


def other():
    # Weizmann is acquired separately by prepare_weizmann_mirror.py because the
    # original host was unreliable. Do not fetch unused robust_view data.
    items = [
        (
            f"https://fenix.ur.edu.pl/~mkepski/ds/data/fall-{n:02}-cam0.mp4",
            OUT / "urfd" / f"fall-{n:02}-cam0.mp4",
        )
        for n in range(17, 31)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda pair: fetch(*pair), items))
    (OUT / "other-acquisition.json").write_text(
        json.dumps(
            [
                dict(url=u, path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
                for u, p in items
            ],
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["utk", "index", "other"])
    a = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "other":
        other()
    else:
        utk(a.part == "index")
