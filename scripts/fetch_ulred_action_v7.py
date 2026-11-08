"""Fetch only native STS skeleton members from official range-enabled ZIPs; CRC checked."""

import binascii
import hashlib
import io
import json
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/action-v7/inputs/ulred"


class Remote(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        tail = self.fetch("-65536")
        self.tail = tail
        end = tail.rfind(b"PK\x05\x06")
        fields = struct.unpack("<4s4H2LH", tail[end : end + 22])
        self.size = fields[5] + fields[6] + len(tail) - end

    def fetch(self, span):
        if not span.startswith("-"):
            a, b = map(int, span.split("-"))
            if b - a + 1 > 262144:
                return b"".join(self.fetch(f"{lo}-{min(b, lo + 262143)}") for lo in range(a, b + 1, 262144))
        cache = OUT / "ranges" / hashlib.sha256((self.url + span).encode()).hexdigest()
        if cache.exists():
            return cache.read_bytes()
        data = subprocess.check_output(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-all-errors",
                "--max-time",
                "60",
                "-sS",
                "-r",
                span,
                self.url,
            ]
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos + n if whence == 1 else self.size + n
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if self.pos >= self.size - len(self.tail):
            data = self.tail[self.pos - (self.size - len(self.tail)) :][:n]
        else:
            data = self.fetch(f"{self.pos}-{self.pos + n - 1}")
        self.pos += len(data)
        return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = []
    for i in [1, 2, 3]:
        url = f"https://datacat.liverpool.ac.uk/2729/{i}/S{i:02}.zip"
        remote = Remote(url)
        z = zipfile.ZipFile(remote)
        names = z.namelist()
        (OUT / f"S{i:02}-members.json").write_text(json.dumps(names, indent=2))
        selected = [
            info
            for info in z.infolist()
            if "SitToStandR3" in info.filename
            and not info.is_dir()
            and not info.filename.lower().endswith((".mp4", ".avi", ".png", ".npy"))
        ]
        for info in selected:
            if (
                "depth/" in info.filename.lower()
                or info.filename.endswith(".csv")
                or "/marker/raw/" in info.filename
            ):
                continue
            print("fetch", info.filename, info.compress_size, flush=True)
            # Fetch local header + compressed member as a single bounded public range.
            end = (
                info.header_offset
                + 30
                + len(info.filename.encode())
                + len(info.extra)
                + info.compress_size
                + 512
            )
            data = remote.fetch(f"{info.header_offset}-{end}")
            head = struct.unpack("<4s5H3L2H", data[:30])
            offset = 30 + head[-2] + head[-1]
            packed = data[offset : offset + info.compress_size]
            content = zlib.decompress(packed, -15) if info.compress_type == 8 else packed
            assert len(content) == info.file_size and binascii.crc32(content) & 0xFFFFFFFF == info.CRC
            dest = OUT / Path(info.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            audit.append(
                dict(
                    url=url,
                    member=info.filename,
                    bytes=len(content),
                    crc32=info.CRC,
                    sha256=hashlib.sha256(content).hexdigest(),
                    path=str(dest.relative_to(ROOT)),
                )
            )
            print(info.filename, len(content), flush=True)
        (OUT / "acquisition.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
