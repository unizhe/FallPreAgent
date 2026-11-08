"""Read only the public Weizmann walk/bend files from a pinned Kaggle mirror."""

import concurrent.futures
import json
import zipfile
from pathlib import Path

import prepare_action_v3 as base

URL = "https://www.kaggle.com/api/v1/datasets/download/wendguan/weizmann"
SIZE = 345063325
base.UTK_URL = URL
base.UTK_SIZE = SIZE


class MirrorZip(base.RemoteZip):
    block = 1 << 20

    def __init__(self):
        self.position = 0
        self.parts = base.OUT / "weizmann-ranges"
        self.parts.mkdir(parents=True, exist_ok=True)


def main():
    remote = MirrorZip()
    with zipfile.ZipFile(remote) as z:
        selected = [f for f in z.infolist() if "/bend/" in f.filename or "/walk/" in f.filename]
        selected = [f for f in selected if f.filename.endswith(".avi")]
        if not selected:
            raise ValueError("Expected original subject/action filenames missing")
        blocks = set()
        for f in selected:
            blocks.update(
                range(
                    f.header_offset // remote.block,
                    (f.header_offset + f.compress_size + len(f.filename) + 100) // remote.block + 1,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(remote.part, sorted(blocks)))
        records = []
        for f in selected:
            path = Path(f.filename)
            action = path.parts[-2]
            dest = base.OUT / "weizmann" / action / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(f))
            records.append(dict(member=f.filename, bytes=f.file_size, sha256=base.sha(dest)))
        result = dict(
            original="https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html",
            mirror=URL,
            mirror_archive_bytes=SIZE,
            zip_crc_verified=True,
            original_archive_checksum_unavailable=True,
            records=records,
        )
        (base.OUT / "weizmann/acquisition.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Weizmann videos", len(records), flush=True)


if __name__ == "__main__":
    main()
