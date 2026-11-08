"""Download fixed public Telkom STS participants, validate publisher MD5."""

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/action-v4/inputs"


def main():
    metadata = json.loads((OUT / "telkom-metadata.json").read_text())["data"]["latestVersion"]
    chosen = []
    for item in metadata["files"]:
        f = item["dataFile"]
        if not f["filename"].lower().startswith("out_subject_"):
            continue
        subject = int(f["filename"].split("_")[2])
        if subject not in [11, 12, 13, 14, 15]:
            continue
        chosen.append((subject, f))

    def fetch(pair):
        subject, f = pair
        when = f["filename"].split("_")[3].split(".")[0]
        path = OUT / f"telkom-{subject}-{when}.mp4"
        url = f"https://dataverse.telkomuniversity.ac.id/api/access/datafile/{f['id']}"
        if not path.exists():
            temp = path.with_suffix(".part")
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "2", "--max-time", "180", "-sS", url, "-o", str(temp)],
                check=True,
            )
            temp.replace(path)
        assert hashlib.md5(path.read_bytes()).hexdigest() == f["checksum"]["value"]
        print(path.name, path.stat().st_size, flush=True)
        return dict(
            subject=subject,
            when=when,
            path=str(path.relative_to(ROOT)),
            url=url,
            md5=f["checksum"]["value"],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            license="CC0-1.0",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(fetch, chosen))
    (OUT / "acquisition.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
