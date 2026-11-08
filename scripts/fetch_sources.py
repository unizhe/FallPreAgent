"""Clone pinned upstream sources without altering an existing checkout."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    sources = json.loads((ROOT / "sources.lock.json").read_text())
    for name, source in sources.items():
        path = ROOT / source["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if source.get("ref"):
                git("clone", "--depth", "1", "--branch", source["ref"], source["url"], str(path))
            else:
                git("clone", "--depth", "1", "--no-checkout", source["url"], str(path))
                git("fetch", "--depth", "1", "origin", source["commit"], cwd=path)
                git("checkout", "--detach", source["commit"], cwd=path)
        actual = git("rev-parse", "HEAD", cwd=path)
        if actual != source["commit"]:
            raise SystemExit(f"{name}: unexpected commit {actual}; existing checkout left unchanged")
        if git("status", "--porcelain", cwd=path):
            raise SystemExit(f"{name}: checkout has changes; left unchanged")
        print(f"{name}: {actual} ({source['license']})")


if __name__ == "__main__":
    main()
