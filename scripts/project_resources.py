#!/usr/bin/env python3
"""Inspect/restore runtime resources without downloading personal or training videos."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fallpre.runtime_resources import (  # noqa: E402
    inspect_resources,
    install_resource,
    manifest,
    preparation_hint,
    resource_path,
    verify_mmaction_source,
)


def fetch_sources(root: Path = ROOT) -> None:
    """Restore pinned MMAction2 only; existing trees are checked, never rewritten."""
    spec = manifest(root)['sources']['mmaction2']
    destination = root / spec['path']
    if not destination.exists():
        # Do not turn a missing directory behind a shared symlink into a write.
        for parent in [destination, *destination.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f'Refusing source install through symlink: {parent}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        spec['url'], str(destination)], check=True)
        subprocess.run(['git', 'sparse-checkout', 'set', *spec['sparse_paths']],
                       cwd=destination, check=True)
        subprocess.run(['git', 'checkout', '--detach', spec['commit']],
                       cwd=destination, check=True)
    verify_mmaction_source(root=root)
    for name in ('mmaction-config', 'ntu60-labels'):
        resource_path(name, root=root)
    print(f'Verified MMAction2 {spec["commit"]}')


def fetch_public(names: list[str], root: Path = ROOT) -> None:
    data = manifest(root)
    for name in names:
        spec = data['resources'][name]
        try:
            print(f'Verified {name}: {resource_path(name, root=root)}')
            continue
        except FileNotFoundError:
            pass
        if spec['kind'] != 'public-weight' or not spec.get('url'):
            print(f'Missing {name}: {preparation_hint(name, spec)}')
            continue
        print(f'Downloading {name} ({spec["size_bytes"]} bytes)', flush=True)
        # Temporary downloads never reuse a user's partial file or replace an
        # existing artifact. install_resource verifies before the final write.
        with tempfile.TemporaryDirectory(prefix='fallpre-download-') as folder:
            downloaded = Path(folder) / 'artifact'
            with urllib.request.urlopen(spec['url'], timeout=60) as response, downloaded.open('xb') as output:
                while block := response.read(1 << 20):
                    output.write(block)
            installed = install_resource(name, downloaded, root=root)
        print(f'Verified {name}: {installed}')


def main(argv=None) -> int:
    data = manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='Read-only asset checks; no downloads')
    doctor.add_argument('--profile', choices=data['profiles'], default='perception-chain')
    doctor.add_argument('--json', action='store_true')
    commands.add_parser('fetch-sources', help='Restore pinned MMAction2 source/config/labels only')
    fetch = commands.add_parser('fetch-public', help='Explicitly download verified public weights only')
    select = fetch.add_mutually_exclusive_group(required=True)
    select.add_argument('--profile', choices=data['profiles'])
    select.add_argument('--resource', choices=data['resources'])
    importer = commands.add_parser('import', help='Verify and install a separately supplied model artifact')
    importer.add_argument('resource', choices=data['resources'])
    importer.add_argument('file', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            result = inspect_resources(args.profile)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                for row in result['resources']:
                    print(f'{row["status"]:8s} {row["id"]}: {row.get("path", row["canonical_path"])}')
                    if row['status'] != 'verified':
                        print(f'  {row["error"]}\n  {row["hint"]}')
                print(f'{args.profile}: {"ready" if result["ok"] else "incomplete"} (resources only)')
                if args.profile in {'temporal', 'perception-chain'}:
                    print('MMAction dependencies: requirements-weekly-action.txt; use '
                          'FALLPRE_ACTION_DEPS or .runtime/action-deps. This check does not run inference.')
            return 0 if result['ok'] else 1
        if args.command == 'fetch-sources':
            fetch_sources()
        elif args.command == 'fetch-public':
            fetch_public([args.resource] if args.resource else data['profiles'][args.profile])
        else:
            print(install_resource(args.resource, args.file))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
