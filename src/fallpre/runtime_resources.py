"""Pinned runtime assets, independent of optional ML packages and private data.

Reads may use verified legacy paths (including existing shared symlinks).
Explicit installation only writes new files below models/ and never overwrites
existing files or writes through symlinks. Nothing downloads at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def manifest(root: Path = ROOT) -> dict:
    return json.loads((root / 'configs/runtime-resources.json').read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Invalid resource path: {value}')
    return root / relative


def verify_file(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise ValueError(f'Resource is not a readable file: {path}; left unchanged')
    if path.stat().st_size != spec['size_bytes'] or sha256(path) != spec['sha256']:
        raise ValueError(f'Resource size/checksum mismatch: {path}; left unchanged')


def preparation_hint(name: str, spec: dict) -> str:
    if spec['kind'] == 'source-file':
        return 'python scripts/project_resources.py fetch-sources'
    if spec.get('url'):
        return f'python scripts/project_resources.py fetch-public --resource {name}'
    return (f'Obtain the verified {name} artifact from the project maintainer, then run: '
            f'python scripts/project_resources.py import {name} /path/to/artifact')


def resource_path(name: str, *, root: Path = ROOT) -> Path:
    """Resolve and verify, preferring canonical paths; corrupt files never fall back."""
    spec = manifest(root)['resources'][name]
    for value in [spec['path'], *spec.get('legacy_paths', [])]:
        path = _relative(root, value)
        if path.exists() or path.is_symlink():
            verify_file(path, spec)
            return path
    raise FileNotFoundError(f'Missing runtime resource {name}: {spec["path"]}. '
                            f'{preparation_hint(name, spec)}')


def inspect_resources(profile: str, *, root: Path = ROOT) -> dict:
    data = manifest(root)
    rows = []
    for name in data['profiles'][profile]:
        spec = data['resources'][name]
        row = dict(id=name, kind=spec['kind'], canonical_path=spec['path'],
                   hint=preparation_hint(name, spec))
        try:
            path = resource_path(name, root=root)
            row.update(status='verified', path=str(path.relative_to(root)),
                       legacy=path != root / spec['path'])
        except FileNotFoundError as error:
            row.update(status='missing', error=str(error))
        except (ValueError, OSError) as error:
            row.update(status='invalid', error=str(error))
        rows.append(row)
    if any(row['kind'] == 'source-file' for row in rows):
        spec = data['sources']['mmaction2']
        row = dict(id='mmaction2-source', kind='source', canonical_path=spec['path'],
                   hint='python scripts/project_resources.py fetch-sources')
        try:
            verify_mmaction_source(root=root)
            row.update(status='verified', path=spec['path'], revision=spec['commit'])
        except FileNotFoundError as error:
            row.update(status='missing', error=str(error))
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            row.update(status='invalid', error=str(error))
        rows.append(row)
    return dict(profile=profile, ok=all(row['status'] == 'verified' for row in rows),
                resources=rows)


def verify_mmaction_source(*, root: Path = ROOT) -> None:
    spec = manifest(root)['sources']['mmaction2']
    directory = _relative(root, spec['path'])
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing MMAction2 source: {directory}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
    changes = subprocess.check_output(['git', 'status', '--porcelain'], cwd=directory, text=True).strip()
    if revision != spec['commit'] or changes:
        raise ValueError('MMAction2 revision/local changes differ from the lock; checkout left unchanged')


def install_resource(name: str, source: Path, *, root: Path = ROOT) -> Path:
    """Verify an explicitly supplied artifact, then install once without overwrites.

    An already verified legacy file is reused instead of copied. A failed import
    never removes, replaces, or edits either the supplied source or a model.
    """
    spec = manifest(root)['resources'][name]
    if spec['kind'] not in {'team-artifact', 'public-weight'}:
        raise ValueError('Source files must be restored with fetch-sources')
    verify_file(source, spec)
    try:
        return resource_path(name, root=root)
    except FileNotFoundError:
        pass
    destination = _relative(root, spec['path'])
    if Path(spec['path']).parts[0] != 'models':
        raise ValueError('Resource installation is restricted to models/')
    relative = destination.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Refusing to write through resource symlink: {current}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.resource-', dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
        verify_file(temporary, spec)
        # Hard-link creation is atomic and fails if another installer created
        # destination meanwhile; it never replaces existing files or symlinks.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def action_dependency_path(*, root: Path = ROOT) -> Path | None:
    """Optional isolated MMAction dependencies; installed packages remain usable."""
    override = os.environ.get('FALLPRE_ACTION_DEPS')
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f'FALLPRE_ACTION_DEPS directory missing: {path}')
        return path
    for value in ('.runtime/action-deps', 'tmp/group-meeting/action-deps'):
        path = root / value
        if path.is_dir():
            return path
    return None
