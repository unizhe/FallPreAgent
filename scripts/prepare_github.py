"""Audit/export the Git-visible source tree without staging, committing or pushing.

No ignored resource or symlink is expanded. Reports list locations, never secret
values. Export only succeeds when the source-boundary checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
FORBIDDEN_ROOTS = {'runs', 'models', 'tmp', 'output', 'third_party', '.git', '.vscode', '.playwright-cli'}
FORBIDDEN_SUFFIXES = {'.mp4', '.webm', '.mov', '.jpg', '.png', '.zip', '.pt', '.pth', '.npz', '.onnx', '.sqlite3'}


def source_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('path_outside_source_tree')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink_not_publishable')
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError('path_outside_source_tree')
    return current


def audit(root=ROOT):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, check=True, capture_output=True)
    names = sorted(set(n for n in result.stdout.decode().split('\0') if n))
    ignored = subprocess.run(['git', 'ls-files', '-z', '--cached', '--ignored', '--exclude-standard'],
                             cwd=root, check=True, capture_output=True)
    ignored_names = set(ignored.stdout.decode().split('\0'))
    files, errors = [], []
    for name in names:
        if name in ignored_names:
            errors.append(dict(file=name, reason='tracked_but_excluded_from_publication'))
            continue
        try:
            path = source_path(root, name)
        except ValueError as error:
            errors.append(dict(file=name, reason=str(error)))
            continue
        if not path.is_file():
            errors.append(dict(file=name, reason='missing_or_not_regular'))
            continue
        if Path(name).parts[0] in FORBIDDEN_ROOTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(dict(file=name, reason='private_or_generated_resource'))
            continue
        data = path.read_bytes()
        if len(data) > 2_000_000:
            errors.append(dict(file=name, reason='file_exceeds_2MB_source_limit'))
        if b'\0' in data:
            errors.append(dict(file=name, reason='binary_in_source_export'))
        text = data.decode('utf-8', errors='replace')
        if SECRET.search(text):
            errors.append(dict(file=name, reason='possible_credential'))
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return dict(schema_version=1, passed=not errors, file_count=len(files),
                total_bytes=sum(f['bytes'] for f in files), errors=errors, files=files,
                scope='Git-visible files only; not a proof that arbitrary secrets are absent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'output/github-preparation/manifest.json')
    parser.add_argument('--export', type=Path, help='Optional source-only zip, inside ignored output/')
    args = parser.parse_args()
    report = audit()
    for path in [args.report, args.export]:
        if path and not path.resolve().is_relative_to((ROOT/'output').resolve()):
            parser.error('Artifacts must be inside ignored output/')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        if args.export.exists():
            parser.error('Export exists; choose a new filename')
        with zipfile.ZipFile(args.export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for item in report['files']:
                data = source_path(ROOT, item['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise RuntimeError('File changed after audit: '+item['path'])
                archive.writestr(item['path'], data)
        print(f'Exported {args.export}')


if __name__ == '__main__':
    main()
