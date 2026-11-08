"""Portable setup uses tiny fixtures; no model weights, videos, or network needed."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from fallpre.runtime_resources import (
    action_dependency_path,
    inspect_resources,
    install_resource,
    resource_path,
)


@pytest.fixture
def resource_tree(tmp_path):
    payload = b'verified tiny test artifact\n'
    (tmp_path / 'configs').mkdir()
    data = dict(profiles={'synthetic': [], 'test': ['private']}, resources={
        'private': dict(kind='team-artifact', path='models/test.bin',
                        legacy_paths=['runs/legacy/test.bin'], size_bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest()),
    })
    (tmp_path / 'configs/runtime-resources.json').write_text(json.dumps(data))
    source = tmp_path / 'team-transfer.bin'
    source.write_bytes(payload)
    return tmp_path, source


def test_fresh_clone_doctor_and_missing_actionable_message(resource_tree):
    root, _ = resource_tree
    assert inspect_resources('synthetic', root=root)['ok']
    report = inspect_resources('test', root=root)
    assert not report['ok']
    assert report['resources'][0]['status'] == 'missing'
    assert 'import private' in report['resources'][0]['hint']
    with pytest.raises(FileNotFoundError, match='project_resources.py import private'):
        resource_path('private', root=root)
    assert not (root / 'models').exists()


def test_import_checks_hash_and_never_changes_source(resource_tree):
    root, source = resource_tree
    original = source.read_bytes()
    installed = install_resource('private', source, root=root)
    assert installed == root / 'models/test.bin'
    assert installed.read_bytes() == original == source.read_bytes()
    assert inspect_resources('test', root=root)['ok']
    assert install_resource('private', source, root=root) == installed
    assert list(installed.parent.iterdir()) == [installed]


def test_bad_import_does_not_create_models(resource_tree):
    root, source = resource_tree
    source.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum mismatch'):
        install_resource('private', source, root=root)
    assert not (root / 'models').exists()
    assert source.read_bytes() == b'wrong'


def test_verified_legacy_symlink_reused_without_duplicate(resource_tree):
    root, source = resource_tree
    legacy = root / 'runs/legacy/test.bin'
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(source)
    assert install_resource('private', source, root=root) == legacy
    report = inspect_resources('test', root=root)
    assert report['ok'] and report['resources'][0]['legacy']
    assert legacy.is_symlink()
    assert not (root / 'models').exists()


def test_corrupt_canonical_never_falls_back_or_overwrites(resource_tree):
    root, source = resource_tree
    legacy = root / 'runs/legacy/test.bin'
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(source)
    target = root / 'models/test.bin'
    target.parent.mkdir()
    target.write_bytes(b'corrupt')
    assert inspect_resources('test', root=root)['resources'][0]['status'] == 'invalid'
    with pytest.raises(ValueError, match='checksum mismatch'):
        install_resource('private', source, root=root)
    assert target.read_bytes() == b'corrupt'
    assert legacy.is_symlink()


def test_import_will_not_write_through_shared_directory(resource_tree):
    root, source = resource_tree
    shared = root / 'shared'
    shared.mkdir()
    (root / 'models').symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        install_resource('private', source, root=root)
    assert not list(shared.iterdir())
    assert (root / 'models').is_symlink()


def test_import_rejects_manifest_path_escape(resource_tree):
    root, source = resource_tree
    path = root / 'configs/runtime-resources.json'
    data = json.loads(path.read_text())
    data['resources']['private']['path'] = '../escape.bin'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Invalid resource path'):
        install_resource('private', source, root=root)


def test_dependency_path_prefers_explicit_then_portable_then_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv('FALLPRE_ACTION_DEPS', raising=False)
    assert action_dependency_path(root=tmp_path) is None
    legacy = tmp_path / 'tmp/group-meeting/action-deps'
    legacy.mkdir(parents=True)
    assert action_dependency_path(root=tmp_path) == legacy
    canonical = tmp_path / '.runtime/action-deps'
    canonical.mkdir(parents=True)
    assert action_dependency_path(root=tmp_path) == canonical
    monkeypatch.setenv('FALLPRE_ACTION_DEPS', str(legacy))
    assert action_dependency_path(root=tmp_path) == legacy
    monkeypatch.setenv('FALLPRE_ACTION_DEPS', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError, match='FALLPRE_ACTION_DEPS'):
        action_dependency_path(root=tmp_path)


def test_public_fetch_does_not_fetch_private_artifact(resource_tree, monkeypatch, capsys):
    root, _ = resource_tree
    script = Path(__file__).resolve().parents[1] / 'scripts/project_resources.py'
    spec = importlib.util.spec_from_file_location('project_resources_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def network_forbidden(*args, **kwargs):
        pytest.fail('Private artifact must not be fetched from network')

    monkeypatch.setattr(module.urllib.request, 'urlopen', network_forbidden)
    module.fetch_public(['private'], root=root)
    assert 'import private' in capsys.readouterr().out
    assert not (root / 'models').exists()
