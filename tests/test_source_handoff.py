"""Portable handoff checks: safe publication and causal observation replay."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from test_perception_chain import source

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = script('replay_perception_chain')
publication = script('prepare_github')


def observations(path, times=(0, .1, .2)):
    rows = [dict(source_time_s=t, result=source(t)) for t in times]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return rows


def test_replay_without_private_resources_preserves_input_and_reports_outputs(tmp_path):
    incoming = tmp_path / 'input.jsonl'
    observations(incoming)
    original = incoming.read_bytes()
    report = replay.replay(incoming, tmp_path / 'out', 'camera-a',
                           runtime_factory=lambda: pytest.fail('Cached mode must not load models'))
    records = [json.loads(line) for line in (tmp_path / 'out/pipeline.jsonl').read_text().splitlines()]
    assert incoming.read_bytes() == original
    assert report['status'] == 'complete'
    assert report['counts']['samples'] == 3
    assert report['temporal_mode'] == 'cached_input'
    assert not report['external_action_performed']
    assert records[-1]['pipeline']['current_action']['label'] == 'standing'
    assert records[-1]['pipeline']['display']['color'] == 'green'


@pytest.mark.parametrize('times', [(1, .5), (0, 0), (-1,), (float('nan'),)])
def test_replay_rejects_invalid_clock_before_creating_outputs(tmp_path, times):
    incoming = tmp_path / 'input.jsonl'
    observations(incoming, times)
    with pytest.raises(ValueError, match='timestamps'):
        replay.replay(incoming, tmp_path / 'out', 'camera-a')
    assert not (tmp_path / 'out').exists()


def test_replay_rejects_scalar_ui_logs_and_conflicting_timestamps(tmp_path):
    incoming = tmp_path / 'input.jsonl'
    incoming.write_text(json.dumps(dict(source_time_s=0, action='standing')))
    with pytest.raises(ValueError, match='Input line 1'):
        replay.read_observations(incoming)
    row = observations(incoming, (0,))[0]
    row['result']['source_time'] = 1
    incoming.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='disagree'):
        replay.read_observations(incoming)


def test_replay_never_overwrites_previous_run(tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        replay.replay(tmp_path / 'unused-input', output, 'camera-a')


def test_new_inference_is_not_available_before_measured_completion():
    class Runtime:
        def predict(self, window):
            return dict(status='ready', target_epoch=window.target_epoch, generation=window.generation,
                        window=window.metadata,
                        window_end_s=window.metadata['end_s'], inference_ms=250)

    engine = replay.CausalTemporalReplay(Runtime())
    outputs = [engine.update(source(i / 10), i / 10) for i in range(21)]
    assert outputs[15]['status'] == 'pending'
    assert outputs[17]['status'] == 'pending'
    assert outputs[18]['status'] == 'ready'
    assert outputs[18]['window_end_s'] == 1.5
    changed = source(2.1)
    changed['action']['selection']['target_epoch'] = 9
    assert engine.update(changed, 2.1)['status'] != 'ready'


@pytest.mark.parametrize('field', ['pose_fall_observation', 'rgb_raw_candidates', 'fall_mode'])
def test_replay_requires_event_inputs_before_creating_outputs(tmp_path, field):
    incoming = tmp_path / 'input.jsonl'
    row = observations(incoming, (0,))[0]
    del row['result'][field]
    incoming.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=field):
        replay.replay(incoming, tmp_path / 'out', 'camera-a')
    assert not (tmp_path / 'out').exists()


def test_replay_rejects_missing_quality_reason(tmp_path):
    incoming = tmp_path / 'input.jsonl'
    row = observations(incoming, (0,))[0]
    del row['result']['quality']['reason']
    incoming.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='quality.reason'):
        replay.replay(incoming, tmp_path / 'out', 'camera-a')
    assert not (tmp_path / 'out').exists()


def test_replay_rejects_scalar_temporal_result(tmp_path):
    incoming = tmp_path / 'input.jsonl'
    row = observations(incoming, (0,))[0]
    row['result']['temporal'] = 'unavailable'
    incoming.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Input line 1: result.temporal'):
        replay.replay(incoming, tmp_path / 'out', 'camera-a')
    assert not (tmp_path / 'out').exists()


def test_publication_ignores_resources_and_does_not_follow_links(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('/private/\n')
    (tmp_path / 'private').mkdir()
    (tmp_path / 'private/video.mp4').write_bytes(b'private bytes')
    (tmp_path / 'README.md').write_text('Source handoff\n')
    report = publication.audit(tmp_path)
    assert report['passed']
    assert {item['path'] for item in report['files']} == {'.gitignore', 'README.md'}
    (tmp_path / 'linked-video').symlink_to(tmp_path / 'private/video.mp4')
    report = publication.audit(tmp_path)
    assert not report['passed']
    assert report['errors'] == [dict(file='linked-video', reason='symlink_not_publishable')]


def test_publication_rejects_accidental_credentials_without_reporting_value(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    # Construct a fake test token so the source scanner does not flag its fixture.
    fake = 'gh' + 'p_' + 'x' * 36
    (tmp_path / 'config.txt').write_text(fake)
    report = publication.audit(tmp_path)
    assert not report['passed']
    assert report['errors'][0]['reason'] == 'possible_credential'
    assert fake not in json.dumps(report)


def test_publication_rejects_tracked_files_behind_directory_symlinks(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    folder = tmp_path / 'scripts'
    folder.mkdir()
    (folder / 'example.py').write_text('pass\n')
    subprocess.run(['git', 'add', 'scripts/example.py'], cwd=tmp_path, check=True)
    folder.rename(tmp_path / 'shared')
    folder.symlink_to(tmp_path / 'shared', target_is_directory=True)
    report = publication.audit(tmp_path)
    assert not report['passed']
    assert dict(file='scripts/example.py', reason='symlink_not_publishable') in report['errors']


def test_publication_rejects_forced_tracked_private_files(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('docs/local-history/\n.env\n')
    private = tmp_path / 'docs/local-history/private.json'
    private.parent.mkdir(parents=True)
    private.write_text('{}')
    (tmp_path / '.env').write_text('PRIVATE_VALUE=example')
    subprocess.run(['git', 'add', '-f', 'docs/local-history/private.json', '.env'], cwd=tmp_path, check=True)
    report = publication.audit(tmp_path)
    assert not report['passed']
    assert {error['reason'] for error in report['errors']} == {'tracked_but_excluded_from_publication'}
