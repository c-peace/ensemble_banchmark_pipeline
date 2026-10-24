"""Exercise evaluation plumbing with CPU fake inference, never gated model downloads."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from ensemble_benchmark.config import load_config
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = json.loads((ROOT / 'sam_audio_pretrained.ipynb').read_text())
CELLS = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


def test_notebook_and_config_contract():
    config = load_config(ROOT / 'configs/experiments/sam_audio_pretrained.yaml')
    from ensemble_benchmark.baseline import CLASSES
    assert set(config['parameters']['prompts']) == set(CLASSES)
    assert len(CELLS) == 6
    assert "splits=('validation',)" in CELLS[1]
    assert 'exp/peace/Sam_audio' in CELLS[0]
    for index, source in enumerate(CELLS):
        compile(source, f'sam-cell-{index}', 'exec')
        assert 'train_local(' not in source
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []


@pytest.fixture
def evaluation_scope(tmp_path, monkeypatch):
    from ensemble_benchmark import scoring
    # Real scorer, reduced expected benchmark classes for this synthetic fixture.
    monkeypatch.setattr(scoring, 'EXPECTED_CLASSES', {'URMP': ['Violin']})
    valid = tmp_path / 'validation'
    valid.mkdir()
    wave = (np.sin(np.arange(269123) * 0.03) * 0.1).astype('float32')
    audio_path = valid / 'audio.wav'
    sf.write(audio_path, wave, 44100, subtype='FLOAT')
    scene = dict(scene_id='urmp_43', dataset='URMP', parent_id='urmp_43', work_group='work',
        split='validation', sample_rate=44100, num_samples=len(wave),
        target_classes=['Violin'], mixture='audio.wav', references={'Violin': 'audio.wav'},
        sha256={'audio.wav': hashlib.sha256(audio_path.read_bytes()).hexdigest()})
    manifest = dict(complete=True, scenes=[scene], expected_scene_ids=['urmp_43'])
    (valid / 'manifest.json').write_text(json.dumps(manifest))
    config = load_config(ROOT / 'configs/experiments/sam_audio_pretrained.yaml')
    config['parameters'].update(dtype='float32', chunk_seconds=2.0, context_seconds=.25)
    snapshot = tmp_path / 'config.yaml'
    snapshot.write_text(yaml.safe_dump(config))
    calls = []

    def resample(wave, source, destination):
        divisor = math.gcd(source, destination)
        return torch.from_numpy(resample_poly(wave.numpy(), destination // divisor,
            source // divisor, axis=-1).copy())

    audio_module = ModuleType('torchaudio')
    audio_module.functional = SimpleNamespace(resample=resample)
    monkeypatch.setitem(sys.modules, 'torchaudio', audio_module)
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1234)

    def process(audios, descriptions):
        calls.append((audios[0].shape[-1], descriptions))
        batch = SimpleNamespace(audios=audios[0].unsqueeze(0))
        batch.to = lambda device: batch
        return batch

    process.audio_sampling_rate = 48000

    def separate(batch, **kwargs):
        assert kwargs['predict_spans'] is False
        assert kwargs['reranking_candidates'] == 1
        assert not torch.is_grad_enabled()
        return SimpleNamespace(target=[batch.audios[0, 0]])

    events = []
    run = SimpleNamespace(summary={}, log=events.append)
    return dict(config=config, p=config['parameters'], MODEL_CONFIG=copy.deepcopy(config),
        CONFIG_SNAPSHOT=snapshot, MODEL_INFO={'model_id': 'fake'}, Path=Path,
        yaml=yaml, copy=copy, json=json, torch=torch, VALID_ROOT=valid, RUN_DIR=tmp_path,
        processor=process, model=SimpleNamespace(separate=separate), DTYPE=torch.float32,
        run=run, calls=calls, original_wave=wave)


@pytest.mark.parametrize('partial', [False, True])
def test_real_sdr_after_chunked_fake_inference(evaluation_scope, partial, monkeypatch):
    scope = evaluation_scope
    if partial:
        scope['p']['max_scenes'] = 1
        scope['MODEL_CONFIG'] = copy.deepcopy(scope['config'])
        scope['CONFIG_SNAPSHOT'].write_text(yaml.safe_dump(scope['config']))
    exec(CELLS[4], scope)
    prediction, rate = sf.read(scope['PREDICTIONS'] / 'urmp_43/Violin.wav')
    assert rate == 44100 and len(prediction) == len(scope['original_wave'])
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction - scope['original_wave'])) < .005
    assert len(scope['calls']) == 4  # Includes a short final chunk.
    assert all(n <= 120000 and prompts == ['violin playing'] for n, prompts in scope['calls'])
    report = json.loads(scope['REPORT_PATH'].read_text())
    assert report['diagnostic'] is partial
    assert (report['overall'] is None) is partial
    assert 'evaluation' in scope['run'].summary
    assert not (scope['RUN_DIR'] / 'train').exists()

    class Artifact:
        def __init__(self, name, type, metadata):
            assert type == 'evaluation'
            self.files = []
            self.waited = False

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append(name)

        def wait(self):
            self.waited = True

    artifacts, finished = [], []
    scope['run'].log_artifact = lambda artifact: artifacts.append(artifact) or artifact
    scope['run'].finish = lambda: finished.append(True)
    import subprocess
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'sam_audio==0.1.0')
    scope.update(wandb=SimpleNamespace(Artifact=Artifact), sys=sys,
                 RUN_ID='test-evaluation', experiment={'wandb_mode': 'online'})
    exec(CELLS[5], scope)
    assert finished == [True] and artifacts[0].waited
    assert set(artifacts[0].files) == {
        'evaluation.json', 'config.yaml', 'environment.txt', 'evaluation-manifest.json'}


def test_missing_prompt_and_changed_settings_fail_before_inference(evaluation_scope):
    scope = evaluation_scope
    scope['p']['prompts'].pop('Violin')
    with pytest.raises(ValueError, match='설정'):
        exec(CELLS[4], scope)
    scope['MODEL_CONFIG'] = copy.deepcopy(scope['config'])
    scope['CONFIG_SNAPSHOT'].write_text(yaml.safe_dump(scope['config']))
    with pytest.raises(ValueError, match='프롬프트'):
        exec(CELLS[4], scope)
    assert scope['calls'] == []
    assert scope['REPORT_PATH'] is None


def test_corrupt_audio_fails_before_inference(evaluation_scope):
    scope = evaluation_scope
    path = scope['VALID_ROOT'] / 'audio.wav'
    audio, rate = sf.read(path)
    sf.write(path, audio * .5, rate, subtype='FLOAT')
    with pytest.raises(ValueError, match='체크섬'):
        exec(CELLS[4], scope)
    assert scope['calls'] == []
