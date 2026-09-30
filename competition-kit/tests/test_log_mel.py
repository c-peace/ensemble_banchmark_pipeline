"""Controlled loss ablation, deterministic initialization, and resume safeguards."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from ensemble_benchmark.baseline import Trainer, _loss
from ensemble_benchmark.losses import LOG_MEL_CONFIG, LogMelLoss, loss_config
from ensemble_benchmark import runtime


def test_log_mel_identity_silence_and_differentiable_error():
    torch.set_num_threads(1)
    criterion = LogMelLoss()
    target = torch.sin(torch.arange(4096)*0.06)[None, None]*0.1
    assert criterion(target, target).item() == 0
    silence = torch.zeros_like(target)
    assert criterion(silence, silence).item() == 0
    prediction = (target*0.5).requires_grad_()
    error = criterion(prediction, target)
    assert torch.isfinite(error) and error > 0
    error.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert criterion.filters.shape == (128, 1025)
    assert (criterion.filters.sum(1) > 0).all()


def test_log_mel_does_not_change_seeded_initialization_or_default_loss():
    baseline = Trainer(tiny=True, seed=42)
    alternate = Trainer(tiny=True, seed=42, loss_type='baseline_log_mel')
    for key, value in baseline.models.state_dict().items():
        torch.testing.assert_close(value, alternate.models.state_dict()[key], rtol=0, atol=0)
    model = baseline.models['strings']
    target = torch.randn(1, 4, 1024)*0.1
    estimates = model(target.sum(1, keepdim=True))
    original = _loss(model, estimates, target)
    total, parts = _loss(model, estimates, target, return_components=True)
    torch.testing.assert_close(original, total, rtol=0, atol=0)
    torch.testing.assert_close(total, parts['stft_mse']+10*parts['weighted_sdr'], rtol=0, atol=0)
    augmented, parts = _loss(model, estimates, target, log_mel=alternate.log_mel,
                             log_mel_weight=.1, return_components=True)
    torch.testing.assert_close(augmented, original + .1*parts['log_mel'], rtol=0, atol=0)
    assert parts['log_mel'] > 0
    augmented.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_loss_checkpoint_roundtrip_legacy_and_reject_mismatch(tmp_path):
    baseline = Trainer(tiny=True)
    path = tmp_path/'baseline.pt'
    baseline.save(path)
    # Simulate a checkpoint generated before optional loss was implemented.
    state = torch.load(path, weights_only=True)
    del state['loss_settings']
    torch.save(state, path)
    assert Trainer.load(path).loss_settings == {'loss_type': 'baseline'}
    with pytest.raises(ValueError, match='loss settings differ'):
        Trainer.load(path, loss_type='baseline_log_mel')
    alternate = Trainer(tiny=True, loss_type='baseline_log_mel', log_mel_weight=.2)
    alternate.save(path)
    restored = Trainer.load(path)
    assert restored.loss_settings == alternate.loss_settings
    assert restored.loss_settings['log_mel_config'] == LOG_MEL_CONFIG
    for kwargs in ({'loss_type': 'baseline'}, {'log_mel_weight': .1}):
        with pytest.raises(ValueError, match='loss settings differ'):
            Trainer.load(path, **kwargs)
    state = torch.load(path, weights_only=True)
    state['loss_settings']['log_mel_config']['n_fft'] = 512
    torch.save(state, path)
    with pytest.raises(ValueError, match='incompatible'):
        Trainer.load(path)


def test_runtime_resume_preserves_metadata_and_logs_components(tmp_path):
    audio = (np.sin(np.arange(2048)*.06)*.1).astype('float32')
    sf.write(tmp_path/'Violin.wav', audio, 44100, subtype='FLOAT')
    (tmp_path/'manifest.json').write_text(json.dumps({'tracks':[{'stems':{'Violin':'Violin.wav'}}]}))
    release = {'release_id':'fixture', 'train':[{'id':'fixture'}]}
    run = tmp_path/'run'
    kwargs = dict(tiny=True, steps_per_shard=1, segment_seconds=.02,
                  loss_type='baseline_log_mel')
    checkpoint = runtime.train(release, tmp_path, run, **kwargs)
    restored = Trainer.load(checkpoint)
    assert restored.step == 1
    records = [json.loads(line) for line in (run/'train-log.jsonl').read_text().splitlines()]
    for family, total in records[0]['losses'].items():
        pieces = records[0]['loss_components'][family]
        assert total == pieces['total']
        assert total == pytest.approx(pieces['stft_mse']+10*pieces['weighted_sdr']+.1*pieces['log_mel'], rel=1e-5)
    before = (run/'run.json').read_bytes()
    for modification in ({'loss_type':'baseline'}, {'log_mel_weight': .2}):
        with pytest.raises(ValueError, match='loss settings differ'):
            runtime.train(release, tmp_path, run, **dict(kwargs, **modification))
        assert (run/'run.json').read_bytes() == before
    assert runtime.train(release, tmp_path, run, **kwargs) == checkpoint
    assert len((run/'train-log.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('kind,weight', [('bad', .1), ('baseline_log_mel', 0),
                                         ('baseline', -1), ('baseline_log_mel', float('nan'))])
def test_invalid_loss_options(kind, weight):
    with pytest.raises(ValueError):
        loss_config(kind, weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple MPS unavailable')
def test_log_mel_mps_gradient():
    criterion = LogMelLoss().to('mps')
    prediction = torch.randn(1, 2, 4096, device='mps', requires_grad=True)
    reference = torch.zeros_like(prediction)
    loss = criterion(prediction, reference)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_log_mel_resume_matches_uninterrupted_updates(tmp_path):
    audio = (np.sin(np.arange(2048)*.06)*.1).astype('float32')
    sf.write(tmp_path/'Violin.wav', audio, 44100, subtype='FLOAT')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'tracks':[{'stems':{'Violin':'Violin.wav'}}]}))
    arguments = dict(steps=2, segment_seconds=.02, seed=718)
    uninterrupted = Trainer(tiny=True, loss_type='baseline_log_mel', seed=92)
    uninterrupted.train_shard(manifest, **arguments)
    interrupted = Trainer(tiny=True, loss_type='baseline_log_mel', seed=92)
    checkpoint = tmp_path/'partial.pt'

    class Interrupted(Exception):
        pass

    def stop(info, losses):
        interrupted.save(checkpoint, progress=info)
        raise Interrupted()

    with pytest.raises(Interrupted):
        interrupted.train_shard(manifest, on_step=stop, **arguments)
    restored = Trainer.load(checkpoint, loss_type='baseline_log_mel', log_mel_weight=.1)
    restored.train_shard(manifest, start_step=restored.progress['batch_index'], **arguments)
    for key, value in uninterrupted.models.state_dict().items():
        torch.testing.assert_close(value, restored.models.state_dict()[key], rtol=0, atol=0)
    assert restored.step == uninterrupted.step == 2
