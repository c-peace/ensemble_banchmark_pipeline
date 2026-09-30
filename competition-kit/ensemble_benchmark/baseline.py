"""SynthSOD X-UMX architecture adapted for sharded Colab training.

Upstream: repertorium/SynthSOD-Baseline (conf.yml, train.py); architecture
vendored from Asteroid v0.7.0, with complex-STFT compatibility fixes.
Changes: shard sampling, checkpoint API, bounded inference, default identity
input statistics (learnable), no channel swap for mono. This is not a claim
of reproducing published scores or a released pretrained checkpoint.
"""
import itertools
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from tqdm.auto import tqdm

from .losses import LogMelLoss, loss_config
from .vendor.x_umx import XUMX

SAMPLE_RATE = 44100
FAMILIES = {
    'strings': ['Violin', 'Viola', 'Cello', 'Bass'],
    'woodwinds': ['Flute', 'Clarinet', 'Oboe', 'Bassoon'],
    'brass': ['Horn', 'Trumpet', 'Trombone', 'Tuba'],
    'other': ['Harp', 'Timpani', 'untunedpercussion'],
}
CLASSES = sum(FAMILIES.values(), [])


def _safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Audio path escapes data root: {relative}')
    return path


def load_segment(track, root, length, rng, augment=True):
    """Aligned random crop; sum ALL available stems to build mixture."""
    stems = track['stems']
    track_gain = float(track.get('gain', 1.0))
    if not math.isfinite(track_gain) or track_gain <= 0:
        raise ValueError('Track gain must be positive and finite')
    if not stems or set(stems) - set(CLASSES):
        raise ValueError('Track has missing or unknown stem classes')
    infos = {key: sf.info(_safe_path(root, value)) for key, value in stems.items()}
    frames = next(iter(infos.values())).frames
    if any(x.samplerate != SAMPLE_RATE or x.channels != 1 or x.frames != frames
           for x in infos.values()):
        raise ValueError('Training stems must be aligned 44100 Hz mono')
    if frames <= 0:
        raise ValueError('Empty audio')
    start = rng.randrange(max(1, frames - length + 1))
    sources = np.zeros((len(CLASSES), length), dtype=np.float32)
    for name, relative in stems.items():
        audio, _ = sf.read(_safe_path(root, relative), start=start,
                           frames=length, dtype='float32')
        if not np.isfinite(audio).all():
            raise ValueError(f'Non-finite source: {relative}')
        gain = rng.uniform(0.25, 1.25) if augment else 1.0
        sources[CLASSES.index(name), :len(audio)] = audio * track_gain * gain
    return sources.sum(axis=0), sources


def _loss(model, estimates, target, *, log_mel=None, log_mel_weight=0.1, return_components=False):
    """Combination magnitude MSE + 10 * weighted SDR, as in SynthSOD.

    The weighted-SDR mixture is the sum of family reference targets, matching
    the original MultiDomainLoss, not the full-orchestra network input.
    """
    spec_hat, time_hat = estimates
    batch, count, length = target.shape
    truth_spec = model.encoder[0](target)
    # Encoder spectrogram's mono reduction must not collapse target classes.
    truth_spec = truth_spec.pow(2).sum(-1).sqrt().permute(3, 0, 1, 2)
    truth_time = target.permute(1, 0, 2)[..., :time_hat.shape[-1]]
    time_hat = time_hat[:, :, 0]
    mix = truth_time.sum(0)
    f_losses, t_losses = [], []
    for size in range(1, count):
        for indices in itertools.combinations(range(count), size):
            idx = list(indices)
            pred_f = spec_hat[idx].sum(0)[:, :, 0]
            ref_f = truth_spec[:, :, idx].sum(2)
            f_losses.append((pred_f - ref_f).square().mean())
            pred = time_hat[idx].sum(0)
            ref = truth_time[idx].sum(0)
            noise, pred_noise = mix - ref, mix - pred
            eps = 1e-10
            alpha = ref.square().sum(-1) / (ref.square().sum(-1) + noise.square().sum(-1) + eps)
            def similarity(a, b):
                return (a*b).sum(-1) / ((a.square().sum(-1)+eps).sqrt() * (b.square().sum(-1)+eps).sqrt() + eps)
            t_losses.append((1 - alpha * similarity(pred, ref) - (1-alpha) * similarity(pred_noise, noise)).mean())
    spectral = torch.stack(f_losses).mean()
    temporal = torch.stack(t_losses).mean()
    total = spectral + 10 * temporal
    mel = total.new_zeros(())
    if log_mel is not None:
        mel = log_mel(time_hat, truth_time)
        total = total + log_mel_weight * mel
    if return_components:
        return total, dict(stft_mse=spectral, weighted_sdr=temporal,
                           log_mel=mel, total=total)
    return total


class Trainer:
    """Keep ONE instance across shards, or restore it with ``Trainer.load``."""
    def __init__(self, device='cpu', tiny=False, seed=42,
                 loss_type='baseline', log_mel_weight=0.1,
                 learning_rate=1e-4, weight_decay=1e-5):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer between 0 and 2**63 - 1')
        for name, value in [('learning_rate', learning_rate), ('weight_decay', weight_decay)]:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name == 'learning_rate' and value == 0)):
                raise ValueError(f'Invalid {name}')
        self.optimizer_settings = dict(learning_rate=learning_rate, weight_decay=weight_decay)
        self.loss_settings = loss_config(loss_type, log_mel_weight)
        self.log_mel_weight = log_mel_weight
        self.device = torch.device(device)
        self.log_mel = LogMelLoss().to(self.device) if loss_type == 'baseline_log_mel' else None
        self.loss_components = {}
        self.tiny = tiny
        self.seed = seed
        self.step = 0
        self.progress = {}
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        nfft = 64 if tiny else 4096
        self.config = dict(window_length=nfft, in_chan=nfft,
                           n_hop=16 if tiny else 1024,
                           hidden_size=8 if tiny else 512,
                           nb_layers=1 if tiny else 3, nb_channels=1,
                           sample_rate=SAMPLE_RATE,
                           max_bin=nfft//2+1 if tiny else int(16000*nfft/SAMPLE_RATE)+1,
                           return_time_signals=True)
        self.models = nn.ModuleDict({name: XUMX(sources=classes, **self.config)
                                     for name, classes in FAMILIES.items()}).to(self.device)
        self.optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                           for name, model in self.models.items()}

    def train_shard(self, manifest_path, steps=None, batch_size=1, seed=None,
                    root=None, samples_per_track=64, segment_seconds=6.0,
                    max_grad_norm=5.0, start_step=0, on_step=None, progress_desc='Train'):
        """One shuffled shard pass, or explicit bounded smoke steps.

        Default pass samples each track ``samples_per_track`` times. A non-None
        steps value deliberately caps this and must not be called a full epoch.
        Returns loss scalars without resetting models or Adam state.
        Resume with the same seed and training settings plus start_step equal
        to the next batch index saved by on_step(metadata, losses). Callback
        runs after every complete four-family update; metadata contains
        batch_index (next index), num_steps, global_step. ``steps`` is the
        total pass length, not the number of remaining updates. Pass a distinct
        seed for every epoch/shard; the default is this trainer's initial seed.
        """
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        tracks = data['tracks']
        root = Path(root) if root else manifest_path.parent
        if not tracks or batch_size < 1 or samples_per_track < 1 or segment_seconds <= 0:
            raise ValueError('Invalid training configuration or empty shard')
        if steps is not None and steps < 1:
            raise ValueError('steps must be positive or None')
        shard_seed = self.seed if seed is None else seed
        rng = random.Random(shard_seed)
        order = list(range(len(tracks))) * samples_per_track
        rng.shuffle(order)
        nsteps = math.ceil(len(order)/batch_size) if steps is None else steps
        if not isinstance(start_step, int) or not 0 <= start_step <= nsteps:
            raise ValueError('start_step must be between zero and total steps')
        length = max(self.config['in_chan']+1, round(segment_seconds*SAMPLE_RATE))
        self.models.train()
        history = []
        with tqdm(total=nsteps, initial=start_step, desc=progress_desc, unit='batch',
                  mininterval=1.0, dynamic_ncols=True) as progress:
            for step in range(start_step, nsteps):
                batch_rng = random.Random(f'{shard_seed}:batch:{step}')
                indexes = order[step*batch_size:(step+1)*batch_size]
                if steps is not None and not indexes:
                    indexes = [batch_rng.randrange(len(tracks)) for _ in range(batch_size)]
                batch = [load_segment(tracks[i], root, length, batch_rng) for i in indexes]
                mix = torch.from_numpy(np.stack([x[0] for x in batch]))[:, None].to(self.device)
                targets = torch.from_numpy(np.stack([x[1] for x in batch])).to(self.device)
                losses = {}
                self.loss_components = {}
                for family, model in self.models.items():
                    optimizer = self.optimizers[family]
                    optimizer.zero_grad(set_to_none=True)
                    truth = targets[:, [CLASSES.index(x) for x in FAMILIES[family]]]
                    loss, components = _loss(model, model(mix), truth,
                        log_mel=self.log_mel, log_mel_weight=self.log_mel_weight,
                        return_components=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss in {family}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    losses[family] = float(loss.detach().cpu())
                    self.loss_components[family] = {key: float(value.detach().cpu())
                                                     for key, value in components.items()}
                self.step += 1
                history.append(losses)
                progress.set_postfix(loss=f'{sum(losses.values())/len(losses):.4f}', refresh=False)
                progress.update(1)
                if on_step is not None:
                    on_step({'batch_index': step+1, 'num_steps': nsteps,
                             'global_step': self.step}, losses)
        return history

    def save(self, path, progress=None):
        """Atomic checkpoint includes Adam, all models and RNG state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            self.progress = progress
        state = dict(format_version=1, tiny=self.tiny, seed=self.seed, step=self.step,
                     progress=self.progress, loss_settings=self.loss_settings,
                     optimizer_settings=self.optimizer_settings, models=self.models.state_dict(),
                     optimizers={k:v.state_dict() for k,v in self.optimizers.items()},
                     rng=self.rng.getstate(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     mps_rng=torch.mps.get_rng_state() if self.device.type == 'mps' else None)
        temporary = path.with_name(path.name+'.tmp')
        torch.save(state, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device='cpu', *, loss_type=None, log_mel_weight=None):
        # Checkpoints contain only tensors and Python primitives. Never unpickle
        # arbitrary participant-provided Python objects with weights_only=False.
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format')
        saved_loss = state.get('loss_settings', loss_config())
        canonical = loss_config(saved_loss['loss_type'], saved_loss.get('log_mel_weight', 0.1))
        if saved_loss != canonical:
            raise ValueError('Checkpoint log-mel configuration is incompatible with this code')
        if loss_type is not None or log_mel_weight is not None:
            requested = loss_config(loss_type or saved_loss['loss_type'],
                                    saved_loss.get('log_mel_weight', 0.1) if log_mel_weight is None else log_mel_weight)
            if requested != saved_loss:
                raise ValueError('Checkpoint loss settings differ; use a new run directory')
        trainer = cls(device=device, tiny=state['tiny'], seed=state['seed'],
                      loss_type=saved_loss['loss_type'],
                      log_mel_weight=saved_loss.get('log_mel_weight', 0.1),
                      **state.get('optimizer_settings', dict(learning_rate=1e-4, weight_decay=1e-5)))
        trainer.models.load_state_dict(state['models'])
        for name, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])
        trainer.step, trainer.progress = state['step'], state['progress']
        trainer.rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'])
        if trainer.device.type == 'cuda' and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if trainer.device.type == 'mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'])
        return trainer

    @torch.no_grad()
    def separate(self, mixture, target_classes, chunk_seconds=20.0, wiener=True, show_progress=False):
        """Overlap-add inference, joint mono Wiener EM once per chunk.

        Chunking bounds GPU use; it differs from unchunked paper inference.
        All 15 estimates enter one joint Wiener stage, matching upstream; only
        requested classes are returned.
        """
        mixture = np.asarray(mixture, dtype=np.float32)
        if mixture.ndim != 1 or not len(mixture) or not np.isfinite(mixture).all():
            raise ValueError('Expected non-empty finite mono mixture')
        if not target_classes or len(set(target_classes)) != len(target_classes) or set(target_classes)-set(CLASSES):
            raise ValueError('Invalid target class list')
        if chunk_seconds <= 0:
            raise ValueError('chunk_seconds must be positive')
        self.models.eval()
        nfft, hop = self.config['in_chan'], self.config['n_hop']
        size = max(nfft*2, int(chunk_seconds*SAMPLE_RATE))
        stride = max(1, size//2)
        output = {name: np.zeros(len(mixture), dtype=np.float64) for name in target_classes}
        weights = np.zeros(len(mixture), dtype=np.float64)
        chunks = math.ceil(max(0, len(mixture)-size)/stride)+1
        with tqdm(total=chunks, desc='Audio chunks', unit='chunk', leave=False,
                  disable=not show_progress, mininterval=1.0, dynamic_ncols=True) as progress:
            for start in range(0, len(mixture), stride):
                end = min(len(mixture), start+size)
                raw = mixture[start:end]
                padded_length = math.ceil(max(len(raw), nfft+1)/hop)*hop
                padded = np.pad(raw, (0, padded_length-len(raw)))
                audio = torch.from_numpy(padded)[None, None].to(self.device)
                # Positive endpoints ensure samples at boundaries are not erased.
                window = np.maximum(np.hanning(len(raw)+2)[1:-1], 1e-6)
                magnitudes, time_outputs = [], {}
                for family, model in self.models.items():
                    if not wiener and not set(FAMILIES[family]).intersection(target_classes):
                        continue
                    magnitude, time_audio = model(audio)
                    magnitudes.append(magnitude.cpu())
                    for index, name in enumerate(FAMILIES[family]):
                        if name in target_classes:
                            time_outputs[name] = time_audio[index,0,0].cpu().numpy()
                if wiener:
                    import norbert
                    # Joint Wiener stage across all family outputs (15 classes).
                    v = torch.cat(magnitudes)[:, :, 0].permute(1,3,2,0).numpy().astype(np.float64)
                    reference_model = self.models['strings']
                    transform = reference_model.encoder[0](audio)[0].permute(2,1,0,3).contiguous()
                    x = torch.view_as_complex(transform).cpu().numpy().astype(np.complex128)
                    y = norbert.wiener(v, x, iterations=1, use_softmask=False)
                    for name in target_classes:
                        spectrum = y[:, :, 0, CLASSES.index(name)].T.copy()
                        # MPS supports complex64, not the complex128 Wiener output.
                        if self.device.type == 'mps':
                            spectrum = spectrum.astype(np.complex64)
                        z = torch.from_numpy(spectrum).to(self.device)
                        time_outputs[name] = torch.istft(z, n_fft=nfft, hop_length=hop,
                            window=reference_model.encoder[0].window,
                            length=len(padded)).cpu().numpy()
                for name in target_classes:
                    wave = time_outputs[name]
                    wave = np.pad(wave, (0,max(0,len(raw)-len(wave))))[:len(raw)]
                    output[name][start:end] += wave*window
                weights[start:end] += window
                progress.update(1)
                if end == len(mixture):
                    break
        return {name: (wave/weights).astype(np.float32) for name,wave in output.items()}
