"""Optional waveform log-mel objective; no additional packages required.

Magnitude (not power) STFT, periodic Hann, constant centered padding, HTK mel
scale with unit-area triangular filters, natural log after a 1e-5 floor.
Applied to individual family stems including silent targets, not combinations.
"""
import math

import torch
from torch import nn

LOG_MEL_CONFIG = dict(sample_rate=44100, n_fft=2048, hop_length=512,
                      n_mels=128, f_min=0.0, f_max=22050.0,
                      magnitude_floor=1e-5, mel_scale='htk', norm='area',
                      spectrum='magnitude', log='natural', center=True,
                      pad_mode='constant', window='hann_periodic')


def loss_config(loss_type='baseline', log_mel_weight=0.1):
    if loss_type not in ('baseline', 'baseline_log_mel'):
        raise ValueError('loss_type must be baseline or baseline_log_mel')
    if not math.isfinite(log_mel_weight) or log_mel_weight < 0:
        raise ValueError('log_mel_weight must be finite and nonnegative')
    if loss_type == 'baseline':
        return {'loss_type': 'baseline'}
    if log_mel_weight == 0:
        raise ValueError('baseline_log_mel requires a positive log_mel_weight')
    return dict(loss_type=loss_type, log_mel_weight=float(log_mel_weight),
                log_mel_config=dict(LOG_MEL_CONFIG))


class LogMelLoss(nn.Module):
    """Mean absolute difference of log-mel magnitudes, differentiable on device."""
    def __init__(self):
        super().__init__()
        cfg = LOG_MEL_CONFIG
        low = 2595 * math.log10(1 + cfg['f_min']/700)
        high = 2595 * math.log10(1 + cfg['f_max']/700)
        mel = torch.linspace(low, high, cfg['n_mels']+2)
        edges = 700 * (10 ** (mel/2595) - 1)
        freq = torch.linspace(0, cfg['sample_rate']/2, cfg['n_fft']//2+1)
        rising = (freq[None]-edges[:-2, None])/(edges[1:-1]-edges[:-2])[:, None]
        falling = (edges[2:, None]-freq[None])/(edges[2:]-edges[1:-1])[:, None]
        bank = torch.minimum(rising, falling).clamp_min(0)
        bank *= (2/(edges[2:]-edges[:-2]))[:, None]
        self.register_buffer('filters', bank)
        self.register_buffer('window', torch.hann_window(cfg['n_fft']))

    def features(self, waveform):
        cfg = LOG_MEL_CONFIG
        flat = waveform.reshape(-1, waveform.shape[-1])
        spectrum = torch.stft(flat, n_fft=cfg['n_fft'],
                              hop_length=cfg['hop_length'], window=self.window,
                              center=True, pad_mode='constant', return_complex=True)
        mel = self.filters @ spectrum.abs()
        return mel.clamp_min(cfg['magnitude_floor']).log()

    def forward(self, prediction, reference):
        if prediction.shape != reference.shape:
            raise ValueError('Log-mel prediction and reference shapes must match')
        return (self.features(prediction)-self.features(reference)).abs().mean()
