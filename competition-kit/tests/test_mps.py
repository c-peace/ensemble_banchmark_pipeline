"""Run with access to Apple GPU; CPU-only runners skip this integration check."""
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from ensemble_benchmark.baseline import Trainer


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple MPS unavailable')
def test_mps_train_checkpoint_rng_and_wiener_inference(tmp_path):
    torch.set_num_threads(4)
    audio=(np.sin(np.arange(4096)*.05)*.1).astype('float32')
    sf.write(tmp_path/'Violin.wav',audio,44100,subtype='FLOAT')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'tracks':[{'stems':{'Violin':'Violin.wav'}}]}))
    trainer=Trainer(device='mps',tiny=True)
    history=trainer.train_shard(manifest,steps=1,segment_seconds=.02)
    assert all(np.isfinite(list(history[0].values())))
    checkpoint=tmp_path/'latest.pt'
    trainer.save(checkpoint)
    rng=torch.mps.get_rng_state().clone()
    loaded=Trainer.load(checkpoint,device='mps')
    assert torch.equal(torch.mps.get_rng_state(),rng)
    assert loaded.step==1
    # Check resumed optimizer state can take an actual MPS training step.
    loaded.train_shard(manifest,steps=1,segment_seconds=.02)
    assert loaded.step==2
    result=loaded.separate(audio,['Violin'],chunk_seconds=.02,wiener=True)
    assert result['Violin'].shape==audio.shape
    assert np.isfinite(result['Violin']).all()
    # Device portability: an MPS checkpoint still loads on CPU.
    assert Trainer.load(checkpoint,device='cpu').step==1
