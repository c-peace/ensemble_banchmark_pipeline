import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ensemble_benchmark.baseline import Trainer, load_segment, CLASSES


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        t = np.arange(1024)/44100
        a = (0.1*np.sin(2*np.pi*440*t)).astype('float32')
        b = (0.07*np.sin(2*np.pi*660*t)).astype('float32')
        sf.write(self.root/'a.wav', a, 44100, subtype='FLOAT')
        sf.write(self.root/'b.wav', b, 44100, subtype='FLOAT')
        self.track = {'track_id':'tiny','stems':{'Violin':'a.wav','Flute':'b.wav'}}
        self.path = self.root/'manifest.json'
        self.path.write_text(json.dumps({'tracks':[self.track]}))
        self.mix = a+b

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_stems_form_mix_and_absent_zero(self):
        import random
        mix, targets = load_segment(self.track,self.root,1024,random.Random(1),augment=False)
        np.testing.assert_allclose(mix,self.mix)
        np.testing.assert_allclose(targets[CLASSES.index('Cello')],0)

    def test_actual_training_checkpoint_resume_and_inference(self):
        trainer = Trainer(tiny=True)
        before = next(p for p in trainer.models.parameters() if p.requires_grad).detach().clone()
        result = trainer.train_shard(self.path,steps=1,segment_seconds=.02)
        self.assertEqual(len(result),1)
        self.assertTrue(all(np.isfinite(list(result[0].values()))))
        self.assertFalse(torch.equal(before,next(p for p in trainer.models.parameters() if p.requires_grad)))
        checkpoint = self.root/'latest.pt'
        trainer.save(checkpoint,progress={'next_shard':1})
        loaded = Trainer.load(checkpoint)
        self.assertEqual(loaded.progress, {'next_shard':1})
        self.assertEqual(loaded.step,1)
        # Same Adam state and same random state must produce same next update.
        trainer.train_shard(self.path,steps=1,segment_seconds=.02,seed=88)
        loaded.train_shard(self.path,steps=1,segment_seconds=.02,seed=88)
        for a,b in zip(trainer.models.parameters(),loaded.models.parameters()):
            torch.testing.assert_close(a,b)
        for use_wiener in (False,True):
            out = loaded.separate(self.mix,['Violin','Flute'],chunk_seconds=.01,wiener=use_wiener)
            self.assertEqual(set(out),{'Violin','Flute'})
            for value in out.values():
                self.assertEqual(value.shape,self.mix.shape)
                self.assertTrue(np.isfinite(value).all())

    def test_track_storage_gain(self):
        import random
        track = dict(self.track, gain=1.5)
        mix, targets = load_segment(track, self.root, 1024, random.Random(1), augment=False)
        np.testing.assert_allclose(mix, self.mix*1.5, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(mix, targets.sum(axis=0))
        for invalid in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                load_segment(dict(self.track, gain=invalid), self.root, 1024, random.Random(1))

    def test_mid_shard_resume_exact_updates(self):
        uninterrupted = Trainer(tiny=True, seed=92)
        uninterrupted.train_shard(self.path, steps=4, seed=718,
                                  segment_seconds=.02)
        interrupted = Trainer(tiny=True, seed=92)
        checkpoint = self.root/'mid-shard.pt'
        seen = []
        class StopAfterCheckpoint(Exception):
            pass
        def callback(meta, losses):
            seen.append(meta.copy())
            if meta['batch_index'] == 2:
                interrupted.save(checkpoint, progress=meta)
                raise StopAfterCheckpoint()
        with self.assertRaises(StopAfterCheckpoint):
            interrupted.train_shard(self.path, steps=4, seed=718,
                segment_seconds=.02, on_step=callback)
        self.assertEqual(seen[-1], {'batch_index':2, 'num_steps':4, 'global_step':2})
        resumed = Trainer.load(checkpoint)
        history = resumed.train_shard(self.path, steps=4, seed=718,
            segment_seconds=.02, start_step=resumed.progress['batch_index'])
        self.assertEqual(len(history), 2)
        self.assertEqual(resumed.step, 4)
        for name, tensor in uninterrupted.models.state_dict().items():
            torch.testing.assert_close(tensor, resumed.models.state_dict()[name], rtol=0, atol=0)
        for family in uninterrupted.optimizers:
            a = uninterrupted.optimizers[family].state_dict()['state']
            b = resumed.optimizers[family].state_dict()['state']
            for key in a:
                for field in a[key]:
                    torch.testing.assert_close(a[key][field], b[key][field], rtol=0, atol=0)
        self.assertEqual(resumed.train_shard(self.path, steps=4, start_step=4), [])
        with self.assertRaises(ValueError):
            resumed.train_shard(self.path, steps=4, start_step=5)

    def test_validation_rejects_escape_and_bad_shapes(self):
        import random
        with self.assertRaises(ValueError):
            load_segment({'stems':{'Violin':'../escape.wav'}},self.root,20,random.Random())
        with self.assertRaises(ValueError):
            Trainer(tiny=True).separate(np.zeros((2,10)),['Violin'])

    def test_resumed_progress_starts_at_saved_batch_and_closes_on_interrupt(self):
        from unittest.mock import patch
        trainer = Trainer(tiny=True)
        with patch('ensemble_benchmark.baseline.tqdm') as factory:
            progress = factory.return_value.__enter__.return_value
            def stop(meta, losses):
                raise RuntimeError('interrupt after completed batch')
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                trainer.train_shard(self.path, steps=3, start_step=2,
                                    segment_seconds=.02, on_step=stop)
            self.assertEqual(factory.call_args.kwargs['initial'], 2)
            self.assertEqual(factory.call_args.kwargs['total'], 3)
            progress.update.assert_called_once_with(1)
            self.assertIn('loss', progress.set_postfix.call_args.kwargs)
            factory.return_value.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
