"""Seed, optimizer and event contracts exercised with the actual tiny model."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch
from ensemble_benchmark import runtime
from ensemble_benchmark.baseline import Trainer


@pytest.fixture
def training(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "audio"
    data.mkdir()
    audio = (0.1 * np.sin(np.arange(2048) * 0.08)).astype("float32")
    sf.write(data / "violin.wav", audio, 44100, subtype="FLOAT")
    (data / "manifest.json").write_text(
        json.dumps({"tracks": [{"track_id": "one", "stems": {"Violin": "violin.wav"}}]})
    )
    release = {"release_id": "tiny", "train": [{"id": "tiny"}]}

    def execute(run="run", **kwargs):
        return runtime.train(
            release,
            data,
            tmp_path / run,
            tiny=True,
            steps_per_shard=2,
            segment_seconds=0.02,
            **kwargs,
        )

    return execute


def test_events_metadata_preservation_and_resume(training, tmp_path):
    events = []
    path = training(
        event_callback=events.append,
        save_every=1,
        seed=17,
        learning_rate=0.002,
        weight_decay=0,
        max_grad_norm=2,
    )
    assert [event["type"] for event in events] == [
        "checkpoint",
        "train",
        "checkpoint",
        "train",
        "checkpoint",
        "checkpoint",
    ]
    assert [e["global_step"] for e in events if e["type"] == "checkpoint"] == [0, 1, 2, 2]
    batches = [e for e in events if e["type"] == "train"]
    assert all(e["losses"] and e["loss_components"] for e in batches)
    assert 0 <= batches[0]["elapsed_seconds"] <= batches[1]["elapsed_seconds"]
    assert all(e["path"] == str(path) for e in events if e["type"] == "checkpoint")
    record = tmp_path / "run/run.json"
    metadata = json.loads(record.read_text())
    metadata["initial_lineage"] = {"commit": "preserve-me"}
    record.write_text(json.dumps(metadata))
    original = record.read_bytes()
    checkpoint = path.read_bytes()
    training(seed=17, learning_rate=0.002, weight_decay=0, max_grad_norm=2)
    assert record.read_bytes() == original
    assert path.read_bytes() == checkpoint
    for changed in (
        {"seed": 18},
        {"learning_rate": 0.003},
        {"weight_decay": 0.01},
        {"max_grad_norm": 3},
    ):
        options = dict(seed=17, learning_rate=0.002, weight_decay=0, max_grad_norm=2)
        options.update(changed)
        with pytest.raises(ValueError, match="다른 데이터/학습 설정"):
            training(**options)
    assert record.read_bytes() == original
    loaded = Trainer.load(path)
    assert loaded.seed == 17
    assert loaded.optimizer_settings == {"learning_rate": 0.002, "weight_decay": 0}
    assert all(o.param_groups[0]["lr"] == 0.002 for o in loaded.optimizers.values())


def test_seed_changes_actual_model_and_default_signature(training, tmp_path):
    first = Trainer.load(training(run="first"))
    second = Trainer.load(training(run="second", seed=43))
    assert any(
        not torch.equal(a, b) for a, b in zip(first.models.parameters(), second.models.parameters())
    )
    settings = json.loads((tmp_path / "first/run.json").read_text())["settings"]
    assert settings == dict(
        tiny=True, steps_per_shard=2, batch_size=1, samples_per_track=64, segment_seconds=0.02
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
        {"seed": 1.5},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"max_grad_norm": float("inf")},
        {"event_callback": "not callable"},
    ],
)
def test_invalid_configuration_creates_no_run(training, tmp_path, kwargs):
    with pytest.raises(ValueError):
        training(**kwargs)
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "cache").exists()


def test_callback_errors_are_propagated(training):
    def fail(event):
        raise RuntimeError("caller failure")

    with pytest.raises(RuntimeError, match="caller failure"):
        training(event_callback=fail)


def test_legacy_checkpoint_optimizer_defaults(tmp_path):
    trainer = Trainer(tiny=True)
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    del state["optimizer_settings"]
    torch.save(state, path)
    loaded = Trainer.load(path)
    assert loaded.optimizer_settings == {"learning_rate": 1e-4, "weight_decay": 1e-5}


def test_nondefault_interrupted_resume_matches_uninterrupted(training):
    options = dict(
        seed=91, learning_rate=0.0003, weight_decay=0.0002, max_grad_norm=1.5, save_every=1
    )
    uninterrupted = Trainer.load(training(run="whole", **options))

    def interrupt(event):
        if event["type"] == "checkpoint" and event["global_step"] == 1:
            raise RuntimeError("disconnect")

    with pytest.raises(RuntimeError, match="disconnect"):
        training(run="resumed", event_callback=interrupt, **options)
    resumed = Trainer.load(training(run="resumed", **options))
    assert resumed.step == uninterrupted.step == 2
    for key, tensor in uninterrupted.models.state_dict().items():
        torch.testing.assert_close(tensor, resumed.models.state_dict()[key], rtol=0, atol=0)
    for family in uninterrupted.optimizers:
        expected = uninterrupted.optimizers[family].state_dict()["state"]
        actual = resumed.optimizers[family].state_dict()["state"]
        for parameter, state in expected.items():
            for key, value in state.items():
                torch.testing.assert_close(value, actual[parameter][key], rtol=0, atol=0)
