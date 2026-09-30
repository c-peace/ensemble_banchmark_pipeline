"""Permanent local data must survive training, interruption and evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from ensemble_benchmark import runtime


class FakeTrainer:
    """Deterministic counter models saved model + optimizer progress, without GPU."""

    interruption = False
    calls = []

    def __init__(self, device="cpu", tiny=False):
        self.progress = {}
        self.updates = 0
        self.tiny = tiny

    @classmethod
    def load(cls, path, device="cpu"):
        obj = cls()
        state = json.loads(Path(path).read_text())
        obj.progress, obj.updates = state["progress"], state["updates"]
        return obj

    def save(self, path, progress):
        self.progress = progress
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dict(progress=progress, updates=self.updates)))

    def train_shard(self, manifest, steps, start_step, on_step, **kwargs):
        self.calls.append((Path(manifest).parent.name, start_step, self.updates))
        for index in range(start_step, steps):
            self.updates += 1
            on_step(
                dict(batch_index=index + 1, num_steps=steps, global_step=self.updates),
                {"loss": 1.0},
            )
            if type(self).interruption:
                type(self).interruption = False
                raise RuntimeError("simulated Colab disconnect")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    selection = {
        "train": {"allowed_sod_ids": ["35", "507", "968"]},
        "synthsod_scenes": [],
        "urmp": [{"id": 43, "split": "validation"}],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(runtime, "_SELECTION_PATH", selection_path)
    for split in ("train", "validation"):
        folder = root / split
        folder.mkdir(parents=True)
        audio = folder / "Violin.wav"
        sf.write(
            audio, np.sin(np.arange(44100) * 0.03).astype("float32") * 0.1, 44100, subtype="FLOAT"
        )
        record = {
            "split": split,
            "sample_rate": 44100,
            "num_samples": 44100,
            "sha256": {"Violin.wav": runtime.sha256(audio)},
        }
        if split == "train":
            record.update(track_id="sod_35", stems={"Violin": "Violin.wav"})
            manifest = dict(split=split, complete=False, tracks=[record])
        else:
            record.update(
                scene_id="urmp_43",
                dataset="URMP",
                target_classes=["Violin"],
                mixture="Violin.wav",
                references={"Violin": "Violin.wav"},
            )
            manifest = dict(
                split=split, complete=True, scenes=[record], expected_scene_ids=["urmp_43"]
            )
        (folder / "manifest.json").write_text(json.dumps(manifest))
    return root


def local_release(root, **kwargs):
    with pytest.warns(UserWarning, match="silent"):
        return runtime.build_local_release(root, **kwargs)


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, runtime.sha256(p))
        for p in root.rglob("*")
        if p.is_file()
    }


def test_persistent_resume_without_network_or_source_mutation(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "Trainer", FakeTrainer)
    release = local_release(prepared)
    assert (release["train_count"], release["validation_count"]) == (1, 1)
    before = snapshot(prepared)
    FakeTrainer.interruption = True
    FakeTrainer.calls = []
    run = tmp_path / "run"
    args = dict(epochs=2, steps_per_shard=2, save_every=1, tiny=True)
    with pytest.raises(RuntimeError, match="disconnect"):
        runtime.train_local(release, prepared, run, **args)
    checkpoint = runtime.train_local(release, prepared, run, **args)
    assert json.loads(checkpoint.read_text())["updates"] == 4
    assert [call[1] for call in FakeTrainer.calls] == [0, 1, 0]
    assert snapshot(prepared) == before
    with pytest.raises(ValueError, match="다른 데이터"):
        runtime.train_local(release, prepared, run, **dict(args, batch_size=2))


@pytest.mark.parametrize("location", ["same", "child", "parent"])
def test_reject_output_overlap(prepared, location):
    run = {"same": prepared, "child": prepared / "runs", "parent": prepared.parent}[location]
    release = local_release(prepared)
    before = snapshot(prepared)
    with pytest.raises(ValueError, match="overlap"):
        runtime.train_local(release, prepared, run)
    assert snapshot(prepared) == before


def test_changed_manifest_rejected_before_output(prepared, tmp_path):
    release = local_release(prepared)
    path = prepared / "train/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tracks"][0]["gain"] = 2.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Local data changed"):
        runtime.train_local(release, prepared, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_unknown_missing_train_and_incomplete_validation_rejected(prepared):
    path = prepared / "train/manifest.json"
    original = path.read_text()
    manifest = json.loads(original)
    manifest["tracks"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing train"):
        runtime.build_local_release(prepared)
    path.write_text(original)
    path = prepared / "validation/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        runtime.build_local_release(prepared)


def test_audio_path_escape_rejected(prepared):
    path = prepared / "train/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tracks"][0]["stems"]["Violin"] = "../validation/Violin.wav"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        runtime.build_local_release(prepared)


def test_smoke_evaluation_reads_only_excerpt_and_keeps_source(prepared, tmp_path, monkeypatch):
    class Separator(FakeTrainer):
        def separate(self, mixture, targets, **kwargs):
            assert len(mixture) == 4410
            return {target: mixture.copy() for target in targets}

    monkeypatch.setattr(runtime, "Trainer", Separator)
    Separator.interruption = False
    release = local_release(prepared, smoke=True)
    run = tmp_path / "run"
    before = snapshot(prepared)
    runtime.train_local(release, prepared, run, tiny=True, steps_per_shard=1)
    seen_frames = []
    read = runtime.sf.read

    def bounded_read(path, **kwargs):
        if Path(path).is_relative_to(prepared):
            seen_frames.append(kwargs.get("frames"))
        return read(path, **kwargs)

    monkeypatch.setattr(runtime.sf, "read", bounded_read)
    monkeypatch.setattr(runtime, "score_dataset", lambda *args, **kwargs: {"score": None})
    report, output = runtime.evaluate_local(
        release, prepared, run, allow_partial=True, max_seconds=0.1
    )
    assert report["diagnostic"] is True and output.exists()
    assert seen_frames and all(x == 4410 for x in seen_frames)
    assert snapshot(prepared) == before
    snap = json.loads((run / "local-manifests/validation/manifest.json").read_text())
    assert [s["scene_id"] for s in snap["scenes"]] == ["urmp_43"]
    with pytest.raises(ValueError, match="allow_partial"):
        runtime.evaluate_local(release, prepared, run)


def test_fingerprint_survives_copy_and_detects_same_size_audio_edit(prepared, tmp_path):
    import shutil

    release = local_release(prepared)
    copied = tmp_path / "copied"
    shutil.copytree(prepared, copied, copy_function=shutil.copyfile)
    assert local_release(copied)["fingerprint"] == release["fingerprint"]
    audio = copied / "train/Violin.wav"
    with audio.open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(b"\x01\x02\x03\x04")
    with pytest.raises(ValueError, match="SHA256 differs"):
        runtime.build_local_release(copied)
