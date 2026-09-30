import json

import numpy as np
import pytest
import soundfile as sf
from ensemble_benchmark import evaluation_metrics as metrics


def test_joint_metric_direction_and_nonfinite():
    rng = np.random.default_rng(410)
    reference = rng.normal(size=(2, 24000))
    clean = reference + rng.normal(scale=0.005, size=reference.shape)
    interference = clean + 0.5 * reference[::-1]
    artifacts = clean + rng.normal(scale=0.5, size=reference.shape)
    good = metrics.score_scene_arrays(reference, clean, sample_rate=8000, filters_len=8)
    mixed = metrics.score_scene_arrays(reference, interference, sample_rate=8000, filters_len=8)
    noisy = metrics.score_scene_arrays(reference, artifacts, sample_rate=8000, filters_len=8)
    assert good[0]["sir"] > mixed[0]["sir"] + 10
    assert good[0]["sar"] > noisy[0]["sar"] + 10
    reference[:, :8000] = 0
    silent = metrics.score_scene_arrays(reference, clean, sample_rate=8000, filters_len=8)
    assert silent[0]["active_frames"] == 2
    muted = metrics.score_scene_arrays(
        reference, np.zeros_like(reference), sample_rate=8000, filters_len=8
    )
    assert muted[0]["sir"] is None
    assert muted[0]["sir_diagnostics"]["reason"] == "at_least_one_entire_source_is_silent"


def fixture_run(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictions = tmp_path / "predictions"
    (predictions / "scene").mkdir(parents=True)
    sf.write(predictions / "scene" / "Violin.wav", np.ones(441), 44100, subtype="FLOAT")
    scene = dict(
        scene_id="scene",
        target_classes=["Violin"],
        num_samples=441,
        references={"Violin": "Violin.wav"},
        sample_rate=44100,
        dataset="SynthSOD",
        parent_id="parent",
        work_group="work",
        part_count=4,
    )
    manifest = {"scenes": [scene]}
    report = dict(
        checkpoint_sha256=metrics.file_hash(checkpoint),
        release_id="release",
        scene_scores=[
            dict(
                scene_id="scene",
                instrument="Violin",
                sdr=3.0,
                dataset="SynthSOD",
                parent_id="parent",
                work_group="work",
            )
        ],
    )
    return checkpoint, predictions, manifest, report


def test_legacy_adoption_hash_validation_and_checkpoint_refusal(tmp_path):
    checkpoint, predictions, manifest, report = fixture_run(tmp_path)
    path = tmp_path / "provenance.json"
    args = (report, manifest, predictions, checkpoint, "release", path)
    with pytest.raises(ValueError, match="Legacy"):
        metrics.validate_reuse(*args)
    adopted = metrics.validate_reuse(*args, trust_legacy=True)
    assert adopted["legacy_adopted"]
    assert metrics.validate_reuse(*args) == adopted
    sf.write(predictions / "scene" / "Violin.wav", np.zeros(441), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="prediction_sha256"):
        metrics.validate_reuse(*args)
    checkpoint.write_bytes(b"new checkpoint")
    with pytest.raises(ValueError, match="different checkpoint"):
        metrics.validate_reuse(*args)


def test_scene_cache_and_ensemble_grouping(tmp_path, monkeypatch):
    checkpoint, predictions, manifest, report = fixture_run(tmp_path)
    sf.write(tmp_path / "Violin.wav", np.ones(441), 44100, subtype="FLOAT")
    calls = []

    def score(*args):
        calls.append(1)
        return [{"sir": 2.0, "sar": 4.0}]

    monkeypatch.setattr(metrics, "score_scene_arrays", score)
    result = metrics.add_supplementary(report, manifest, tmp_path, predictions, tmp_path / "cache")
    assert result["supplementary"]["sir"]["overall"] == 2
    assert result["ensemble_groups"]["part_count=4"]["sdr"]["overall"] == 3
    assert report["scene_scores"][0].get("sir") is None
    assert (
        metrics.add_supplementary(report, manifest, tmp_path, predictions, tmp_path / "cache")
        == result
    )
    assert len(calls) == 1
    sf.write(tmp_path / "Violin.wav", np.full(441, 0.5), 44100, subtype="FLOAT")
    metrics.add_supplementary(report, manifest, tmp_path, predictions, tmp_path / "cache")
    assert len(calls) == 2


def test_no_silent_good_mean():
    rows = [
        dict(
            scene_id="x",
            dataset="SynthSOD",
            instrument="Violin",
            parent_id="p",
            work_group="w",
            sir=None,
        ),
        dict(
            scene_id="y",
            dataset="SynthSOD",
            instrument="Violin",
            parent_id="q",
            work_group="z",
            sir=20.0,
        ),
    ]
    result = metrics.summarize(rows, "sir")
    assert result["overall"] is None
    assert result["datasets"]["SynthSOD"]["instruments"]["Violin"] == 20
    assert result["datasets"]["SynthSOD"]["instrument_coverage"]["Violin"]["complete"] is False


def test_single_source_silence_has_explicit_partial_coverage():
    rng = np.random.default_rng(413)
    refs = rng.normal(size=(2, 24000))
    refs[1, :8000] = 0
    estimates = refs + rng.normal(scale=0.01, size=refs.shape)
    rows = metrics.score_scene_arrays(refs, estimates, sample_rate=8000, filters_len=8)
    assert rows[0]["sir"] is not None
    assert rows[0]["sir_diagnostics"]["valid_frames"] == 2
    assert rows[0]["sir_diagnostics"]["undefined_frames"] == 1
    assert rows[0]["sir_diagnostics"]["complete"] is False


def test_runtime_reuse_never_loads_model_or_overwrites_sdr(tmp_path, monkeypatch):
    from ensemble_benchmark import runtime

    checkpoint, predictions, manifest, report = fixture_run(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    checkpoint.rename(tmp_path / "checkpoints" / "latest.pt")
    (tmp_path / "run.json").write_text(json.dumps({"release_id": "release"}))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    results = tmp_path / "results"
    results.mkdir()
    original = json.dumps(report)
    (results / "validation.json").write_text(original)

    def fail(*args, **kwargs):
        raise AssertionError("Model must not be loaded during prediction reuse")

    monkeypatch.setattr(runtime.Trainer, "load", fail)
    monkeypatch.setattr(
        metrics, "add_supplementary", lambda report, *a: dict(report, supplementary={"test": True})
    )
    output, path = runtime.evaluate(
        {"release_id": "release", "validation": {}},
        tmp_path,
        tmp_path,
        reuse_predictions=True,
        trust_legacy_predictions=True,
        include_metrics=True,
    )
    assert output["supplementary"] == {"test": True}
    assert path.name == "validation-extra.json"
    assert (results / "validation.json").read_text() == original
    assert (results / "prediction-provenance.json").is_file()


def test_reuse_requires_same_known_inference_settings(tmp_path):
    checkpoint, predictions, manifest, report = fixture_run(tmp_path)
    path = tmp_path / "provenance.json"
    provenance = metrics.prediction_provenance(
        manifest, predictions, checkpoint, "release", {"chunk_seconds": 20.0, "wiener": True}
    )
    metrics.write_json(path, provenance)
    args = (report, manifest, predictions, checkpoint, "release", path)
    metrics.validate_reuse(*args, inference_settings={"chunk_seconds": 20.0, "wiener": True})
    with pytest.raises(ValueError, match="chunk_seconds"):
        metrics.validate_reuse(*args, inference_settings={"chunk_seconds": 6.0, "wiener": True})
