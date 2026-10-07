"""Execute the copy/paste cells with real training and local Git, fake cloud boundaries."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from ensemble_benchmark.config import load_config
from test_local_runtime import prepared  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = json.loads((ROOT / "pipeline.ipynb").read_text())
CELLS = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]


def test_notebook_compiles_and_has_no_stored_outputs():
    assert len(CELLS) == 6
    for i, source in enumerate(CELLS):
        compile(source, f"cell-{i + 1}", "exec")
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []


class FakeArtifact:
    def __init__(self, name, type, **kwargs):
        self.files = {}
        self.waited = False

    def add_file(self, path, name):
        assert Path(path).is_file()
        self.files[name] = Path(path)

    def wait(self):
        self.waited = True


class FakeSummary:
    """Match W&B's keyed access without dict membership/iteration support."""

    def __init__(self):
        self.values = {}

    def keys(self):
        return list(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]


class FakeRun:
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.summary = FakeSummary()
        self.events = []
        self.finished = False
        self.url = "https://wandb.example/test"

    def log(self, event):
        self.events.append(event)

    def log_artifact(self, artifact):
        self.artifact = artifact
        return artifact

    def finish(self):
        self.finished = True


@pytest.mark.parametrize("evaluate", [True, False])
def test_cells_data_to_real_training_and_artifact(prepared, tmp_path, monkeypatch, evaluate):  # noqa: F811
    import numpy as np
    import soundfile as sf
    import yaml
    from ensemble_benchmark import cloud, runtime

    # Real SDR needs at least five active one-second frames.
    for split in ("train", "validation"):
        audio = prepared / split / "Violin.wav"
        sf.write(audio, np.sin(np.arange(6 * 44100) * .03).astype("float32") * .1,
                 44100, subtype="FLOAT")
        manifest_path = prepared / split / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = manifest["tracks" if split == "train" else "scenes"][0]
        record["num_samples"] = 6 * 44100
        if split == "validation":
            record.update(parent_id="urmp_43", work_group="fixture")
        record["sha256"]["Violin.wav"] = runtime.sha256(audio)
        manifest_path.write_text(json.dumps(manifest))
    config = load_config(ROOT / "configs/experiments/smoke.yaml")
    config["training"]["steps_per_shard"] = 1
    config["evaluation"].update(enabled=evaluate, max_seconds=6.0)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    colab = ModuleType("google.colab")
    colab.auth = SimpleNamespace(authenticate_user=lambda **kw: calls.append(("auth", kw)))
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def download(uri, destination, **kwargs):
        calls.append(("download", uri, kwargs))
        return prepared

    monkeypatch.setattr(cloud, "download_data", download)
    sdk = ModuleType("wandb")
    sdk.run = None
    sdk.login = lambda **kw: None
    sdk.init = lambda **kw: FakeRun(**kw)
    sdk.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    scope = dict(CODE_ROOT=tmp_path, CONFIG_FILE=path.name, Path=Path,
                 REPOSITORY="owner/repo", CODE_COMMIT="a" * 40,
                 secret=lambda name: None)
    for index in range(1, 6):
        source = CELLS[index].replace('"/content/ensemble-runs"', repr(str(tmp_path / "runs")))
        exec(compile(source, f"cell-{index + 1}", "exec"), scope)
    run = scope["run"]
    assert calls[0][0] == "auth" and calls[1][0] == "download"
    assert run.config["github_commit"] == "a" * 40
    assert len(run.events) == 1 and "train/strings" in run.events[0]
    assert scope["CHECKPOINT"].is_file()
    assert run.finished and run.artifact.waited
    assert set(run.artifact.files) == {"checkpoint.pt", "config.yaml"} | (
        {"evaluation.json"} if evaluate else set()
    )
    if evaluate:
        assert run.summary["evaluation"]["diagnostic"]
    else:
        assert scope["REPORT_PATH"] is None


def test_github_cell_download_rerun_dirty_and_changed_ref(tmp_path, monkeypatch):
    """Run actual Git fetch/checkout locally; no remote access or pip install."""
    repo = tmp_path / "remote"
    repo.mkdir()
    original = subprocess.run

    def git(*args):
        return original(["git", "-C", str(repo), *args], check=True,
                        capture_output=True, text=True).stdout.strip()

    original(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "sample.txt").write_text("first")
    git("add", ".")
    git("commit", "-m", "Test fixture")
    target = tmp_path / "checkout"
    installs = []

    def run(command, **kwargs):
        if command[:3] == [sys.executable, "-m", "pip"]:
            installs.append(command)
            return subprocess.CompletedProcess(command, 0)
        if command[-4:-1] == ["remote", "add", "origin"]:
            command = [*command[:-1], str(repo)]
        if command[-3:] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(command, 0,
                stdout="https://github.com/c-peace/ensemble_banchmark_pipeline.git\n")
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    source = CELLS[0].replace('"/content/ensemble-code"', repr(str(target)))
    scope = {}
    exec(source, scope)
    first_commit = scope["CODE_COMMIT"]
    assert (target / "sample.txt").read_text() == "first"
    exec(source, scope)
    assert len(installs) == 2
    (target / "sample.txt").write_text("my unsaved work")
    with pytest.raises(ValueError, match="수정"):
        exec(source, scope)
    assert (target / "sample.txt").read_text() == "my unsaved work"
    (target / "sample.txt").write_text("first")
    (repo / "sample.txt").write_text("second")
    git("add", ".")
    git("commit", "-m", "Updated fixture")
    with pytest.raises(ValueError, match="변경"):
        exec(source, scope)
    assert original(["git", "-C", str(target), "rev-parse", "HEAD"],
                    check=True, capture_output=True, text=True).stdout.strip() == first_commit


def test_save_rejects_stale_report_and_changed_config(tmp_path):
    import yaml

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"updated model")
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    config = load_config(ROOT / "configs/experiments/smoke.yaml")
    snapshot = tmp_path / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config))
    scope = dict(Path=Path, yaml=yaml, CHECKPOINT=checkpoint, REPORT_PATH=report,
                 config=config, CONFIG_SNAPSHOT=snapshot)
    with pytest.raises(ValueError, match="현재 모델"):
        exec(CELLS[5], scope)
    config["training"]["epochs"] += 1
    for cell in CELLS[3:]:
        with pytest.raises(ValueError, match="설정이 변경"):
            exec(cell, scope)


def test_training_failure_clears_previous_results(tmp_path, monkeypatch):
    import yaml
    from ensemble_benchmark import runtime

    config = load_config(ROOT / "configs/experiments/smoke.yaml")
    snapshot = tmp_path / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config))
    run = FakeRun(config=config)
    run.summary["evaluation"] = {"old": True}
    scope = dict(config=config, CONFIG_SNAPSHOT=snapshot, yaml=yaml, run=run,
                 CHECKPOINT=tmp_path / "old.pt", REPORT_PATH=tmp_path / "old.json",
                 DATA_ROOT=tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("data unavailable")

    monkeypatch.setattr(runtime, "build_local_release", fail)
    with pytest.raises(RuntimeError, match="data unavailable"):
        exec(CELLS[3], scope)
    assert scope["CHECKPOINT"] is None and scope["REPORT_PATH"] is None
    assert "evaluation" not in run.summary.keys()
