"""Validate the single-upload notebook without downloading gated weights."""
import ast
import gc
import json
from pathlib import Path

import yaml
from test_sam_audio_notebook import evaluation_scope  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = json.loads((ROOT / 'sam_audio_large_colab.ipynb').read_text())
CELLS = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


def test_upload_notebook_has_embedded_full_evaluation_settings():
    for index, source in enumerate(CELLS):
        compile(source, f'colab-cell-{index}', 'exec')
    config_node = next(node for node in ast.parse(CELLS[1]).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'CONFIG_YAML'
                               for t in node.targets))
    config = yaml.safe_load(ast.literal_eval(config_node.value))
    assert config['parameters']['model_id'] == 'facebook/sam-audio-large'
    assert config['experiment']['name'] == 'sam-audio-large-pretrained'
    assert "if p['model_id'] != 'facebook/sam-audio-large':" in CELLS[2]
    assert 'facebook/sam-audio-small' not in ''.join(CELLS)
    assert config['parameters']['max_scenes'] is None
    assert config['parameters']['model_revision'] == '5f2cd3a9471a08c7282c06036be6893e18de8b70'
    assert config['data']['smoke'] is False
    assert "splits=('validation',)" in CELLS[1]
    assert 'files.download(str(bundle))' in CELLS[5]
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None


def test_upload_notebook_runs_real_scoring_with_fake_inference(evaluation_scope):  # noqa: F811
    scope = evaluation_scope
    scope['gc'] = gc
    exec(CELLS[4], scope)
    report = json.loads(scope['REPORT_PATH'].read_text())
    assert report['diagnostic'] is False
    assert report['overall'] is not None
    assert len(scope['calls']) == 4
    assert 'model' not in scope  # GPU memory is freed before CPU scoring.


def test_install_logs_forward_stdout_stderr_and_propagate_failure(capsys):
    import subprocess
    import sys

    import pytest

    node = next(node for node in ast.parse(CELLS[0]).body
                if isinstance(node, ast.FunctionDef) and node.name == 'run_live')
    scope = {'subprocess': subprocess}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<run_live>', 'exec'), scope)
    scope['run_live']([sys.executable, '-u', '-c',
                       "import sys; print('install log'); print('error log', file=sys.stderr)"])
    output = capsys.readouterr().out
    assert 'install log' in output and 'error log' in output
    with pytest.raises(subprocess.CalledProcessError) as error:
        scope['run_live']([sys.executable, '-c', 'raise SystemExit(2)'])
    assert error.value.returncode == 2


def test_install_preserves_existing_gpu_stack():
    from types import SimpleNamespace

    node = next(node for node in ast.parse(CELLS[0]).body
                if isinstance(node, ast.FunctionDef)
                and node.name == 'preserved_gpu_requirements')
    packages = [('torch', '2.9.0+cu128'), ('torchaudio', '2.9.0+cu128'),
                ('torchvision', '0.24.0+cu128'), ('nvidia_cublas_cu12', '12.8.4.1'),
                ('triton', '3.5.0'), ('unrelated', '1.0')]
    scope = {'metadata': SimpleNamespace(distributions=lambda: [
        SimpleNamespace(metadata={'Name': name}, version=version)
        for name, version in packages])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<constraints>', 'exec'), scope)
    assert scope['preserved_gpu_requirements']() == sorted([
        'torch==2.9.0+cu128', 'torchaudio==2.9.0+cu128', 'torchvision==0.24.0+cu128',
        'nvidia-cublas-cu12==12.8.4.1', 'triton==3.5.0'])
    assert "'--constraint', str(constraints)" in CELLS[0]
    assert "if shutil.which('ffmpeg'):" in CELLS[0]
    assert 'torch==2.10.0' not in CELLS[0]
    assert "'--upgrade'" not in CELLS[0]


def test_pytorch_backend_is_selected_before_model_imports():
    for source in (CELLS[0], CELLS[3]):
        tree = ast.parse(source)
        prefix = tree.body[:4]
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'USE_TF': '1', 'USE_TORCH': '0', 'USE_FLAX': '1'}):
            exec(compile(ast.Module(body=prefix, type_ignores=[]), '<backend>', 'exec'), {})
            assert os.environ['USE_TF'] == '0'
            assert os.environ['USE_FLAX'] == '0'
            assert os.environ['USE_TORCH'] == '1'
        assert source.index("os.environ['USE_TF']") < source.index('import subprocess' if source == CELLS[0] else 'import torch')
