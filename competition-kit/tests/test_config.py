from pathlib import Path

import pytest
from ensemble_benchmark.config import load_config, validate_config, validate_run_id

ROOT = Path(__file__).resolve().parents[2]


def test_configs_only_differ_in_loss():
    a = load_config(ROOT / "configs/experiments/baseline.yaml")
    b = load_config(ROOT / "configs/experiments/log_mel.yaml")
    assert b["training"].pop("loss_type") == "baseline_log_mel"
    assert a["training"].pop("loss_type") == "baseline"
    a.pop("experiment")
    b.pop("experiment")
    assert a == b


@pytest.mark.parametrize(
    "settings",
    [
        {"training": {"seed": True}},
        {"training": {"learning_rate": float("nan")}},
        {"training": {"batch_szie": 8}},
        {"data": {"sample_rate": 16000}},
        {"training": {"steps_per_shard": 1}},
        {"evaluation": {"max_seconds": 2}},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        validate_config(settings)


def test_duplicate_yaml_and_unsafe_run_ids(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("training:\n  seed: 42\n  seed: 1\n")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)
    for value in ("../old", "", "foo/bar", ".secret"):
        with pytest.raises(ValueError):
            validate_run_id(value)




def test_custom_model_parameters_need_no_shared_code_change():
    supplied = {"parameters": {"dropout": 0.2, "hidden_size": 128,
                               "layers": [64, 128], "custom_loss": "my_loss"}}
    config = validate_config(supplied)
    assert config["parameters"] == supplied["parameters"]
    config["parameters"]["layers"].append(256)
    assert supplied["parameters"]["layers"] == [64, 128]


@pytest.mark.parametrize("value", [[], {"x": float("nan")}, {"x": object()}])
def test_invalid_custom_parameters(value):
    with pytest.raises(ValueError, match="parameters"):
        validate_config({"parameters": value})
