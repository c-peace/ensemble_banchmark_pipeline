"""YAML experiment settings for the copy-and-paste notebook cells."""

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "parameters": {},
    "experiment": {
        "name": "ensemble",
        "wandb_project": "ensemble",
        "wandb_entity": None,
        "wandb_mode": "online",
    },
    "data": {
        "gcs_uri": "",
        "gcp_project": None,
        "sample_rate": 44100,
        "smoke": False,
        "excluded_train_ids": ["sod_507", "sod_968"],
    },
    "training": {
        "epochs": 3,
        "tiny": False,
        "steps_per_shard": None,
        "batch_size": 16,
        "samples_per_track": 64,
        "segment_seconds": 6.0,
        "save_every": 50,
        "seed": 42,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_grad_norm": 5.0,
        "loss_type": "baseline",
        "log_mel_weight": 0.1,
    },
    "evaluation": {
        "enabled": True,
        "chunk_seconds": 20.0,
        "allow_partial": False,
        "max_seconds": None,
        "reuse_predictions": True,
        "include_metrics": False,
    },
}


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently using the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("Configuration keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
# YAML 1.2 / JSON scientific notation (PyYAML otherwise treats 1e-05 as text).
UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def validate_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("run ID must contain 1–96 letters, digits, underscores or hyphens")
    return value


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown experiment section or invalid configuration")
    result = deepcopy(DEFAULTS)
    for section, supplied in value.items():
        if section == "parameters":
            if not isinstance(supplied, dict):
                raise ValueError("parameters must be a mapping of custom model settings")
            try:
                json.dumps(supplied, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("parameters must contain finite JSON-compatible values") from error
            result[section] = deepcopy(supplied)
            continue
        if section == "schema_version":
            if type(supplied) is not int or supplied != 1:
                raise ValueError("Unsupported schema_version")
            continue
        if not isinstance(supplied, dict) or set(supplied) - set(result[section]):
            raise ValueError(f"Unknown or invalid {section} settings")
        result[section].update(supplied)
    data, train, evaluation = (result[s] for s in ("data", "training", "evaluation"))
    experiment = result["experiment"]
    validate_run_id(experiment["name"])
    if not isinstance(experiment["wandb_project"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", experiment["wandb_project"]
    ):
        raise ValueError("wandb_project must contain letters, digits, underscores or hyphens")
    if experiment["wandb_entity"] is not None and (
        not isinstance(experiment["wandb_entity"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment["wandb_entity"])
    ):
        raise ValueError("wandb_entity must be an account/team name or null")
    if experiment["wandb_mode"] not in ("online", "offline", "disabled"):
        raise ValueError("wandb_mode must be online, offline or disabled")
    if data["gcp_project"] is not None and (
        not isinstance(data["gcp_project"], str) or not data["gcp_project"].strip()
    ):
        raise ValueError("gcp_project must be a project ID or null")
    if not isinstance(data["gcs_uri"], str):
        raise ValueError("gcs_uri must be a string")
    if type(data["sample_rate"]) is not int or data["sample_rate"] != 44100:
        raise ValueError("The current model requires sample_rate=44100")
    for group, names in (
        (data, ["smoke"]),
        (train, ["tiny"]),
        (evaluation, ["enabled", "allow_partial", "reuse_predictions", "include_metrics"]),
    ):
        for name in names:
            if type(group[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
    excluded = data["excluded_train_ids"]
    if (
        not isinstance(excluded, list)
        or any(not isinstance(x, str) for x in excluded)
        or len(set(excluded)) != len(excluded)
        or set(excluded) - {"sod_507", "sod_968"}
    ):
        raise ValueError("Only known silent training tracks can be excluded")
    for name in (
        "epochs",
        "batch_size",
        "samples_per_track",
        "save_every",
        "seed",
        "steps_per_shard",
    ):
        number = train[name]
        if name == "steps_per_shard" and number is None:
            continue
        if type(number) is not int or number < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a valid integer")
    if train["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    for group, names in (
        (
            train,
            ["segment_seconds", "learning_rate", "weight_decay", "max_grad_norm", "log_mel_weight"],
        ),
        (evaluation, ["chunk_seconds", "max_seconds"]),
    ):
        for name in names:
            number = group[name]
            if name == "max_seconds" and number is None:
                continue
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            if number < 0 or (number == 0 and name not in ("weight_decay", "log_mel_weight")):
                raise ValueError(f"{name} is outside the supported range")
    if train["loss_type"] not in ("baseline", "baseline_log_mel"):
        raise ValueError("Unknown loss_type")
    if train["loss_type"] == "baseline_log_mel" and train["log_mel_weight"] <= 0:
        raise ValueError("log-mel loss requires a positive weight")
    if (
        data["smoke"]
        or train["tiny"]
        or train["steps_per_shard"] is not None
        or evaluation["max_seconds"] is not None
    ) and not evaluation["allow_partial"]:
        raise ValueError("Smoke/bounded experiments require evaluation.allow_partial=true")
    return result


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.load(stream, Loader=UniqueLoader))
