# yaml config + `--set section.key=value` overrides.
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)   # true / 0.2 / null / [a,b] keep their types
    except yaml.YAMLError:
        return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Unknown config path: {key}")
            cursor = cursor[part]
        # only override keys that already exist, otherwise a typo silently adds dead config
        if parts[-1] not in cursor:
            raise KeyError(f"Unknown config key: {key}")
        cursor[parts[-1]] = _coerce(value)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)   # keep key order, so runs diff
