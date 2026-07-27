"""Load and lightly validate config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "runtime" not in config or "base_url" not in config["runtime"]:
        raise ValueError("config.yaml must define runtime.base_url")
    if not config.get("models"):
        raise ValueError("config.yaml must define at least one entry under models")

    return config
