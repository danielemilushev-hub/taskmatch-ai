"""Load and lightly validate config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


EXAMPLE_PATH = Path("config.example.yaml")


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        # config.yaml is deliberately gitignored (it holds your own runtime
        # URL, model list and judge choice), so "missing" is the normal state
        # of a fresh clone -- say how to fix it rather than just what's wrong.
        if EXAMPLE_PATH.exists():
            raise FileNotFoundError(
                f"{path} not found. It's gitignored because it's your personal config; "
                f"create it from the tracked template:  copy {EXAMPLE_PATH} {path}"
            )
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "runtime" not in config or "base_url" not in config["runtime"]:
        raise ValueError("config.yaml must define runtime.base_url")
    if not config.get("models"):
        raise ValueError("config.yaml must define at least one entry under models")

    return config
