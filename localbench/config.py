"""Load and lightly validate config.yaml."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml


def _find_example_path() -> Path | None:
    """Locate the config template, from either a source checkout or an
    installed package.

    A git clone has config.example.yaml at the repo root, next to cli.py --
    that's the CWD-relative path a developer expects and the one the README
    documents. But a `pip install`-ed console script can be run from any
    working directory, and the repo root doesn't travel with the wheel, only
    package data does. A packaged copy lives inside localbench/ specifically
    so it can be found via importlib.resources regardless of CWD. The two
    must stay identical; CI diffs them so they can't silently drift apart.
    """
    cwd_path = Path("config.example.yaml")
    if cwd_path.exists():
        return cwd_path
    try:
        packaged = importlib.resources.files("localbench").joinpath("config.example.yaml")
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return None


EXAMPLE_PATH = _find_example_path()


def bootstrap_config(path: str | Path = "config.yaml") -> bool:
    """Create config.yaml from the tracked template on first run.

    config.yaml is gitignored (it is per-user), so a fresh clone never has
    one. Making the user copy it by hand is a step that exists only because
    of how the repo is laid out, not because it needs a decision from them --
    the template is a working default. Returns True if a file was created."""
    path = Path(path)
    if path.exists() or EXAMPLE_PATH is None:
        return False
    path.write_text(EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        # config.yaml is deliberately gitignored (it holds your own runtime
        # URL, model list and judge choice), so "missing" is the normal state
        # of a fresh clone -- say how to fix it rather than just what's wrong.
        if EXAMPLE_PATH is not None:
            raise FileNotFoundError(
                f"{path} not found. It's gitignored because it's your personal config; "
                f"create it from the template:  copy {EXAMPLE_PATH} {path}"
            )
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "runtime" not in config or "base_url" not in config["runtime"]:
        raise ValueError("config.yaml must define runtime.base_url")
    if not config.get("models"):
        raise ValueError("config.yaml must define at least one entry under models")

    return config
