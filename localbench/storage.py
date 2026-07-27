"""Persist RunRecords as one timestamped JSON file per run under results/runs/.

Each file is fully self-contained (hardware snapshot + every model's suite
results) so sharing a benchmark run with someone else is just handing them
that one file.
"""

from __future__ import annotations

import json
from pathlib import Path

from .results import RunRecord


def save_run(run: RunRecord, results_dir: str | Path = "results") -> Path:
    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.run_id}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return path


def list_runs(results_dir: str | Path = "results") -> list[dict]:
    """Return lightweight metadata for every saved run, newest first."""
    runs_dir = Path(results_dir) / "runs"
    if not runs_dir.exists():
        return []

    summaries = []
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(
            {
                "run_id": data.get("run_id", path.stem),
                "started_at": data.get("started_at"),
                "models": list(data.get("models", {}).keys()),
                "hardware": data.get("hardware", {}),
                "path": str(path),
            }
        )
    return summaries


def validate_run_id(run_id: str) -> str:
    """run_id ends up in a filesystem path -- reject anything that isn't the
    plain timestamp-shaped id we generate ourselves, to rule out path
    traversal via a crafted id in an API request."""
    if not run_id or any(c in run_id for c in ("/", "\\", "..")):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def load_run(run_id: str, results_dir: str | Path = "results") -> dict:
    run_id = validate_run_id(run_id)
    path = Path(results_dir) / "runs" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def delete_run(run_id: str, results_dir: str | Path = "results") -> None:
    run_id = validate_run_id(run_id)
    results_dir = Path(results_dir)
    json_path = results_dir / "runs" / f"{run_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(run_id)
    json_path.unlink()

    # Best-effort cleanup of rendered markdown report and active state
    md_path = results_dir / f"{run_id}.md"
    if md_path.exists():
        md_path.unlink()

    active_path = results_dir / "active" / f"{run_id}.json"
    if active_path.exists():
        active_path.unlink()
