"""Persist RunRecords as one timestamped JSON file per run under results/runs/.

Each file is fully self-contained (hardware snapshot + every model's suite
results) so sharing a benchmark run with someone else is just handing them
that one file.
"""

from __future__ import annotations

import json
import re
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


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_run_id(run_id: str) -> str:
    """run_id ends up in a filesystem path, so validate it with a strict
    ALLOWLIST rather than by blocking known-bad characters.

    A denylist here was genuinely unsafe on Windows: blocking only "/", "\\"
    and ".." still let a drive-relative path like "D:evil" through, and
    Path("results/runs") / "D:evil.json" resolves against the *current
    directory on drive D*, not under results/ at all -- which made both the
    raw-JSON download and the delete endpoint reach arbitrary files outside
    the results directory. Only the characters we actually generate
    (timestamps and uuid4 hex) are permitted."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
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
