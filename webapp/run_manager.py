"""In-memory tracking of in-progress benchmark runs for the web dashboard.

run_benchmark() is synchronous and blocks on input() for manual model
switches -- neither works directly inside a FastAPI request. Each run gets
its own background thread; the manual-switch pause becomes a
threading.Event the browser sets via POST /api/run/{id}/continue instead of
a terminal keypress.

Run state is also persisted to disk (results/active/<run_id>.json) purely
so its log history survives a server restart for post-mortem inspection --
NOT to make an in-progress run resumable. A background thread dies with its
process; there is no way to "reconnect" to work that no longer exists. Any
run left in a non-terminal state (running/waiting_confirm) when the manager
starts up is therefore immediately marked as errored rather than left
looking like it's still progressing -- a stale "running: true" that can
never change or fail is worse than an honest error, because nothing ever
tells the client the truth.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from localbench.runner import run_benchmark

_ORPHANED_MESSAGE = (
    "server restarted while this run was in progress -- its outcome is unknown; "
    "start a new run"
)


class ActiveRun:
    def __init__(self, run_id: str, config: dict, active_dir: Path):
        self.run_id = run_id
        self.config = config
        self.log_lines: list[str] = []
        self.status = "running"  # running | waiting_confirm | done | error
        self.pending_message: str | None = None
        self.confirm_event = threading.Event()
        self.result_run_id: str | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.created_at = time.time()
        self.state_file = active_dir / f"{self.run_id}.json"

    def _persist(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass  # best-effort only -- the in-memory state is authoritative while the process is alive

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)
        self._persist()

    def confirm(self, message: str) -> None:
        self.pending_message = message
        self.status = "waiting_confirm"
        self._persist()
        self.confirm_event.clear()
        self.confirm_event.wait()
        self.status = "running"
        self.pending_message = None
        self._persist()

    def start(self) -> None:
        def target() -> None:
            try:
                self._persist()
                record = run_benchmark(self.config, progress_cb=self.log, confirm_cb=self.confirm)
                self.result_run_id = record.run_id
                self.status = "done"
                self._persist()
            except Exception as e:  # noqa: BLE001 -- surface any failure to the UI, don't crash the thread silently
                self.error = str(e)
                self.status = "error"
                self._persist()

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "log": self.log_lines,
            "pending_message": self.pending_message,
            "error": self.error,
            "result_run_id": self.result_run_id,
            "done": self.status in ("done", "error"),
        }


class RunManager:
    def __init__(self, results_dir: str | Path = "results"):
        self._runs: dict[str, ActiveRun] = {}
        self._lock = threading.Lock()
        self.active_dir = Path(results_dir) / "active"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self._mark_orphaned_runs_on_startup()

    def _mark_orphaned_runs_on_startup(self) -> None:
        """Any run left in a non-terminal state from a previous process is
        dead -- its thread doesn't exist anymore. Mark it as such immediately
        so a client polling it gets a real answer instead of a frozen
        "still running" that will never resolve."""
        for state_file in self.active_dir.glob("*.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not data.get("done", False):
                data["status"] = "error"
                data["error"] = _ORPHANED_MESSAGE
                data["done"] = True
                try:
                    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except OSError:
                    pass

    def start_run(self, config: dict) -> str:
        run_id = uuid.uuid4().hex[:8]
        active = ActiveRun(run_id, config, self.active_dir)
        with self._lock:
            self._runs[run_id] = active
        active.start()
        return run_id

    def get(self, run_id: str) -> ActiveRun | dict | None:
        active = self._runs.get(run_id)
        if active is not None:
            return active
        # Not in this process's memory -- check disk in case it's a run from
        # before a restart. _mark_orphaned_runs_on_startup() already ensured
        # anything non-terminal here has a real, honest error status.
        state_file = self.active_dir / f"{run_id}.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def continue_run(self, run_id: str) -> bool:
        active = self._runs.get(run_id)
        if active is None:
            return False
        active.confirm_event.set()
        return True
