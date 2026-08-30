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

from localbench.live_monitor import LiveHardwareMonitor
from localbench.runner import RunCancelled, run_benchmark
from localbench.storage import validate_run_id

_ORPHANED_MESSAGE = (
    "server restarted while this run was in progress -- its outcome is unknown; "
    "start a new run"
)


class ActiveRun:
    def __init__(self, run_id: str, config: dict, active_dir: Path, run_frontier_graded: bool = False, profile: str = "full"):
        self.run_id = run_id
        self.config = config
        self.run_frontier_graded = run_frontier_graded
        self.profile = profile
        self.log_lines: list[str] = []
        self.status = "running"  # running | waiting_confirm | done | error
        self.pending_message: str | None = None
        self.confirm_event = threading.Event()
        self.cancel_event = threading.Event()
        self.result_run_id: str | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.created_at = time.time()
        self.state_file = active_dir / f"{self.run_id}.json"
        self.live_monitor = LiveHardwareMonitor()
        # Most-recently-completed problem's speed, for the live HUD -- not
        # persisted to the saved run record (that already has per-problem
        # figures); this is real-time-only, like resource_samples.
        self.live_tokens_per_sec: float | None = None
        self.live_ttft_seconds: float | None = None
        self.live_prefill_tokens_per_sec: float | None = None
        self.live_problem_id: str | None = None

    def _persist(self) -> None:
        try:
            # Deliberately persists to_dict(include_samples=False): the live
            # hardware samples are real-time-only by design, and writing them
            # here would rewrite a ~40KB rolling buffer to disk on every
            # single log line (now one per problem), for data that is
            # meaningless after the process that produced it has exited.
            self.state_file.write_text(
                json.dumps(self.to_dict(include_samples=False), indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # best-effort only -- the in-memory state is authoritative while the process is alive

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)
        self._persist()

    def update_live_stats(
        self,
        tokens_per_sec: float | None,
        ttft_seconds: float | None,
        prefill_tokens_per_sec: float | None = None,
        problem_id: str | None = None,
    ) -> None:
        self.live_tokens_per_sec = tokens_per_sec
        self.live_ttft_seconds = ttft_seconds
        self.live_prefill_tokens_per_sec = prefill_tokens_per_sec
        self.live_problem_id = problem_id

    def confirm(self, message: str) -> None:
        self.pending_message = message
        self.status = "waiting_confirm"
        self._persist()
        self.confirm_event.clear()
        while not self.confirm_event.wait(timeout=0.5):
            if self.cancel_event.is_set():
                # unblock the waiting thread; run_benchmark's next cancel
                # check turns this into a clean RunCancelled
                return
        self.status = "running"
        self.pending_message = None
        self._persist()

    def start(self) -> None:
        def target() -> None:
            self.live_monitor.start()
            try:
                self._persist()
                record = run_benchmark(
                    self.config,
                    progress_cb=self.log,
                    confirm_cb=self.confirm,
                    run_frontier_graded=self.run_frontier_graded,
                    should_cancel=self.cancel_event.is_set,
                    profile=self.profile,
                    on_stats=self.update_live_stats,
                )
                self.result_run_id = record.run_id
                self.status = "done"
                self._persist()
            except RunCancelled as e:
                # Deliberately NOT saved. A partial run has fewer problems per
                # suite than it claims, so its pass rates and confidence
                # intervals would be wrong, and it would quietly skew any
                # comparison it appeared in.
                self.log(f"RUN STOPPED: {e}. Partial results were discarded.")
                self.status = "cancelled"
                self._persist()
            except Exception as e:  # noqa: BLE001 -- surface any failure to the UI, don't crash the thread silently
                self.error = str(e)
                self.status = "error"
                self._persist()
            finally:
                self.live_monitor.stop()

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def to_dict(self, include_samples: bool = True) -> dict:
        data = {
            "run_id": self.run_id,
            "status": self.status,
            "log": self.log_lines,
            "pending_message": self.pending_message,
            "error": self.error,
            "result_run_id": self.result_run_id,
            "done": self.status in ("done", "error", "cancelled"),
            "live_tokens_per_sec": self.live_tokens_per_sec,
            "live_ttft_seconds": self.live_ttft_seconds,
            "live_prefill_tokens_per_sec": self.live_prefill_tokens_per_sec,
            "live_problem_id": self.live_problem_id,
        }
        if include_samples:
            data["resource_samples"] = self.live_monitor.latest_samples()
        return data


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

    def start_run(self, config: dict, run_frontier_graded: bool = False, profile: str = "full") -> str:
        run_id = uuid.uuid4().hex[:8]
        active = ActiveRun(run_id, config, self.active_dir, run_frontier_graded=run_frontier_graded, profile=profile)
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
        # run_id comes straight from the URL here, so it must be validated
        # before it's used to build a path (see storage.validate_run_id).
        try:
            run_id = validate_run_id(run_id)
        except ValueError:
            return None
        state_file = self.active_dir / f"{run_id}.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def cancel_run(self, run_id: str) -> bool:
        active = self._runs.get(run_id)
        if active is None:
            return False
        active.cancel_event.set()
        active.confirm_event.set()  # release a run parked on a manual switch
        return True

    def continue_run(self, run_id: str) -> bool:
        active = self._runs.get(run_id)
        if active is None:
            return False
        active.confirm_event.set()
        return True
