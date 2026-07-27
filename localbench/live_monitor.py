"""Continuous, run-scoped hardware sampling for the dashboard's live monitor
-- distinct from resource_monitor.py's per-suite baseline/peak tracking,
which exists to compute the *recorded* delta metrics saved in each run's
report. This module is for real-time observation only (never persisted to
the saved run JSON): so users can watch RAM/CPU/GPU/disk while a run is in
progress and notice a spillover as it happens, not just after the fact.

CPU/RAM/disk throughput come from psutil (fast, in-process, reliable). GPU
memory + utilization come from Windows' own performance counters via a
short-lived `Get-Counter` call -- the only mechanism that works for ANY GPU
vendor (nvidia-smi is NVIDIA-only; these counters are what Task Manager's
own GPU tab reads from, tracked by the OS's display driver framework, not a
vendor tool). Each Get-Counter call takes ~2-3s regardless of whether the
PowerShell process is fresh or reused -- that overhead is inherent to
Get-Counter itself (confirmed by testing both ways), so GPU samples refresh
on a slower cadence than CPU/RAM/disk rather than trying to fight it with a
persistent-process protocol that wouldn't actually be faster.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from collections import deque

import psutil

_GPU_SCRIPT = r"""
$ErrorActionPreference = "SilentlyContinue"
$gpuMem = Get-Counter -Counter "\GPU Adapter Memory(*)\Dedicated Usage"
$gpuEngine = Get-Counter -Counter "\GPU Engine(*)\Utilization Percentage"
$gpuMemTotal = ($gpuMem.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum
$gpuUtilMax = ($gpuEngine.CounterSamples | Measure-Object -Property CookedValue -Maximum).Maximum
[PSCustomObject]@{ gpu_mem_bytes = $gpuMemTotal; gpu_util_percent = $gpuUtilMax } | ConvertTo-Json -Compress
"""


def _query_gpu_windows() -> dict | None:
    if platform.system() != "Windows":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _GPU_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        mem_bytes = data.get("gpu_mem_bytes")
        return {
            "gpu_mem_used_bytes": mem_bytes,
            "gpu_util_percent": data.get("gpu_util_percent"),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, IndexError):
        return None


class LiveHardwareMonitor:
    """Samples CPU/RAM/disk every tick and refreshes GPU data every 3rd
    tick (its query is much slower), keeping a rolling buffer of the most
    recent readings for the dashboard to poll mid-run."""

    MAX_SAMPLES = 180

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._samples: deque[dict] = deque(maxlen=self.MAX_SAMPLES)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_disk_io = psutil.disk_io_counters()
        self._last_disk_time = time.monotonic()
        self._tick = 0
        self._last_gpu: dict | None = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the non-blocking baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect()
            with self._lock:
                self._samples.append(sample)
            self._tick += 1
            self._stop_event.wait(self.interval)

    def _collect(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        now = time.monotonic()
        disk_io = psutil.disk_io_counters()
        elapsed = max(now - self._last_disk_time, 0.001)
        read_bytes_sec = (disk_io.read_bytes - self._last_disk_io.read_bytes) / elapsed
        write_bytes_sec = (disk_io.write_bytes - self._last_disk_io.write_bytes) / elapsed
        self._last_disk_io = disk_io
        self._last_disk_time = now

        if self._tick % 3 == 0:
            gpu = _query_gpu_windows()
            if gpu is not None:
                self._last_gpu = gpu

        gpu_data = self._last_gpu or {}
        gpu_mem_bytes = gpu_data.get("gpu_mem_used_bytes")

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "disk_read_mb_s": round(read_bytes_sec / (1024**2), 2),
            "disk_write_mb_s": round(write_bytes_sec / (1024**2), 2),
            "gpu_util_percent": gpu_data.get("gpu_util_percent"),
            "gpu_mem_used_gb": round(gpu_mem_bytes / (1024**3), 2) if gpu_mem_bytes is not None else None,
        }

    def latest_samples(self) -> list[dict]:
        with self._lock:
            return list(self._samples)
