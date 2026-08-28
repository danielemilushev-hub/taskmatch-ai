"""Continuous, run-scoped hardware sampling for the dashboard's live monitor
-- distinct from resource_monitor.py's per-suite baseline/peak tracking,
which exists to compute the *recorded* delta metrics saved in each run's
report. This module is for real-time observation only (never persisted to
the saved run JSON): so users can watch RAM/CPU/GPU/disk while a run is in
progress and notice a spillover as it happens, not just after the fact.

CPU/RAM/disk throughput come from psutil (fast, in-process, reliable). GPU
memory + utilization come from the shared gpu_probe module, so the live
panel and the per-suite deltas saved into a run always report the same
figure from the same source rather than drifting apart. That probe costs
~0.6s per call, so GPU samples refresh on a slower cadence than
CPU/RAM/disk.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import psutil

from .gpu_probe import query_gpu


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
        try:
            self._last_disk_io = psutil.disk_io_counters()
        except Exception:
            self._last_disk_io = None
        self._last_disk_time = time.monotonic()
        self._tick = 0
        self._last_gpu: dict | None = None
        try:
            from .hardware import _gpu_info
            self._detected_gpus = _gpu_info()
        except Exception:
            self._detected_gpus = None

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
            try:
                sample = self._collect()
            except Exception:
                self._tick += 1
                self._stop_event.wait(self.interval)
                continue
            with self._lock:
                self._samples.append(sample)
            self._tick += 1
            self._stop_event.wait(self.interval)

    def _collect(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        try:
            swap_percent = psutil.swap_memory().percent
        except Exception:
            swap_percent = None

        now = time.monotonic()
        try:
            disk_io = psutil.disk_io_counters()
        except Exception:
            disk_io = None

        read_bytes_sec = 0.0
        write_bytes_sec = 0.0
        if disk_io is not None and self._last_disk_io is not None:
            elapsed = max(now - self._last_disk_time, 0.001)
            read_bytes_sec = max(0.0, (disk_io.read_bytes - self._last_disk_io.read_bytes) / elapsed)
            write_bytes_sec = max(0.0, (disk_io.write_bytes - self._last_disk_io.write_bytes) / elapsed)
        self._last_disk_io = disk_io
        self._last_disk_time = now

        if self._tick % 3 == 0:
            gpu = query_gpu()
            if gpu is not None:
                self._last_gpu = gpu

        gpu_data = self._last_gpu or {}
        gpu_mem_mb = gpu_data.get("used_mb")
        raw_gpus = gpu_data.get("gpus") or []

        detected = self._detected_gpus if isinstance(self._detected_gpus, list) else []
        gpus_summary = []
        for i, g in enumerate(raw_gpus):
            det = detected[i] if i < len(detected) else {}
            name = det.get("name")
            if not name:
                if (g.get("used_mb") or 0) == 0 and (g.get("util_percent") or 0) == 0:
                    continue
                name = f"GPU {i+1}"
            mem_str = det.get("memory") or ""
            total_gb = None
            if mem_str and "gb" in mem_str.lower():
                try:
                    total_gb = float(mem_str.lower().replace("gb", "").strip())
                except ValueError:
                    total_gb = None

            u_mb = g.get("used_mb")
            short_name = name.replace("AMD Radeon ", "").replace("NVIDIA GeForce ", "").strip()
            gpus_summary.append({
                "index": i,
                "name": name,
                "short_name": short_name,
                "util_percent": g.get("util_percent"),
                "mem_used_gb": round(u_mb / 1024, 2) if u_mb is not None else None,
                "mem_total_gb": total_gb,
            })

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "swap_percent": swap_percent,
            "disk_read_mb_s": round(read_bytes_sec / (1024**2), 2),
            "disk_write_mb_s": round(write_bytes_sec / (1024**2), 2),
            "gpu_util_percent": gpu_data.get("util_percent"),
            "gpu_mem_used_gb": round(gpu_mem_mb / 1024, 2) if gpu_mem_mb is not None else None,
            "gpus": gpus_summary,
        }

    def latest_samples(self) -> list[dict]:
        with self._lock:
            return list(self._samples)
