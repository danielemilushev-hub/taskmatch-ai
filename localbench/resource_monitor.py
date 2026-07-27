"""Background sampling of RAM/CPU/VRAM usage during a suite run.

Honesty matters more than coverage here: `psutil.virtual_memory().used` is
TOTAL SYSTEM RAM in use across the whole machine -- not memory attributable
to the model, not a delta, and it has NO relationship to VRAM (psutil can't
see GPU memory at all). Reporting that raw number as "peak RAM" implied a
per-model VRAM proxy it never was. Two fixes:

1. A baseline is captured before the suite starts, so what's reported is the
   RAM *increase* during the suite -- still not model-specific (anything else
   on the machine could account for some of it), but at least scoped to what
   changed while the suite ran, not your OS/browser/everything-else's
   pre-existing footprint.
2. Actual VRAM delta is sampled via `nvidia-smi` when present. That's the
   only vendor tool available that exposes a live memory-used query at all --
   there is no AMD/Intel equivalent this tool can rely on cross-platform, so
   for those GPUs vram_delta_mb stays None (reported as "not available",
   never guessed).
"""

from __future__ import annotations

import shutil
import subprocess
import threading

import psutil


def _query_nvidia_vram_used_mb() -> float | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        # first GPU's figure; multi-GPU systems would need per-index handling
        return float(proc.stdout.strip().splitlines()[0])
    except Exception:
        return None


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._baseline_ram_gb: float | None = None
        self._peak_ram_gb = 0.0
        self._peak_cpu_percent = 0.0
        self._cpu_samples: list[float] = []
        self._baseline_vram_mb = _query_nvidia_vram_used_mb()
        self._peak_vram_mb: float | None = self._baseline_vram_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the internal counter
        self._baseline_ram_gb = psutil.virtual_memory().used / (1024**3)
        self._peak_ram_gb = self._baseline_ram_gb
        while not self._stop.is_set():
            ram_gb = psutil.virtual_memory().used / (1024**3)
            cpu = psutil.cpu_percent(interval=None)
            self._peak_ram_gb = max(self._peak_ram_gb, ram_gb)
            self._peak_cpu_percent = max(self._peak_cpu_percent, cpu)
            self._cpu_samples.append(cpu)
            if self._baseline_vram_mb is not None:
                vram = _query_nvidia_vram_used_mb()
                if vram is not None:
                    self._peak_vram_mb = max(self._peak_vram_mb or 0, vram)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None
        ram_delta_gb = None
        if self._baseline_ram_gb is not None:
            ram_delta_gb = round(max(0.0, self._peak_ram_gb - self._baseline_ram_gb), 2)
        vram_delta_mb = None
        if self._baseline_vram_mb is not None and self._peak_vram_mb is not None:
            vram_delta_mb = round(max(0.0, self._peak_vram_mb - self._baseline_vram_mb), 1)
        return {
            "ram_delta_gb": ram_delta_gb,
            "peak_ram_gb_total_system": round(self._peak_ram_gb, 2),
            "peak_cpu_percent": round(self._peak_cpu_percent, 1),
            "avg_cpu_percent": round(avg_cpu, 1) if avg_cpu is not None else None,
            "vram_delta_mb": vram_delta_mb,  # None means "not available" (non-NVIDIA GPU), never a guess
            "samples": len(self._cpu_samples),
        }
