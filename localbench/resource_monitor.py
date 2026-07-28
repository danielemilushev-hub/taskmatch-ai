"""Background sampling of RAM/CPU/VRAM usage during a suite run.

Honesty matters more than coverage here, and the two memory figures measure
genuinely different things:

`ram_delta_gb` is the increase in TOTAL SYSTEM RAM over a baseline captured
right before the suite starts. It is NOT model-attributable: the model is
already loaded (into VRAM, on a GPU setup) before any suite begins, so for a
fully GPU-resident model this number is mostly system noise plus whatever
the runtime allocates host-side for the KV cache. A large value here is
meaningful -- it usually means long prompts or a model that didn't fit in
VRAM -- but a small value does NOT mean "this model is small".

`vram_delta_mb` is the increase in GPU memory in use over the same baseline,
which is the number that actually tracks a GPU-loaded model's working set.
It comes from gpu_probe (nvidia-smi where available, otherwise Windows' own
GPU performance counters), so unlike the previous nvidia-smi-only version it
reports real figures on AMD/Intel too instead of "not available".

Sampling cadence differs deliberately: RAM/CPU come from psutil and are
essentially free, so they're sampled every `interval`. A GPU query costs
~0.6s of subprocess work, so it runs on a much slower cadence -- frequent
enough to catch a suite's peak, infrequent enough that the monitor doesn't
meaningfully inflate the CPU number it is itself reporting.
"""

from __future__ import annotations

import threading
import time

import psutil

from .gpu_probe import query_gpu

# How long to wait between GPU queries. Each costs ~0.6s of subprocess work;
# at this cadence that's ~10% duty cycle on a single core, which is a
# deliberate trade for having real VRAM numbers at all.
_GPU_SAMPLE_INTERVAL_SECONDS = 6.0


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._baseline_ram_gb: float | None = None
        self._peak_ram_gb = 0.0
        self._peak_cpu_percent = 0.0
        self._cpu_samples: list[float] = []
        self._baseline_vram_mb: float | None = None
        self._peak_vram_mb: float | None = None
        self._peak_gpu_util: float | None = None
        self._gpu_source: str | None = None
        self._gpu_samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_gpu(self) -> None:
        try:
            gpu = query_gpu()
        except Exception:
            # query_gpu() is documented to never raise, but this thread has
            # no supervisor -- if a probe's contract is ever violated (as
            # gpu_probe.py's macOS probe briefly was), the failure must
            # degrade to "no GPU sample this tick", not silently kill
            # tracking for the rest of the suite.
            return
        if gpu is None:
            return
        used = gpu["used_mb"]
        self._gpu_source = gpu["source"]
        self._gpu_samples += 1
        if self._baseline_vram_mb is None:
            self._baseline_vram_mb = used
            self._peak_vram_mb = used
        else:
            self._peak_vram_mb = max(self._peak_vram_mb or 0.0, used)
        util = gpu.get("util_percent")
        if util is not None:
            self._peak_gpu_util = max(self._peak_gpu_util or 0.0, float(util))

    def _run(self) -> None:
        try:
            psutil.cpu_percent(interval=None)  # prime the internal counter
            self._baseline_ram_gb = psutil.virtual_memory().used / (1024**3)
            self._peak_ram_gb = self._baseline_ram_gb
        except Exception:
            # Observed for real: psutil.swap_memory() has raised RuntimeError
            # on a real machine here when Windows Performance Counters are
            # disabled -- an environment issue, not proof CPU/RAM tracking
            # is broken. Leaves _baseline_ram_gb as None, which summary()
            # already reports honestly rather than a fabricated delta.
            pass
        self._sample_gpu()  # already self-guarded; baseline before generation starts
        last_gpu = time.monotonic()

        while not self._stop.is_set():
            try:
                ram_gb = psutil.virtual_memory().used / (1024**3)
                cpu = psutil.cpu_percent(interval=None)
                self._peak_ram_gb = max(self._peak_ram_gb, ram_gb)
                self._peak_cpu_percent = max(self._peak_cpu_percent, cpu)
                self._cpu_samples.append(cpu)
            except Exception:
                # One bad tick must not end tracking for the rest of the
                # suite -- this thread has no supervisor to restart it.
                pass

            now = time.monotonic()
            if now - last_gpu >= _GPU_SAMPLE_INTERVAL_SECONDS:
                self._sample_gpu()
                last_gpu = time.monotonic()

            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread:
            # generous join: a GPU query may be mid-flight when we stop
            self._thread.join(timeout=25)

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
            # None means "no GPU probe worked here", never a guess.
            "vram_delta_mb": vram_delta_mb,
            "peak_vram_mb_total": round(self._peak_vram_mb, 1) if self._peak_vram_mb is not None else None,
            "baseline_vram_mb": round(self._baseline_vram_mb, 1) if self._baseline_vram_mb is not None else None,
            "peak_gpu_util_percent": round(self._peak_gpu_util, 1) if self._peak_gpu_util is not None else None,
            "gpu_source": self._gpu_source,
            "samples": len(self._cpu_samples),
            "gpu_samples": self._gpu_samples,
        }
