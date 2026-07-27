"""One cross-vendor way to ask "how much GPU memory is in use right now?".

Shared by resource_monitor.py (per-suite baseline->peak deltas saved into a
run) and live_monitor.py (real-time dashboard sampling), so both report the
same number from the same source instead of drifting apart.

Two probes, in order:

1. `nvidia-smi` -- fast (~50ms) and unambiguous, but NVIDIA-only.
2. Windows GPU performance counters via CIM -- the same counters Task
   Manager's GPU tab reads, provided by the OS display-driver framework
   rather than any vendor tool, so they work on AMD/Intel/NVIDIA alike.
   Measured at ~0.6s per query on this machine (vs ~1.7s for the equivalent
   `Get-Counter` PDH call, which is why CIM is used here).

Returns None rather than a guess when neither works (e.g. Linux/macOS
without NVIDIA), so callers can honestly report "not available".
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess

# Dedicated (on-card) memory summed across adapters, plus the busiest engine's
# utilization. Instance names are opaque LUIDs, so aggregate rather than trying
# to map them to a specific physical card.
# GPU engine counters are per (process, engine) -- instance names look like
# pid_6120_luid_..._eng_0_engtype_3D -- so several processes report against the
# same physical engine. A bare Maximum across instances undercounts when work is
# split across processes, and a bare Sum across everything double-counts
# unrelated engines (3D + Copy + VideoDecode), which is how a "210%" reading
# happens. Task Manager's model is the correct one: total each ENGINE TYPE
# across processes, then take the busiest engine type. Utilization is a
# fraction of wall-clock time an engine was busy, so it cannot exceed 100% by
# definition; anything above is counter overshoot and is clamped.
_CIM_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$mem = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
$eng = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine
$util = 0
if ($eng) {
  $perType = $eng | Group-Object { ($_.Name -split 'engtype_')[-1] } | ForEach-Object {
    ($_.Group | Measure-Object -Property UtilizationPercentage -Sum).Sum
  }
  $util = ($perType | Measure-Object -Maximum).Maximum
}
[PSCustomObject]@{
  used_bytes = [double](($mem | Measure-Object -Property DedicatedUsage -Sum).Sum)
  util       = [double]$util
} | ConvertTo-Json -Compress
"""


def _clamp_percent(value) -> float | None:
    """Utilization is a share of wall-clock time and cannot exceed 100%.
    Clamp rather than surface a physically impossible number."""
    if value is None:
        return None
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return None


def _probe_nvidia() -> dict | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        # sum across GPUs so multi-GPU boxes aren't silently under-reported
        used_mb = 0.0
        utils = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            used_mb += float(parts[0])
            utils.append(float(parts[1]))
        return {
            "used_mb": used_mb,
            "util_percent": _clamp_percent(max(utils)) if utils else None,
            "source": "nvidia-smi",
        }
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _probe_windows_cim() -> dict | None:
    if platform.system() != "Windows":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _CIM_SCRIPT],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        used_bytes = data.get("used_bytes")
        if used_bytes is None:
            return None
        return {
            "used_mb": float(used_bytes) / (1024**2),
            "util_percent": _clamp_percent(data.get("util")),
            "source": "windows_gpu_counters",
        }
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, OSError, IndexError):
        return None


def query_gpu() -> dict | None:
    """{"used_mb": float, "util_percent": float|None, "source": str} or None."""
    for probe in (_probe_nvidia, _probe_windows_cim):
        result = probe()
        if result is not None:
            return result
    return None
