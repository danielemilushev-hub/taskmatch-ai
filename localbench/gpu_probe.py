"""One cross-vendor way to ask "how much GPU memory is in use right now?".

Shared by resource_monitor.py (per-suite baseline->peak deltas saved into a
run) and live_monitor.py (real-time dashboard sampling), so both report the
same number from the same source instead of drifting apart.

Probes, in order:

1. `nvidia-smi` -- fast (~50ms) and unambiguous, but NVIDIA-only. Not gated
   to any OS: it ships with the NVIDIA driver on Linux too, so this already
   covers Linux+NVIDIA with no extra code.
2. Windows GPU performance counters via CIM -- the same counters Task
   Manager's GPU tab reads, provided by the OS display-driver framework
   rather than any vendor tool, so they work on AMD/Intel/NVIDIA alike.
   Measured at ~0.6s per query on this machine (vs ~1.7s for the equivalent
   `Get-Counter` PDH call, which is why CIM is used here).
3. `rocm-smi` / `amd-smi` -- AMD on Linux. Unlike the two probes above,
   THIS ONE IS UNVERIFIED against real ROCm hardware: written defensively
   (broad exception handling, falls through to the next tool/returns None
   on anything unexpected) against the key names most commonly documented
   for these CLIs, but ROCm's own JSON output has changed shape across
   versions and no ROCm machine was available to confirm this against a
   real GPU. Treat a `source: rocm-smi`/`amd-smi` reading as plausible, not
   confirmed, until checked on real hardware.
4. `ioreg -r -d 1 -c IOAccelerator` on macOS -- reads GPU utilization from
   the IOKit registry without root (unlike `powermetrics`, which needs sudo
   on every call and is a poor fit for a per-second background sampler).
   ALSO UNVERIFIED: no Mac was available to confirm either the
   `IOAccelerator` class name or its `PerformanceStatistics` sub-keys
   against real hardware, and Apple Silicon's GPU driver stack has been
   reported to expose this under a different class name (e.g.
   `AGXAccelerator`) depending on chip generation -- this probe tries both.

Returns None rather than a guess when nothing works, so callers can
honestly report "not available".
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


# Above 100% but within this bound is treated as genuine counter overshoot
# (several engine contexts billing slightly overlapping time) and clamped.
# Anything beyond it isn't a percentage at all and is discarded -- see
# _clamp_percent.
_UTIL_OVERSHOOT_CEILING = 150.0


def _clamp_percent(value) -> float | None:
    """Normalise a GPU utilization reading, or discard it as unusable.

    Utilization is a share of wall-clock time, so >100% is impossible. Small
    overshoots are routine and get clamped. But these are uint64 performance
    counters: if a sampled process exits mid-interval, or the counter's time
    base resets, the formatted delta goes negative and wraps into an enormous
    number -- a real observed reading was 696534349797534.

    Clamping that to 100 would be worse than dropping it, because it silently
    asserts "the GPU was pegged" when the truth is the sample is garbage. A
    corrupt reading returns None so it can be reported as unavailable rather
    than as a confident wrong answer.
    """
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if val < 0 or val > _UTIL_OVERSHOOT_CEILING:
        return None
    return min(100.0, val)


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


def _rocm_smi_json_to_reading(data: dict) -> dict | None:
    """Pull a utilization %/used-VRAM reading out of `rocm-smi --json` output.

    rocm-smi nests per-card data under keys like "card0", each a flat dict
    of human-readable-label -> string value; the exact labels have shifted
    across ROCm releases, so this matches by substring on the most commonly
    documented ones ("GPU use (%)", "VRAM Total Used Memory (B)") rather
    than one exact key, and simply finds nothing (returns None) if the
    installed version uses different labels -- see the module docstring's
    verification caveat.
    """
    used_mb = 0.0
    utils: list[float] = []
    found_any = False
    for card_data in data.values():
        if not isinstance(card_data, dict):
            continue
        for key, value in card_data.items():
            key_lower = key.lower()
            try:
                if "gpu use" in key_lower and "%" in key_lower:
                    utils.append(float(value))
                    found_any = True
                elif "vram total used memory" in key_lower:
                    used_mb += float(value) / (1024**2)
                    found_any = True
            except (TypeError, ValueError):
                continue
    if not found_any:
        return None
    return {
        "used_mb": used_mb if used_mb > 0 else None,
        "util_percent": _clamp_percent(max(utils)) if utils else None,
    }


def _amd_smi_json_to_reading(parsed) -> dict | None:
    """Pull a reading out of `amd-smi metric --usage --vram-usage --json`
    output -- a list of one dict per GPU (or a bare dict on a single-GPU
    box). Never raises: one malformed entry (not a dict, a non-numeric
    stat) is skipped rather than discarding valid data already gathered
    from other entries on a multi-GPU box, or propagating past this
    function -- see the module docstring's verification caveat.
    """
    entries = parsed if isinstance(parsed, list) else [parsed]
    used_mb = 0.0
    utils: list[float] = []
    found_any = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        usage = entry.get("usage") or {}
        vram = entry.get("vram_usage") or entry.get("vram") or {}
        gfx = usage.get("gfx_activity") if isinstance(usage, dict) else None
        used = vram.get("vram_used") if isinstance(vram, dict) else None
        try:
            if gfx is not None:
                utils.append(float(gfx))
                found_any = True
            if used is not None:
                used_mb += float(used) / (1024**2)
                found_any = True
        except (TypeError, ValueError):
            continue
    if not found_any:
        return None
    return {
        "used_mb": used_mb if used_mb > 0 else None,
        "util_percent": _clamp_percent(max(utils)) if utils else None,
        "source": "amd-smi",
    }


def _probe_rocm() -> dict | None:
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            proc = subprocess.run(
                [rocm_smi, "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                reading = _rocm_smi_json_to_reading(json.loads(proc.stdout))
                if reading is not None:
                    return {**reading, "source": "rocm-smi"}
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, OSError):
            pass

    # ROCm 6+ ships a newer unified CLI; try it if rocm-smi isn't present or
    # didn't produce a usable reading. Same verification caveat applies.
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            proc = subprocess.run(
                [amd_smi, "metric", "--usage", "--vram-usage", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                reading = _amd_smi_json_to_reading(json.loads(proc.stdout))
                if reading is not None:
                    return reading
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            pass

    return None


def _ioreg_entries_to_reading(entries) -> dict | None:
    """Pull a utilization/used-memory reading out of parsed `ioreg -a` plist
    entries. Never raises: this parses an undocumented, unverified structure
    (exact key names unconfirmed against real hardware), so anything
    unexpected -- entries not a list, an entry not a dict, a non-numeric
    stat -- must degrade to None, the same contract every other probe in
    this module upholds.
    """
    if not entries or not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stats = entry.get("PerformanceStatistics")
        if not isinstance(stats, dict):
            continue
        util = None
        for key in ("Device Utilization %", "GPU Activity(%)", "Utilization %"):
            if key in stats:
                util = stats[key]
                break
        used_mb = None
        for key in ("In use system memory", "vramUsedBytes", "gpuVramUsedBytes"):
            if key in stats:
                try:
                    used_mb = float(stats[key]) / (1024**2)
                except (TypeError, ValueError):
                    used_mb = None
                break
        util_percent = _clamp_percent(util) if util is not None else None
        # A key being *present* isn't enough -- e.g. a garbage/unparseable
        # value degrades to None via _clamp_percent, and reporting a
        # "successful" reading where every field is None would be reporting
        # nothing while claiming otherwise. Only return once something
        # actually usable was extracted; otherwise keep looking at the next
        # entry (there can be more than one accelerator).
        if util_percent is None and used_mb is None:
            continue
        return {"used_mb": used_mb, "util_percent": util_percent, "source": "ioreg"}
    return None


def _probe_macos_ioreg() -> dict | None:
    if platform.system() != "Darwin":
        return None
    import plistlib

    # Try both class names since which one carries GPU stats has reportedly
    # varied by chip generation; `-a` asks ioreg for XML plist output, which
    # plistlib can parse reliably (unlike ioreg's default indented-text tree
    # format, which would need fragile regex scraping).
    for accel_class in ("IOAccelerator", "AGXAccelerator"):
        try:
            proc = subprocess.run(
                ["ioreg", "-a", "-r", "-d", "1", "-c", accel_class],
                capture_output=True, timeout=5,
            )
            if proc.returncode != 0 or not proc.stdout:
                continue
            entries = plistlib.loads(proc.stdout)
        except (subprocess.SubprocessError, OSError, plistlib.InvalidFileException):
            continue
        reading = _ioreg_entries_to_reading(entries)
        if reading is not None:
            return reading
    return None


def query_gpu() -> dict | None:
    """{"used_mb": float, "util_percent": float|None, "source": str} or None."""
    for probe in (_probe_nvidia, _probe_windows_cim, _probe_rocm, _probe_macos_ioreg):
        result = probe()
        if result is not None:
            return result
    return None
