"""Best-effort hardware snapshot, attached to every run so results from
different machines are never silently compared as if they were equivalent.

Every piece is wrapped so a missing tool (no nvidia-smi, no psutil) degrades
to "unknown" instead of failing the whole benchmark run.
"""

from __future__ import annotations

import json as _json
import os
import platform
import shutil
import subprocess


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _cpu_friendly_name_windows() -> str | None:
    """WMI's Win32_Processor.Name gives the marketing name (e.g. "AMD Ryzen 5
    7500F 6-Core Processor") -- platform.processor() only gives the raw
    family/model/stepping CPUID string, which is correct but meaningless to
    a human comparing two machines."""
    if platform.system() != "Windows":
        return None
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    name = proc.stdout.strip()
    return name or None


def _cpu_and_ram() -> dict:
    info = {
        "cpu": _try(_cpu_friendly_name_windows) or platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "ram_total_gb": None,
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    return info


def _gpu_info_nvidia() -> list[dict] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    proc = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if proc.returncode != 0:
        return None

    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            gpus.append({"name": parts[0], "memory": parts[1], "source": "nvidia-smi"})
    return gpus or None


# WMI's Win32_VideoController.AdapterRAM is a 32-bit field that overflows and
# caps out at 4GB (or misreports) on any card with more VRAM -- a long-standing
# Windows API limitation, not specific to any one GPU vendor. The fix tools
# like LM Studio itself use: each display adapter's own registry key under
# this class GUID has a 64-bit "HardwareInformation.qwMemorySize" value that
# doesn't have that cap. Verified on this machine: WMI reported 4.0 GB for an
# RX 7800 XT; this registry value gives 17163091968 bytes = 15.98 GB, matching
# LM Studio's own Hardware panel exactly.
_DISPLAY_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"


def _gpu_info_windows_registry() -> list[dict] | None:
    if platform.system() != "Windows":
        return None

    script = (
        f"Get-ChildItem 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{_DISPLAY_CLASS_GUID}' "
        "-ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\\d+$' } | "
        "ForEach-Object { $p = Get-ItemProperty -Path $_.PSPath -ErrorAction SilentlyContinue; "
        "if ($p.'HardwareInformation.qwMemorySize') { "
        "[PSCustomObject]@{ Name = $p.DriverDesc; Bytes = $p.'HardwareInformation.qwMemorySize' } "
        "} } | ConvertTo-Json"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if not proc.stdout.strip():
        return None

    data = _json.loads(proc.stdout)
    if isinstance(data, dict):
        data = [data]

    gpus = []
    for entry in data:
        raw_bytes = entry.get("Bytes")
        if not raw_bytes:
            continue
        gpus.append(
            {
                "name": entry.get("Name") or "unknown",
                "memory": f"{round(int(raw_bytes) / (1024**3), 2)} GB",
                "source": "windows_registry",
            }
        )
    return gpus or None


def _gpu_info_windows_wmi() -> list[dict] | None:
    """Fallback if the registry read fails for some reason -- explicitly
    labeled as approximate since it's the buggy 32-bit field."""
    if platform.system() != "Windows":
        return None

    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if not proc.stdout.strip():
        return None

    data = _json.loads(proc.stdout)
    if isinstance(data, dict):
        data = [data]

    gpus = []
    for entry in data:
        ram_bytes = entry.get("AdapterRAM")
        gpus.append(
            {
                "name": entry.get("Name", "unknown"),
                "memory_approx": (
                    f"{round(ram_bytes / (1024**3), 2)} GB (WMI-reported, may be inaccurate)"
                    if ram_bytes
                    else "unknown"
                ),
                "source": "windows_wmi",
            }
        )
    return gpus or None


def _gpu_info() -> list[dict] | str:
    for probe in (_gpu_info_nvidia, _gpu_info_windows_registry, _gpu_info_windows_wmi):
        result = _try(probe)
        if result:
            return result
    return "unknown (no GPU detection method succeeded on this platform)"


def get_hardware_snapshot() -> dict:
    snapshot = {
        "os": f"{platform.system()} {platform.release()}",
        "machine_arch": platform.machine(),
        "python_version": platform.python_version(),
    }
    snapshot.update(_try(_cpu_and_ram, default={"cpu": "unknown"}))
    snapshot["gpu"] = _try(_gpu_info, default="unknown (detection failed)")
    return snapshot
