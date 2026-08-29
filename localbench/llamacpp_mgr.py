"""localbench/llamacpp_mgr.py -- Discovery and process manager for llama.cpp / llama-server.

Provides automatic detection of llama-server.exe (Vulkan/ROCm/CPU), resolution
of GGUF model files on the host machine, and process lifecycle management
(terminal launching, health checking, and clean teardown).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Any

import requests

_MANAGED_PROCESS: subprocess.Popen | None = None
_MANAGED_PORT: int = 8080


def find_llama_server_binary(custom_path: str | None = None, backend: str | None = None) -> str | None:
    """Find a usable llama-server / llama-server.exe binary on the host.

    `backend` (one of "vulkan"/"rocm"/"cuda"/"cpu"), if given, picks that
    specific backend's binary via discover_llama_backends() rather than the
    auto-scored "best" one -- see that function's docstring for why the old
    auto-pick (Vulkan > ROCm > CUDA > AVX2 by name alone) isn't a safe
    default: it can silently select a backend that fails to launch.
    """
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    if backend:
        backends = discover_llama_backends()
        match = next((b for b in backends if b["id"] == backend), None)
        return match["path"] if match else None

    # 1. Check system PATH
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return os.path.abspath(found)

    # 2. Check LM Studio backends (prioritize Vulkan, then ROCm, then AVX2) --
    # this legacy auto-pick path is kept only for the no-backend-specified
    # case; prefer discover_llama_backends() + an explicit backend id, which
    # is what the dashboard now uses.
    lm_backends = os.path.expanduser("~/.lmstudio/extensions/backends")
    if os.path.isdir(lm_backends):
        candidates = []
        for root, _, files in os.walk(lm_backends):
            for f in files:
                if f.lower() in ("llama-server.exe", "llama-server"):
                    full_path = os.path.join(root, f)
                    lower = full_path.lower()
                    score = 0
                    if "vulkan" in lower:
                        score = 100
                    elif "rocm" in lower or "amd" in lower:
                        score = 90
                    elif "cuda" in lower or "nvidia" in lower:
                        score = 80
                    elif "avx2" in lower:
                        score = 50
                    candidates.append((score, full_path))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][1]

    # 3. Check common Windows installation paths
    common_paths = [
        r"C:\llama.cpp\llama-server.exe",
        r"D:\llama.cpp\llama-server.exe",
        r"C:\Users\Daniel\.docker\bin\inference\llama-server.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    return None


# Directory-name substring -> (backend id, human label). Order matters: a
# ROCm build's path also contains "amd", so ROCm must be checked first or
# an AMD-CUDA-mislabeled path could never happen, but more importantly a
# generic "amd" match must not shadow the more specific "rocm" one.
_BACKEND_NAME_PATTERNS: list[tuple[str, str, str]] = [
    ("rocm", "rocm", "ROCm"),
    ("nvidia-cuda", "cuda", "CUDA"),
    ("cuda", "cuda", "CUDA"),
    ("vulkan", "vulkan", "Vulkan"),
]

_VERSION_RE = re.compile(r"(\d+(?:\.\d+){1,3})")

# "  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 15405 MiB free)"
_DEVICE_LINE_RE = re.compile(
    r"^\s*(?P<id>\S+):\s*(?P<name>.+?)\s*\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\)\s*$"
)


def _classify_backend_dir(dirname: str) -> tuple[str, str] | None:
    lower = dirname.lower()
    for substr, backend_id, label in _BACKEND_NAME_PATTERNS:
        if substr in lower:
            return backend_id, label
    if "avx2" in lower or "cpu" in lower:
        return "cpu", "CPU"
    return None


def _extract_version(dirname: str) -> tuple[int, ...]:
    m = _VERSION_RE.search(dirname)
    if not m:
        return (0,)
    return tuple(int(part) for part in m.group(1).split("."))


def discover_llama_backends() -> list[dict]:
    """Find the latest installed version of each llama.cpp backend type
    (vulkan/rocm/cuda/cpu) under LM Studio's bundled backends directory.

    Does NOT verify the binary actually launches -- see probe_backend_devices
    for that. Returns unverified candidates; the caller decides whether to
    probe them (probing costs a few seconds per backend, so callers that
    just need "what versions are installed" shouldn't pay that cost).
    """
    lm_backends = os.path.expanduser("~/.lmstudio/extensions/backends")
    if not os.path.isdir(lm_backends):
        return []

    best: dict[str, tuple[tuple[int, ...], str, str]] = {}
    for entry in os.listdir(lm_backends):
        full_dir = os.path.join(lm_backends, entry)
        if not os.path.isdir(full_dir):
            continue
        classified = _classify_backend_dir(entry)
        if classified is None:
            continue
        backend_id, label = classified
        binary = os.path.join(full_dir, "llama-server.exe")
        if not os.path.isfile(binary):
            binary = os.path.join(full_dir, "llama-server")
            if not os.path.isfile(binary):
                continue
        version_tuple = _extract_version(entry)
        version_str = ".".join(str(p) for p in version_tuple)
        current = best.get(backend_id)
        if current is None or version_tuple > current[0]:
            best[backend_id] = (version_tuple, version_str, binary)

    return [
        {"id": backend_id, "label": label_for(backend_id), "path": path, "version": version_str}
        for backend_id, (_, version_str, path) in best.items()
    ]


def label_for(backend_id: str) -> str:
    return {"vulkan": "Vulkan", "rocm": "ROCm", "cuda": "CUDA", "cpu": "CPU"}.get(backend_id, backend_id)


def probe_backend_devices(binary_path: str, timeout: float = 15.0) -> dict:
    """Actually attempt to launch `binary_path --list-devices` and parse the
    result -- the only reliable way to know a backend truly works, as
    opposed to trusting that its files exist. A backend can be fully
    installed and still fail to launch (missing runtime DLL, unsupported
    GPU generation, etc.) -- LM Studio's own compatibility survey can be
    optimistic about this (observed live: it marked a ROCm build
    "Compatible" on a machine where directly invoking that exact binary
    failed with STATUS_DLL_NOT_FOUND), so this doesn't trust any external
    survey either, including LM Studio's -- it only trusts a real,
    successful launch of the exact binary this app would actually use.

    Returns {"available": bool, "devices": [...], "error": str | None}.
    Never raises.
    """
    try:
        proc = subprocess.run(
            [binary_path, "--list-devices"],
            cwd=os.path.dirname(binary_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "devices": [], "error": f"timed out after {timeout}s"}
    except OSError as e:
        return {"available": False, "devices": [], "error": str(e)}

    if proc.returncode != 0:
        # Windows STATUS_DLL_NOT_FOUND surfaces as a huge unsigned return
        # code (3221225781 = 0xC0000135) rather than a normal small exit
        # status -- name it plainly rather than just printing the number.
        detail = proc.stderr.strip() or proc.stdout.strip()
        if proc.returncode == 3221225781 or "0xC0000135" in detail:
            detail = (detail + " -- ").strip(" -") + "missing a required runtime DLL (commonly fixed by installing the Microsoft Visual C++ Redistributable)"
        return {
            "available": False,
            "devices": [],
            "error": detail or f"exited with code {proc.returncode}",
        }

    devices = []
    for line in proc.stdout.splitlines():
        m = _DEVICE_LINE_RE.match(line)
        if m:
            devices.append(
                {
                    "id": m.group("id"),
                    "name": m.group("name"),
                    "total_mb": int(m.group("total")),
                    "free_mb": int(m.group("free")),
                }
            )

    # A clean exit means the binary launched and ran successfully -- an
    # empty device list is expected and correct for a CPU-only build (there
    # is no GPU to enumerate, not a failure), so it does not mark this
    # backend unavailable.
    return {"available": True, "devices": devices, "error": None}


def list_llama_backends_with_status() -> list[dict]:
    """discover_llama_backends() + a real launch probe for each -- the
    combined result the dashboard actually shows: which backends exist,
    which of those genuinely work on this machine right now, and their
    real device lists for GPU selection.
    """
    backends = discover_llama_backends()
    for b in backends:
        status = probe_backend_devices(b["path"])
        b.update(status)
    return backends


def list_local_ggufs(search_dirs: list[str] | None = None) -> list[str]:
    """Scan local directories for GGUF model files (excluding mmproj projector weights)."""
    if search_dirs is None:
        try:
            from . import settings_store
            search_dirs = settings_store.get_model_directories()
        except Exception:
            search_dirs = [
                os.path.expanduser("~/.lmstudio/models"),
                os.path.expanduser("~/.cache/lm-studio/models"),
                os.path.expanduser("~/.cache/huggingface/hub"),
            ]

    ggufs = []
    seen = set()
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        try:
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(".gguf") and not f.lower().startswith("mmproj"):
                        full_p = os.path.abspath(os.path.join(root, f))
                        if full_p.lower() not in seen:
                            seen.add(full_p.lower())
                            ggufs.append(full_p)
        except Exception:
            continue
    return ggufs


def parse_gguf_metadata(file_path: str) -> dict:
    """Extract model name, size, quant, architecture, and params from a GGUF file path."""
    fname = os.path.basename(file_path)
    stem = fname[:-5] if fname.lower().endswith(".gguf") else fname
    
    # Try to extract parameter size (e.g. 9B, 14B, 24B, 35B, 70B, 0.5B)
    params_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", stem)
    params = params_match.group(0).upper() if params_match else None

    # Try to extract quantization (e.g. Q4_K_M, Q8_0, Q3_K_L, FP16, BF16, IQ4_XS)
    quant_match = re.search(r"(?:q\d+_[a-z0-9_]+|fp16|bf16|iq\d+_[a-z0-9_]+)", stem, re.IGNORECASE)
    quant = quant_match.group(0).upper() if quant_match else None

    # Try to guess architecture / family
    lower_stem = stem.lower()
    arch = None
    if "qwen" in lower_stem:
        arch = "qwen"
    elif "mistral" in lower_stem or "ministral" in lower_stem or "devstral" in lower_stem:
        arch = "mistral"
    elif "llama" in lower_stem:
        arch = "llama"
    elif "gemma" in lower_stem:
        arch = "gemma"
    elif "deepseek" in lower_stem:
        arch = "deepseek"
    elif "glm" in lower_stem:
        arch = "glm"

    try:
        size_bytes = os.path.getsize(file_path)
    except Exception:
        size_bytes = None

    return {
        "display_name": stem,
        "file_path": file_path,
        "size_bytes": size_bytes,
        "params": params,
        "quantization": quant,
        "architecture": arch,
        "format": "gguf",
    }


def scan_all_local_models(search_dirs: list[str] | None = None) -> dict[str, dict]:
    """Return dictionary of {model_key: metadata} for all local GGUFs."""
    ggufs = list_local_ggufs(search_dirs)
    catalog = {}
    for g in ggufs:
        meta = parse_gguf_metadata(g)
        key = meta["display_name"].lower()
        catalog[key] = meta
    return catalog


def auto_scan_drives() -> list[str]:
    """Scan all drive letters (C:\\, D:\\, etc.) for directories containing .gguf files."""
    import string
    found_dirs = []
    seen = set()

    # Priority check for user home AI directories
    known = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.ollama/models"),
        r"D:\models",
        r"C:\models",
        r"D:\LLM",
        r"C:\LLM",
        r"D:\AI",
        r"C:\AI",
    ]
    for k in known:
        if os.path.isdir(k) and os.path.normpath(k).lower() not in seen:
            seen.add(os.path.normpath(k).lower())
            found_dirs.append(os.path.normpath(k))

    # Fast shallow scan on all available drive roots
    drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    for d in drives:
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir():
                        elower = entry.name.lower()
                        if any(kw in elower for kw in ("model", "gguf", "llm", "weights", "ai")):
                            norm = os.path.normpath(entry.path)
                            if norm.lower() not in seen:
                                seen.add(norm.lower())
                                found_dirs.append(norm)
        except Exception:
            continue

    return found_dirs


def find_model_gguf(model_name: str, search_dirs: list[str] | None = None) -> str | None:
    """Resolve a model identifier or name to the exact local GGUF file path."""
    if os.path.isfile(model_name):
        return os.path.abspath(model_name)

    ggufs = list_local_ggufs(search_dirs)
    if not ggufs:
        return None

    # Exact filename match
    clean_target = os.path.basename(model_name).lower()
    for g in ggufs:
        if os.path.basename(g).lower() == clean_target or os.path.splitext(os.path.basename(g))[0].lower() == clean_target:
            return g

    # Clean tokens for fuzzy scoring
    # e.g. "qwen/qwen3.5-9b" -> ["3.5", "9b"]
    raw_clean = re.sub(r"[^a-zA-Z0-9\.]+", " ", model_name).lower()
    tokens = [t for t in raw_clean.split() if t not in ("qwen", "mistral", "unsloth", "lmstudio", "community", "gguf", "chat", "instruct")]

    best_match = None
    best_score = 0

    for g in ggufs:
        base_clean = re.sub(r"[^a-zA-Z0-9\.]+", " ", os.path.basename(g)).lower()
        score = 0
        for t in tokens:
            if t in base_clean:
                score += len(t)
        if score > best_score:
            best_score = score
            best_match = g
    return best_match if best_score > 0 else None


def build_llama_server_args(model_cfg: dict, gguf_path: str, port: int = 8080) -> list[str]:
    """Build list of command line arguments for llama-server."""
    norm_gguf = os.path.normpath(os.path.abspath(gguf_path))
    settings = model_cfg.get("settings") or model_cfg
    ctx_len = settings.get("context_length") or model_cfg.get("context_length")
    parallel = settings.get("parallel") or model_cfg.get("parallel")
    kv = settings.get("gpu_kv") or model_cfg.get("gpu_kv")
    fa = settings.get("flash_attention") if "flash_attention" in settings else model_cfg.get("flash_attention", True)
    mmap = settings.get("mmap") if "mmap" in settings else model_cfg.get("mmap", True)
    mlock = settings.get("mlock") or model_cfg.get("mlock", False)
    batch = settings.get("batch_size") or model_cfg.get("batch_size", 2048)
    split = settings.get("split_mode") or model_cfg.get("split_mode")
    ngl = settings.get("gpu_offload_layers") or model_cfg.get("gpu_offload_layers", 99)
    devices = settings.get("devices") or model_cfg.get("devices")

    args = ["-m", norm_gguf, "--port", str(port), "-ngl", str(ngl)]

    if devices:
        # Restrict the run to specific GPU device ids (e.g. ["Vulkan0"]), as
        # reported by `--list-devices` / probe_backend_devices(). Selecting a
        # single device here is also how ROCm is kept from ever being asked
        # to split work across mismatched GPU generations -- the caller is
        # responsible for only ever passing one id for a ROCm backend.
        args += ["-dev", ",".join(devices)]

    # Thread & CPU performance optimization: Physical cores avoid thread contention
    try:
        phys_cores = psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 4) // 2)
    except Exception:
        phys_cores = 6
    args += ["-t", str(max(1, phys_cores))]

    if ctx_len:
        args += ["-c", str(ctx_len)]
    if parallel and parallel != 1:
        args += ["-np", str(parallel)]
    if kv and kv != "f16":
        args += ["-ctk", str(kv), "-ctv", str(kv)]
    if fa is True:
        args += ["-fa", "on"]
    elif fa is False:
        args += ["-fa", "off"]
    if mmap is False:
        args.append("--no-mmap")
    if mlock:
        args.append("--mlock")
    if batch:
        args += ["-b", str(batch), "-ub", "512"]
    if split and split != "none":
        args += ["-sm", str(split)]

    return args


def wait_for_server_ready(base_url: str = "http://localhost:8080/v1", timeout_seconds: float = 45.0) -> bool:
    """Poll the /v1/models endpoint until the server reports ready."""
    url = base_url.rstrip("/") + "/models"
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return True
        except Exception:
            pass
        time.sleep(0.8)
    return False


def stop_llama_server(port: int = 8080) -> bool:
    """Terminate any running llama-server process on the designated port."""
    global _MANAGED_PROCESS

    stopped = False
    if _MANAGED_PROCESS is not None:
        try:
            _MANAGED_PROCESS.terminate()
            _MANAGED_PROCESS.wait(timeout=3)
        except Exception:
            try:
                _MANAGED_PROCESS.kill()
            except Exception:
                pass
        _MANAGED_PROCESS = None
        stopped = True

    # Also kill any orphaned llama-server.exe process on Windows
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, check=False)
            stopped = True
        except Exception:
            pass

    time.sleep(1.0)
    return stopped


def launch_llama_server(
    model_cfg: dict,
    port: int = 8080,
    in_terminal: bool = True,
    custom_binary: str | None = None,
    backend: str | None = None,
) -> tuple[bool, str]:
    """Launch llama-server with the specified model and settings in a terminal or background.

    `backend`, if given (one of "vulkan"/"rocm"/"cuda"/"cpu"), selects that
    specific compute backend's binary rather than the legacy auto-pick --
    callers should only pass a backend the dashboard already verified via
    probe_backend_devices() as actually launchable on this machine.
    """
    global _MANAGED_PROCESS, _MANAGED_PORT
    _MANAGED_PORT = port

    model_name = model_cfg.get("name") or "model"
    raw_binary = find_llama_server_binary(custom_binary, backend=backend)
    if not raw_binary:
        reason = f" for backend '{backend}'" if backend else ""
        return False, f"Could not find llama-server.exe binary{reason} on this machine."
    binary = os.path.normpath(os.path.abspath(raw_binary))

    gguf_path = find_model_gguf(model_name)
    if not gguf_path:
        return False, f"Could not locate GGUF model file matching '{model_name}' in local models directories."
    norm_gguf = os.path.normpath(os.path.abspath(gguf_path))

    # Stop any previous instance to ensure port and VRAM are clean
    stop_llama_server(port)

    args = build_llama_server_args(model_cfg, norm_gguf, port=port)
    bin_dir = os.path.dirname(binary)

    if in_terminal and os.name == "nt":
        # Launch visible interactive Windows Command Prompt window with /k so it stays open
        title = f"llama.cpp Server [{os.path.basename(norm_gguf)}] on :{port}"
        launch_cmd = ["cmd.exe", "/c", "start", title, "/D", bin_dir, "cmd.exe", "/k", binary] + args
        try:
            _MANAGED_PROCESS = subprocess.Popen(launch_cmd, cwd=bin_dir)
            return True, f"Launched llama-server in terminal for '{os.path.basename(norm_gguf)}' on port {port}"
        except Exception as e:
            return False, f"Failed to open terminal for llama-server: {e}"
    else:
        try:
            full_cmd = [binary] + args
            _MANAGED_PROCESS = subprocess.Popen(full_cmd, cwd=bin_dir)
            return True, f"Started llama-server background process for '{os.path.basename(norm_gguf)}' on port {port}"
        except Exception as e:
            return False, f"Failed to start llama-server process: {e}"
