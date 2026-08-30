"""localbench/llamacpp_mgr.py -- Discovery and process manager for llama.cpp / llama-server.

Provides automatic detection of llama-server.exe (Vulkan/ROCm/CPU), resolution
of GGUF model files on the host machine, and process lifecycle management
(terminal launching, health checking, and clean teardown).
"""

from __future__ import annotations

import json
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

    # 3. Check common manual-install locations. Home-relative paths work on
    # every platform; drive-letter paths are only meaningful on Windows.
    common_paths = [
        os.path.expanduser("~/llama.cpp/llama-server"),
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        "/usr/local/bin/llama-server",
        "/opt/llama.cpp/llama-server",
    ]
    if os.name == "nt":
        common_paths = [
            r"C:\llama.cpp\llama-server.exe",
            r"D:\llama.cpp\llama-server.exe",
            os.path.expanduser(r"~\llama.cpp\llama-server.exe"),
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


def _resolve_vendor_dirs(backend_dir: str) -> list[str]:
    """Read the backend's own backend-manifest.json for the vendor runtime
    package(s) it depends on, and resolve each to a real directory under the
    shared backends/vendor/ folder (a sibling of every versioned backend
    folder).

    This exists because LM Studio does NOT ship a ROCm/CUDA build's actual
    GPU runtime libraries (amdhip64.dll, cudart64_*.dll, etc.) inside the
    versioned backend folder itself -- those live in a separate shared
    vendor package and LM Studio's own launcher must put it on PATH before
    starting llama-server.exe. Without this, invoking the exact same binary
    directly fails with a missing-DLL error that looks like a generic
    runtime problem but is actually just this missing PATH entry -- verified
    live on this machine: the ROCm build failed with
    "api-ms-win-crt-heap-l1-1-0.dll" missing until the vendor package's `bin`
    directory (containing amdhip64_7.dll et al.) was added to PATH, at which
    point --list-devices succeeded and correctly enumerated both GPUs.
    """
    manifest_path = os.path.join(backend_dir, "backend-manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []

    pkg_names = manifest.get("vendor_lib_package_names") or []
    if not pkg_names:
        return []

    # backends/vendor/ is a sibling of every llama.cpp-win-x86_64-* folder.
    vendor_root = os.path.join(os.path.dirname(backend_dir), "vendor")
    dirs = []
    for name in pkg_names:
        pkg_dir = os.path.join(vendor_root, name)
        if os.path.isdir(pkg_dir):
            dirs.append(pkg_dir)
            bin_dir = os.path.join(pkg_dir, "bin")
            if os.path.isdir(bin_dir):
                dirs.append(bin_dir)
    return dirs


def _env_with_vendor_dirs(binary_path: str) -> dict:
    """os.environ plus the binary's vendor runtime directories prepended to
    PATH -- see _resolve_vendor_dirs. Always returns a full env dict (never
    just the additions) since it's meant to be passed directly as
    subprocess's `env=`."""
    env = os.environ.copy()
    vendor_dirs = _resolve_vendor_dirs(os.path.dirname(binary_path))
    if vendor_dirs:
        env["PATH"] = os.pathsep.join(vendor_dirs) + os.pathsep + env.get("PATH", "")
    return env


# Last successful probe per binary, for this process only. A GPU backend that
# enumerated devices once is known to work; a later probe of the same binary
# can still fail for transient reasons (the GPU is busy serving the model just
# benchmarked, driver init is slow under load, the call times out). Falling
# back to the last good answer keeps a working backend selectable instead of
# making it unavailable until the app is restarted -- the concrete complaint
# behind "the NVIDIA card got lost after one test". Never persisted, so a
# genuine hardware change is picked up on the next launch.
_LAST_GOOD_PROBE: dict[str, dict] = {}

# GPU runtimes can take a while to initialise on first call -- notably CUDA on
# a laptop with hybrid graphics, or any backend while the GPU is already busy.
# 15s was tight enough to time out in exactly those cases.
_PROBE_TIMEOUT_SECONDS = 40.0

# How long to watch a freshly-started llama-server before calling the launch a
# success. Long enough to catch a model that fails to load (those die well
# under a second), short enough not to stall on a real load, which takes far
# longer and completes in the background.
_EARLY_EXIT_GRACE_SECONDS = 2.5


def _probe_failure(binary_path: str, error: str) -> dict:
    """Result for a failed probe, reusing this binary's last successful device
    list if it has one.

    A backend that enumerated devices earlier in this session demonstrably
    works; a probe failing now is far more likely to be transient (busy GPU,
    slow driver init, timeout) than the hardware having vanished. Reporting it
    as unavailable made a working GPU backend unselectable until restart.
    `stale` marks that the devices come from the earlier probe, not this one.
    """
    cached = _LAST_GOOD_PROBE.get(binary_path)
    if cached:
        return {
            "available": True,
            "devices": cached["devices"],
            "error": None,
            "stale": True,
            "last_error": error,
        }
    return {"available": False, "devices": [], "error": error}


def probe_backend_devices(binary_path: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> dict:
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
            env=_env_with_vendor_dirs(binary_path),
        )
    except subprocess.TimeoutExpired:
        return _probe_failure(binary_path, f"timed out after {timeout}s")
    except OSError as e:
        return _probe_failure(binary_path, str(e))

    if proc.returncode != 0:
        # Windows STATUS_DLL_NOT_FOUND surfaces as a huge unsigned return
        # code (3221225781 = 0xC0000135) rather than a normal small exit
        # status -- name it plainly rather than just printing the number.
        detail = proc.stderr.strip() or proc.stdout.strip()
        if proc.returncode == 3221225781 or "0xC0000135" in detail:
            # Already tried adding this backend's known vendor runtime dirs
            # (see _env_with_vendor_dirs) before this failure happened, so a
            # missing GPU-vendor driver component (e.g. no NVIDIA driver
            # installed at all) is the likely remaining cause, not a generic
            # VC++ redistributable gap.
            detail = (detail + " -- ").strip(" -") + "missing a required runtime DLL even with known vendor runtime paths added (the GPU driver itself may be missing, or this GPU vendor isn't present on this machine)"
        return _probe_failure(binary_path, detail or f"exited with code {proc.returncode}")

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

    # A clean exit means the binary launched and ran successfully -- an empty
    # device list is expected and correct for a CPU-only build (there is no
    # GPU to enumerate, not a failure), so it does not mark this backend
    # unavailable. Only a non-empty list is worth remembering as known-good.
    if devices:
        _LAST_GOOD_PROBE[binary_path] = {"devices": devices}
    return {"available": True, "devices": devices, "error": None}


# Which GPU vendor each backend targets -- "any" means it isn't
# vendor-specific (Vulkan runs on any card, CPU needs none at all).
_BACKEND_VENDOR = {"vulkan": "any", "rocm": "amd", "cuda": "nvidia", "cpu": "any"}

_VENDOR_NAME_HINTS = {
    "amd": ("amd", "radeon"),
    "nvidia": ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"),
}


def _classify_vendors(device_names: set[str]) -> set[str]:
    vendors = set()
    for name in device_names:
        lower = name.lower()
        for vendor, hints in _VENDOR_NAME_HINTS.items():
            if any(h in lower for h in hints):
                vendors.add(vendor)
    return vendors


def list_llama_backends_with_status() -> list[dict]:
    """discover_llama_backends() + a real launch probe for each -- the
    combined result the dashboard actually shows: which backends exist,
    which of those genuinely work on this machine right now, and their
    real device lists for GPU selection.

    A backend whose target vendor is positively known to be absent is dropped
    (e.g. CUDA on a machine where enumeration found only AMD cards) rather
    than shown as a perpetually-disabled option, since that mismatch can
    never be fixed by anything the user does here.

    Crucially, this requires POSITIVE evidence of the GPU inventory. If no
    probe enumerated any device at all, we have learned nothing about what
    hardware exists and must not conclude a vendor is missing: probes fail
    for ordinary, temporary reasons -- the GPU is busy serving a model just
    benchmarked, VRAM is full, a probe timed out. Treating that silence as
    "no NVIDIA present" made the CUDA backend disappear from the dashboard
    after a successful run on an NVIDIA laptop, with no way to get it back
    short of restarting. Absence of evidence is not evidence of absence.
    """
    backends = discover_llama_backends()
    for b in backends:
        status = probe_backend_devices(b["path"])
        b.update(status)

    all_device_names = set()
    for b in backends:
        all_device_names.update(d["name"] for d in b.get("devices", []))
    detected_vendors = _classify_vendors(all_device_names)

    # No device seen anywhere -> inventory unknown -> filter nothing.
    if not all_device_names:
        return backends

    def keep(b: dict) -> bool:
        vendor = _BACKEND_VENDOR.get(b["id"], "any")
        if vendor == "any":
            return True
        if vendor in detected_vendors:
            return True
        # A backend that enumerated its own devices is self-evidently usable,
        # even if those device names match no vendor hint (e.g. Intel Arc, or
        # a card named in a way this heuristic doesn't recognise).
        return bool(b.get("devices"))

    return [b for b in backends if keep(b)]


def is_auxiliary_gguf(filename: str) -> bool:
    """True for .gguf files that are companion weights, not standalone models.

    These cannot be loaded as a main model and llama-server refuses them
    outright -- e.g. loading a projector fails with "CLIP cannot be used as
    main model, use it with --mmproj instead", which is what a user saw after
    picking one from the model list.

    Matching is on the whole filename, not just its start: real-world names put
    the marker at the end (`Qwen3.5-2B.BF16-mmproj.gguf`, `qwen36_mtp.gguf`),
    so an older `startswith("mmproj")` check let every one of them through.
    The `mtp`/`draft` markers are matched only as a delimited token so a
    legitimate model whose name merely contains those letters isn't hidden.
    """
    name = filename.lower()
    if not name.endswith(".gguf"):
        return False
    stem = name[: -len(".gguf")]
    if "mmproj" in stem or "projector" in stem:
        return True
    # Multi-token-prediction / speculative-decoding draft weights.
    for marker in ("mtp", "draft", "eagle"):
        if stem == marker or stem.endswith(f"-{marker}") or stem.endswith(f"_{marker}") or stem.endswith(f".{marker}"):
            return True
    return False


def list_local_ggufs(search_dirs: list[str] | None = None) -> list[str]:
    """Scan local directories for GGUF model files (excluding companion
    weights like mmproj projectors -- see is_auxiliary_gguf)."""
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
                    if f.lower().endswith(".gguf") and not is_auxiliary_gguf(f):
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
    """Scan likely storage roots for directories that look like model folders.

    On Windows those roots are the drive letters; on Linux/macOS they're the
    home directory plus the conventional mount points (/mnt, /media, /Volumes),
    since a POSIX box has no drive letters to walk and would otherwise find
    nothing at all here.
    """
    import string
    found_dirs = []
    seen = set()

    # Priority check for user home AI directories
    known = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.ollama/models"),
    ]
    if os.name == "nt":
        known += [r"D:\models", r"C:\models", r"D:\LLM", r"C:\LLM", r"D:\AI", r"C:\AI"]
    else:
        known += ["/usr/share/ollama/.ollama/models", os.path.expanduser("~/models")]

    for k in known:
        if os.path.isdir(k) and os.path.normpath(k).lower() not in seen:
            seen.add(os.path.normpath(k).lower())
            found_dirs.append(os.path.normpath(k))

    # Fast shallow scan of each storage root
    if os.name == "nt":
        roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    else:
        roots = [p for p in (os.path.expanduser("~"), "/mnt", "/media", "/Volumes", "/opt") if os.path.isdir(p)]
    for d in roots:
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
    spec_mtp = settings.get("speculative_mtp") if "speculative_mtp" in settings else model_cfg.get("speculative_mtp", False)
    spec_n_max = settings.get("speculative_n_max") or model_cfg.get("speculative_n_max")

    args = ["-m", norm_gguf, "--port", str(port), "-ngl", str(ngl)]

    if spec_mtp:
        # Multi-Token Prediction: some models (e.g. Qwen3.8-27B) ship an MTP
        # head in the weights themselves, so speculative decoding needs no
        # separate draft model -- llama.cpp just has to be told to use it.
        # `--spec-type` defaults to "none", so without this the MTP head is
        # simply never used. A model without an MTP head ignores the request
        # rather than failing, which is why this is safe to offer per-run.
        args += ["--spec-type", "draft-mtp"]
        if spec_n_max:
            # How many tokens to draft per step. llama.cpp's default is 3;
            # higher can help when the draft is usually accepted, and hurt
            # when it isn't, so it's worth being able to measure both.
            args += ["--spec-draft-n-max", str(spec_n_max)]

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


def serving_model_path(base_url: str = "http://localhost:8080/v1", timeout: float = 2.0) -> str | None:
    """Absolute GGUF path llama-server is currently serving, or None.

    llama-server reports the model by its full file path, which is what makes
    a reliable "is this already the model I want?" check possible.
    """
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data") or []
        if not data:
            return None
        path = data[0].get("id") or data[0].get("name")
        return os.path.normpath(os.path.abspath(path)) if path else None
    except Exception:
        return None


def is_model_already_serving(gguf_path: str, base_url: str = "http://localhost:8080/v1") -> bool:
    """True when llama-server is already serving exactly this GGUF.

    Lets a benchmark reuse a server the user already started ("Load Model" /
    "Launch in Terminal") instead of tearing it down and loading the identical
    weights again -- launch_llama_server() always stops any existing server
    first, so without this check pressing Load and then Start Benchmark loaded
    the same multi-GB model twice in a row.
    """
    current = serving_model_path(base_url)
    if not current:
        return False
    return current == os.path.normpath(os.path.abspath(gguf_path))


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

    # No placeholder default: "model" fuzzy-matches a real model.gguf, which
    # turned a nameless request into loading an arbitrary model.
    model_name = (model_cfg.get("name") or "").strip()
    if not model_name:
        return False, "no model name was given, so there is nothing to load."
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
    # ROCm/CUDA builds need their real GPU runtime libraries (amdhip64.dll,
    # cudart64_*.dll, etc.) on PATH -- LM Studio keeps those in a separate
    # shared vendor package, not inside the versioned backend folder itself
    # (see _resolve_vendor_dirs). A spawned child process inherits this via
    # normal Windows process creation, including through `cmd.exe /c start`
    # in the terminal-launch path below.
    launch_env = _env_with_vendor_dirs(binary)

    if in_terminal and os.name == "nt":
        # Launch visible interactive Windows Command Prompt window with /k so it stays open
        title = f"llama.cpp Server [{os.path.basename(norm_gguf)}] on :{port}"
        launch_cmd = ["cmd.exe", "/c", "start", title, "/D", bin_dir, "cmd.exe", "/k", binary] + args
        try:
            _MANAGED_PROCESS = subprocess.Popen(launch_cmd, cwd=bin_dir, env=launch_env)
            return True, f"Launched llama-server in terminal for '{os.path.basename(norm_gguf)}' on port {port}"
        except Exception as e:
            return False, f"Failed to open terminal for llama-server: {e}"
    else:
        try:
            full_cmd = [binary] + args
            _MANAGED_PROCESS = subprocess.Popen(full_cmd, cwd=bin_dir, env=launch_env)

            # Starting the process is not the same as loading the model. A
            # model llama.cpp cannot read (unsupported architecture, wrong
            # backend for the file, corrupt GGUF) makes llama-server exit
            # within a second -- but reporting only "process started" showed
            # the user a success message for a load that had already failed.
            # A short grace period catches that class of failure without
            # waiting out a genuine multi-GB load, which continues in the
            # background and is confirmed separately by wait_for_server_ready.
            time.sleep(_EARLY_EXIT_GRACE_SECONDS)
            exit_code = _MANAGED_PROCESS.poll()
            if exit_code is not None:
                _MANAGED_PROCESS = None
                return False, (
                    f"llama-server exited immediately (code {exit_code}) while loading "
                    f"'{os.path.basename(norm_gguf)}'. Use 'Load in Terminal' to see the "
                    f"reason -- a common cause is a model architecture this compute "
                    f"backend cannot read."
                )
            return True, f"Started llama-server background process for '{os.path.basename(norm_gguf)}' on port {port}"
        except Exception as e:
            return False, f"Failed to start llama-server process: {e}"
