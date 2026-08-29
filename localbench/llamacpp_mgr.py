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


def find_llama_server_binary(custom_path: str | None = None) -> str | None:
    """Find a usable llama-server / llama-server.exe binary on the host."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # 1. Check system PATH
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return os.path.abspath(found)

    # 2. Check LM Studio backends (prioritize Vulkan, then ROCm, then AVX2)
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

    args = ["-m", norm_gguf, "--port", str(port), "-ngl", str(ngl)]

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
        args += ["-b", str(batch)]
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
) -> tuple[bool, str]:
    """Launch llama-server with the specified model and settings in a terminal or background."""
    global _MANAGED_PROCESS, _MANAGED_PORT
    _MANAGED_PORT = port

    model_name = model_cfg.get("name") or "model"
    raw_binary = find_llama_server_binary(custom_binary)
    if not raw_binary:
        return False, "Could not find llama-server.exe binary on this machine."
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
