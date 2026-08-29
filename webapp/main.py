"""FastAPI backend for the localbench dashboard.

Run via `python cli.py serve`, or directly with:
    uvicorn webapp.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# allow `import localbench` when this module is launched standalone (not as
# part of the localbench package) via `uvicorn webapp.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from localbench import report, settings_store, storage
from localbench.config import load_config
from localbench.profiles import DEFAULT_PROFILE, PROFILES, problems_for
from localbench.engine import RunContext
from localbench.json_extract import extract_json
from webapp.run_manager import RunManager

app = FastAPI(title="TaskMatch AI dashboard")
run_manager = RunManager(results_dir=load_config().get("output", {}).get("results_dir", "results"))

STATIC_DIR = Path(__file__).parent / "static"
ALL_SUITES = [
    "json_schema",
    "coding",
    "logic_math",
    "instruction_following",
    "pattern_reasoning",
    "long_context",
    "tool_calling",
    "multi_turn",
]


# The dashboard's JS/CSS are edited constantly and served from localhost with
# no content hashing in the URL, so a browser that heuristically caches them
# will happily keep running a stale app.js against a freshly-restarted server
# -- which looks exactly like "my change did nothing" or "this feature is
# broken". Always revalidate; there's no bandwidth argument for caching on
# loopback anyway.
_NO_CACHE = {"Cache-Control": "no-cache, must-revalidate", "Pragma": "no-cache"}


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):  # type: ignore[override]
        resp = super().file_response(*args, **kwargs)
        resp.headers.update(_NO_CACHE)
        return resp


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE)


def _get_suite_task_count(name: str, config: dict) -> int:
    s_cfg = config.get("suites", {}).get(name, {})
    count = problems_for(name, "full", s_cfg.get("num_problems"))
    if count is not None:
        return count
    return s_cfg.get("num_problems", 0)


@app.get("/api/config")
def get_config() -> dict:
    config = load_config()
    return {
        "base_url": config["runtime"]["base_url"],
        "models": [m["name"] for m in config["models"]],
        "suites": {
            name: {
                "enabled": config.get("suites", {}).get(name, {}).get("enabled", True),
                "task_count": _get_suite_task_count(name, config),
                "quick_count": problems_for(
                    name, "quick", config.get("suites", {}).get(name, {}).get("num_problems")
                ),
                "full_count": problems_for(
                    name, "full", config.get("suites", {}).get(name, {}).get("num_problems")
                ),
            }
            for name in ALL_SUITES
        },
    }


_hw_specs_cache = None

@app.get("/api/hardware/specs")
def get_hardware_specs() -> dict:
    global _hw_specs_cache
    if _hw_specs_cache is None:
        try:
            from localbench.hardware import get_hardware_snapshot
            _hw_specs_cache = get_hardware_snapshot()
        except Exception as e:
            _hw_specs_cache = {"error": str(e)}
    return _hw_specs_cache


@app.get("/api/hardware/live")
def get_live_hardware_telemetry() -> dict:
    """Return real-time live hardware resource metrics (CPU, RAM, GPUs)."""
    try:
        from localbench.live_monitor import get_instant_hardware_sample
        return get_instant_hardware_sample()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/models/detect")
def detect_models() -> dict:
    """Query the configured runtime directly for what's actually loadable
    right now, and merge any local GGUF models found in configured model directories."""
    from localbench import llamacpp_mgr
    config = load_config()
    base_url = config["runtime"]["base_url"]
    runtime_ids = []
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=4)
        if resp.status_code == 200:
            runtime_ids = [m.get("id") for m in resp.json().get("data", []) if m.get("id")]
    except Exception:
        pass

    # Discover local GGUF files
    local_ggufs = llamacpp_mgr.scan_all_local_models()

    seen = set()
    combined = []
    # 1. Add all verified local GGUF models
    for key, meta in local_ggufs.items():
        name = meta.get("display_name") or key
        if name and name.lower() not in seen:
            seen.add(name.lower())
            combined.append(name)

    # 2. Add runtime IDs (normalized)
    for mid in runtime_ids:
        if not mid:
            continue
        clean_name = os.path.splitext(os.path.basename(mid))[0] if mid.lower().endswith(".gguf") else mid
        if clean_name.lower() not in seen:
            seen.add(clean_name.lower())
            combined.append(clean_name)

    if not combined:
        raise HTTPException(status_code=502, detail=f"No models detected on runtime ({base_url}) or local model folders.")
    return {"models": combined}


@app.get("/api/models/catalog")
def models_catalog() -> dict:
    """Richer per-model metadata (size, quantization, params, context length,
    vision/tool-use capability flags) for every model already downloaded from
    LM Studio, Ollama, and all local GGUF folders."""
    from localbench import llamacpp_mgr
    catalog = {}

    # 1. LMS catalog if available
    lms_path = shutil.which("lms")
    if lms_path:
        try:
            proc = subprocess.run(
                [lms_path, "ls", "--json"], capture_output=True, text=True, timeout=3
            )
            if proc.returncode == 0 and proc.stdout.strip():
                entries = json.loads(proc.stdout)
                for entry in entries:
                    key = entry.get("modelKey") or entry.get("indexedModelIdentifier")
                    if not key:
                        continue
                    file_p = entry.get("path")
                    if file_p and not os.path.exists(file_p):
                        # Ghost model -- file was deleted from disk
                        continue
                    quant = entry.get("quantization") or {}
                    catalog[key] = {
                        "display_name": entry.get("displayName"),
                        "publisher": entry.get("publisher"),
                        "size_bytes": entry.get("sizeBytes"),
                        "params": entry.get("paramsString"),
                        "architecture": entry.get("architecture"),
                        "quantization": quant.get("name"),
                        "context_length": entry.get("maxContextLength"),
                        "vision": entry.get("vision"),
                        "tool_use": entry.get("trainedForToolUse"),
                        "type": entry.get("type"),
                        "file_path": file_p,
                    }
        except Exception:
            pass

    # 2. Local GGUF scanner
    local_ggufs = llamacpp_mgr.scan_all_local_models()
    for key, meta in local_ggufs.items():
        display_name = meta.get("display_name") or key
        matched_key = None
        for k in catalog:
            if k.lower() == key or k.lower() == display_name.lower():
                matched_key = k
                break
        if matched_key:
            if not catalog[matched_key].get("file_path") and meta.get("file_path"):
                catalog[matched_key]["file_path"] = meta["file_path"]
        else:
            catalog[display_name] = {
                "display_name": display_name,
                "publisher": "local",
                "size_bytes": meta.get("size_bytes"),
                "params": meta.get("params"),
                "architecture": meta.get("architecture"),
                "quantization": meta.get("quantization"),
                "context_length": 32768,
                "vision": False,
                "tool_use": False,
                "type": "gguf",
                "file_path": meta.get("file_path"),
            }

    return {"models": catalog}


@app.get("/api/settings/directories")
def get_directories() -> dict:
    from localbench import settings_store
    dirs = settings_store.get_model_directories()
    stats = []
    for d in dirs:
        count = 0
        if os.path.isdir(d):
            try:
                for root, _, files in os.walk(d):
                    count += sum(1 for f in files if f.lower().endswith(".gguf") and not f.lower().startswith("mmproj"))
            except Exception:
                pass
        stats.append({"path": d, "exists": os.path.isdir(d), "gguf_count": count})
    return {"directories": stats}


@app.post("/api/settings/directories")
def add_directory(payload: dict) -> dict:
    from localbench import settings_store
    path = payload.get("path")
    if not path or not path.strip():
        raise HTTPException(400, "Directory path is required")
    try:
        settings_store.add_model_directory(path.strip())
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/settings/directories")
def remove_directory(payload: dict) -> dict:
    from localbench import settings_store
    path = payload.get("path")
    if not path:
        raise HTTPException(400, "Directory path is required")
    settings_store.remove_model_directory(path)
    return {"ok": True}


@app.post("/api/settings/directories/scan")
def scan_drives() -> dict:
    from localbench import llamacpp_mgr, settings_store
    found = llamacpp_mgr.auto_scan_drives()
    for f in found:
        try:
            settings_store.add_model_directory(f)
        except Exception:
            pass
    return {"ok": True, "discovered": found}


@app.get("/api/settings")
def get_settings() -> dict:
    from localbench.judge.factory import PROVIDER_PACKAGES, sdk_installed

    return {
        "runtime": settings_store.get_runtime_settings(),
        "judge": settings_store.get_judge_settings(),
        "keys": settings_store.key_status(),
        # The judge SDKs are optional installs, so a configured provider with
        # a valid key can still be unrunnable. Report that up front rather
        # than letting the run fail on its first judge call.
        "sdks": {p: sdk_installed(p) for p in settings_store.PROVIDER_ENV_VARS},
        "sdk_packages": PROVIDER_PACKAGES,
    }


@app.post("/api/settings/runtime")
def update_settings_runtime(payload: dict) -> dict:
    try:
        return settings_store.update_runtime_settings(payload)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/settings/judge")
def update_settings_judge(payload: dict) -> dict:
    try:
        return settings_store.update_judge_settings(payload)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/settings/judge/models")
def judge_models(provider: str) -> dict:
    if provider not in settings_store.PROVIDER_ENV_VARS:
        raise HTTPException(400, f"unknown provider '{provider}'")
    models = settings_store.list_judge_models(provider)
    # OpenRouter publishes real per-token pricing; surfacing it inline in the
    # picker means the cost of a judge choice is visible at the moment you
    # make it. No other provider exposes this, so their picker simply has no
    # price column rather than a guessed one.
    pricing = settings_store.list_openrouter_pricing() if provider == "openrouter" else {}
    return {"models": models, "pricing": pricing}


@app.get("/api/settings/judge/history")
def judge_history(provider: str, model: str) -> dict:
    """Best-effort: the most recent past run's measured frontier-graded
    stats for this exact provider/model combo, so the New Run screen can
    show a real, previously-observed time estimate instead of guessing
    before this judge has ever actually been run. Returns found=False (not
    an error) if no matching history exists yet -- that's the normal state
    for a judge that's never been used."""
    config = load_config()
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return {"found": False}

    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        judge_info = data.get("config_summary", {}).get("frontier_judge") or {}
        if judge_info.get("provider") != provider or judge_info.get("model") != model:
            continue
        for model_result in data.get("models", {}).values():
            suite = model_result.get("suites", {}).get("frontier_graded")
            if suite and suite.get("total") and suite.get("avg_seconds_per_task") is not None:
                return {
                    "found": True,
                    "run_id": data.get("run_id"),
                    "avg_seconds_per_task": suite["avg_seconds_per_task"],
                    "total_tasks": suite["total"],
                    "avg_judge_prompt_tokens": suite.get("avg_judge_prompt_tokens"),
                    "avg_judge_completion_tokens": suite.get("avg_judge_completion_tokens"),
                }
    return {"found": False}


@app.get("/api/settings/judge/cost-estimate")
def judge_cost_estimate(provider: str, model: str, num_tasks: int) -> dict:
    """Real dollar estimate using OpenRouter's own public, live per-token
    pricing plus real token usage observed in a previous run with this
    judge. Only available for OpenRouter -- Anthropic/OpenAI/Gemini don't
    expose pricing via a stable public API, and hardcoding their rates here
    would just go stale and mislead, so we deliberately don't guess for
    them."""
    if provider != "openrouter":
        return {
            "available": False,
            "reason": "Live pricing is only available for OpenRouter (a public marketplace API). "
            "Check your provider's own pricing page for a per-token rate.",
        }

    history = judge_history(provider, model)
    # Both token figures are needed and can legitimately diverge -- a provider
    # can report prompt tokens while omitting completion tokens (e.g. Gemini
    # leaves candidates_token_count off a blocked/empty response), and each
    # averages independently. Checking only one left the other free to be
    # None, which crashed the multiplication below with a TypeError.
    if (
        not history["found"]
        or history.get("avg_judge_prompt_tokens") is None
        or history.get("avg_judge_completion_tokens") is None
    ):
        return {
            "available": False,
            "reason": "No previous run with this exact model yet -- token usage can't be estimated until after a first run.",
        }

    pricing = settings_store.get_openrouter_pricing(model)
    if pricing is None:
        return {
            "available": False,
            "reason": f"Could not find live pricing for '{model}' in OpenRouter's catalog -- check the model id.",
        }

    cost_per_task = (
        history["avg_judge_prompt_tokens"] * pricing["prompt"]
        + history["avg_judge_completion_tokens"] * pricing["completion"]
    )
    return {
        "available": True,
        "total_cost_usd": round(cost_per_task * num_tasks, 4),
        "cost_per_task_usd": round(cost_per_task, 6),
        "prompt_rate_per_1m": round(pricing["prompt"] * 1_000_000, 4),
        "completion_rate_per_1m": round(pricing["completion"] * 1_000_000, 4),
        "based_on_run": history["run_id"],
    }


@app.post("/api/settings/keys")
def set_settings_key(payload: dict) -> dict:
    provider = payload.get("provider")
    api_key = payload.get("api_key")
    try:
        settings_store.set_api_key(provider, api_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "keys": settings_store.key_status()}


@app.delete("/api/settings/keys/{provider}")
def clear_settings_key(provider: str) -> dict:
    try:
        settings_store.clear_api_key(provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "keys": settings_store.key_status()}


@app.post("/api/run")
def start_run(payload: dict) -> dict:
    config = load_config()

    models_filter = payload.get("models")
    model_settings = payload.get("model_settings") or {}
    if models_filter:
        by_name = {m["name"]: dict(m) for m in config.get("models", [])}
        new_models = []
        for item in models_filter:
            if isinstance(item, dict):
                name = item["name"]
                base_cfg = by_name.get(name, {"name": name}).copy()
                base_cfg.update(item)
                new_models.append(base_cfg)
            else:
                name = str(item)
                base_cfg = by_name.get(name, {"name": name}).copy()
                if name in model_settings:
                    base_cfg.update(model_settings[name])
                new_models.append(base_cfg)
        config["models"] = new_models

    suites_filter = payload.get("suites")
    if suites_filter is not None:
        suites_cfg = config.setdefault("suites", {})
        for suite_name in ALL_SUITES:
            suites_cfg.setdefault(suite_name, {})["enabled"] = suite_name in suites_filter

    profile = payload.get("profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile '{profile}', expected one of {list(PROFILES)}")

    run_frontier_graded = bool(payload.get("run_frontier_graded"))
    if run_frontier_graded:
        judge_cfg = config.setdefault("judge", {})
        if not judge_cfg.get("enabled"):
            raise HTTPException(400, "frontier judge requested but judge.enabled is false -- enable it in Settings first")

        # Per-run override: let the New Run screen pick a different judge
        # provider/model than the one saved in Settings, without persisting
        # that choice to config.yaml.
        judge_override = payload.get("judge_override") or {}
        override_provider = judge_override.get("provider")
        if override_provider:
            if override_provider not in settings_store.PROVIDER_ENV_VARS:
                raise HTTPException(400, f"unknown judge provider '{override_provider}'")
            judge_cfg["provider"] = override_provider
        override_model = judge_override.get("model")
        if override_model:
            judge_cfg["model"] = override_model

        provider = judge_cfg.get("provider")
        if not settings_store.key_status().get(provider):
            raise HTTPException(
                400,
                f"frontier judge requested but no API key is set for provider '{provider}' -- add one in Settings first",
            )

        from localbench.judge.factory import PROVIDER_PACKAGES, sdk_installed

        if not sdk_installed(provider):
            raise HTTPException(
                400,
                f"frontier judge requested but the {provider} SDK isn't installed. "
                f"Install it with:  pip install {PROVIDER_PACKAGES.get(provider, provider)}",
            )

    run_id = run_manager.start_run(config, run_frontier_graded=run_frontier_graded, profile=profile)
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/status")
def run_status(run_id: str) -> dict:
    active = run_manager.get(run_id)
    if active is None:
        raise HTTPException(404, "run not found")
    if isinstance(active, dict):
        return active
    return active.to_dict()


@app.post("/api/run/{run_id}/continue")
def run_continue(run_id: str) -> dict:
    if not run_manager.continue_run(run_id):
        raise HTTPException(404, "run not found")
    return {"ok": True}


@app.post("/api/run/{run_id}/cancel")
def run_cancel(run_id: str) -> dict:
    """Stop an in-progress run at the next problem boundary. Partial results
    are discarded rather than saved -- a half-finished suite has fewer
    problems than it reports, so its pass rate and confidence interval would
    both be wrong and would skew any comparison it appeared in."""
    if not run_manager.cancel_run(run_id):
        raise HTTPException(404, "run not found or already finished")
    return {"ok": True}


@app.get("/api/runs")
def list_runs() -> list:
    return storage.list_runs()


def _load_run_or_http_error(run_id: str) -> dict:
    """Shared guard: an invalid run_id is a 400 (bad request), a valid-but-
    unknown one is a 404. Without this, validate_run_id's ValueError escapes
    as an unhandled 500."""
    try:
        return storage.load_run(run_id)
    except ValueError:
        raise HTTPException(400, "invalid run_id")
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _load_run_or_http_error(run_id)


@app.get("/api/runs/{run_id}/report.md")
def get_run_markdown(run_id: str) -> PlainTextResponse:
    data = _load_run_or_http_error(run_id)
    return PlainTextResponse(report.render_markdown(data), media_type="text/markdown")


@app.get("/api/runs/{run_id}/report.pdf")
def get_run_pdf(run_id: str) -> FileResponse:
    data = _load_run_or_http_error(run_id)
    tmp_path = Path(tempfile.gettempdir()) / f"localbench_{run_id}.pdf"
    report.render_pdf(data, tmp_path)
    return FileResponse(tmp_path, media_type="application/pdf", filename=f"{run_id}.pdf")


@app.get("/api/runs/{run_id}/raw.json")
def get_run_raw(run_id: str) -> FileResponse:
    try:
        run_id = storage.validate_run_id(run_id)
    except ValueError:
        raise HTTPException(400, "invalid run_id")
    config = load_config()
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    path = results_dir / "runs" / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "run not found")
    return FileResponse(path, media_type="application/json", filename=f"{run_id}.json")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    config = load_config()
    results_dir = config.get("output", {}).get("results_dir", "results")
    try:
        storage.delete_run(run_id, results_dir=results_dir)
    except ValueError:
        raise HTTPException(400, "invalid run_id")
    except FileNotFoundError:
        raise HTTPException(404, "run not found")
    return {"ok": True}


@app.post("/api/custom")
def run_custom(payload: dict) -> dict:
    """Ad-hoc playground: run one prompt against one model, optionally
    validating the response against a user-supplied JSON schema. Not part of
    any saved benchmark run -- for quickly probing a model's behavior on a
    specific workflow."""
    config = load_config()
    model = payload.get("model")
    prompt = payload.get("prompt")
    system_prompt = payload.get("system_prompt")
    temperature = payload.get("temperature")
    max_tokens = payload.get("max_tokens", 1536)

    if not model or not prompt:
        raise HTTPException(400, "both 'model' and 'prompt' are required")

    ctx = RunContext(
        base_url=config["runtime"]["base_url"],
        model=model,
        api_key=config["runtime"].get("api_key", "not-needed"),
        timeout_seconds=payload.get("timeout_seconds", 120),
        max_tokens=max_tokens,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    call_kwargs = {}
    if temperature is not None:
        call_kwargs["temperature"] = float(temperature)
    if payload.get("detect_loops"):
        call_kwargs["detect_loops"] = True

    chat = ctx.call(messages, **call_kwargs)

    result = {
        "success": chat.success,
        "content": chat.content,
        "reasoning_content": chat.reasoning_content,
        "error": chat.error,
        "truncated": chat.truncated,
        "loop_detected": chat.loop_detected,
        "latency_seconds": chat.latency_seconds,
        "ttft_seconds": chat.ttft_seconds,
        "tokens_per_sec": chat.tokens_per_sec,
    }

    schema = payload.get("schema")
    if schema and chat.success:
        value, extract_err = extract_json(chat.content)
        if extract_err:
            result["schema_valid"] = False
            result["schema_error"] = extract_err
        else:
            try:
                jsonschema.validate(instance=value, schema=schema)
                result["schema_valid"] = True
            except jsonschema.ValidationError as e:
                result["schema_valid"] = False
                result["schema_error"] = f"{list(e.absolute_path)}: {e.message}"

    return result


@app.get("/api/models/active")
def get_active_model() -> dict:
    """Check what model is currently loaded in llama.cpp (:8080) or LM Studio (:1234)."""
    # 1. Check llama.cpp on port 8080
    try:
        r = requests.get("http://localhost:8080/v1/models", timeout=0.8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                m_id = data[0].get("id") or data[0].get("name")
                return {"active": True, "model": m_id, "runtime": "llamacpp", "port": 8080}
    except Exception:
        pass

    # 2. Check LM Studio on port 1234
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=0.8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                m_id = data[0].get("id") or data[0].get("name")
                return {"active": True, "model": m_id, "runtime": "lmstudio", "port": 1234}
    except Exception:
        pass

    return {"active": False, "model": None, "runtime": None, "port": None}


@app.post("/api/models/load")
def load_model_endpoint(payload: dict) -> dict:
    """Load a model into the specified runtime engine."""
    from localbench import llamacpp_mgr
    flavor = payload.get("runtime_flavor") or "llamacpp"
    name = payload.get("name") or "model"
    in_terminal = bool(payload.get("in_terminal", False))

    if flavor == "llamacpp":
        ok, msg = llamacpp_mgr.launch_llama_server(payload, in_terminal=in_terminal, port=payload.get("port", 8080))
        if not ok:
            raise HTTPException(400, msg)
        return {"ok": True, "message": msg, "model": name, "runtime": "llamacpp"}
    elif flavor == "lmstudio":
        lms_path = shutil.which("lms")
        if not lms_path:
            raise HTTPException(400, "LM Studio CLI ('lms') not found on system PATH")
        cmd = f'lms load "{name}" -y --gpu {payload.get("gpu_offload", "max")}'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(400, f"LM Studio load failed: {proc.stderr or proc.stdout}")
        return {"ok": True, "message": f"Loaded '{name}' in LM Studio", "model": name, "runtime": "lmstudio"}
    else:
        raise HTTPException(400, f"Unsupported runtime flavor: {flavor}")


@app.post("/api/models/unload")
def unload_model_endpoint(payload: dict | None = None) -> dict:
    """Unload a specific model or stop its server to free VRAM."""
    from localbench import llamacpp_mgr
    payload = payload or {}
    flavor = payload.get("runtime_flavor") or "llamacpp"
    name = payload.get("name")

    if flavor == "llamacpp" or not flavor:
        llamacpp_mgr.stop_llama_server(port=payload.get("port", 8080))
    if flavor == "lmstudio" and name:
        lms_path = shutil.which("lms")
        if lms_path:
            subprocess.run(f'lms unload "{name}"', shell=True, capture_output=True)
    return {"ok": True, "message": "Model unloaded and VRAM freed"}


@app.post("/api/models/unload_all")
def unload_all_endpoint() -> dict:
    """Unload all models across all runtimes (llama.cpp + LM Studio) and reclaim 100% VRAM."""
    from localbench import llamacpp_mgr
    llamacpp_mgr.stop_llama_server()
    lms_path = shutil.which("lms")
    if lms_path:
        try:
            subprocess.run("lms unload --all", shell=True, capture_output=True)
        except Exception:
            pass
    return {"ok": True, "message": "All models unloaded — VRAM is 100% free"}


@app.post("/api/llamacpp/launch")
def launch_llamacpp(payload: dict) -> dict:
    """Launch llama-server.exe in a dedicated terminal window with the specified model and flags."""
    from localbench import llamacpp_mgr
    port = payload.get("port", 8080)
    in_terminal = payload.get("in_terminal", True)
    ok, msg = llamacpp_mgr.launch_llama_server(payload, port=port, in_terminal=in_terminal)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@app.post("/api/llamacpp/stop")
def stop_llamacpp(payload: dict | None = None) -> dict:
    """Stop any currently running llama-server process."""
    from localbench import llamacpp_mgr
    port = (payload or {}).get("port", 8080)
    stopped = llamacpp_mgr.stop_llama_server(port=port)
    return {"ok": True, "stopped": stopped}


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
