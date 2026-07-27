"""FastAPI backend for the localbench dashboard.

Run via `python cli.py serve`, or directly with:
    uvicorn webapp.main:app --port 8000
"""

from __future__ import annotations

import json
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

from localbench import report, storage
from localbench.config import load_config
from localbench.engine import RunContext
from localbench.json_extract import extract_json
from webapp.run_manager import RunManager

app = FastAPI(title="localbench dashboard")
run_manager = RunManager(results_dir=load_config().get("output", {}).get("results_dir", "results"))

STATIC_DIR = Path(__file__).parent / "static"
ALL_SUITES = [
    "json_schema",
    "coding",
    "logic_math",
    "instruction_following",
    "pattern_reasoning",
    "long_context",
]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _get_suite_task_count(name: str, config: dict) -> int:
    s_cfg = config.get("suites", {}).get(name, {})
    if "num_problems" in s_cfg:
        return s_cfg["num_problems"]
    if name == "json_schema":
        try:
            from localbench.data.json_schema_problems import PROBLEMS
            return len(PROBLEMS)
        except Exception:
            return 5
    if name == "coding":
        try:
            from localbench.data.coding_problems import PROBLEMS
            return len(PROBLEMS)
        except Exception:
            return 8
    defaults = {
        "logic_math": 20,
        "instruction_following": 16,
        "pattern_reasoning": 10,
        "long_context": 4,
    }
    return defaults.get(name, 0)


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
            }
            for name in ALL_SUITES
        },
    }


@app.get("/api/models/detect")
def detect_models() -> dict:
    """Query the configured runtime directly for what's actually loadable
    right now -- lets the dashboard offer live model selection instead of
    only whatever's hand-typed into config.yaml."""
    config = load_config()
    base_url = config["runtime"]["base_url"]
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=10)
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", [])]
        return {"models": ids}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"could not reach runtime at {base_url}: {e}")


@app.get("/api/models/catalog")
def models_catalog() -> dict:
    """Richer per-model metadata (size, quantization, params, context length,
    vision/tool-use capability flags) for every model already downloaded --
    not just the currently-loaded one. Best-effort: only works when LM
    Studio's `lms` CLI is on PATH; returns an empty catalog otherwise so the
    UI can fall back to bare model IDs from /api/models/detect rather than
    fabricate data for runtimes this doesn't support (Ollama/llama.cpp)."""
    lms_path = shutil.which("lms")
    if not lms_path:
        return {"models": {}}
    try:
        proc = subprocess.run(
            [lms_path, "ls", "--json"], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"models": {}}
        entries = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return {"models": {}}

    catalog = {}
    for entry in entries:
        key = entry.get("modelKey") or entry.get("indexedModelIdentifier")
        if not key:
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
        }
    return {"models": catalog}


@app.post("/api/run")
def start_run(payload: dict) -> dict:
    config = load_config()

    models_filter = payload.get("models")
    if models_filter:
        by_name = {m["name"]: m for m in config["models"]}
        # A model picked from live detection (not in config.yaml) has no
        # known switch command -- fall back to manual-switch mode for it
        # rather than rejecting the run outright.
        config["models"] = [by_name.get(name, {"name": name}) for name in models_filter]

    suites_filter = payload.get("suites")
    if suites_filter is not None:
        suites_cfg = config.setdefault("suites", {})
        for suite_name in ALL_SUITES:
            suites_cfg.setdefault(suite_name, {})["enabled"] = suite_name in suites_filter

    run_id = run_manager.start_run(config)
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


@app.get("/api/runs")
def list_runs() -> list:
    return storage.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return storage.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.get("/api/runs/{run_id}/report.md")
def get_run_markdown(run_id: str) -> PlainTextResponse:
    try:
        data = storage.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")
    return PlainTextResponse(report.render_markdown(data), media_type="text/markdown")


@app.get("/api/runs/{run_id}/report.pdf")
def get_run_pdf(run_id: str) -> FileResponse:
    try:
        data = storage.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")
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
    if not model or not prompt:
        raise HTTPException(400, "both 'model' and 'prompt' are required")

    ctx = RunContext(
        base_url=config["runtime"]["base_url"],
        model=model,
        api_key=config["runtime"].get("api_key", "not-needed"),
        timeout_seconds=payload.get("timeout_seconds", 120),
        max_tokens=payload.get("max_tokens", 1024),
    )
    chat = ctx.call([{"role": "user", "content": prompt}])

    result = {
        "success": chat.success,
        "content": chat.content,
        "reasoning_content": chat.reasoning_content,
        "error": chat.error,
        "truncated": chat.truncated,
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
