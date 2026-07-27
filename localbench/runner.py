"""Orchestrates a full benchmark run: sequential model switching + suites.

Models are never loaded concurrently -- these are large models on limited
VRAM. For each model, if config defines switch.load_cmd/unload_cmd (e.g. LM
Studio's `lms load`/`lms unload`), those commands are shelled out to and
awaited; otherwise the runner pauses for a manual switch (Ollama/llama.cpp/
manual control).
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess

import requests

from .engine import RunContext
from .hardware import get_hardware_snapshot
from .profiles import DEFAULT_PROFILE, problems_for
from .resource_monitor import ResourceMonitor
from .results import ModelRunResult, RunRecord, SuiteRunResult
from .storage import save_run
from .suites import (
    coding_suite,
    frontier_graded_suite,
    instruction_following_suite,
    json_schema_suite,
    logic_math_suite,
    long_context_suite,
    pattern_reasoning_suite,
)

class RunCancelled(Exception):
    """Raised to unwind a run the user stopped from the dashboard.

    Cancellation is delivered through the per-problem progress callback,
    which every suite already calls after each problem. That avoids
    threading a stop-flag parameter through six suite signatures, and it
    means a run can only be interrupted at a problem boundary -- never
    midway through grading, which would produce a half-written result.
    """


DEFAULT_PAUSE_PROMPT = (
    "\n>>> Load model '{model}' in your runtime, then press Enter to continue... "
)


def _default_confirm(message: str) -> None:
    input(message)


def _make_progress_logger(log, model_name: str, suite_name: str, should_cancel=None):
    def on_progress(idx: int, total: int, problem_id: str, passed: bool) -> None:
        status = "PASS" if passed else "FAIL"
        log(f"    [{model_name}] {suite_name} [{idx}/{total}] {problem_id}: {status}")
        if should_cancel is not None and should_cancel():
            raise RunCancelled(f"stopped during {suite_name} after {idx}/{total} problems")

    return on_progress


def _switch_to_model(model_cfg: dict, base_url: str, unload_all_cmd: str | None, log, confirm) -> None:
    switch = model_cfg.get("switch") or {}
    load_cmd = switch.get("load_cmd")
    name = model_cfg["name"]

    if load_cmd:
        # Some runtimes (e.g. LM Studio's `lms load`) will stack a second
        # instance of a model instead of reusing an already-loaded one, so
        # always start from a clean slate before loading the target model.
        if unload_all_cmd:
            log(f"clearing loaded models via: {unload_all_cmd}")
            subprocess.run(unload_all_cmd, shell=True, check=True)

        cmd = load_cmd.format(model=name)
        log(f"loading '{name}' via: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        _sanity_check_model_present(base_url, name, log)
    else:
        confirm(DEFAULT_PAUSE_PROMPT.format(model=name))


def _unload_model(model_cfg: dict, log) -> None:
    switch = model_cfg.get("switch") or {}
    unload_cmd = switch.get("unload_cmd")
    if unload_cmd:
        log(f"unloading via: {unload_cmd}")
        subprocess.run(unload_cmd, shell=True, check=True)


def _sanity_check_model_present(base_url: str, model_name: str, log) -> None:
    """Best-effort check that the model shows up after a load command -- not a
    readiness poll (lms load already blocks until loaded), just a fast-fail if
    the load silently didn't work."""
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=10)
        ids = [m.get("id") for m in resp.json().get("data", [])]
        if model_name not in ids:
            log(f"WARNING: '{model_name}' not found in {base_url}/models after load command")
    except requests.exceptions.RequestException as e:
        log(f"WARNING: could not verify model list after load: {e}")


def _capture_lms_load_info(model_name: str) -> dict | None:
    """Best-effort: if LM Studio's `lms` CLI is available, record what it
    reports about the currently loaded model (size, quantization, context
    length, status). Returns None silently if `lms` isn't installed or the
    query fails -- this is purely diagnostic, never required for a run."""
    if not shutil.which("lms"):
        return None
    try:
        proc = subprocess.run(
            ["lms", "ps", "--json"], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            return None
        entries = json.loads(proc.stdout)
        for entry in entries:
            if entry.get("modelKey") == model_name or entry.get("identifier") == model_name:
                return {
                    "size_bytes": entry.get("sizeBytes"),
                    "context_length": entry.get("contextLength"),
                    "quantization": entry.get("quantization"),
                    "status": entry.get("status"),
                }
        return None
    except Exception:
        return None


def run_benchmark(
    config: dict,
    progress_cb=None,
    confirm_cb=None,
    run_frontier_graded: bool = False,
    should_cancel=None,
    profile: str = DEFAULT_PROFILE,
) -> RunRecord:
    log = progress_cb or print
    confirm = confirm_cb or _default_confirm

    def _check_cancel(where: str) -> None:
        if should_cancel is not None and should_cancel():
            raise RunCancelled(f"stopped before {where}")

    runtime_cfg = config["runtime"]
    base_url = runtime_cfg["base_url"]
    api_key = runtime_cfg.get("api_key", "not-needed")
    timeout_seconds = runtime_cfg.get("request_timeout_seconds", 120)
    unload_all_cmd = runtime_cfg.get("unload_all_cmd")
    sampling = config.get("sampling", {})
    suites_cfg = config.get("suites", {})

    # Resolve the profile into concrete per-suite counts once, up front, so
    # the record below states exactly what was actually run.
    for _suite, _cfg in suites_cfg.items():
        _n = problems_for(_suite, profile, _cfg.get("num_problems"))
        if _n is not None:
            _cfg["num_problems"] = _n

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    started_at = datetime.datetime.now().isoformat(timespec="seconds")

    run = RunRecord(
        run_id=run_id,
        started_at=started_at,
        hardware=get_hardware_snapshot(),
        config_summary={
            "base_url": base_url,
            "suites_enabled": {k: v.get("enabled", True) for k, v in suites_cfg.items()},
            "sampling": sampling,
            "profile": profile,
            "problems_per_suite": {
                k: v.get("num_problems")
                for k, v in suites_cfg.items()
                if v.get("enabled", True) and v.get("num_problems") is not None
            },
        },
    )

    judge_cfg = config.get("judge", {})
    judge_client = None
    if run_frontier_graded:
        if not judge_cfg.get("enabled"):
            raise ValueError(
                "run_frontier_graded=True but judge.enabled is false in config -- "
                "set judge.enabled: true and configure a provider/model first"
            )
        run.config_summary["frontier_judge"] = {
            "provider": judge_cfg.get("provider"),
            "model": judge_cfg.get("model"),
            "pass_threshold": judge_cfg.get("pass_threshold", 7),
        }
        from .judge.factory import get_judge_client

        log(f"initializing frontier judge: {judge_cfg['provider']}/{judge_cfg['model']}")
        judge_client = get_judge_client(judge_cfg["provider"], judge_cfg["model"])

    models = config["models"]
    for i, model_cfg in enumerate(models):
        model_name = model_cfg["name"]
        _check_cancel(f"loading {model_name}")
        log(f"[{i + 1}/{len(models)}] switching to model: {model_name}")
        _switch_to_model(model_cfg, base_url, unload_all_cmd, log, confirm)

        ctx = RunContext(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            temperature=sampling.get("temperature", 0.2),
            max_tokens=sampling.get("max_tokens", 1024),
        )

        model_result = ModelRunResult(
            model=model_name, runtime_load_info=_capture_lms_load_info(model_name)
        )

        json_schema_cfg = suites_cfg.get("json_schema", {})
        if json_schema_cfg.get("enabled", True):
            log(f"  [{model_name}] running json_schema suite...")
            with ResourceMonitor() as mon:
                problems = json_schema_suite.run(
                    ctx,
                    generated=json_schema_cfg.get("generated", True),
                    num_problems=json_schema_cfg.get("num_problems", 20),
                    seed=json_schema_cfg.get("seed", 42),
                    max_tokens=json_schema_cfg.get("max_tokens"),
                    call_timeout_seconds=json_schema_cfg.get("call_timeout_seconds"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="json_schema"),
                )
            model_result.suites["json_schema"] = SuiteRunResult(
                suite="json_schema", problems=problems, resource_usage=mon.summary()
            )

        coding_cfg = suites_cfg.get("coding", {})
        if coding_cfg.get("enabled", True):
            log(f"  [{model_name}] running coding suite...")
            with ResourceMonitor() as mon:
                problems = coding_suite.run(
                    ctx,
                    timeout_seconds=coding_cfg.get("timeout_seconds", 10),
                    generated=coding_cfg.get("generated", True),
                    num_problems=coding_cfg.get("num_problems", 12),
                    seed=coding_cfg.get("seed", 42),
                    max_tokens=coding_cfg.get("max_tokens"),
                    call_timeout_seconds=coding_cfg.get("call_timeout_seconds"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="coding"),
                )
            model_result.suites["coding"] = SuiteRunResult(
                suite="coding", problems=problems, resource_usage=mon.summary()
            )

        logic_math_cfg = suites_cfg.get("logic_math", {})
        if logic_math_cfg.get("enabled", True):
            log(f"  [{model_name}] running logic_math suite...")
            with ResourceMonitor() as mon:
                problems = logic_math_suite.run(
                    ctx,
                    num_problems=logic_math_cfg.get("num_problems", 20),
                    seed=logic_math_cfg.get("seed", 42),
                    max_tokens=logic_math_cfg.get("max_tokens"),
                    call_timeout_seconds=logic_math_cfg.get("call_timeout_seconds"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="logic_math"),
                )
            model_result.suites["logic_math"] = SuiteRunResult(
                suite="logic_math", problems=problems, resource_usage=mon.summary()
            )

        instruction_following_cfg = suites_cfg.get("instruction_following", {})
        if instruction_following_cfg.get("enabled", True):
            log(f"  [{model_name}] running instruction_following suite...")
            with ResourceMonitor() as mon:
                problems = instruction_following_suite.run(
                    ctx,
                    num_problems=instruction_following_cfg.get("num_problems", 16),
                    seed=instruction_following_cfg.get("seed", 42),
                    max_tokens=instruction_following_cfg.get("max_tokens"),
                    call_timeout_seconds=instruction_following_cfg.get("call_timeout_seconds"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="instruction_following"),
                )
            model_result.suites["instruction_following"] = SuiteRunResult(
                suite="instruction_following", problems=problems, resource_usage=mon.summary()
            )

        pattern_reasoning_cfg = suites_cfg.get("pattern_reasoning", {})
        if pattern_reasoning_cfg.get("enabled", True):
            log(f"  [{model_name}] running pattern_reasoning suite...")
            with ResourceMonitor() as mon:
                problems = pattern_reasoning_suite.run(
                    ctx,
                    num_problems=pattern_reasoning_cfg.get("num_problems", 10),
                    seed=pattern_reasoning_cfg.get("seed", 42),
                    max_tokens=pattern_reasoning_cfg.get("max_tokens"),
                    call_timeout_seconds=pattern_reasoning_cfg.get("call_timeout_seconds"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="pattern_reasoning"),
                )
            model_result.suites["pattern_reasoning"] = SuiteRunResult(
                suite="pattern_reasoning", problems=problems, resource_usage=mon.summary()
            )

        long_context_cfg = suites_cfg.get("long_context", {})
        if long_context_cfg.get("enabled", True):
            log(f"  [{model_name}] running long_context suite...")
            with ResourceMonitor() as mon:
                problems = long_context_suite.run(
                    ctx,
                    num_problems=long_context_cfg.get("num_problems", 4),
                    seed=long_context_cfg.get("seed", 42),
                    source_file=long_context_cfg.get("source_file"),
                    window_lines=long_context_cfg.get("window_lines", 1000),
                    timeout_seconds=long_context_cfg.get("timeout_seconds", 180),
                    max_tokens=long_context_cfg.get("max_tokens"),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, suite_name="long_context"),
                )
            model_result.suites["long_context"] = SuiteRunResult(
                suite="long_context", problems=problems, resource_usage=mon.summary()
            )

        if judge_client is not None:
            log(f"  [{model_name}] running frontier_graded suite (paid, non-deterministic)...")
            problems = frontier_graded_suite.run(
                ctx,
                judge_client,
                num_tasks=judge_cfg.get("num_tasks", 6),
                categories=judge_cfg.get("categories"),
                pass_threshold=judge_cfg.get("pass_threshold", 7),
            )
            model_result.suites["frontier_graded"] = SuiteRunResult(
                suite="frontier_graded", problems=problems
            )

        run.models[model_name] = model_result

        if (model_cfg.get("switch") or {}).get("unload_cmd"):
            log(f"  unloading model: {model_name}")
        _unload_model(model_cfg, log)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(run, results_dir=results_dir)
    log(f"run saved to {path}")

    return run
