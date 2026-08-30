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
    hardware_perf_suite,
    instruction_following_suite,
    json_schema_suite,
    logic_math_suite,
    long_context_suite,
    multi_turn_suite,
    pattern_reasoning_suite,
    tool_calling_suite,
)

class RunCancelled(Exception):
    """Raised to unwind a run the user stopped from the dashboard.

    Cancellation is delivered through the per-problem progress callback,
    which every suite already calls after each problem. That avoids
    threading a stop-flag parameter through six suite signatures, and it
    means a run can only be interrupted at a problem boundary -- never
    midway through grading, which would produce a half-written result.
    """


class ModelSwitchError(Exception):
    """A model's load command failed -- e.g. the model isn't downloaded on
    this machine. Raised so the run loop can skip just that model instead of
    aborting the entire run (and every model after it)."""


DEFAULT_PAUSE_PROMPT = (
    "\n>>> Load model '{model}' in your runtime, then press Enter to continue... "
)


def _default_confirm(message: str) -> None:
    input(message)


def _make_progress_logger(log, model_name: str, suite_name: str, should_cancel=None, on_stats=None):
    def on_progress(idx: int, total: int, problem_id: str, passed: bool, result=None) -> None:
        # Update live speed stats before logging, so the persisted state
        # written by log()'s own _persist() call already reflects them --
        # keeps the live HUD current within one problem, not one behind.
        if result is not None and on_stats is not None:
            # Also pass prefill throughput and the problem id: for a prefill
            # probe (huge prompt, tiny max_tokens) decode tok/s is a
            # structurally tiny number -- 8 tokens over a latency dominated by
            # processing thousands of prompt tokens -- and shown unlabelled as
            # "Generation Speed" it badly understates the model. The UI needs
            # to know which metric is the meaningful one for this task.
            on_stats(
                result.tokens_per_sec,
                result.ttft_seconds,
                result.prefill_tokens_per_sec,
                problem_id,
            )
        status = "PASS" if passed else "FAIL"
        log(f"    [{model_name}] {suite_name} [{idx}/{total}] {problem_id}: {status}")
        if should_cancel is not None and should_cancel():
            raise RunCancelled(f"stopped during {suite_name} after {idx}/{total} problems")

    return on_progress


def _build_runtime_load_cmd(model_cfg: dict) -> str | None:
    switch = model_cfg.get("switch") or {}
    name = model_cfg["name"]
    load_cmd = switch.get("load_cmd")
    flavor = model_cfg.get("runtime_flavor")

    ctx_len = model_cfg.get("context_length")
    kv = model_cfg.get("gpu_kv")
    gpu = model_cfg.get("gpu_offload")
    fa = model_cfg.get("flash_attention")
    mmap = model_cfg.get("mmap")
    mlock = model_cfg.get("mlock")
    batch = model_cfg.get("batch_size")
    split = model_cfg.get("split_mode")
    parallel = model_cfg.get("parallel")

    if load_cmd:
        cmd = load_cmd.format(model=name)
        if cmd.startswith("lms ") or "lms.exe" in cmd:
            if ctx_len and "-c " not in cmd and "--context-length" not in cmd:
                cmd += f" -c {ctx_len}"
            if parallel and "--parallel" not in cmd:
                cmd += f" --parallel {parallel}"
            return cmd

        if ctx_len and "-c " not in cmd and "--context-length" not in cmd and "--max-model-len" not in cmd:
            cmd += f" -c {ctx_len}"
        if parallel and "-np " not in cmd and "--parallel" not in cmd and "--max-num-seqs" not in cmd:
            cmd += f" -np {parallel}"
        if kv and "--gpu-kv" not in cmd and "-ctk" not in cmd and "--kv-cache-dtype" not in cmd:
            cmd += f" -ctk {kv} -ctv {kv}"
        if fa is True and "--flash-attention" not in cmd and "-fa" not in cmd:
            cmd += " -fa"
        if mmap is False and "--no-mmap" not in cmd:
            cmd += " --no-mmap"
        if mlock is True and "--mlock" not in cmd:
            cmd += " --mlock"
        if batch and "-b " not in cmd:
            cmd += f" -b {batch}"
        return cmd

    if flavor == "lmstudio" or (not flavor and shutil.which("lms")):
        gpu_val = gpu if gpu is not None else "max"
        cmd = f'lms load "{name}" -y --gpu {gpu_val}'
        if ctx_len:
            cmd += f" -c {ctx_len}"
        if parallel:
            cmd += f" --parallel {parallel}"
        # Accepts either key: speculative_mtp is what the dashboard sets
        # (shared with the llama.cpp path), speculative_draft_mtp is the
        # older config-file name kept working for existing configs.
        if model_cfg.get("speculative_mtp") or model_cfg.get("speculative_draft_mtp"):
            cmd += " --speculative-draft-mtp"
            n_max = model_cfg.get("speculative_n_max")
            if n_max:
                cmd += f" --speculative-draft-max-tokens {n_max}"
        return cmd

    if flavor == "llamacpp":
        ngl = model_cfg.get("gpu_offload_layers", 99)
        cmd = f'llama-server -m "{name}" -ngl {ngl}'
        if ctx_len:
            cmd += f" -c {ctx_len}"
        if parallel:
            cmd += f" -np {parallel}"
        if kv and kv != "f16":
            cmd += f" -ctk {kv} -ctv {kv}"
        if fa is True:
            cmd += " -fa"
        if mmap is False:
            cmd += " --no-mmap"
        if mlock is True:
            cmd += " --mlock"
        if batch:
            cmd += f" -b {batch}"
        if split:
            cmd += f" -sm {split}"
        return cmd

    if flavor == "vllm":
        gpu_util = gpu if (gpu and isinstance(gpu, (int, float))) else 0.95
        cmd = f'vllm serve "{name}" --gpu-memory-utilization {gpu_util}'
        if ctx_len:
            cmd += f" --max-model-len {ctx_len}"
        if parallel:
            cmd += f" --max-num-seqs {parallel}"
        if kv and kv != "f16":
            cmd += f" --kv-cache-dtype {kv}"
        return cmd

    return None


def _switch_to_model(model_cfg: dict, base_url: str, unload_all_cmd: str | None, log, confirm) -> None:
    name = model_cfg["name"]
    flavor = model_cfg.get("runtime_flavor") or (model_cfg.get("settings") or {}).get("runtime_flavor")

    if flavor == "llamacpp":
        from . import llamacpp_mgr
        target_url = "http://localhost:8080/v1"

        # Reuse a server that is already serving exactly this model. The user
        # can start one themselves from the config modal ("Load Model" /
        # "Launch in Terminal"), and launch_llama_server() unconditionally
        # stops any running server before starting a new one -- so without
        # this check, doing that and then pressing Start Benchmark unloaded
        # and reloaded the identical multi-GB weights, doubling the wait for
        # no benefit.
        existing_gguf = llamacpp_mgr.find_model_gguf(name)
        if existing_gguf and llamacpp_mgr.is_model_already_serving(existing_gguf, target_url):
            log(f"'{name}' is already loaded and serving on {target_url} -- reusing it (no reload)")
            return

        log(f"launching llama-server terminal for '{name}'...")
        # Without this, launch_llama_server() falls back to its legacy
        # auto-pick (Vulkan > ROCm > CUDA > AVX2 by substring score), which
        # ignores whatever compute backend was actually selected in the
        # dashboard -- and since -dev device ids are backend-specific (e.g.
        # "ROCm0" only means something to the ROCm binary), passing a ROCm
        # device id to an auto-picked Vulkan binary doesn't restrict
        # anything: llama.cpp silently falls back to using every GPU.
        backend = model_cfg.get("backend") or (model_cfg.get("settings") or {}).get("backend")
        ok, msg = llamacpp_mgr.launch_llama_server(model_cfg, in_terminal=True, port=8080, backend=backend)
        if not ok:
            raise ModelSwitchError(msg)
        log(f"{msg} -- waiting for readiness on {target_url}...")
        if not llamacpp_mgr.wait_for_server_ready(target_url, timeout_seconds=45.0):
            raise ModelSwitchError(
                f"llama-server was launched but did not report ready on {target_url}/models within 45s. Check the terminal window for details."
            )
        log(f"llama-server is online and ready on {target_url}")
        return

    cmd = _build_runtime_load_cmd(model_cfg)

    if cmd:
        if unload_all_cmd:
            log(f"clearing loaded models via: {unload_all_cmd}")
            try:
                subprocess.run(unload_all_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                log(f"WARNING: unload-all command failed (exit {e.returncode}); continuing")

        log(f"loading '{name}' via: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ModelSwitchError(
                f"load command for '{name}' failed (exit {e.returncode}). "
                f"Is this model downloaded on this machine? Command was: {cmd}"
            ) from e
        _sanity_check_model_present(base_url, name, log)
    else:
        confirm(DEFAULT_PAUSE_PROMPT.format(model=name))


def _unload_model(model_cfg: dict, log) -> None:
    flavor = model_cfg.get("runtime_flavor") or (model_cfg.get("settings") or {}).get("runtime_flavor")
    if flavor == "llamacpp":
        from . import llamacpp_mgr
        log("stopping llama.cpp server...")
        llamacpp_mgr.stop_llama_server()
        return

    switch = model_cfg.get("switch") or {}
    unload_cmd = switch.get("unload_cmd")
    if unload_cmd:
        log(f"unloading via: {unload_cmd}")
        try:
            subprocess.run(unload_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            # Non-fatal: the model's results are already recorded; don't let a
            # flaky unload throw away a completed benchmark.
            log(f"WARNING: unload command failed (exit {e.returncode}); continuing")


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
    on_stats=None,
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
    skipped_models: list[str] = []
    for i, model_cfg in enumerate(models):
        model_name = model_cfg["name"]
        _check_cancel(f"loading {model_name}")
        log(f"[{i + 1}/{len(models)}] switching to model: {model_name}")
        flavor = model_cfg.get("runtime_flavor") or (model_cfg.get("settings") or {}).get("runtime_flavor")
        if flavor == "llamacpp":
            effective_base_url = "http://localhost:8080/v1"
        elif flavor == "ollama":
            effective_base_url = "http://localhost:11434/v1"
        elif flavor == "vllm":
            effective_base_url = "http://localhost:8000/v1"
        else:
            effective_base_url = model_cfg.get("base_url") or base_url

        try:
            _switch_to_model(model_cfg, effective_base_url, unload_all_cmd, log, confirm)
        except ModelSwitchError as e:
            log(f"SKIPPED [{model_name}]: {e}")
            skipped_models.append(model_name)
            continue

        model_sampling = model_cfg.get("sampling") or {}
        effective_temp = model_sampling.get("temperature", sampling.get("temperature", 0.2))
        effective_max_tokens = model_sampling.get("max_tokens", sampling.get("max_tokens", 1024))

        ctx = RunContext(
            base_url=effective_base_url,
            model=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            temperature=effective_temp,
            max_tokens=effective_max_tokens,
        )

        # Pre-flight health probe: verify the runtime actually returns a chat completion
        probe = ctx.call([{"role": "user", "content": "1+1="}], max_tokens=8)
        if not probe.success:
            log(f"SKIPPED [{model_name}]: Runtime at {effective_base_url} is not responding ({probe.error}). Ensure the model is loaded in VRAM.")
            skipped_models.append(model_name)
            _unload_model(model_cfg, log)
            continue

        runtime_info = _capture_lms_load_info(model_name) or {}
        for k in [
            "context_length",
            "gpu_kv",
            "flash_attention",
            "mmap",
            "mlock",
            "batch_size",
            "split_mode",
            "runtime_flavor",
            "gpu_offload",
            "backend",
            "devices",
            "gpuSelection",
            "speculative_mtp",
            "speculative_n_max",
        ]:
            if model_cfg.get(k) is not None:
                runtime_info[k] = model_cfg[k]

        model_result = ModelRunResult(
            model=model_name, runtime_load_info=runtime_info or None
        )

        # Runs first, deliberately -- a hardware/speed baseline (prefill
        # scaling, sustained decode) that every other suite's numbers for
        # this model can be read against, before any accuracy suite has run.
        hardware_perf_cfg = suites_cfg.get("hardware_perf", {})
        if hardware_perf_cfg.get("enabled", True):
            log(f"  [{model_name}] running hardware_perf suite...")
            with ResourceMonitor() as mon:
                # The largest prefill tiers only run if they fit this model's
                # configured context window -- otherwise they'd overflow it
                # and produce a failed call rather than a slow measurement.
                model_ctx = model_cfg.get("context_length") or (model_cfg.get("settings") or {}).get("context_length")
                problems = hardware_perf_suite.run(
                    ctx,
                    seed=hardware_perf_cfg.get("seed", 42),
                    call_timeout_seconds=hardware_perf_cfg.get("call_timeout_seconds", 300),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="hardware_perf"),
                    max_context_tokens=model_ctx,
                )
            model_result.suites["hardware_perf"] = SuiteRunResult(
                suite="hardware_perf", problems=problems, resource_usage=mon.summary()
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
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="json_schema"),
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
                    detect_loops=coding_cfg.get("detect_loops", False),
                    early_exit_check=coding_cfg.get("early_exit_check", True),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="coding"),
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
                    detect_loops=logic_math_cfg.get("detect_loops", False),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="logic_math"),
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
                    detect_loops=instruction_following_cfg.get("detect_loops", False),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="instruction_following"),
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
                    detect_loops=pattern_reasoning_cfg.get("detect_loops", False),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="pattern_reasoning"),
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
                    detect_loops=long_context_cfg.get("detect_loops", False),
                    on_progress=_make_progress_logger(log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="long_context"),
                )
            model_result.suites["long_context"] = SuiteRunResult(
                suite="long_context", problems=problems, resource_usage=mon.summary()
            )

        tool_calling_cfg = suites_cfg.get("tool_calling", {})
        if tool_calling_cfg.get("enabled", True):
            log(f"  [{model_name}] running tool_calling suite...")
            with ResourceMonitor() as mon:
                problems = tool_calling_suite.run(
                    ctx,
                    num_problems=tool_calling_cfg.get("num_problems", 16),
                    seed=tool_calling_cfg.get("seed", 42),
                    config=tool_calling_cfg,
                    on_progress=_make_progress_logger(
                        log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="tool_calling"
                    ),
                )
            model_result.suites["tool_calling"] = SuiteRunResult(
                suite="tool_calling", problems=problems, resource_usage=mon.summary()
            )

        multi_turn_cfg = suites_cfg.get("multi_turn", {})
        if multi_turn_cfg.get("enabled", True):
            log(f"  [{model_name}] running multi_turn suite...")
            with ResourceMonitor() as mon:
                problems = multi_turn_suite.run(
                    ctx,
                    num_problems=multi_turn_cfg.get("num_problems", 12),
                    seed=multi_turn_cfg.get("seed", 42),
                    config=multi_turn_cfg,
                    on_progress=_make_progress_logger(
                        log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="multi_turn"
                    ),
                )
            model_result.suites["multi_turn"] = SuiteRunResult(
                suite="multi_turn", problems=problems, resource_usage=mon.summary()
            )

        if judge_client is not None:
            log(f"  [{model_name}] running frontier_graded suite (paid, non-deterministic)...")
            problems = frontier_graded_suite.run(
                ctx,
                judge_client,
                num_tasks=judge_cfg.get("num_tasks", 6),
                categories=judge_cfg.get("categories"),
                pass_threshold=judge_cfg.get("pass_threshold", 7),
                on_progress=_make_progress_logger(
                    log, model_name, should_cancel=should_cancel, on_stats=on_stats, suite_name="frontier_graded"
                ),
            )
            model_result.suites["frontier_graded"] = SuiteRunResult(
                suite="frontier_graded", problems=problems
            )

        run.models[model_name] = model_result

        if (model_cfg.get("switch") or {}).get("unload_cmd"):
            log(f"  unloading model: {model_name}")
        _unload_model(model_cfg, log)

    if skipped_models:
        log(
            f"NOTE: {len(skipped_models)} model(s) skipped because their load "
            f"command failed: {', '.join(skipped_models)}"
        )
    if not run.models:
        raise RuntimeError(
            "no models were benchmarked -- every selected model failed to load. "
            "Check that the models are downloaded on this machine (use 'Detect "
            "live models' in the dashboard to see what's actually available)."
        )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(run, results_dir=results_dir)
    log(f"run saved to {path}")

    return run
