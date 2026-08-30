"""Render a RunRecord (see storage.py / results.py) into a markdown leaderboard,
or an HTML/PDF version of the same report for sharing outside a terminal.

Operates on plain dicts (as loaded from JSON) so it works identically whether
fed a freshly-built run or one loaded back from results/runs/*.json.
"""

from __future__ import annotations

import html
from pathlib import Path


def _fmt_pct(x: float | None) -> str:
    """One decimal place, not a whole-number round -- matches the dashboard's
    fmtPct(). Suite sizes as small as 4-10 problems mean a single problem's
    difference is already a real percentage-point gap (e.g. 1/12 = 8.3
    points) that whole-number rounding could visually collapse two close-
    but-different pass rates into the same displayed number. Trims to a
    bare integer when the rate is exact (100%, 50%, 0%) so it doesn't read
    as falsely precise.
    """
    if x is None:
        return "n/a"
    rounded = round(x * 1000) / 10
    return f"{rounded:.0f}%" if rounded == int(rounded) else f"{rounded:.1f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if x is not None else "n/a"


def _suite_names(run: dict) -> list[str]:
    seen: dict[str, None] = {}
    for model_result in run["models"].values():
        for suite_name in model_result["suites"]:
            seen.setdefault(suite_name, None)
    return list(seen.keys())


def _hardware_block(hardware: dict) -> str:
    gpu = hardware.get("gpu")
    if isinstance(gpu, list):
        gpu_str = ", ".join(
            f"{g['name']} ({g.get('memory') or g.get('memory_approx', 'unknown')})" for g in gpu
        )
    else:
        gpu_str = str(gpu)
    return (
        f"- **OS:** {hardware.get('os', 'unknown')} ({hardware.get('machine_arch', 'unknown')})\n"
        f"- **CPU:** {hardware.get('cpu', 'unknown')} "
        f"({hardware.get('cpu_count_physical', '?')}c/{hardware.get('cpu_count_logical', '?')}t)\n"
        f"- **RAM:** {hardware.get('ram_total_gb', 'unknown')} GB\n"
        f"- **GPU:** {gpu_str}\n"
    )


def _models_summary_block(run: dict) -> str:
    """Model size/quantization at a glance -- useful for comparing an
    ultra-compressed model (e.g. end-to-end low-bit trained) against a
    traditionally post-training-quantized model of similar parameter count:
    does it actually keep pass rate while using a fraction of the size?"""
    lines = [
        "| Model | Size on Disk | Quantization | Context Length | Engine / Backend / GPUs |",
        "|---|---|---|---|---|",
    ]
    for model_name, model_result in run["models"].items():
        info = model_result.get("runtime_load_info") or {}
        size_bytes = info.get("size_bytes")
        size_str = f"{size_bytes / (1024**3):.2f} GB" if size_bytes else "unknown"
        quant = info.get("quantization")
        quant_str = quant.get("name") if isinstance(quant, dict) else "unknown"
        ctx_len = info.get("context_length", "unknown")
        lines.append(
            f"| {model_name} | {size_str} | {quant_str} | {ctx_len} | {_engine_summary(model_result, run.get('hardware', {}))} |"
        )
    return "\n".join(lines)


# hardware_perf problem ids, in the order the suite generates them, paired
# with the short column label each represents. Kept here (rather than importing
# from the suite) so a saved run from an older/newer version still renders:
# a missing id just shows as "-".
_HW_PREFILL_TIERS = [
    ("prefill_tiny", "~200 tok"),
    ("prefill_small", "~1k tok"),
    ("prefill_medium", "~4k tok"),
    ("prefill_large", "~8k tok"),
    ("prefill_xl", "~16k tok"),
    ("prefill_xxl", "~32k tok"),
]
_HW_DECODE_TIERS = [("decode_short", "1024 gen"), ("decode_long", "4096 gen")]


def _hw_problem(suite: dict, problem_id: str) -> dict | None:
    for p in suite.get("problems", []):
        if p.get("problem_id") == problem_id:
            return p
    return None


def _hardware_perf_rows(run: dict) -> list[tuple[str, list[str]]]:
    """(model_name, cells) for the hardware_perf speed table.

    This suite measures throughput, not correctness, so a pass rate for it is
    meaningless (everything that completes "passes"). It gets its own table
    built from prefill_tokens_per_sec / tokens_per_sec instead -- without this
    the headline numbers of the whole suite never appeared in a report at all.
    """
    rows = []
    for model_name, model_result in run["models"].items():
        suite = model_result["suites"].get("hardware_perf")
        if suite is None:
            continue
        cells = []
        for pid, _ in _HW_PREFILL_TIERS:
            p = _hw_problem(suite, pid)
            cells.append(_fmt_num(p.get("prefill_tokens_per_sec"), 0) if p else "-")
        for pid, _ in _HW_DECODE_TIERS:
            p = _hw_problem(suite, pid)
            cells.append(_fmt_num(p.get("tokens_per_sec"), 1) if p else "-")
        tiny = _hw_problem(suite, "prefill_tiny")
        cells.append(_fmt_num(tiny.get("ttft_seconds"), 2) if tiny else "-")
        rows.append((model_name, cells))
    return rows


_HW_NOTE = (
    "Raw hardware throughput for this model on this machine -- speed only, not "
    "correctness, so it carries no pass rate and is deliberately excluded from "
    "the accuracy tables below. Prefill is prompt processing (prompt tokens / "
    "time-to-first-token) measured across escalating context lengths; decode "
    "is sustained generation speed. Prefill usually climbs sharply with "
    "context length as the GPU saturates, which a single-length test would "
    "hide. A tier shows '-' when it was skipped because it exceeded that "
    "model's configured context window."
)


def _outcome_counts(suite: dict) -> str:
    """Failure states broken out, rather than lumping every non-pass together.

    `truncated`, `loop_detected` and `early_exit` are tracked per problem and
    shown in the dashboard, but were previously invisible in reports -- yet
    they mean very different things: hitting the token budget, getting stuck
    repeating, or answering correctly but never stopping.
    """
    problems = suite.get("problems", [])
    truncated = sum(1 for p in problems if p.get("truncated"))
    looped = sum(1 for p in problems if p.get("loop_detected"))
    early = sum(1 for p in problems if p.get("early_exit"))
    bits = []
    if truncated:
        bits.append(f"{truncated} truncated")
    if looped:
        bits.append(f"{looped} loop-detected")
    if early:
        bits.append(f"{early} early-exit")
    return ", ".join(bits) if bits else "-"


def _pass_with_ci(suite: dict) -> str:
    """Pass rate plus its 95% confidence interval.

    The interval is already computed and stored per suite; omitting it from
    reports overstated precision, since on a 6-12 problem suite a single
    problem is a large swing.
    """
    base = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
    ci = suite.get("pass_rate_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None:
        return f"{base} [{_fmt_pct(ci[0])}-{_fmt_pct(ci[1])}]"
    return base


def _run_gpu_description(model_result: dict, hardware: dict) -> str:
    """Which GPU(s) a run actually used, distinguishing recorded from inferred.

    `hardware["gpu"]` lists every GPU *installed* on the host, captured once
    per run and independent of which devices the run was restricted to.
    Reporting it as the run's GPUs misstates a single-GPU run on a multi-GPU
    machine as having used every card -- an unmeasured claim. See the matching
    describeRunGpus() in app.js.
    """
    info = model_result.get("runtime_load_info") or {}
    recorded = info.get("gpuSelection") or info.get("devices")
    if isinstance(recorded, list) and recorded:
        return " + ".join(str(g) for g in recorded)

    gpu = hardware.get("gpu")
    host = [str(g.get("name", "GPU")) for g in gpu] if isinstance(gpu, list) else []
    # split_mode "none" means one GPU either way (the config modal writes it
    # when a single GPU is picked; llama.cpp's own `-sm none` likewise means
    # "use one GPU"), so the count is known even when the identity isn't.
    if info.get("split_mode") == "none" and len(host) > 1:
        return f"1 of {len(host)} GPUs (which one not recorded)"
    if host:
        return " + ".join(host) + ("" if len(host) == 1 else " (host GPUs; per-run selection not recorded)")
    return "unknown"


def _engine_summary(model_result: dict, hardware: dict | None = None) -> str:
    """Which runtime/compute backend and GPUs actually produced these numbers.

    Speed numbers are meaningless without it: the same model on the same box
    can differ several-fold between compute backends or GPU selections, so a
    report that omits this can't be compared against another run honestly.
    """
    info = model_result.get("runtime_load_info") or {}
    bits = []
    flavor = info.get("runtime_flavor")
    backend = info.get("backend")
    if flavor:
        bits.append(str(flavor))
    if backend:
        bits.append(str(backend))
    if info.get("speculative_mtp"):
        bits.append("MTP")
    if hardware is not None:
        bits.append(_run_gpu_description(model_result, hardware))
    else:
        gpus = info.get("gpuSelection") or info.get("devices")
        if isinstance(gpus, list) and gpus:
            bits.append(" + ".join(str(g) for g in gpus))
    return " / ".join(bits) if bits else "unknown"


def render_markdown(run: dict) -> str:
    lines = [f"# TaskMatch AI report — `{run['run_id']}`", ""]
    lines.append(f"**Started:** {run.get('started_at', 'unknown')}")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append(_models_summary_block(run))
    lines.append("")
    lines.append("## Hardware")
    lines.append("")
    lines.append(_hardware_block(run.get("hardware", {})))
    lines.append("")
    lines.append(
        "*`VRAM used` is total GPU memory in use at each suite's peak -- the model's "
        "actual footprint. `VRAM Δ` and `RAM Δ` are the increase over a baseline "
        "sampled just before that suite started; both are normally small, because "
        "the model is already loaded before any suite begins, so they mostly track "
        "KV-cache growth rather than model size. A small `RAM Δ` does NOT mean a "
        "small model -- a GPU-resident model lives in VRAM, not RAM. GPU figures "
        "come from `nvidia-smi` where available, otherwise Windows' own GPU "
        "performance counters (which work on AMD/Intel too); `not captured` means "
        "no GPU probe was available, never a guess.*"
    )

    # hardware_perf measures speed, not correctness -- it gets its own table
    # and is kept out of the accuracy loop, matching how the dashboard treats
    # it. Rendering it as a normal suite would show a meaningless ~100% "pass
    # rate" alongside real accuracy scores.
    if "hardware_perf" in _suite_names(run):
        lines.append("## hardware_perf — raw speed")
        lines.append("")
        lines.append(f"*{_HW_NOTE}*")
        lines.append("")
        header = ["Model"] + [f"Prefill {lbl}" for _, lbl in _HW_PREFILL_TIERS] + [f"Decode {lbl}" for _, lbl in _HW_DECODE_TIERS] + ["TTFT (s)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for model_name, cells in _hardware_perf_rows(run):
            lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("*Prefill and decode figures are tokens/sec; higher is better.*")
        lines.append("")

    deterministic_suites = [s for s in _suite_names(run) if s not in ("frontier_graded", "hardware_perf")]

    for suite_name in deterministic_suites:
        lines.append(f"## {suite_name}")
        lines.append("")
        lines.append(
            "| Model | Pass Rate (95% CI) | Outcomes | Avg Latency (s) | Avg TTFT (s) | Avg Tokens/sec | "
            "VRAM used (GB) | VRAM Δ (MB) | GPU % | RAM Δ (GB) | Peak CPU % |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get(suite_name)
            if suite is None:
                continue
            pass_str = _pass_with_ci(suite)
            resource = suite.get("resource_usage") or {}
            vram_total = resource.get("peak_vram_mb_total")
            lines.append(
                f"| {model_name} | {pass_str} | {_outcome_counts(suite)} | "
                f"{_fmt_num(suite['avg_latency_seconds'])} | "
                f"{_fmt_num(suite.get('avg_ttft_seconds'))} | "
                f"{_fmt_num(suite['avg_tokens_per_sec'])} | "
                f"{_fmt_num(vram_total / 1024 if vram_total is not None else None, 2)} | "
                f"{_fmt_num(resource.get('vram_delta_mb'), 0)} | "
                f"{_fmt_num(resource.get('peak_gpu_util_percent'), 0)} | "
                f"{_fmt_num(resource.get('ram_delta_gb'), 2)} | "
                f"{_fmt_num(resource.get('peak_cpu_percent'), 1)} |"
            )
        lines.append("")

        failures = []
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get(suite_name)
            if suite is None:
                continue
            for problem in suite["problems"]:
                if not problem["passed"]:
                    failures.append((model_name, problem))
        if failures:
            lines.append("<details><summary>Failures</summary>")
            lines.append("")
            for model_name, problem in failures:
                lines.append(f"- **{model_name} / {problem['problem_id']}**: {problem['error']}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    if "frontier_graded" in _suite_names(run):
        lines.append("## Frontier-Graded (non-deterministic, paid -- directional signal only)")
        lines.append("")
        lines.append(
            "*Scored by a frontier-model judge per [TASK_SPEC.md](TASK_SPEC.md), never "
            "mixed into the deterministic pass-rate numbers above.*"
        )
        lines.append("")
        lines.append("| Model | Avg Score (/10) | Pass Rate (score >= threshold) |")
        lines.append("|---|---|---|")
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get("frontier_graded")
            if suite is None:
                continue
            scores = [p["score"] for p in suite["problems"] if p.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else None
            pass_str = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
            lines.append(f"| {model_name} | {_fmt_num(avg_score, 1)} | {pass_str} |")
        lines.append("")
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get("frontier_graded")
            if suite is None:
                continue
            for problem in suite["problems"]:
                lines.append(
                    f"- **{model_name} / {problem['problem_id']}** "
                    f"(score {problem.get('score', 'n/a')}): "
                    f"{problem.get('rationale') or problem.get('error') or ''}"
                )
        lines.append("")

    return "\n".join(lines)


def render_comparison_markdown(runs: list[dict]) -> str:
    """Compare the same suite/model across multiple saved runs (e.g. different
    hardware, or re-runs over time)."""
    lines = ["# TaskMatch AI comparison", ""]
    lines.append(
        "| Run | Started | Host GPUs (installed) | Engine / Backend / GPUs used | Model | Suite | "
        "Pass Rate (95% CI) | Avg Latency (s) | Tok/s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        hw = run.get("hardware", {})
        gpu = hw.get("gpu")
        # Every detected GPU, not just the first -- on a multi-GPU box naming
        # only gpu[0] silently misattributes the run to one card.
        if isinstance(gpu, list) and gpu:
            gpu_name = " + ".join(str(g.get("name", "GPU")) for g in gpu)
        else:
            gpu_name = "unknown"
        for model_name, model_result in run["models"].items():
            engine = _engine_summary(model_result, hw)
            for suite_name, suite in model_result["suites"].items():
                # hardware_perf grades completion, not correctness, so its
                # ~100% "pass rate" is not comparable to an accuracy suite's
                # and must not sit in the same column as one.
                if suite_name == "hardware_perf":
                    pass_str = "n/a (speed-only suite)"
                else:
                    pass_str = _pass_with_ci(suite)
                lines.append(
                    f"| {run['run_id']} | {run.get('started_at', '?')} | {gpu_name} | {engine} | "
                    f"{model_name} | {suite_name} | {pass_str} | "
                    f"{_fmt_num(suite['avg_latency_seconds'])} | "
                    f"{_fmt_num(suite['avg_tokens_per_sec'])} |"
                )
    return "\n".join(lines)


_PDF_STYLE = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #1a1a1a; }
  h1 { font-size: 18pt; margin-bottom: 2pt; }
  h2 { font-size: 13pt; margin-top: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
  table { width: 100%; border-collapse: collapse; margin-top: 6pt; }
  th, td { border: 1px solid #ccc; padding: 4pt 6pt; text-align: left; font-size: 9pt; }
  th { background-color: #f0f0f0; }
  .meta { color: #555; font-size: 9pt; margin-bottom: 4pt; }
  .fail { color: #b00020; font-size: 8pt; }
</style>
"""


def render_html(run: dict) -> str:
    """HTML version of the same report, used both for on-screen viewing and as
    the source document for PDF export (see render_pdf)."""
    esc = html.escape
    parts = [_PDF_STYLE, f"<h1>TaskMatch AI report &mdash; {esc(run['run_id'])}</h1>"]
    parts.append(f"<div class='meta'>Started: {esc(str(run.get('started_at', 'unknown')))}</div>")

    parts.append("<h2>Models</h2>")
    parts.append(
        "<table><tr><th>Model</th><th>Size on Disk</th><th>Quantization</th>"
        "<th>Context Length</th><th>Engine / Backend / GPUs</th></tr>"
    )
    for model_name, model_result in run["models"].items():
        info = model_result.get("runtime_load_info") or {}
        size_bytes = info.get("size_bytes")
        size_str = f"{size_bytes / (1024**3):.2f} GB" if size_bytes else "unknown"
        quant = info.get("quantization")
        quant_str = quant.get("name") if isinstance(quant, dict) else "unknown"
        ctx_len = info.get("context_length", "unknown")
        parts.append(
            f"<tr><td>{esc(model_name)}</td><td>{size_str}</td><td>{esc(str(quant_str))}</td>"
            f"<td>{ctx_len}</td><td>{esc(_engine_summary(model_result, run.get('hardware', {})))}</td></tr>"
        )
    parts.append("</table>")

    hw = run.get("hardware", {})
    gpu = hw.get("gpu")
    gpu_str = (
        ", ".join(f"{g['name']} ({g.get('memory') or g.get('memory_approx', 'unknown')})" for g in gpu)
        if isinstance(gpu, list)
        else str(gpu)
    )
    parts.append("<h2>Hardware</h2>")
    parts.append(
        "<div class='meta'>"
        f"OS: {esc(hw.get('os', 'unknown'))} ({esc(hw.get('machine_arch', 'unknown'))})<br/>"
        f"CPU: {esc(hw.get('cpu', 'unknown'))} "
        f"({hw.get('cpu_count_physical', '?')}c/{hw.get('cpu_count_logical', '?')}t)<br/>"
        f"RAM: {hw.get('ram_total_gb', 'unknown')} GB<br/>"
        f"GPU: {esc(gpu_str)}"
        "</div>"
    )

    # Speed-only suite: own table, kept out of the accuracy loop (see _HW_NOTE).
    if "hardware_perf" in _suite_names(run):
        parts.append("<h2>hardware_perf &mdash; raw speed</h2>")
        parts.append(f"<div class='meta'>{esc(_HW_NOTE)}</div>")
        header = ["Model"] + [f"Prefill {lbl}" for _, lbl in _HW_PREFILL_TIERS] + [f"Decode {lbl}" for _, lbl in _HW_DECODE_TIERS] + ["TTFT (s)"]
        parts.append("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>")
        for model_name, cells in _hardware_perf_rows(run):
            parts.append(f"<tr><td>{esc(model_name)}</td>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>")
        parts.append("</table>")
        parts.append("<div class='meta'>Prefill and decode figures are tokens/sec; higher is better.</div>")

    for suite_name in [s for s in _suite_names(run) if s not in ("frontier_graded", "hardware_perf")]:
        parts.append(f"<h2>{esc(suite_name)}</h2>")
        parts.append(
            "<table><tr><th>Model</th><th>Pass Rate (95% CI)</th><th>Outcomes</th>"
            "<th>Avg Latency (s)</th><th>Avg TTFT (s)</th><th>Avg Tokens/sec</th>"
            "<th>RAM &Delta; (GB)</th><th>VRAM &Delta; (MB)</th><th>Peak CPU %</th></tr>"
        )
        failures = []
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get(suite_name)
            if suite is None:
                continue
            pass_str = _pass_with_ci(suite)
            resource = suite.get("resource_usage") or {}
            parts.append(
                f"<tr><td>{esc(model_name)}</td><td>{pass_str}</td><td>{esc(_outcome_counts(suite))}</td>"
                f"<td>{_fmt_num(suite['avg_latency_seconds'])}</td>"
                f"<td>{_fmt_num(suite.get('avg_ttft_seconds'))}</td>"
                f"<td>{_fmt_num(suite['avg_tokens_per_sec'])}</td>"
                f"<td>{_fmt_num(resource.get('ram_delta_gb'), 2)}</td>"
                f"<td>{_fmt_num(resource.get('vram_delta_mb'), 0)}</td>"
                f"<td>{_fmt_num(resource.get('peak_cpu_percent'), 1)}</td></tr>"
            )
            for problem in suite["problems"]:
                if not problem["passed"]:
                    failures.append((model_name, problem))
        parts.append("</table>")
        for model_name, problem in failures:
            parts.append(
                f"<div class='fail'>{esc(model_name)} / {esc(problem['problem_id'])}: "
                f"{esc(str(problem['error']))}</div>"
            )

    if "frontier_graded" in _suite_names(run):
        parts.append("<h2>Frontier-Graded (non-deterministic, paid)</h2>")
        parts.append(
            "<div class='meta'>Scored by a frontier-model judge per TASK_SPEC.md, "
            "never mixed into the deterministic pass-rate numbers above.</div>"
        )
        parts.append("<table><tr><th>Model</th><th>Avg Score (/10)</th><th>Pass Rate</th></tr>")
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get("frontier_graded")
            if suite is None:
                continue
            scores = [p["score"] for p in suite["problems"] if p.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else None
            pass_str = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
            parts.append(f"<tr><td>{esc(model_name)}</td><td>{_fmt_num(avg_score, 1)}</td><td>{pass_str}</td></tr>")
        parts.append("</table>")

    return "\n".join(parts)


def render_pdf(run: dict, output_path: str | Path) -> Path:
    """Render the report to a PDF file using xhtml2pdf (pure Python, no
    system/GTK dependencies -- import is deferred so it's only required when
    PDF export is actually used)."""
    from xhtml2pdf import pisa

    output_path = Path(output_path)
    html_doc = f"<html><body>{render_html(run)}</body></html>"
    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(html_doc, dest=f)
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s)")
    return output_path
