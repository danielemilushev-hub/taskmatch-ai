"""Render a RunRecord (see storage.py / results.py) into a markdown leaderboard,
or an HTML/PDF version of the same report for sharing outside a terminal.

Operates on plain dicts (as loaded from JSON) so it works identically whether
fed a freshly-built run or one loaded back from results/runs/*.json.
"""

from __future__ import annotations

import html
from pathlib import Path


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "n/a"


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
    lines = ["| Model | Size on Disk | Quantization | Context Length |", "|---|---|---|---|"]
    for model_name, model_result in run["models"].items():
        info = model_result.get("runtime_load_info") or {}
        size_bytes = info.get("size_bytes")
        size_str = f"{size_bytes / (1024**3):.2f} GB" if size_bytes else "unknown"
        quant = info.get("quantization")
        quant_str = quant.get("name") if isinstance(quant, dict) else "unknown"
        ctx_len = info.get("context_length", "unknown")
        lines.append(f"| {model_name} | {size_str} | {quant_str} | {ctx_len} |")
    return "\n".join(lines)


def render_markdown(run: dict) -> str:
    lines = [f"# localbench report — `{run['run_id']}`", ""]
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

    deterministic_suites = [s for s in _suite_names(run) if s != "frontier_graded"]

    for suite_name in deterministic_suites:
        lines.append(f"## {suite_name}")
        lines.append("")
        lines.append(
            "| Model | Pass Rate | Avg Latency (s) | Avg TTFT (s) | Avg Tokens/sec | "
            "VRAM used (GB) | VRAM Δ (MB) | GPU % | RAM Δ (GB) | Peak CPU % |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get(suite_name)
            if suite is None:
                continue
            pass_str = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
            resource = suite.get("resource_usage") or {}
            vram_total = resource.get("peak_vram_mb_total")
            lines.append(
                f"| {model_name} | {pass_str} | "
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
    lines = ["# localbench comparison", ""]
    lines.append("| Run | Started | Hardware | Model | Suite | Pass Rate | Avg Latency (s) | Tok/s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in runs:
        hw = run.get("hardware", {})
        gpu = hw.get("gpu")
        gpu_name = gpu[0]["name"] if isinstance(gpu, list) and gpu else "unknown"
        for model_name, model_result in run["models"].items():
            for suite_name, suite in model_result["suites"].items():
                pass_str = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
                lines.append(
                    f"| {run['run_id']} | {run.get('started_at', '?')} | {gpu_name} | "
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
    parts = [_PDF_STYLE, f"<h1>localbench report &mdash; {esc(run['run_id'])}</h1>"]
    parts.append(f"<div class='meta'>Started: {esc(str(run.get('started_at', 'unknown')))}</div>")

    parts.append("<h2>Models</h2>")
    parts.append("<table><tr><th>Model</th><th>Size on Disk</th><th>Quantization</th><th>Context Length</th></tr>")
    for model_name, model_result in run["models"].items():
        info = model_result.get("runtime_load_info") or {}
        size_bytes = info.get("size_bytes")
        size_str = f"{size_bytes / (1024**3):.2f} GB" if size_bytes else "unknown"
        quant = info.get("quantization")
        quant_str = quant.get("name") if isinstance(quant, dict) else "unknown"
        ctx_len = info.get("context_length", "unknown")
        parts.append(f"<tr><td>{esc(model_name)}</td><td>{size_str}</td><td>{esc(str(quant_str))}</td><td>{ctx_len}</td></tr>")
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

    for suite_name in [s for s in _suite_names(run) if s != "frontier_graded"]:
        parts.append(f"<h2>{esc(suite_name)}</h2>")
        parts.append(
            "<table><tr><th>Model</th><th>Pass Rate</th>"
            "<th>Avg Latency (s)</th><th>Avg TTFT (s)</th><th>Avg Tokens/sec</th>"
            "<th>RAM &Delta; (GB)</th><th>VRAM &Delta; (MB)</th><th>Peak CPU %</th></tr>"
        )
        failures = []
        for model_name, model_result in run["models"].items():
            suite = model_result["suites"].get(suite_name)
            if suite is None:
                continue
            pass_str = f"{_fmt_pct(suite['pass_rate'])} ({suite['pass_count']}/{suite['total']})"
            resource = suite.get("resource_usage") or {}
            parts.append(
                f"<tr><td>{esc(model_name)}</td><td>{pass_str}</td>"
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
