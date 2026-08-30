"""Job suite: raw hardware/model throughput -- prefill speed and sustained
decode speed -- rather than correctness. Runs first, before the accuracy
suites, so every model's subsequent numbers can be read against a known
speed baseline for this specific hardware.

Deliberately excluded from pass-rate aggregates elsewhere in the app
(verdict banner, radar chart, mean pass rate -- see app.js's suite-set
filtering, same treatment as frontier_graded): "passed" here means the
call completed at all, not that an answer was correct. Mixing a suite
where nearly everything trivially "passes" into an accuracy mean would
inflate it without saying anything about accuracy.
"""

from __future__ import annotations

from typing import Callable

from ..data.hardware_perf_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = "You are a helpful assistant."


def _run_one(problem: dict, ctx: RunContext, timeout_seconds: float) -> ProblemResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]
    chat = ctx.call(
        messages, max_tokens=problem["max_tokens"], timeout_seconds=timeout_seconds
    )

    prompt_preview = f"[{problem['task_type']} probe, {len(problem['prompt'])} chars] " + problem["prompt"][:200]

    if not chat.success:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"call failed: {chat.error}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=prompt_preview,
        )

    # This suite measures speed, not correctness -- a passing result just
    # means the hardware/model completed the workload and produced real
    # timing data. A truncated or looping response is still a valid
    # measurement of how fast (or not) that generation ran.
    passed = chat.ttft_seconds is not None and bool(chat.completion_tokens)
    return ProblemResult(
        problem_id=problem["id"],
        passed=passed,
        error=None if passed else "no timing data captured (empty response)",
        latency_seconds=chat.latency_seconds,
        ttft_seconds=chat.ttft_seconds,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        prompt=prompt_preview,
        response_content=chat.content,
        # A prefill probe sets max_tokens to 8 on purpose -- stopping at the
        # limit is the intended outcome, not a fault, and it happens on every
        # single one. Recording that as `truncated` carried no information and
        # made a fully passing suite display a TRUNCATED flag on every row.
        # Decode probes keep the real value, where hitting the cap is
        # genuinely meaningful.
        truncated=False if problem.get("task_type") == "prefill" else chat.truncated,
    )


def run(
    ctx: RunContext,
    seed: int = 42,
    call_timeout_seconds: float = 300,
    problems: list[dict] | None = None,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
    max_context_tokens: int | None = None,
) -> list[ProblemResult]:
    # Fixed set regardless of Quick/Full profile -- see the module docstring
    # in data/hardware_perf_problems.py for why: this characterizes
    # hardware, it isn't a statistically-sampled accuracy measurement.
    # max_context_tokens only drops prefill tiers that cannot fit the model's
    # context window, so a small-context model runs fewer probes rather than
    # failing the largest ones.
    problems = problems if problems is not None else generate_problems(seed, max_context_tokens)
    results: list[ProblemResult] = []

    for idx, problem in enumerate(problems):
        result = _run_one(problem, ctx, call_timeout_seconds)
        results.append(result)
        if on_progress:
            on_progress(idx + 1, len(problems), result.problem_id, result.passed, result=result)

    return results
