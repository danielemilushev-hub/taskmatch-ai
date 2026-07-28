"""Job suite: ARC-AGI-style abstract grid-transformation puzzles. Infer the
rule from a couple of examples, apply it to a new grid, graded by exact
match against the programmatically-computed expected grid.
"""

from __future__ import annotations

import re
from typing import Callable

from ..data.pattern_reasoning_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are solving abstract grid-transformation puzzles. Infer the single "
    "consistent rule from the examples, then apply it to the new input grid. "
    "Respond with ONLY the output grid: one row per line, values separated by "
    "single spaces, no extra text, no labels, no code fences, no blank lines."
)

_LINE_RE = re.compile(r"^[\d\s]+$")


def _grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _build_prompt(problem: dict) -> str:
    parts = []
    for idx, example in enumerate(problem["examples"], start=1):
        parts.append(
            f"Example {idx} input:\n{_grid_to_text(example['input'])}\n\n"
            f"Example {idx} output:\n{_grid_to_text(example['output'])}"
        )
    parts.append(f"New input:\n{_grid_to_text(problem['test_input'])}\n\nOutput:")
    return "\n\n".join(parts)


def _parse_grid(
    text: str | None, expected_rows: int, expected_cols: int
) -> tuple[list[list[int]] | None, str | None]:
    if not text:
        return None, "empty response"
    lines = [ln.strip() for ln in text.strip().splitlines()]
    digit_lines = [ln for ln in lines if ln and _LINE_RE.match(ln)]
    if len(digit_lines) < expected_rows:
        return None, f"found only {len(digit_lines)} numeric rows, expected {expected_rows}"

    candidate_lines = digit_lines[-expected_rows:]
    grid = []
    for ln in candidate_lines:
        values = ln.split()
        if len(values) != expected_cols:
            return None, f"row {ln!r} has {len(values)} values, expected {expected_cols}"
        try:
            grid.append([int(v) for v in values])
        except ValueError:
            return None, f"row {ln!r} contains a non-integer value"
    return grid, None


def _run_one(problem: dict, ctx: RunContext, call_kwargs: dict) -> ProblemResult:
    prompt_val = _build_prompt(problem)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_val},
    ]

    chat = ctx.call(messages, **call_kwargs)

    if not chat.success:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"call failed: {chat.error}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=prompt_val,
        )

    if chat.loop_detected:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=(
                f"generation aborted after ~{chat.completion_tokens} tokens: "
                "detected a repetition loop (the model was re-deriving the same "
                "content rather than converging) -- stopped early instead of "
                "waiting for max_tokens"
            ),
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            loop_detected=True,
        )

    if chat.truncated:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error="response truncated at max_tokens before completion (finish_reason=length)",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            truncated=True,
        )

    expected = problem["expected_output"]
    expected_rows, expected_cols = len(expected), len(expected[0])
    grid, parse_error = _parse_grid(chat.content, expected_rows, expected_cols)

    if grid is None:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"could not parse output grid: {parse_error}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
        )

    passed = grid == expected
    return ProblemResult(
        problem_id=problem["id"],
        passed=passed,
        error=None if passed else f"expected grid {expected}, got {grid}",
        latency_seconds=chat.latency_seconds,
        ttft_seconds=chat.ttft_seconds,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        prompt=prompt_val,
        response_content=chat.content,
        reasoning_content=chat.reasoning_content,
    )


def run(
    ctx: RunContext,
    num_problems: int = 10,
    seed: int = 42,
    problems: list[dict] | None = None,
    max_tokens: int | None = None,
    call_timeout_seconds: float | None = None,
    detect_loops: bool = False,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
) -> list[ProblemResult]:
    problems = problems if problems is not None else generate_problems(num_problems, seed)
    results: list[ProblemResult] = []
    call_kwargs = {}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if call_timeout_seconds is not None:
        call_kwargs["timeout_seconds"] = call_timeout_seconds
    if detect_loops:
        call_kwargs["detect_loops"] = True

    for idx, problem in enumerate(problems):
        result = _run_one(problem, ctx, call_kwargs)
        results.append(result)
        if on_progress:
            on_progress(idx + 1, len(problems), result.problem_id, result.passed, result=result)

    return results
