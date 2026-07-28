"""Job suite: synthetically generated logic/math problems, graded by exact match."""

from __future__ import annotations

import re
from typing import Callable

from ..data.logic_math_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Follow the exact answer-format "
    "instructions given in each question."
)

_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _extract_answer_line(content: str | None) -> str | None:
    if not content:
        return None
    matches = _ANSWER_LINE_RE.findall(content)
    if not matches:
        return None
    return matches[-1].strip()


def _grade(answer_type: str, expected, raw_answer: str | None) -> tuple[bool, str | None]:
    if raw_answer is None:
        return False, "no 'Answer: ...' line found in model output"

    if answer_type == "int":
        match = _INT_RE.search(raw_answer)
        if not match:
            return False, f"could not parse an integer from answer line: {raw_answer!r}"
        got = int(match.group(0))
        if got == expected:
            return True, None
        return False, f"expected {expected}, got {got}"

    if answer_type == "yes_no":
        normalized = raw_answer.strip().lower()
        if normalized.startswith("yes"):
            got = "yes"
        elif normalized.startswith("no"):
            got = "no"
        else:
            return False, f"could not parse yes/no from answer line: {raw_answer!r}"
        if got == expected:
            return True, None
        return False, f"expected {expected}, got {got}"

    if answer_type == "bool":
        normalized = raw_answer.strip().lower()
        if normalized.startswith("true"):
            got = "True"
        elif normalized.startswith("false"):
            got = "False"
        else:
            return False, f"could not parse True/False from answer line: {raw_answer!r}"
        if got == expected:
            return True, None
        return False, f"expected {expected}, got {got}"

    raise ValueError(f"unknown answer_type: {answer_type}")


def _run_one(problem: dict, ctx: RunContext, call_kwargs: dict) -> ProblemResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]

    chat = ctx.call(messages, **call_kwargs)
    prompt_val = problem["prompt"]

    if not chat.success:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"call failed: {chat.error}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=prompt_val,
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

    raw_answer = _extract_answer_line(chat.content)
    passed, error = _grade(problem["answer_type"], problem["expected"], raw_answer)

    return ProblemResult(
        problem_id=problem["id"],
        passed=passed,
        error=error,
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
    num_problems: int = 20,
    seed: int = 42,
    problems: list[dict] | None = None,
    max_tokens: int | None = None,
    call_timeout_seconds: float | None = None,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
) -> list[ProblemResult]:
    problems = problems if problems is not None else generate_problems(num_problems, seed)
    results: list[ProblemResult] = []
    call_kwargs = {}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if call_timeout_seconds is not None:
        call_kwargs["timeout_seconds"] = call_timeout_seconds

    for idx, problem in enumerate(problems):
        result = _run_one(problem, ctx, call_kwargs)
        results.append(result)
        if on_progress:
            on_progress(idx + 1, len(problems), result.problem_id, result.passed, result=result)

    return results
