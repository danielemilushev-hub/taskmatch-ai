"""Multi-turn conversation benchmark suite.

Evaluates conversational memory, entity-attribute binding, arithmetic state
tracking, and constraint retention across multiple chat turns.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..data.multi_turn_problems import generate_multi_turn_problems
from ..engine import RunContext
from ..results import ProblemResult


def _evaluate_dialogue(
    eval_type: str,
    expected: str,
    responses: list[str],
) -> tuple[bool, str | None]:
    final_resp = responses[-1] if responses else ""

    if eval_type == "exact_contains":
        if expected.lower() in final_resp.lower():
            return True, None
        return (
            False,
            f"expected to find '{expected}' in response, but got: '{final_resp.strip()[:200]}'",
        )

    if eval_type == "starts_with":
        for i, resp in enumerate(responses):
            cleaned = resp.strip()
            if not cleaned.startswith(expected):
                return (
                    False,
                    f"turn {i + 1} violated persistent prefix rule: expected to start with '{expected}', got '{cleaned[:100]}'",
                )
        return True, None

    if eval_type == "forbidden_word":
        pattern = rf"\b{re.escape(expected)}\b"
        for i, resp in enumerate(responses):
            if re.search(pattern, resp, re.IGNORECASE):
                return (
                    False,
                    f"turn {i + 1} violated negative constraint: used forbidden word '{expected}'",
                )
        return True, None

    if eval_type == "numerical_exact":
        # Look for the expected number as a distinct integer
        pattern = rf"\b{re.escape(expected)}\b"
        if re.search(pattern, final_resp):
            return True, None
        return (
            False,
            f"expected final numerical state '{expected}', but got response: '{final_resp.strip()[:200]}'",
        )

    return False, f"unknown evaluation type '{eval_type}'"


def _run_one(problem: dict[str, Any], ctx: RunContext, config: dict) -> ProblemResult:
    messages: list[dict[str, str]] = []
    responses: list[str] = []
    total_latency = 0.0
    ttft_first: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    turns = problem["turns"]
    last_reasoning: str | None = None

    for i, user_prompt in enumerate(turns):
        messages.append({"role": "user", "content": user_prompt})
        chat = ctx.call(
            messages,
            timeout_seconds=config.get("call_timeout_seconds", 120),
            max_tokens=config.get("max_tokens", 1024),
        )

        total_latency += chat.latency_seconds
        if ttft_first is None and chat.ttft_seconds is not None:
            ttft_first = chat.ttft_seconds
        if chat.prompt_tokens:
            prompt_tokens += chat.prompt_tokens
        if chat.completion_tokens:
            completion_tokens += chat.completion_tokens
        last_reasoning = chat.reasoning_content

        if not chat.success:
            return ProblemResult(
                problem_id=problem["id"],
                passed=False,
                error=f"turn {i + 1} failed: {chat.error}",
                latency_seconds=total_latency,
                ttft_seconds=ttft_first,
                prompt="\n---\n".join(f"Turn {idx+1}: {t}" for idx, t in enumerate(turns[: i + 1])),
                response_content=chat.content,
                reasoning_content=chat.reasoning_content,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
            )

        resp_text = chat.content or ""
        responses.append(resp_text)
        messages.append({"role": "assistant", "content": resp_text})

    passed, error = _evaluate_dialogue(problem["eval_type"], problem["expected"], responses)

    full_dialogue_prompt = "\n---\n".join(
        f"User (Turn {idx+1}): {t}" for idx, t in enumerate(turns)
    )

    return ProblemResult(
        problem_id=problem["id"],
        passed=passed,
        error=error,
        latency_seconds=total_latency,
        ttft_seconds=ttft_first,
        prompt=full_dialogue_prompt,
        response_content=responses[-1] if responses else None,
        reasoning_content=last_reasoning,
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
    )


def run(
    ctx: RunContext,
    num_problems: int = 12,
    seed: int = 42,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
    config: dict | None = None,
) -> list[ProblemResult]:
    config = config or {}
    problems = generate_multi_turn_problems(num_problems=num_problems, seed=seed)
    results: list[ProblemResult] = []

    for i, problem in enumerate(problems):
        res = _run_one(problem, ctx, config)
        results.append(res)
        if on_progress:
            on_progress(i + 1, len(problems), problem["id"], res.passed, result=res)

    return results
