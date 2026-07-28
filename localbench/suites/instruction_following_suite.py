"""Job suite: IFEval-style instruction following, graded by mechanical
rule-checking (counting/regex) on the raw response -- never an LLM judge.
This measures whether the model follows a literal formatting instruction,
not whether the content is good.
"""

from __future__ import annotations

import re
from typing import Callable

from ..data.instruction_following_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are a precise writing assistant. Follow formatting instructions exactly, "
    "including any constraints on length, wording, capitalization, or punctuation."
)


def _check_paragraph_count(content: str, params: dict) -> tuple[bool, str | None]:
    paragraphs = [p for p in re.split(r"\n\s*\n", content.strip()) if p.strip()]
    n = params["n"]
    if len(paragraphs) == n:
        return True, None
    return False, f"expected {n} paragraphs, got {len(paragraphs)}"


def _check_forbidden_letter(content: str, params: dict) -> tuple[bool, str | None]:
    letter = params["letter"]
    if letter.lower() not in content.lower():
        return True, None
    return False, f"response contains forbidden letter '{letter}'"


def _check_ending_phrase(content: str, params: dict) -> tuple[bool, str | None]:
    phrase = params["phrase"]
    if content.strip().endswith(phrase):
        return True, None
    return False, f"response does not end with the exact required phrase: {phrase!r}"


def _check_word_count_max(content: str, params: dict) -> tuple[bool, str | None]:
    n = params["n"]
    count = len(content.split())
    if count <= n:
        return True, None
    return False, f"expected at most {n} words, got {count}"


def _check_keyword_count(content: str, params: dict) -> tuple[bool, str | None]:
    word, n = params["word"], params["n"]
    count = len(re.findall(rf"\b{re.escape(word)}\b", content, re.IGNORECASE))
    if count >= n:
        return True, None
    return False, f"expected word '{word}' at least {n} times, found {count}"


def _check_all_lowercase(content: str, params: dict) -> tuple[bool, str | None]:
    if content == content.lower():
        return True, None
    return False, "response contains uppercase letters"


def _check_no_commas(content: str, params: dict) -> tuple[bool, str | None]:
    if "," not in content:
        return True, None
    return False, "response contains a comma"


def _check_starts_with(content: str, params: dict) -> tuple[bool, str | None]:
    word = params["word"]
    if content.strip().lower().startswith(word.lower()):
        return True, None
    return False, f"response does not start with '{word}'"


_CHECKERS = {
    "paragraph_count": _check_paragraph_count,
    "forbidden_letter": _check_forbidden_letter,
    "ending_phrase": _check_ending_phrase,
    "word_count_max": _check_word_count_max,
    "keyword_count": _check_keyword_count,
    "all_lowercase": _check_all_lowercase,
    "no_commas": _check_no_commas,
    "starts_with": _check_starts_with,
}


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

    checker = _CHECKERS[problem["constraint_type"]]
    passed, error = checker(chat.content or "", problem["params"])

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
    num_problems: int = 16,
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
