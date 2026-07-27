"""Job suite: give the model a large (1000+ line) source excerpt and ask it
to retrieve a planted marker or locate a mechanically-injected bug. Graded by
exact match -- see data/long_context_problems.py for how ground truth is
constructed (we control the needle/mutation, so it's always known exactly).
"""

from __future__ import annotations

import re

from ..data.long_context_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are a careful code reviewer. Read the entire numbered source excerpt "
    "provided, then answer precisely. Follow the exact answer-format "
    "instructions given in the question."
)

_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _extract_answer_line(content: str | None) -> str | None:
    if not content:
        return None
    matches = _ANSWER_LINE_RE.findall(content)
    return matches[-1].strip() if matches else None


def _build_prompt(problem: dict) -> str:
    if problem["task_type"] == "needle_retrieval":
        instruction = (
            "Somewhere in the numbered source excerpt below is a comment line "
            "containing the text 'LOCALBENCH_NEEDLE: verification_code_is_<CODE>'. "
            "Find that exact line and report ONLY the code that follows "
            "'verification_code_is_'.\n\nEnd your response with exactly:\nAnswer: <the code>"
        )
    else:
        instruction = (
            "The numbered source excerpt below has been slightly altered: exactly "
            "one line contains a comparison operator that was changed from the "
            "original, introducing a subtle logic bug (e.g. == became !=, or <= "
            "became <). Find the exact line NUMBER (shown at the start of that "
            "line) where this altered comparison occurs.\n\n"
            "End your response with exactly:\nAnswer: <line number>"
        )
    return (
        f"{instruction}\n\n"
        f"----- SOURCE EXCERPT ({problem['num_lines']} lines) -----\n"
        f"{problem['code_text']}\n"
        f"----- END EXCERPT -----"
    )


def _grade(problem: dict, raw_answer: str | None) -> tuple[bool, str | None]:
    if raw_answer is None:
        return False, "no 'Answer: ...' line found in model output"

    if problem["task_type"] == "needle_retrieval":
        # tolerate the model wrapping the code in stray punctuation/quotes
        cleaned = raw_answer.strip().strip("\"'.")
        if cleaned == str(problem["expected"]):
            return True, None
        return False, f"expected code {problem['expected']!r}, got {cleaned!r}"

    match = _INT_RE.search(raw_answer)
    if not match:
        return False, f"could not parse a line number from answer line: {raw_answer!r}"
    got = int(match.group(0))
    if got == problem["expected"]:
        return True, None
    return False, f"expected line {problem['expected']}, got line {got}"


def run(
    ctx: RunContext,
    num_problems: int = 4,
    seed: int = 42,
    source_file: str | None = None,
    window_lines: int = 1000,
    timeout_seconds: float = 180,
    problems: list[dict] | None = None,
    max_tokens: int | None = None,
) -> list[ProblemResult]:
    problems = problems if problems is not None else generate_problems(
        num_problems, seed, source_file=source_file, window_lines=window_lines
    )
    results: list[ProblemResult] = []
    call_kwargs = {"timeout_seconds": timeout_seconds}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    for problem in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(problem)},
        ]

        chat = ctx.call(messages, **call_kwargs)

        prompt_val = _build_prompt(problem)
        if not chat.success:
            results.append(
                ProblemResult(
                    problem_id=problem["id"],
                    passed=False,
                    error=f"call failed: {chat.error}",
                    latency_seconds=chat.latency_seconds,
                    ttft_seconds=chat.ttft_seconds,
                    prompt=prompt_val,
                )
            )
            continue

        if chat.truncated:
            results.append(
                ProblemResult(
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
            )
            continue

        raw_answer = _extract_answer_line(chat.content)
        passed, error = _grade(problem, raw_answer)

        results.append(
            ProblemResult(
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
        )

    return results
