"""Job suite: HumanEval-style coding problems, executed in a sandboxed subprocess.

"Sandboxed" here means: the model's generated code runs in its own throwaway
subprocess with a hard wall-clock timeout and no shared state with this
process -- not full OS-level isolation (no seccomp/cgroups/chroot). That's a
deliberate, documented limitation on Windows; see README.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ..data.coding_problems import PROBLEMS
from ..data.generated_coding_problems import generate_problems
from ..engine import RunContext
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are a Python code generation engine. Respond with ONLY a single Python "
    "code block implementing exactly the requested function. Do not include "
    "example usage, tests, explanations, or any text outside the code block."
)

_RESULT_MARKER = "###LOCALBENCH_RESULTS###"

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_RUNNER_TEMPLATE = '''\
import json

{code}

_tests = {tests_repr}
_results = []
for _t in _tests:
    try:
        _actual = {entry_point}(*_t["args"])
        _results.append({{"ok": _actual == _t["expected"], "actual": repr(_actual)}})
    except Exception as _e:
        _results.append({{"ok": False, "error": f"{{type(_e).__name__}}: {{_e}}"}})

print("{marker}")
print(json.dumps(_results))
'''


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    if "def " in text:
        return text
    return None


def _build_test_error(problem: dict, test_results: list[dict]) -> str:
    lines = []
    for test, result in zip(problem["tests"], test_results):
        if result.get("ok"):
            continue
        if "error" in result:
            lines.append(f"args={test['args']!r} raised {result['error']}")
        else:
            lines.append(
                f"args={test['args']!r} expected {test['expected']!r}, got {result['actual']}"
            )
    return "; ".join(lines)


def _run_one(problem: dict, ctx: RunContext, call_kwargs: dict, timeout_seconds: float) -> ProblemResult:
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

    code = extract_code(chat.content)
    if code is None:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error="no Python code found in model output",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
        )

    script = _RUNNER_TEMPLATE.format(
        code=code,
        tests_repr=repr(problem["tests"]),
        entry_point=problem["entry_point"],
        marker=_RESULT_MARKER,
    )

    with tempfile.TemporaryDirectory(prefix="localbench_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return ProblemResult(
                problem_id=problem["id"],
                passed=False,
                error=f"execution timed out after {timeout_seconds}s",
                latency_seconds=chat.latency_seconds,
                ttft_seconds=chat.ttft_seconds,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
                prompt=prompt_val,
                response_content=chat.content,
                reasoning_content=chat.reasoning_content,
            )

    if _RESULT_MARKER not in proc.stdout:
        error = f"sandbox process crashed before producing results: {proc.stderr[-500:]}"
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=error,
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
        )

    results_json = proc.stdout.split(_RESULT_MARKER, 1)[1].strip()
    try:
        test_results = json.loads(results_json)
    except json.JSONDecodeError:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"could not parse sandbox output: {results_json[:300]}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            prompt=prompt_val,
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
        )

    all_pass = all(r.get("ok") for r in test_results)
    return ProblemResult(
        problem_id=problem["id"],
        passed=all_pass,
        error=None if all_pass else _build_test_error(problem, test_results),
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
    problems: list[dict] | None = None,
    timeout_seconds: float = 10,
    max_tokens: int | None = None,
    call_timeout_seconds: float | None = None,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
    generated: bool = True,
    num_problems: int = 12,
    seed: int = 42,
) -> list[ProblemResult]:
    # Generated problems by default. The fixed set (factorial, fibonacci,
    # is_prime, ...) is in every training corpus, so passing it can be pure
    # recall -- it measures memorisation, not coding. Set `generated: false`
    # in config to fall back to the classic set for comparison.
    if problems is None:
        problems = generate_problems(num_problems, seed) if generated else PROBLEMS
    results: list[ProblemResult] = []
    call_kwargs = {}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if call_timeout_seconds is not None:
        call_kwargs["timeout_seconds"] = call_timeout_seconds

    for idx, problem in enumerate(problems):
        result = _run_one(problem, ctx, call_kwargs, timeout_seconds)
        results.append(result)
        if on_progress:
            on_progress(idx + 1, len(problems), result.problem_id, result.passed)

    return results
