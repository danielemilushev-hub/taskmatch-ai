"""Job suite: prompt the model to produce JSON matching a schema, validate it."""

from __future__ import annotations

import json

import jsonschema

from ..data.json_schema_problems import PROBLEMS
from ..engine import RunContext
from ..json_extract import extract_json
from ..results import ProblemResult

SYSTEM_PROMPT = (
    "You are a JSON generation engine. Respond with ONLY a single valid JSON "
    "value that satisfies the user's request. Do not include markdown code "
    "fences, explanations, or any text other than the JSON itself."
)


def run(
    ctx: RunContext,
    problems: list[dict] | None = None,
    max_tokens: int | None = None,
    call_timeout_seconds: float | None = None,
) -> list[ProblemResult]:
    problems = problems if problems is not None else PROBLEMS
    results: list[ProblemResult] = []
    call_kwargs = {}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if call_timeout_seconds is not None:
        call_kwargs["timeout_seconds"] = call_timeout_seconds

    for problem in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{problem['task']}\n\n"
                    f"The JSON must validate against this JSON Schema:\n"
                    f"{json.dumps(problem['schema'], indent=2)}"
                ),
            },
        ]

        chat = ctx.call(messages, **call_kwargs)

        prompt_val = messages[1]["content"] if len(messages) > 1 else str(messages)
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

        value, extract_error = extract_json(chat.content)
        if extract_error:
            results.append(
                ProblemResult(
                    problem_id=problem["id"],
                    passed=False,
                    error=extract_error,
                    latency_seconds=chat.latency_seconds,
                    ttft_seconds=chat.ttft_seconds,
                    prompt_tokens=chat.prompt_tokens,
                    completion_tokens=chat.completion_tokens,
                    prompt=prompt_val,
                    response_content=chat.content,
                    reasoning_content=chat.reasoning_content,
                )
            )
            continue

        try:
            jsonschema.validate(instance=value, schema=problem["schema"])
            results.append(
                ProblemResult(
                    problem_id=problem["id"],
                    passed=True,
                    latency_seconds=chat.latency_seconds,
                    ttft_seconds=chat.ttft_seconds,
                    prompt_tokens=chat.prompt_tokens,
                    completion_tokens=chat.completion_tokens,
                    prompt=prompt_val,
                    response_content=chat.content,
                    reasoning_content=chat.reasoning_content,
                )
            )
        except jsonschema.ValidationError as e:
            results.append(
                ProblemResult(
                    problem_id=problem["id"],
                    passed=False,
                    error=f"schema validation failed at {list(e.absolute_path)}: {e.message}",
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
