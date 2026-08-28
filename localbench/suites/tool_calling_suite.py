"""Tool / Function Calling benchmark suite.

Evaluates whether local models correctly select and invoke tools via the
OpenAI-compatible `tools` schema, format valid JSON arguments adhering to the
parameter specification, and refrain from invoking tools when standard text
answers are requested (negative control).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import jsonschema

from ..data.tool_calling_problems import generate_tool_calling_problems
from ..engine import RunContext
from ..json_extract import extract_json
from ..results import ProblemResult


def _extract_tool_call(chat_result: Any) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Extract (tool_name, arguments_dict, error) from a ChatResult.

    Supports:
    1. Native `chat.tool_calls` (OpenAI format streamed/returned)
    2. Fallback text parsing: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
    3. Fallback text parsing: ```json {"name": ..., "arguments": ...} ```
    """
    # 1. Native tool_calls
    if chat_result.tool_calls:
        tc = chat_result.tool_calls[0]
        fn = tc.get("function") or {}
        name = fn.get("name")
        args_raw = fn.get("arguments", "")
        if isinstance(args_raw, dict):
            return name, args_raw, None
        try:
            args = json.loads(args_raw) if args_raw else {}
            return name, args, None
        except Exception as e:
            return name, None, f"malformed tool call arguments JSON: {e}"

    # 2. Text fallback
    text = (chat_result.content or "").strip()
    if not text:
        return None, None, None

    # Check for <tool_call> tags
    xml_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        parsed, err = extract_json(xml_match.group(1))
        if err is None and isinstance(parsed, dict):
            name = parsed.get("name") or parsed.get("function")
            args = parsed.get("arguments") or parsed.get("parameters") or parsed
            if isinstance(args, str):
                args, _ = extract_json(args)
            if isinstance(args, dict):
                return name, args, None

    # Check for JSON object in text
    parsed, err = extract_json(text)
    if err is None and isinstance(parsed, dict):
        if "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            name = parsed.get("name")
            args = parsed.get("arguments") or parsed.get("parameters")
            if isinstance(args, str):
                args, _ = extract_json(args)
            if isinstance(args, dict):
                return name, args, None
        if "name" in parsed and "location" in parsed or "query" in parsed:
            # Flat tool call object
            name = parsed.pop("name", None)
            return name, parsed, None

    return None, None, None


def _validate_args(
    called_args: dict[str, Any],
    expected_args: dict[str, Any],
    validator_type: str,
) -> tuple[bool, str | None]:
    for key, expected_val in expected_args.items():
        if key not in called_args:
            return False, f"missing required argument '{key}'"
        actual_val = called_args[key]
        if isinstance(expected_val, str):
            if str(actual_val).strip().lower() != expected_val.strip().lower():
                return (
                    False,
                    f"argument '{key}' value mismatch: expected '{expected_val}', got '{actual_val}'",
                )
        elif isinstance(expected_val, (int, float)):
            try:
                num_actual = float(actual_val)
                if abs(num_actual - float(expected_val)) > 0.01:
                    return (
                        False,
                        f"argument '{key}' numerical mismatch: expected {expected_val}, got {actual_val}",
                    )
            except (ValueError, TypeError):
                return False, f"argument '{key}' could not be parsed as number: {actual_val}"
    return True, None


def _run_one(problem: dict[str, Any], ctx: RunContext, config: dict) -> ProblemResult:
    messages = [{"role": "user", "content": problem["prompt"]}]
    tools = problem["tools"]
    extra_params = {"tools": tools}

    chat = ctx.call(
        messages,
        extra_params=extra_params,
        timeout_seconds=config.get("call_timeout_seconds", 120),
        max_tokens=config.get("max_tokens", 1024),
    )

    if not chat.success:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"call failed: {chat.error}",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    called_name, called_args, extract_err = _extract_tool_call(chat)

    # Negative control: Model should NOT call any tool
    if problem["is_negative"]:
        if called_name is not None:
            return ProblemResult(
                problem_id=problem["id"],
                passed=False,
                error=f"negative control failed: model hallucinated tool call '{called_name}' for general knowledge question",
                latency_seconds=chat.latency_seconds,
                ttft_seconds=chat.ttft_seconds,
                prompt=problem["prompt"],
                response_content=chat.content,
                reasoning_content=chat.reasoning_content,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )
        # Passed negative control
        return ProblemResult(
            problem_id=problem["id"],
            passed=True,
            error=None,
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    # Positive tool call
    if extract_err is not None:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=extract_err,
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    if not called_name:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"expected tool call '{problem['expected_tool']}', but model returned text without invoking a tool",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    if called_name != problem["expected_tool"]:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=f"wrong tool selected: expected '{problem['expected_tool']}', got '{called_name}'",
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    # Schema validation against tool parameters
    target_tool_def = next(
        (t for t in tools if t["function"]["name"] == called_name), None
    )
    if target_tool_def and "parameters" in target_tool_def["function"]:
        param_schema = target_tool_def["function"]["parameters"]
        try:
            jsonschema.validate(instance=called_args or {}, schema=param_schema)
        except jsonschema.ValidationError as ve:
            path = ".".join(str(p) for p in ve.path) or "root"
            return ProblemResult(
                problem_id=problem["id"],
                passed=False,
                error=f"tool arguments schema validation failed at '{path}': {ve.message}",
                latency_seconds=chat.latency_seconds,
                ttft_seconds=chat.ttft_seconds,
                prompt=problem["prompt"],
                response_content=chat.content,
                reasoning_content=chat.reasoning_content,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )

    # Validate argument semantics & extracted entities
    args_ok, val_err = _validate_args(
        called_args or {}, problem["expected_args"], problem["validator_type"]
    )
    if not args_ok:
        return ProblemResult(
            problem_id=problem["id"],
            passed=False,
            error=val_err,
            latency_seconds=chat.latency_seconds,
            ttft_seconds=chat.ttft_seconds,
            prompt=problem["prompt"],
            response_content=chat.content,
            reasoning_content=chat.reasoning_content,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    return ProblemResult(
        problem_id=problem["id"],
        passed=True,
        error=None,
        latency_seconds=chat.latency_seconds,
        ttft_seconds=chat.ttft_seconds,
        prompt=problem["prompt"],
        response_content=chat.content,
        reasoning_content=chat.reasoning_content,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
    )


def run(
    ctx: RunContext,
    num_problems: int = 16,
    seed: int = 42,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
    config: dict | None = None,
) -> list[ProblemResult]:
    config = config or {}
    problems = generate_tool_calling_problems(num_problems=num_problems, seed=seed)
    results: list[ProblemResult] = []

    for i, problem in enumerate(problems):
        res = _run_one(problem, ctx, config)
        results.append(res)
        if on_progress:
            on_progress(i + 1, len(problems), problem["id"], res.passed, result=res)

    return results
