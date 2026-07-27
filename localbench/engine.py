"""Runtime-agnostic caller for OpenAI-compatible /v1/chat/completions endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class ChatResult:
    success: bool
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    ttft_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    requested_max_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True if the response was cut off at the token budget.

        finish_reason == "length" is the documented signal, but it is not
        reliable on its own: measured across real runs, 21% of
        pattern_reasoning responses ended at exactly max_tokens while only
        5.7% carried that finish_reason -- the rest were silently clipped and
        would have been scored as ordinary wrong answers. Landing exactly on
        the ceiling is not a coincidence a model produces 11 times, so the
        token count is treated as corroborating evidence.
        """
        if self.finish_reason == "length":
            return True
        if (
            self.requested_max_tokens
            and self.completion_tokens
            and self.completion_tokens >= self.requested_max_tokens
        ):
            return True
        return False

    @property
    def tokens_per_sec(self) -> float | None:
        if self.completion_tokens is None or self.latency_seconds <= 0:
            return None
        return self.completion_tokens / self.latency_seconds


@dataclass
class RunContext:
    """Bundles connection + sampling config so suites don't juggle loose args."""

    base_url: str
    model: str
    api_key: str = "not-needed"
    timeout_seconds: float = 120
    temperature: float = 0.2
    max_tokens: int = 1024

    def call(self, messages: list[dict[str, str]], **overrides: Any) -> "ChatResult":
        kwargs = dict(
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        kwargs.update(overrides)
        return chat_completion(self.base_url, self.model, messages, **kwargs)


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "not-needed",
    timeout_seconds: float = 120,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    extra_params: dict[str, Any] | None = None,
) -> ChatResult:
    """POST to {base_url}/chat/completions (streamed) and return a ChatResult.

    Streams the response so time-to-first-token (ttft_seconds) can be
    measured separately from total latency -- prefill time on a long prompt
    and decode speed are different things, and conflating them into one
    latency number hides which one a slow model is actually slow at.

    Never raises: network errors, timeouts, non-2xx responses, and malformed
    JSON are all captured in ChatResult.error so callers (job suites) can
    grade a failed call the same way they grade a bad answer.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_params:
        payload.update(extra_params)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            url, json=payload, headers=headers, timeout=timeout_seconds, stream=True
        )
    except requests.exceptions.Timeout:
        return ChatResult(
            success=False,
            error=f"request timed out after {timeout_seconds}s",
            latency_seconds=time.perf_counter() - start,
        )
    except requests.exceptions.ConnectionError as e:
        return ChatResult(
            success=False,
            error=f"connection failed (is the server running at {base_url}?): {e}",
            latency_seconds=time.perf_counter() - start,
        )
    except requests.exceptions.RequestException as e:
        return ChatResult(
            success=False,
            error=f"request failed: {e}",
            latency_seconds=time.perf_counter() - start,
        )

    if resp.status_code != 200:
        return ChatResult(
            success=False,
            error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            latency_seconds=time.perf_counter() - start,
        )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage: dict = {}
    ttft: float | None = None
    got_any_chunk = False

    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            got_any_chunk = True
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                piece_content = delta.get("content")
                piece_reasoning = delta.get("reasoning_content")
                if ttft is None and (piece_content or piece_reasoning):
                    ttft = time.perf_counter() - start
                if piece_content:
                    content_parts.append(piece_content)
                if piece_reasoning:
                    reasoning_parts.append(piece_reasoning)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
    except requests.exceptions.RequestException as e:
        return ChatResult(
            success=False,
            error=f"stream read failed: {e}",
            latency_seconds=time.perf_counter() - start,
        )

    latency = time.perf_counter() - start

    if not got_any_chunk:
        return ChatResult(
            success=False,
            error="no data received from streamed response",
            latency_seconds=latency,
        )

    return ChatResult(
        success=True,
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts) or None,
        finish_reason=finish_reason,
        latency_seconds=latency,
        ttft_seconds=ttft,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        requested_max_tokens=max_tokens,
        raw={"finish_reason": finish_reason, "usage": usage},
    )
