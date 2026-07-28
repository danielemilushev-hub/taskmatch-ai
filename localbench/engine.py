"""Runtime-agnostic caller for OpenAI-compatible /v1/chat/completions endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests


def _has_repetition(text: str, window_chars: int, phrase_len: int, min_repeats: int) -> bool:
    """True if some exact substring of length `phrase_len` recurs at least
    `min_repeats` times within the trailing `window_chars` of `text`.

    Only the most recent window is checked (not the whole response so far)
    so a loop is caught shortly after it starts, not diluted by everything
    generated before it. The exact-match requirement is deliberate: ordinary
    wordy-but-non-looping prose essentially never repeats a 40+ character
    span verbatim, but a model re-deriving the same example grid on every
    "wait, let me re-check" pass does -- that's the actual pattern observed
    in a live gemma-4-12b-qat transcript that this was tuned against.
    """
    window = text[-window_chars:]
    if len(window) < phrase_len * min_repeats:
        return False
    counts: dict[str, int] = {}
    for i in range(len(window) - phrase_len + 1):
        phrase = window[i : i + phrase_len]
        count = counts.get(phrase, 0) + 1
        if count >= min_repeats:
            return True
        counts[phrase] = count
    return False


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
    # True if generation was proactively aborted mid-stream because it looked
    # like a repetition loop, rather than run to completion or to max_tokens.
    # Distinct from `truncated`: truncated means the token budget ran out;
    # loop_detected means we chose to stop early, before that budget was hit,
    # because a live diagnostic against gemma-4-12b-qat showed some
    # pattern_reasoning problems never converge -- 20,000 completion tokens
    # (2.4x the suite's normal cap, well inside the model's context window)
    # still ended in finish_reason=length with zero characters of actual
    # answer content, all of it spent re-deriving the same examples on
    # repeat. No token budget fixes that; only recognizing the loop does.
    loop_detected: bool = False
    # True if generation was proactively stopped because a caller-supplied
    # `early_exit_check` found an already-valid, already-graded-correct
    # answer in the stream. A live gemma transcript showed the model finding
    # correct code in the first few hundred tokens, then never stopping --
    # re-verifying it against new self-invented test cases indefinitely.
    # That answer was real; grading the response as a failure because the
    # model wouldn't stop talking would be inaccurate. See coding_suite.py's
    # early-exit checker for how the "already correct" judgment is made.
    early_exit: bool = False
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
    detect_loops: bool = False,
    # Tuned against live results, not guessed: a looser first pass (4000/40/3
    # -- match anywhere in a wide window) correctly caught genuine stalls but
    # also flagged legitimate, verbose multi-hypothesis reasoning that
    # re-quotes the same example grid at widely separated points while
    # working through different theories. Requiring the repeat to be DENSE
    # (close together, in a narrow trailing window) rather than merely
    # present anywhere in a long response eliminated that false positive in
    # a live A/B (same problem, same model) while still catching two
    # confirmed genuine non-convergent generations.
    loop_window_chars: int = 1200,
    loop_phrase_len: int = 50,
    loop_min_repeats: int = 4,
    loop_min_chars: int = 1500,
    # A second, independent check for a different loop *shape*: a live
    # bug_locator transcript repeated the exact same ~21-char condition
    # ("if a >= 0 and b >= 0:") 14+ times, but wrapped in a line number that
    # changed every time -- no 50-char exact span ever recurred, so the
    # check above completely missed it. A short phrase alone would be too
    # promiscuous (ordinary prose reuses short fragments by chance), so the
    # min-repeat count is raised instead of the phrase length -- six-plus
    # verbatim repeats of even a short, specific span is not a coincidence.
    loop_short_phrase_len: int = 18,
    loop_short_min_repeats: int = 6,
    # Suite-supplied, suite-graded check: given the combined reasoning+content
    # text so far, return True if it already contains a correct, verified
    # answer. Unlike loop detection (a fuzzy text-repetition heuristic), this
    # is grounded in the suite's own real grading logic (e.g. actually
    # running candidate code against test cases) -- so it's safe to trust
    # without the false-positive risk that came with repetition matching.
    early_exit_check: Callable[[str], bool] | None = None,
    early_exit_check_interval_chars: int = 150,
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
    token_estimate = 0
    loop_detected = False
    early_exit = False
    chars_at_last_loop_check = 0
    chars_at_last_early_exit_check = 0

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
                    token_estimate += 1
                if piece_reasoning:
                    reasoning_parts.append(piece_reasoning)
                    token_estimate += 1
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]

            if early_exit_check is not None:
                combined_chars = sum(len(p) for p in reasoning_parts) + sum(
                    len(p) for p in content_parts
                )
                if combined_chars - chars_at_last_early_exit_check >= early_exit_check_interval_chars:
                    chars_at_last_early_exit_check = combined_chars
                    combined_text = "".join(reasoning_parts) + "".join(content_parts)
                    if early_exit_check(combined_text):
                        early_exit = True
                        finish_reason = "early_exit"
                        resp.close()
                        break

            if detect_loops:
                # Reasoning-heavy models loop in the hidden reasoning channel,
                # not the final answer, so both are checked combined. Only
                # re-scan once enough new text has arrived to be worth the
                # pass -- see _has_repetition for why this rarely false-positives.
                combined_chars = sum(len(p) for p in reasoning_parts) + sum(
                    len(p) for p in content_parts
                )
                if (
                    combined_chars >= loop_min_chars
                    and combined_chars - chars_at_last_loop_check >= loop_window_chars // 16
                ):
                    chars_at_last_loop_check = combined_chars
                    combined_text = "".join(reasoning_parts) + "".join(content_parts)
                    if _has_repetition(
                        combined_text, loop_window_chars, loop_phrase_len, loop_min_repeats
                    ) or _has_repetition(
                        combined_text,
                        loop_window_chars,
                        loop_short_phrase_len,
                        loop_short_min_repeats,
                    ):
                        loop_detected = True
                        finish_reason = "loop_detected"
                        resp.close()
                        break
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

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        # Aborting early (loop detection) skips the final usage-bearing
        # chunk entirely, and some servers omit usage outright -- fall back
        # to counting streamed delta events, which is a token each for these
        # OpenAI-compatible APIs.
        completion_tokens = token_estimate

    return ChatResult(
        success=True,
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts) or None,
        finish_reason=finish_reason,
        latency_seconds=latency,
        ttft_seconds=ttft,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        total_tokens=usage.get("total_tokens"),
        requested_max_tokens=max_tokens,
        loop_detected=loop_detected,
        early_exit=early_exit,
        raw={"finish_reason": finish_reason, "usage": usage},
    )
