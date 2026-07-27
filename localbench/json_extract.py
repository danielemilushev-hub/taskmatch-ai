"""Best-effort extraction of a JSON value from free-form LLM text output."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _first_balanced(text: str) -> str | None:
    """Return the substring of the first balanced {...} or [...] in text."""
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            break
    else:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> tuple[object | None, str | None]:
    """Try hard to find a JSON value in `text`. Returns (value, error)."""
    if text is None:
        return None, "no content to parse (empty response)"

    text = text.strip()
    candidates = []

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(text)

    for cand in candidates:
        try:
            return json.loads(cand), None
        except json.JSONDecodeError:
            continue

    for cand in candidates:
        balanced = _first_balanced(cand)
        if balanced is not None:
            try:
                return json.loads(balanced), None
            except json.JSONDecodeError:
                continue

    return None, "no valid JSON found in model output"
