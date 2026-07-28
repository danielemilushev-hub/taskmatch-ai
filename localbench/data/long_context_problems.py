"""Long-context problems: give the model a large (1000+ line) source excerpt
and ask it to either (a) retrieve a unique marker planted somewhere inside it
("needle in a haystack"), or (b) locate the exact line where a comparison
operator was mechanically flipped, introducing a subtle bug.

Both are graded by exact match -- we control the needle/mutation, so ground
truth is certain, never guessed. Real code makes a much more realistic test
than lorem-ipsum filler, so this can optionally read an actual file from the
user's machine (source_file in config.yaml, not bundled/committed -- that
path is local-only). Falls back to a bundled synthetic codebase generator so
the suite still works with no source file configured, e.g. on a fresh clone.
"""

from __future__ import annotations

import random
from pathlib import Path

_COMMENT_STYLES = {
    ".py": "#",
    ".js": "//",
    ".ts": "//",
    ".java": "//",
    ".c": "//",
    ".cpp": "//",
    ".cs": "//",
    ".go": "//",
}

# Ordered so multi-character operators are matched/replaced before the
# shorter substrings they contain (e.g. "===" before "==").
_OPERATOR_FLIPS = [
    ("===", "!=="),
    ("!==", "==="),
    ("<=", "<"),
    (">=", ">"),
    ("==", "!="),
    ("!=", "=="),
]


def _synthetic_codebase(num_lines: int, rng: random.Random) -> list[str]:
    """Bundled fallback so this suite works with no real source file
    configured (e.g. a fresh clone on someone else's machine)."""
    lines = []
    while len(lines) < num_lines:
        i = len(lines)
        op = rng.choice(["+", "-", "*"])
        lines.append(f"def helper_function_{i}(a, b):")
        lines.append("    if a >= 0 and b >= 0:")
        lines.append(f"        return a {op} b  # auto-generated stub {i}")
        lines.append("    return 0")
    return lines[:num_lines]


def _load_source_lines(
    source_file: str | None, window_lines: int, rng: random.Random
) -> tuple[list[str], str]:
    if source_file:
        path = Path(source_file)
        if path.exists():
            all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(all_lines) > window_lines:
                start = rng.randint(0, len(all_lines) - window_lines)
                all_lines = all_lines[start : start + window_lines]
            return all_lines, path.suffix or ".txt"
    return _synthetic_codebase(window_lines, rng), ".py"


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


def _make_needle_problem(idx: int, lines: list[str], comment: str, rng: random.Random) -> dict:
    token = str(rng.randrange(10**6, 10**7))
    insert_at = rng.randint(max(1, len(lines) // 10), max(1, len(lines) - len(lines) // 10))
    needle_lines = lines[:]
    needle_lines.insert(insert_at, f"{comment} LOCALBENCH_NEEDLE: verification_code_is_{token}")
    return {
        "id": f"needle_retrieval_{idx}",
        "task_type": "needle_retrieval",
        "expected": token,
        "code_text": _numbered(needle_lines),
        "num_lines": len(needle_lines),
    }


def _make_bug_locator_problem(idx: int, lines: list[str], rng: random.Random) -> dict | None:
    candidates = [i for i, ln in enumerate(lines) if any(op in ln for op, _ in _OPERATOR_FLIPS)]
    if not candidates:
        return None
    bug_idx = rng.choice(candidates)
    line = lines[bug_idx]
    mutated = line
    for orig, flipped in _OPERATOR_FLIPS:
        if orig in line:
            mutated = line.replace(orig, flipped, 1)
            break
    mutated_lines = lines[:]
    mutated_lines[bug_idx] = mutated
    return {
        "id": f"bug_locator_{idx}",
        "task_type": "bug_locator",
        "expected": bug_idx + 1,  # 1-indexed to match the numbering shown to the model
        "code_text": _numbered(mutated_lines),
        "num_lines": len(mutated_lines),
    }


def generate_problems(
    num_problems: int,
    seed: int = 42,
    source_file: str | None = None,
    window_lines: int = 1000,
) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(num_problems):
        lines, ext = _load_source_lines(source_file, window_lines, rng)
        comment = _COMMENT_STYLES.get(ext, "#")

        if i % 2 == 0:
            problems.append(_make_needle_problem(i, lines, comment, rng))
        else:
            bug_problem = _make_bug_locator_problem(i, lines, rng)
            problems.append(bug_problem or _make_needle_problem(i, lines, comment, rng))
    return problems
