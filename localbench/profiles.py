"""Run profiles: how many problems per suite a run should use.

Quick exists because a full run over 114 problems is slow enough to
discourage the iterative use this tool is for ("does this quant help?").

Quick is deliberately the FIRST HALF of each suite, not a random sample.
Every generator produces problems in a deterministic order from its seed,
so a quick run's problems are a strict prefix of the full run's at the same
seed -- verified for all five generated suites. That means the two are
directly comparable: quick measures the same tasks, just fewer of them, and
therefore carries a wider confidence interval. A random subsample would
break that.
"""

from __future__ import annotations

# suite -> (quick, full)
PROFILE_SIZES: dict[str, tuple[int, int]] = {
    "json_schema": (10, 20),
    "coding": (6, 12),
    "logic_math": (15, 30),
    "instruction_following": (12, 24),
    "pattern_reasoning": (10, 20),
    "long_context": (4, 8),
    "tool_calling": (8, 16),
    "multi_turn": (6, 12),
}

PROFILES = ("quick", "full")
DEFAULT_PROFILE = "full"


def problems_for(suite: str, profile: str, configured: int | None = None) -> int | None:
    """Problem count for a suite under a profile.

    An explicit num_problems in config always wins for `full` -- the profile
    must never silently override a deliberate setting. Under `quick` that
    configured value is halved, so a customised suite still gets a
    proportionally faster run rather than being reset to our default.
    """
    sizes = PROFILE_SIZES.get(suite)
    if profile == "quick":
        if configured is not None:
            return max(1, configured // 2)
        return sizes[0] if sizes else None
    if configured is not None:
        return configured
    return sizes[1] if sizes else None
