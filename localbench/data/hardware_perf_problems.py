"""Problems for the hardware_perf suite -- measuring raw hardware/model
throughput (prefill speed, sustained decode speed), not correctness.

Two distinct probe shapes:

1. Prefill scaling: a long, benign filler prompt paired with a tiny
   `max_tokens` so time-to-first-token is dominated by prefill of that
   prompt, not decode. Run at several context lengths to see how prefill
   throughput (prompt_tokens / ttft_seconds) scales -- this is exactly
   where hardware differences (compute-bound prefill, KV-cache handling on
   long context) actually show up, and a single-length test would miss.

2. Sustained decode: a short prompt with a generous `max_tokens` on an
   open-ended prompt unlikely to stop early, measuring tokens/sec over a
   longer generation -- catches throttling a short generation wouldn't.

Filler text is real (if generic) English sentences, not random tokens or
repeated characters -- some tokenizers/models handle degenerate repeated
input in ways that don't reflect normal prefill cost, and this needs to
approximate what a model actually sees in practice (a real prompt/document).
Exact token counts vary by tokenizer; the point is to land in the right
ballpark per tier, not hit an exact number -- the suite records the real
prompt_tokens returned by the API as ground truth either way.
"""

from __future__ import annotations

import random

_FILLER_SENTENCES = [
    "The history of computing hardware spans many decades of steady innovation.",
    "Modern processors rely on parallel execution units to increase overall throughput.",
    "Memory bandwidth is often the limiting factor in large-scale numerical workloads.",
    "Graphics processing units were originally designed for rendering images quickly.",
    "Machine learning workloads benefit from specialized matrix multiplication hardware.",
    "Cache hierarchies exist to hide the latency of slower main memory access.",
    "Power efficiency has become as important as raw computational speed.",
    "Distributed systems must account for network latency between machines.",
    "Compilers translate human-readable source code into machine instructions.",
    "Operating systems manage the allocation of shared hardware resources.",
    "Data centers consume enormous amounts of electricity to run continuously.",
    "Storage technology has shifted from spinning disks toward solid-state memory.",
    "Network protocols define how information is packaged and transmitted reliably.",
    "Software engineers must balance correctness, performance, and maintainability.",
    "Cloud computing allows workloads to scale elastically with demand.",
    "Semiconductor manufacturing continues to push the limits of physical scale.",
    "Cooling systems are essential for maintaining stable hardware performance.",
    "Virtualization enables multiple isolated environments to share one machine.",
    "Benchmarking is the practice of measuring performance under controlled conditions.",
    "Concurrency introduces subtle correctness challenges that sequential code avoids.",
    "Database systems optimize for both fast reads and durable, consistent writes.",
    "Embedded systems often operate under strict constraints on power and memory.",
    "Version control systems track the history of changes made to a codebase.",
    "Encryption protects data from being read by anyone without the correct key.",
    "User interfaces should be intuitive even for people unfamiliar with the system.",
]

# (problem id, approximate target word count, approximate token count).
# English prose averages roughly 0.75 words per token, so these targets land
# near 200/1k/4k/8k/16k/32k tokens -- the real prompt_tokens value from the
# API is what actually gets recorded and compared, not this estimate.
#
# The 16k and 32k tiers exist because prefill throughput keeps climbing well
# past 8k on capable hardware (a live run went 85 -> 860 tok/s between 200
# and 7k tokens and had clearly not plateaued), so stopping at 8k measured
# the ramp but never found the ceiling. They are skipped automatically when
# they wouldn't fit the model's configured context window -- see
# tiers_for_context().
_PREFILL_TIERS = [
    ("prefill_tiny", 150, 200),
    ("prefill_small", 750, 1_000),
    ("prefill_medium", 3_000, 4_000),
    ("prefill_large", 6_000, 8_000),
    ("prefill_xl", 12_000, 16_000),
    ("prefill_xxl", 24_000, 32_000),
]

# Prefill probes generate only this many tokens -- the point is to time
# prompt processing, not generation.
_PREFILL_MAX_TOKENS = 8

# Token-count estimates from word counts are tokenizer-dependent and can run
# over, so a tier is only included when it fits the context window with room
# to spare. Overflowing the window doesn't produce a slow-but-valid
# measurement, it produces a failed call or a silently truncated prompt --
# either of which would be a meaningless data point.
_CONTEXT_SAFETY_FACTOR = 1.20


def tiers_for_context(max_context_tokens: int | None) -> list[tuple[str, int, int]]:
    """Prefill tiers that fit within `max_context_tokens`.

    None means "no known limit" -- every tier is used. This is what keeps the
    big new tiers from breaking small-context setups: a model configured with
    an 8k window simply runs the tiers up to 8k and skips the rest, instead
    of failing two probes.
    """
    if not max_context_tokens:
        return list(_PREFILL_TIERS)
    return [
        tier for tier in _PREFILL_TIERS
        if tier[2] * _CONTEXT_SAFETY_FACTOR + _PREFILL_MAX_TOKENS <= max_context_tokens
    ]


def num_problems_for_context(max_context_tokens: int | None) -> int:
    """How many problems this suite will actually run for a given context
    window -- so task counts shown in the UI match reality rather than
    assuming every tier always runs."""
    return len(tiers_for_context(max_context_tokens)) + len(_DECODE_TIERS)

# (problem id, max_tokens). A short prompt, so prefill contributes almost
# nothing to the timing -- this isolates decode speed.
_DECODE_TIERS = [
    ("decode_short", 1024),
    ("decode_long", 4096),
]

# Single source of truth for "how many problems does this suite run" at an
# unrestricted context window -- used wherever the task count needs
# reporting without duplicating this suite's shape as a magic number.
# Use num_problems_for_context() when the model's context window is known,
# since the largest prefill tiers are skipped if they wouldn't fit.
NUM_PROBLEMS = len(_PREFILL_TIERS) + len(_DECODE_TIERS)

_DECODE_PROMPT = (
    "Write a detailed, imaginative short story about a lighthouse keeper who "
    "discovers something extraordinary during a storm. Include vivid "
    "descriptions of the setting, the character's thoughts, and a complete "
    "narrative arc. Do not stop until you reach a natural, satisfying ending."
)


def _build_filler_text(rng: random.Random, target_words: int) -> str:
    sentences: list[str] = []
    word_count = 0
    while word_count < target_words:
        s = rng.choice(_FILLER_SENTENCES)
        sentences.append(s)
        word_count += len(s.split())
    return " ".join(sentences)


def generate_problems(seed: int = 42, max_context_tokens: int | None = None) -> list[dict]:
    """Fixed set, deliberately not scaled by Quick/Full profile -- this
    suite characterizes hardware, it isn't a statistically-sampled accuracy
    measurement, so there's no notion of "fewer samples for a faster
    baseline" the way the other suites have.

    `max_context_tokens` only ever removes tiers that physically cannot fit
    the model's context window; it never scales the set for speed.
    """
    rng = random.Random(seed)
    problems: list[dict] = []

    for pid, target_words, est_tokens in tiers_for_context(max_context_tokens):
        filler = _build_filler_text(rng, target_words)
        problems.append(
            {
                "id": pid,
                "task_type": "prefill",
                "prompt": f"{filler}\n\nReply with only the single word: OK",
                "max_tokens": _PREFILL_MAX_TOKENS,
                "est_prompt_tokens": est_tokens,
            }
        )

    for pid, max_tokens in _DECODE_TIERS:
        problems.append(
            {
                "id": pid,
                "task_type": "decode",
                "prompt": _DECODE_PROMPT,
                "max_tokens": max_tokens,
            }
        )

    return problems
