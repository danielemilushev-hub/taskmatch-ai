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

# (problem id, approximate target word count). English prose averages
# roughly 0.75 words per token, so these targets land near 200/1,000/4,000/
# 8,000 tokens -- the real prompt_tokens value from the API is what actually
# gets recorded and compared, not this estimate.
_PREFILL_TIERS = [
    ("prefill_tiny", 150),
    ("prefill_small", 750),
    ("prefill_medium", 3000),
    ("prefill_large", 6000),
]

# (problem id, max_tokens). A short prompt, so prefill contributes almost
# nothing to the timing -- this isolates decode speed.
_DECODE_TIERS = [
    ("decode_short", 1024),
    ("decode_long", 4096),
]

# Single source of truth for "how many problems does this suite run" --
# used wherever the task count needs reporting (e.g. the New Run page's
# task-count summary) without duplicating this suite's fixed shape as a
# magic number elsewhere.
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


def generate_problems(seed: int = 42) -> list[dict]:
    """Fixed set, deliberately not scaled by Quick/Full profile -- this
    suite characterizes hardware, it isn't a statistically-sampled accuracy
    measurement, so there's no notion of "fewer samples for a faster
    baseline" the way the other suites have."""
    rng = random.Random(seed)
    problems: list[dict] = []

    for pid, target_words in _PREFILL_TIERS:
        filler = _build_filler_text(rng, target_words)
        problems.append(
            {
                "id": pid,
                "task_type": "prefill",
                "prompt": f"{filler}\n\nReply with only the single word: OK",
                "max_tokens": 8,
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
