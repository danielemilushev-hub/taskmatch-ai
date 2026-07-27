"""Job suite: a frontier judge generates tasks and grades the local model's
responses against TASK_SPEC.md's rubric. Opt-in only -- costs real money,
never runs as part of a normal deterministic benchmark pass. See
TASK_SPEC.md for the anti-hallucination rules governing what the judge is
and isn't allowed to treat as ground truth.
"""

from __future__ import annotations

from pathlib import Path

from ..engine import RunContext
from ..judge.base import JudgeClient
from ..results import ProblemResult

_TASK_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "TASK_SPEC.md"

DEFAULT_CATEGORIES = [
    "instruction_following",
    "source_grounded_summarization",
    "reasoning_explanation",
    "creative_writing",
    "code_explanation",
    "hallucination_probe",
]


def _load_task_spec() -> str:
    return _TASK_SPEC_PATH.read_text(encoding="utf-8")


def run(
    ctx: RunContext,
    judge: JudgeClient,
    num_tasks: int = 6,
    categories: list[str] | None = None,
    pass_threshold: float = 7,
    task_spec: str | None = None,
) -> list[ProblemResult]:
    categories = categories or DEFAULT_CATEGORIES
    task_spec = task_spec if task_spec is not None else _load_task_spec()
    results: list[ProblemResult] = []

    for i in range(num_tasks):
        category = categories[i % len(categories)]
        problem_id = f"{category}_{i}"

        try:
            task = judge.generate_task(category, task_spec)
        except Exception as e:  # noqa: BLE001 -- any judge/network failure is a graded failure, not a crash
            results.append(
                ProblemResult(
                    problem_id=problem_id,
                    passed=False,
                    error=f"judge task generation failed: {e}",
                )
            )
            continue

        chat = ctx.call([{"role": "user", "content": task["prompt"]}])
        if not chat.success:
            results.append(
                ProblemResult(
                    problem_id=problem_id,
                    passed=False,
                    error=f"call failed: {chat.error}",
                    latency_seconds=chat.latency_seconds,
                    ttft_seconds=chat.ttft_seconds,
                )
            )
            continue

        try:
            verdict = judge.grade(task, chat.content or "", task_spec)
        except Exception as e:  # noqa: BLE001
            results.append(
                ProblemResult(
                    problem_id=problem_id,
                    passed=False,
                    error=f"judge grading failed: {e}",
                    latency_seconds=chat.latency_seconds,
                    ttft_seconds=chat.ttft_seconds,
                    prompt_tokens=chat.prompt_tokens,
                    completion_tokens=chat.completion_tokens,
                )
            )
            continue

        score = verdict.get("score")
        passed = score is not None and score >= pass_threshold
        rationale = verdict.get("rationale")
        issues = verdict.get("issues") or []
        error = None if passed else (rationale or "; ".join(issues) or "score below threshold")

        results.append(
            ProblemResult(
                problem_id=problem_id,
                passed=passed,
                error=error,
                score=score,
                rationale=rationale,
                latency_seconds=chat.latency_seconds,
                ttft_seconds=chat.ttft_seconds,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )
        )

    return results
