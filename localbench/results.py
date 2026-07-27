"""Shared result types produced by job suites, and the RunRecord that bundles
an entire benchmark run (every model, every suite, plus hardware context) into
one self-contained, JSON-serializable object."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# 95% two-sided normal quantile, for the Wilson score interval below.
_Z_95 = 1.959963984540054


def wilson_interval(passes: int, total: int, z: float = _Z_95) -> tuple[float, float] | None:
    """95% confidence interval for a pass rate, via the Wilson score method.

    A pass rate measured over a handful of problems is an estimate, not a
    fact, and reporting it bare overstates what was measured: 5/5 on a
    5-problem suite is statistically consistent with a model that fails 43%
    of the time. Wilson is used rather than the textbook normal
    approximation because it stays sensible at small N and at rates near 0
    or 1 -- exactly this tool's regime -- where the normal approximation
    produces intervals running past 0% or 100%.
    """
    if total <= 0:
        return None
    p = passes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def rates_distinguishable(passes_a: int, total_a: int, passes_b: int, total_b: int) -> bool:
    """Whether two pass rates differ by more than measurement noise.

    Non-overlapping 95% intervals is a deliberately conservative test: it
    implies significance, though the converse doesn't strictly hold. The
    asymmetry is the point -- the expensive error here is telling someone
    model A beats model B when the data cannot support it."""
    ci_a = wilson_interval(passes_a, total_a)
    ci_b = wilson_interval(passes_b, total_b)
    if ci_a is None or ci_b is None:
        return False
    return ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1]


@dataclass
class ProblemResult:
    problem_id: str
    passed: bool
    error: str | None = None
    latency_seconds: float = 0.0
    ttft_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt: str | list[dict] | None = None
    response_content: str | None = None
    reasoning_content: str | None = None
    truncated: bool = False
    # Only set by the frontier-graded suite: a 0-10 score alongside pass/fail
    # (score >= threshold), since that suite's grading is a judged quality
    # signal, not a deterministic exact-match result like the other suites.
    score: float | None = None
    rationale: str | None = None
    # Wall-clock time of the two paid frontier-judge API calls (task
    # generation, grading) -- separate from latency_seconds, which only
    # times the local model's own call. Needed to honestly estimate how
    # long/costly a future frontier-graded run will be, since local suites
    # never make outbound paid calls and have no equivalent.
    judge_generate_seconds: float | None = None
    judge_grade_seconds: float | None = None
    # Real token usage from the judge's own two API calls (generation +
    # grading), summed -- separate from prompt_tokens/completion_tokens
    # above, which are the LOCAL model's (free) tokens. This is what
    # actually costs money and is needed for a real cost estimate.
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None

    @property
    def tokens_per_sec(self) -> float | None:
        if not self.completion_tokens or self.latency_seconds <= 0:
            return None
        return self.completion_tokens / self.latency_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens_per_sec"] = self.tokens_per_sec
        return d


@dataclass
class SuiteRunResult:
    suite: str
    problems: list[ProblemResult] = field(default_factory=list)
    # Peak/avg CPU + RAM usage sampled while this suite ran (see
    # resource_monitor.py). A peak RAM usage far above idle baseline combined
    # with sustained high CPU during generation suggests the model spilled
    # out of VRAM -- correlational, not a direct VRAM measurement.
    resource_usage: dict | None = None

    @property
    def total(self) -> int:
        return len(self.problems)

    @property
    def pass_count(self) -> int:
        return sum(1 for p in self.problems if p.passed)

    @property
    def pass_rate(self) -> float | None:
        return self.pass_count / self.total if self.total else None

    @property
    def pass_rate_ci(self) -> tuple[float, float] | None:
        """95% CI on this suite's pass rate -- the honest width of the estimate."""
        return wilson_interval(self.pass_count, self.total)

    @property
    def avg_latency_seconds(self) -> float | None:
        if not self.problems:
            return None
        return sum(p.latency_seconds for p in self.problems) / len(self.problems)

    @property
    def avg_ttft_seconds(self) -> float | None:
        values = [p.ttft_seconds for p in self.problems if p.ttft_seconds is not None]
        return sum(values) / len(values) if values else None

    @property
    def avg_tokens_per_sec(self) -> float | None:
        rates = [p.tokens_per_sec for p in self.problems if p.tokens_per_sec is not None]
        return sum(rates) / len(rates) if rates else None

    @property
    def avg_score(self) -> float | None:
        # Only meaningful for the frontier-graded suite -- score is None on
        # every other suite's ProblemResult, so this is naturally empty there.
        scores = [p.score for p in self.problems if p.score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_seconds_per_task(self) -> float | None:
        # Only meaningful for the frontier-graded suite: local call latency
        # plus both paid judge calls (generate + grade), so a future run's
        # total wall-clock time can be estimated as num_tasks * this value.
        totals = [
            p.latency_seconds + (p.judge_generate_seconds or 0) + (p.judge_grade_seconds or 0)
            for p in self.problems
            if p.judge_generate_seconds is not None or p.judge_grade_seconds is not None
        ]
        return sum(totals) / len(totals) if totals else None

    @property
    def judge_infrastructure_failed(self) -> bool:
        """True when EVERY problem failed because the judge itself never
        answered (auth/quota/network), rather than because the local model
        did badly.

        Without this the two are indistinguishable in the UI: a depleted API
        quota produces six failed problems and a 0% pass rate, which reads as
        "this model scored zero" when in fact it was never graded. Only
        meaningful for the frontier-graded suite; every other suite grades
        locally and has no judge to fail."""
        if not self.problems:
            return False
        judge_errors = 0
        for p in self.problems:
            if p.passed:
                return False
            err = p.error or ""
            if err.startswith("judge task generation failed:") or err.startswith("judge grading failed:"):
                judge_errors += 1
        return judge_errors == len(self.problems)

    @property
    def avg_judge_prompt_tokens(self) -> float | None:
        values = [p.judge_prompt_tokens for p in self.problems if p.judge_prompt_tokens is not None]
        return sum(values) / len(values) if values else None

    @property
    def avg_judge_completion_tokens(self) -> float | None:
        values = [p.judge_completion_tokens for p in self.problems if p.judge_completion_tokens is not None]
        return sum(values) / len(values) if values else None

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "problems": [p.to_dict() for p in self.problems],
            "total": self.total,
            "pass_count": self.pass_count,
            "pass_rate": self.pass_rate,
            "pass_rate_ci": list(self.pass_rate_ci) if self.pass_rate_ci else None,
            "avg_latency_seconds": self.avg_latency_seconds,
            "avg_ttft_seconds": self.avg_ttft_seconds,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
            "avg_score": self.avg_score,
            "avg_seconds_per_task": self.avg_seconds_per_task,
            "avg_judge_prompt_tokens": self.avg_judge_prompt_tokens,
            "avg_judge_completion_tokens": self.avg_judge_completion_tokens,
            "judge_infrastructure_failed": self.judge_infrastructure_failed,
            "resource_usage": self.resource_usage,
        }


@dataclass
class ModelRunResult:
    model: str
    suites: dict[str, SuiteRunResult] = field(default_factory=dict)
    # Best-effort runtime info captured right after this model was loaded
    # (e.g. LM Studio's reported size/quantization/context length). Note that
    # none of the runtimes we support expose an exact GPU-vs-CPU offload
    # ratio via API/CLI today -- if tokens/sec looks much lower than expected
    # for a model's size, it likely spilled out of VRAM; check the runtime's
    # own UI (e.g. LM Studio's Developer tab) to confirm.
    runtime_load_info: dict | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "suites": {name: suite.to_dict() for name, suite in self.suites.items()},
            "runtime_load_info": self.runtime_load_info,
        }


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    hardware: dict
    config_summary: dict
    models: dict[str, ModelRunResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "hardware": self.hardware,
            "config_summary": self.config_summary,
            "models": {name: m.to_dict() for name, m in self.models.items()},
        }
