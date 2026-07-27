"""Shared result types produced by job suites, and the RunRecord that bundles
an entire benchmark run (every model, every suite, plus hardware context) into
one self-contained, JSON-serializable object."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "problems": [p.to_dict() for p in self.problems],
            "total": self.total,
            "pass_count": self.pass_count,
            "pass_rate": self.pass_rate,
            "avg_latency_seconds": self.avg_latency_seconds,
            "avg_ttft_seconds": self.avg_ttft_seconds,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
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
