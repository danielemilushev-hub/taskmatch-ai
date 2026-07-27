"""Provider-agnostic frontier judge: each provider only implements chat();
task generation and grading prompt/parsing logic lives here once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..json_extract import extract_json


@dataclass
class JudgeChatResult:
    text: str
    # Real token usage from the provider's own API response -- needed to
    # honestly estimate the judge's own cost (separate from the local
    # model's tokens, which are free and tracked elsewhere). None if a
    # provider's SDK response doesn't expose it for some reason.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


_GENERATE_TASK_PROMPT = """\
{task_spec}

---

Generate ONE task for the "{category}" category described above, to test a
local language model. Respond with ONLY a JSON object (no markdown fences,
no extra text) in exactly this shape:

{{
  "category": "{category}",
  "prompt": "the exact task/prompt text to give the local model",
  "rubric": ["list of specific, independently-checkable grading criteria"]
}}
"""

_GRADE_PROMPT = """\
{task_spec}

---

You are grading a local model's response to a "{category}" task.

TASK GIVEN TO THE MODEL:
{task_prompt}

GRADING RUBRIC:
{rubric}

MODEL'S RESPONSE:
{response}

Grade this response. Respond with ONLY a JSON object (no markdown fences,
no extra text) in exactly this shape:

{{
  "score": <integer 0-10>,
  "constraints_met": ["which rubric criteria were satisfied"],
  "issues": ["specific problems found, if any"],
  "rationale": "one or two sentences explaining the score"
}}
"""


class JudgeClient(ABC):
    """A frontier model used to generate/grade tasks for the local model
    under test. Never used for the deterministic suites."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> JudgeChatResult:
        """Send messages to the provider, return the text response plus real
        token usage from that API response. Each provider subclass
        implements this against its own SDK/API."""
        raise NotImplementedError

    def generate_task(self, category: str, task_spec: str) -> tuple[dict, JudgeChatResult]:
        prompt = _GENERATE_TASK_PROMPT.format(task_spec=task_spec, category=category)
        result = self.chat([{"role": "user", "content": prompt}], max_tokens=1024)
        value, error = extract_json(result.text)
        if error or not isinstance(value, dict):
            raise ValueError(f"judge did not return a valid task JSON object: {error or result.text!r}")
        return value, result

    def grade(self, task: dict, response: str, task_spec: str) -> tuple[dict, JudgeChatResult]:
        prompt = _GRADE_PROMPT.format(
            task_spec=task_spec,
            category=task.get("category", "unknown"),
            task_prompt=task.get("prompt", ""),
            rubric="\n".join(f"- {c}" for c in task.get("rubric", [])),
            response=response or "(empty response)",
        )
        result = self.chat([{"role": "user", "content": prompt}], max_tokens=1024)
        value, error = extract_json(result.text)
        if error or not isinstance(value, dict):
            raise ValueError(f"judge did not return a valid grade JSON object: {error or result.text!r}")
        return value, result
