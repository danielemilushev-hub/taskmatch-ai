"""OpenAI (GPT) judge provider. Import of the `openai` package is deferred
to __init__ so it's only required when this provider is actually selected."""

from __future__ import annotations

import os

from .base import JudgeChatResult, JudgeClient


class OpenAIJudge(JudgeClient):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set -- add it to .env to use the OpenAI judge"
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> JudgeChatResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=messages,
        )
        usage = resp.usage
        return JudgeChatResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
