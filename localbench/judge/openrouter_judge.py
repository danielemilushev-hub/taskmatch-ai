"""OpenRouter judge provider -- lets the user pick any frontier model by
string (e.g. "anthropic/claude-sonnet-5", "openai/gpt-5") without a
provider-specific SDK, since OpenRouter's endpoint is OpenAI-compatible.
Reuses the `openai` package pointed at OpenRouter's base_url."""

from __future__ import annotations

import os

from .base import JudgeChatResult, JudgeClient


class OpenRouterJudge(JudgeClient):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set -- add it to .env to use the OpenRouter judge"
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> JudgeChatResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        usage = resp.usage
        return JudgeChatResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
