"""Anthropic (Claude) judge provider. Import of the `anthropic` package is
deferred to __init__ so it's only required when this provider is actually
selected."""

from __future__ import annotations

import os

from .base import JudgeChatResult, JudgeClient


class AnthropicJudge(JudgeClient):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set -- add it to .env to use the Anthropic judge"
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> JudgeChatResult:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return JudgeChatResult(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
