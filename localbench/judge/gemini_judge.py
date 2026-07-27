"""Google Gemini judge provider. Import of the `google-genai` package is
deferred to __init__ so it's only required when this provider is actually
selected."""

from __future__ import annotations

import os

from .base import JudgeChatResult, JudgeClient


class GeminiJudge(JudgeClient):
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set -- add it to .env to use the Gemini judge"
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> JudgeChatResult:
        # our usage is always single-turn (one user message per call), so a
        # simple text join is sufficient rather than a full message-role mapping
        prompt = "\n\n".join(m["content"] for m in messages)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"max_output_tokens": max_tokens},
        )
        usage = getattr(resp, "usage_metadata", None)
        return JudgeChatResult(
            text=resp.text or "",
            prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )
