"""Picks a JudgeClient implementation by provider name and loads .env once."""

from __future__ import annotations

from .base import JudgeClient

_PROVIDERS = {
    "anthropic": "localbench.judge.anthropic_judge.AnthropicJudge",
    "openai": "localbench.judge.openai_judge.OpenAIJudge",
    "gemini": "localbench.judge.gemini_judge.GeminiJudge",
    "openrouter": "localbench.judge.openrouter_judge.OpenRouterJudge",
}

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    _dotenv_loaded = True


def get_judge_client(provider: str, model: str) -> JudgeClient:
    _ensure_dotenv_loaded()

    if provider not in _PROVIDERS:
        raise ValueError(f"unknown judge provider '{provider}', expected one of {list(_PROVIDERS)}")

    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(model)
