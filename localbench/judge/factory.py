"""Picks a JudgeClient implementation by provider name and loads .env once."""

from __future__ import annotations

from .base import JudgeClient

_PROVIDERS = {
    "anthropic": "localbench.judge.anthropic_judge.AnthropicJudge",
    "openai": "localbench.judge.openai_judge.OpenAIJudge",
    "gemini": "localbench.judge.gemini_judge.GeminiJudge",
    "openrouter": "localbench.judge.openrouter_judge.OpenRouterJudge",
}

# The pip package that provides each judge's SDK. These are intentionally NOT
# in requirements.txt (you only need the one provider you actually use), which
# means "not installed" is the normal first-run state -- so it has to produce
# an instruction, not a traceback. OpenRouter is OpenAI-compatible and reuses
# that SDK.
PROVIDER_PACKAGES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google-genai",
    "openrouter": "openai",
}

# Module actually imported at runtime, which differs from the pip name for
# google-genai (pip install google-genai -> from google import genai).
_PROVIDER_IMPORT_NAMES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google.genai",
    "openrouter": "openai",
}


def sdk_installed(provider: str) -> bool:
    """Whether this provider's SDK can actually be imported right now, so the
    UI can say so before a run starts instead of failing partway in."""
    import importlib.util

    module = _PROVIDER_IMPORT_NAMES.get(provider)
    if not module:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False

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

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(model)
    except ImportError as e:
        # Raw ImportErrors here are cryptic ("cannot import name 'genai' from
        # 'google'") and don't tell you the one thing you need to do. The
        # judge SDKs are deliberately optional, so this is the expected
        # first-run state for whichever provider you pick.
        pkg = PROVIDER_PACKAGES.get(provider, provider)
        raise RuntimeError(
            f"the {provider} judge needs its SDK, which isn't installed: {e}. "
            f"Install it with:  pip install {pkg}"
        ) from e
