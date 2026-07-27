"""Read/write the mutable parts of config.yaml (runtime + judge sections)
and the .env file (frontier API keys) for the dashboard's Settings page.

config.yaml is hand-written with extensive explanatory comments -- a plain
yaml.safe_load()/yaml.dump() round-trip would silently discard all of them.
ruamel.yaml's round-trip mode preserves comments/formatting for keys it
doesn't touch, so edits made here only affect the specific values changed.

API keys never round-trip through this module's return values -- callers
only ever get back an "is_set" boolean per provider, never the key itself.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import set_key, unset_key
from ruamel.yaml import YAML

CONFIG_PATH = Path("config.yaml")
ENV_PATH = Path(".env")

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_yaml = YAML()
_yaml.preserve_quotes = True
# Match config.yaml's hand-authored style (dashes indented 2 spaces past
# their parent key) -- ruamel's own defaults (sequence flush with the
# parent key) would reformat the whole file on every write.
_yaml.indent(mapping=2, sequence=4, offset=2)


def _represent_none(representer, _data):
    # ruamel's default represents None as a bare empty value; config.yaml
    # writes it explicitly as `null` (e.g. long_context.source_file) --
    # match that so an untouched null field doesn't get reformatted.
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


_yaml.representer.add_representer(type(None), _represent_none)


def _load() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return _yaml.load(f)


def _save(data: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)


def get_runtime_settings() -> dict:
    data = _load()
    runtime = data.get("runtime", {})
    return {
        "base_url": runtime.get("base_url"),
        "request_timeout_seconds": runtime.get("request_timeout_seconds"),
        "unload_all_cmd": runtime.get("unload_all_cmd"),
    }


def update_runtime_settings(updates: dict) -> dict:
    data = _load()
    runtime = data.setdefault("runtime", {})

    if "base_url" in updates:
        base_url = (updates["base_url"] or "").strip()
        if not base_url:
            raise ValueError("base_url cannot be empty")
        runtime["base_url"] = base_url

    if "request_timeout_seconds" in updates:
        val = updates["request_timeout_seconds"]
        if val is not None:
            val = int(val)
            if val <= 0:
                raise ValueError("request_timeout_seconds must be positive")
        runtime["request_timeout_seconds"] = val

    if "unload_all_cmd" in updates:
        val = (updates["unload_all_cmd"] or "").strip()
        runtime["unload_all_cmd"] = val or None

    _save(data)
    return get_runtime_settings()


def get_judge_settings() -> dict:
    data = _load()
    judge = data.get("judge", {})
    return {
        "enabled": judge.get("enabled", False),
        "provider": judge.get("provider"),
        "model": judge.get("model"),
        "num_tasks": judge.get("num_tasks"),
        "pass_threshold": judge.get("pass_threshold"),
    }


def update_judge_settings(updates: dict) -> dict:
    data = _load()
    judge = data.setdefault("judge", {})

    if "enabled" in updates:
        judge["enabled"] = bool(updates["enabled"])

    if "provider" in updates:
        provider = updates["provider"]
        if provider not in PROVIDER_ENV_VARS:
            raise ValueError(f"unknown provider '{provider}', expected one of {list(PROVIDER_ENV_VARS)}")
        judge["provider"] = provider

    if "model" in updates:
        model = (updates["model"] or "").strip()
        if not model:
            raise ValueError("model cannot be empty")
        judge["model"] = model

    if "num_tasks" in updates:
        val = int(updates["num_tasks"])
        if val <= 0:
            raise ValueError("num_tasks must be positive")
        judge["num_tasks"] = val

    if "pass_threshold" in updates:
        val = int(updates["pass_threshold"])
        if not (0 <= val <= 10):
            raise ValueError("pass_threshold must be between 0 and 10")
        judge["pass_threshold"] = val

    _save(data)
    return get_judge_settings()


def key_status() -> dict:
    """{provider: bool} -- whether a non-empty key is currently on file.
    Never returns the key values themselves."""
    from dotenv import dotenv_values

    env_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    return {
        provider: bool(env_values.get(env_var))
        for provider, env_var in PROVIDER_ENV_VARS.items()
    }


def set_api_key(provider: str, value: str) -> None:
    if provider not in PROVIDER_ENV_VARS:
        raise ValueError(f"unknown provider '{provider}', expected one of {list(PROVIDER_ENV_VARS)}")
    value = (value or "").strip()
    if not value:
        raise ValueError("api_key cannot be empty -- use the clear endpoint to remove a key")
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    set_key(str(ENV_PATH), PROVIDER_ENV_VARS[provider], value, quote_mode="never")


def clear_api_key(provider: str) -> None:
    if provider not in PROVIDER_ENV_VARS:
        raise ValueError(f"unknown provider '{provider}', expected one of {list(PROVIDER_ENV_VARS)}")
    if not ENV_PATH.exists():
        return
    unset_key(str(ENV_PATH), PROVIDER_ENV_VARS[provider])


def _get_key(provider: str) -> str | None:
    from dotenv import dotenv_values

    env_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    return env_values.get(PROVIDER_ENV_VARS.get(provider, ""))


def list_judge_models(provider: str) -> list[str]:
    """Best-effort live model list for the given judge provider -- lets the
    Settings/New Run UI suggest real model ids instead of relying purely on
    hand-typed strings. Returns [] on any failure (no key, network error,
    unexpected response shape) rather than raising -- this is a UX
    convenience, never required to actually run the judge."""
    import requests

    if provider == "openrouter":
        # Public catalog, no key required -- OpenRouter is a marketplace and
        # lists every model + its live pricing this way.
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return []

    key = _get_key(provider)
    if not key:
        return []

    try:
        if provider == "anthropic":
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                timeout=10,
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))

        if provider == "openai":
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            resp.raise_for_status()
            excluded = ("embedding", "whisper", "tts", "dall-e", "moderation")
            ids = [
                m["id"]
                for m in resp.json().get("data", [])
                if m.get("id") and not any(x in m["id"] for x in excluded)
            ]
            return sorted(ids)

        if provider == "gemini":
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
                timeout=10,
            )
            resp.raise_for_status()
            ids = [
                m["name"].removeprefix("models/")
                for m in resp.json().get("models", [])
                if "generateContent" in (m.get("supportedGenerationMethods") or [])
            ]
            return sorted(ids)
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return []

    return []


def list_openrouter_pricing() -> dict:
    """{model_id: {"prompt": $/token, "completion": $/token}} for every
    OpenRouter model, from one call to their public catalog. Lets the model
    picker show a real per-model rate inline instead of making the user go
    look it up. Empty dict on any failure -- never invented numbers."""
    import requests

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        out = {}
        for entry in resp.json().get("data", []):
            mid = entry.get("id")
            pricing = entry.get("pricing") or {}
            if not mid or pricing.get("prompt") is None or pricing.get("completion") is None:
                continue
            try:
                out[mid] = {
                    "prompt": float(pricing["prompt"]),
                    "completion": float(pricing["completion"]),
                }
            except (TypeError, ValueError):
                continue
        return out
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return {}


def get_openrouter_pricing(model: str) -> dict | None:
    """Real, live $/token pricing for a specific OpenRouter model id, from
    their public catalog (no key needed -- it's a marketplace, so pricing is
    part of the product listing, not account-specific). Returns None if the
    model isn't found or the request fails -- never a guessed number."""
    import requests

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("id") == model:
                pricing = entry.get("pricing") or {}
                prompt_rate = pricing.get("prompt")
                completion_rate = pricing.get("completion")
                if prompt_rate is None or completion_rate is None:
                    return None
                return {"prompt": float(prompt_rate), "completion": float(completion_rate)}
        return None
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return None
