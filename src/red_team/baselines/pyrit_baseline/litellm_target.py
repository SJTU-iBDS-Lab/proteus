"""Adapter: PyRIT ``OpenAIChatTarget`` → our LiteLLM proxy.

PyRIT's ``OpenAIChatTarget`` reads endpoint config from a small env-var
contract (``OPENAI_CHAT_ENDPOINT`` / ``OPENAI_CHAT_KEY`` / ``OPENAI_CHAT_MODEL``)
or from explicit kwargs.  This helper builds a configured target pointed
at the local LiteLLM proxy that already routes to GLM-5 / DeepSeek /
GPT-5.4-mini / Kimi the same way Proteus does, so PyRIT runs share the
exact upstream Proteus uses (and the same cost-bookkeeping).
"""
from __future__ import annotations

import os
from typing import Any

_DEFAULT_LITELLM_URL = "http://127.0.0.1:4000"
_DEFAULT_LITELLM_KEY = "sk-claw-local"


def build_litellm_chat_target(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    is_json_supported: bool = False,
    **extra: Any,
):
    """Return a PyRIT ``OpenAIChatTarget`` pointed at our LiteLLM proxy.

    PyRIT 0.12 expects endpoint URL to be the full ``…/chat/completions``
    path; LiteLLM exposes it natively.  ``api_key`` defaults to the
    LiteLLM master key used elsewhere in the codebase.  ``model`` is the
    LiteLLM model id (e.g.\\ ``glm-5``, ``deepseek-v4-pro``).
    """
    from pyrit.prompt_target import OpenAIChatTarget

    # LiteLLM proxy roots both styles ("/chat/completions" suffix and bare
    # base URL).  PyRIT 0.12 prefers bare base URL because the OpenAI SDK
    # appends ``/chat/completions`` itself.
    base = (base_url or os.environ.get("LITELLM_URL", _DEFAULT_LITELLM_URL)).rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    key = api_key or os.environ.get("LITELLM_KEY", _DEFAULT_LITELLM_KEY)

    return OpenAIChatTarget(
        endpoint=base,
        api_key=key,
        model_name=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        is_json_supported=is_json_supported,
        **extra,
    )


__all__ = ["build_litellm_chat_target"]
