"""``LiteLLMClient`` — default :class:`llm.client.LLMClient` implementation.

The client is a thin wrapper around :class:`openai.OpenAI` pointed at the
local LiteLLM proxy.  Three deliberate design choices:

* **Lazy OpenAI import.**  ``openai`` is imported inside
  :func:`_make_openai_client`, not at module load, so unit tests that
  patch the factory can import this module in environments where the
  dependency is missing or the proxy is unreachable.
* **Env-var fallback with documented defaults.**  Matches the pattern in
  ``red_team/agent/runtime.py`` so Wave 2 is a drop-in swap rather than
  a behavioural change.
* **Conditional kwarg forwarding in** :meth:`LiteLLMClient.chat`.  The
  OpenAI client rejects ``tools=None`` / ``tool_choice=None`` /
  ``max_tokens=None``; we only forward keys the caller actually supplied.
"""
from __future__ import annotations

import os
from typing import Any


_DEFAULT_LITELLM_URL = "http://127.0.0.1:4000"
_DEFAULT_LITELLM_KEY = "sk-claw-local"


class LiteLLMClient:
    """LLMClient backed by the local LiteLLM proxy.

    Reads ``LITELLM_URL`` / ``LITELLM_KEY`` env vars with sensible
    defaults so tests can patch :func:`_make_openai_client` without
    touching the environment.  Intentionally does NOT import ``openai``
    at module load time — callers in test environments without the
    package need to be able to import this module.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("LITELLM_URL") or _DEFAULT_LITELLM_URL
        )
        self._api_key = (
            api_key or os.environ.get("LITELLM_KEY") or _DEFAULT_LITELLM_KEY
        )
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Return the cached OpenAI client, building it on first use.

        Construction is deferred so importing this module never touches
        ``openai`` or the network — see the module docstring.
        """
        if self._client is None:
            self._client = _make_openai_client(self._base_url, self._api_key)
        return self._client

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        """Forward to ``client.chat.completions.create``.

        Optional kwargs are only forwarded when non-``None`` — the
        OpenAI client treats explicit ``None`` as an error for several
        of these fields.
        """
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return client.chat.completions.create(**kwargs)


def _make_openai_client(base_url: str, api_key: str) -> Any:
    """Lazy-imported OpenAI client factory.

    Isolated so tests can monkey-patch ``llm.litellm._make_openai_client``
    (or the underlying ``openai.OpenAI``) without caring about import
    ordering.  The lazy import means this module loads even in
    environments where ``openai`` is not installed.

    Per-request timeout defaults to 300 seconds — chosen so reasoning
    models (deepseek-v4-pro, glm-5 with extended chain-of-thought) have
    headroom to finish without tripping a clean timeout, while still
    bounding a 5-round campaign that hits a stuck endpoint.  The
    OpenAI client's own default is 10 minutes which makes a single
    bad request hang the whole loop for an hour.  Override via
    ``LLM_REQUEST_TIMEOUT`` env when needed.
    """
    import os

    from openai import OpenAI

    try:
        timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "300"))
    except (TypeError, ValueError):
        timeout = 300.0
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


__all__ = ["LiteLLMClient", "_make_openai_client"]
