"""Tests for :class:`llm.litellm.LiteLLMClient` (Wave 1, W1-llm).

The tests exercise the four behaviours that Wave 2 relies on when it
migrates ``red_team.agent.runtime._make_client`` onto the new
abstraction:

1. environment-variable fallback with documented defaults,
2. explicit constructor overrides,
3. lazy client construction (``openai`` is never imported at
   module-import time, only when ``chat`` is first called),
4. conditional kwarg forwarding in :meth:`LiteLLMClient.chat`.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest  # noqa: F401  — used implicitly for monkeypatch fixture type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``llm.litellm._make_openai_client`` with a factory that
    returns a shared :class:`MagicMock`.  Returns the mock so tests can
    assert on it.
    """
    fake_client = MagicMock()
    monkeypatch.setattr(
        "llm.litellm._make_openai_client",
        lambda *args, **kwargs: fake_client,
    )
    return fake_client


# ---------------------------------------------------------------------------
# 1. Environment-variable default fallback
# ---------------------------------------------------------------------------


def test_defaults_from_env_vars_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither constructor args nor env vars are provided, the
    documented fallbacks (``127.0.0.1:4000`` / ``sk-claw-local``) apply.
    """
    monkeypatch.delenv("LITELLM_URL", raising=False)
    monkeypatch.delenv("LITELLM_KEY", raising=False)

    from llm.litellm import (
        _DEFAULT_LITELLM_KEY,
        _DEFAULT_LITELLM_URL,
        LiteLLMClient,
    )

    client = LiteLLMClient()
    assert client._base_url == _DEFAULT_LITELLM_URL
    assert client._api_key == _DEFAULT_LITELLM_KEY


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LITELLM_URL`` / ``LITELLM_KEY`` env vars are honoured when the
    constructor is called without explicit overrides.
    """
    monkeypatch.setenv("LITELLM_URL", "http://proxy.example:9000")
    monkeypatch.setenv("LITELLM_KEY", "sk-env-token")

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient()
    assert client._base_url == "http://proxy.example:9000"
    assert client._api_key == "sk-env-token"


# ---------------------------------------------------------------------------
# 2. Explicit overrides beat env vars
# ---------------------------------------------------------------------------


def test_explicit_args_override_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``base_url`` / ``api_key`` arguments win over env vars."""
    monkeypatch.setenv("LITELLM_URL", "http://env.example:1111")
    monkeypatch.setenv("LITELLM_KEY", "sk-env")

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient(
        base_url="http://explicit.example:2222",
        api_key="sk-explicit",
    )
    assert client._base_url == "http://explicit.example:2222"
    assert client._api_key == "sk-explicit"


# ---------------------------------------------------------------------------
# 3. Lazy client construction
# ---------------------------------------------------------------------------


def test_make_openai_client_is_called_lazily_and_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_make_openai_client`` must not fire during ``__init__`` and must
    be called exactly once across multiple ``chat`` invocations (client
    reuse).
    """
    calls: list[tuple[str, str]] = []

    def _fake_factory(base_url: str, api_key: str) -> Any:
        calls.append((base_url, api_key))
        inner = MagicMock()
        inner.chat.completions.create.return_value = MagicMock()
        return inner

    monkeypatch.setattr("llm.litellm._make_openai_client", _fake_factory)

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient(base_url="http://x", api_key="k")
    # No factory call during construction.
    assert calls == []

    client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    client.chat(model="m", messages=[{"role": "user", "content": "hi2"}])

    # Factory called exactly once; the underlying client is reused.
    assert calls == [("http://x", "k")]


# ---------------------------------------------------------------------------
# 4. Conditional kwarg forwarding
# ---------------------------------------------------------------------------


def test_chat_forwards_full_kwargs_when_all_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller supplies tools/tool_choice/max_tokens they all
    land on ``client.chat.completions.create``.
    """
    fake = _install_fake_openai(monkeypatch)

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient(base_url="http://x", api_key="k")

    tools = [{"type": "function", "function": {"name": "noop"}}]
    tool_choice = "auto"
    messages = [{"role": "user", "content": "hi"}]

    client.chat(
        model="m",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.3,
        max_tokens=512,
    )

    fake.chat.completions.create.assert_called_once()
    _, kwargs = fake.chat.completions.create.call_args
    assert kwargs["model"] == "m"
    assert kwargs["messages"] is messages
    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 512


def test_chat_omits_optional_kwargs_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults must not leak ``tools=None`` / ``tool_choice=None`` /
    ``max_tokens=None`` into the upstream call — the OpenAI client
    rejects some of those as ``None``.
    """
    fake = _install_fake_openai(monkeypatch)

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient(base_url="http://x", api_key="k")

    client.chat(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    fake.chat.completions.create.assert_called_once()
    _, kwargs = fake.chat.completions.create.call_args
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "max_tokens" not in kwargs
    # Required/default fields are still present.
    assert kwargs["model"] == "m"
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_returns_upstream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw ``ChatCompletion``-shaped object is returned unchanged —
    the abstraction deliberately stays structural, matching the current
    ``runtime.py`` consumers.
    """
    fake = _install_fake_openai(monkeypatch)
    sentinel = object()
    fake.chat.completions.create.return_value = sentinel

    from llm.litellm import LiteLLMClient

    client = LiteLLMClient(base_url="http://x", api_key="k")

    result = client.chat(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result is sentinel
