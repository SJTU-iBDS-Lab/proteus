"""Tests for the ``LLMClient`` protocol surface (Wave 1, W1-llm).

These tests pin the *shape* of the abstraction — they must pass without
``openai`` being importable, because upstream callers (red-team runtime,
blue-team scanners) will import the protocol from environments where the
proxy is not configured.
"""
from __future__ import annotations

from typing import Protocol


def test_llm_client_is_runtime_checkable_protocol() -> None:
    """``LLMClient`` is a ``typing.Protocol`` decorated with
    ``@runtime_checkable`` so callers can use ``isinstance`` checks.
    """
    from llm.client import LLMClient

    # Must be a Protocol subclass (typing machinery stores this flag).
    assert issubclass(LLMClient, Protocol)  # type: ignore[arg-type]

    # ``runtime_checkable`` sets ``_is_runtime_protocol`` on the class.
    assert getattr(LLMClient, "_is_runtime_protocol", False) is True


def test_llm_client_exposes_chat_method() -> None:
    """The protocol advertises a ``chat`` callable — the only operation
    Wave 2 needs when swapping ``runtime._make_client`` over.
    """
    from llm.client import LLMClient

    assert hasattr(LLMClient, "chat")
    # ``chat`` must be defined on the protocol itself, not inherited.
    assert "chat" in LLMClient.__dict__


def test_litellm_client_structurally_satisfies_protocol() -> None:
    """``LiteLLMClient`` must pass the runtime ``isinstance`` check.

    We do not actually call ``.chat`` here — we only verify the
    structural match so swapping concrete backends is a no-op for
    callers typed against :class:`LLMClient`.
    """
    from llm.client import LLMClient
    from llm.litellm import LiteLLMClient

    client = LiteLLMClient.__new__(LiteLLMClient)  # skip __init__ side-effects
    assert isinstance(client, LLMClient)


def test_package_reexports() -> None:
    """The ``llm`` package re-exports both the protocol and the default
    implementation so callers do not need to reach into submodules.
    """
    import llm

    assert hasattr(llm, "LLMClient")
    assert hasattr(llm, "LiteLLMClient")
