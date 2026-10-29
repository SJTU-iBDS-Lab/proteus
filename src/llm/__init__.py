"""Top-level ``llm`` package — the one and only entry point for every
LLM call in the CLAW codebase (red team + blue team).

Re-exports the :class:`LLMClient` protocol (so callers can type against
the abstraction) and the default :class:`LiteLLMClient` implementation
(so the common case is one import).
"""
from llm.client import LLMClient
from llm.litellm import LiteLLMClient

__all__ = ["LLMClient", "LiteLLMClient"]
