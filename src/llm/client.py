"""``LLMClient`` protocol — the single seam between CLAW modules and
OpenAI-compatible chat-completions endpoints.

The protocol is intentionally minimal: it covers exactly the surface
``red_team.agent.runtime`` and the blue-team scanners already use
against the LiteLLM proxy.  Return values are typed as ``Any`` because
the existing call sites pattern-match on the concrete OpenAI response
schema (``choices[0].message.tool_calls`` etc.); hiding that behind a
structural type would force a much larger refactor than Wave 1 intends.

Adding new provider backends (Anthropic direct, local llama.cpp, …)
means implementing this protocol in a sibling module — see
``llm/README.md`` for the expansion recipe.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Thin abstraction over an OpenAI-compatible chat-completions endpoint."""

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
        """Return the raw OpenAI ``ChatCompletion`` response object.

        The concrete return type is left as ``Any`` because the callers
        already pattern-match on the OpenAI client's response schema
        (``choices[0].message.tool_calls`` etc.).
        """
        ...


__all__ = ["LLMClient"]
