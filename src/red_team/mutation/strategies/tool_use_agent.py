"""Tool-use agent strategy — wraps :func:`red_team.agent.runtime.agent_run`.

The tool-use agent is our headline red-team driver: an OpenAI
function-calling loop against the LiteLLM proxy.  This thin adapter
exposes it as a :class:`MutationStrategy` so evolution stages and
campaign runners only ever depend on the L2 mutation interface.

The adapter is deliberately dumb — it does no policy work of its own.
Every decision (budget, tool-call limit, target model, ablation knobs)
is carried through ``MutationContext``; the returned ``AgentRunResult``
is faithfully re-packaged as a :class:`MutationResult`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from red_team.agent.runtime import agent_run
from red_team.model_defaults import default_mutator_model
from red_team.mutation.base import MutationContext, MutationResult


@dataclass(frozen=True)
class ToolUseAgentStrategy:
    """Adapter that drives the tool-use LLM loop for one mutation round."""

    name: str = "tool_use_agent"

    def mutate(self, ctx: MutationContext) -> MutationResult:
        """Run one tool-use agent round and return the mutation result.

        ``ctx.extra`` is consulted for optional pass-through knobs
        (``ablation``, ``attack_log_path``, ``round_index``,
        ``mutator_model``, ``temperature``, ``memory_enabled``) so
        callers can steer the underlying runtime without widening
        :class:`MutationContext` itself.
        """
        extra: dict[str, Any] = dict(ctx.extra or {})
        call_kwargs: dict[str, Any] = {
            "memory_enabled": extra.get("memory_enabled", True),
            "max_tool_calls": ctx.max_tool_calls,
            "budget_usd": ctx.budget_usd,
            "mutator_model": extra.get("mutator_model", default_mutator_model()),
            "temperature": extra.get("temperature", 0.0),
            "attack_log_path": extra.get("attack_log_path"),
            "round_index": extra.get("round_index", 0),
            "ablation": extra.get("ablation", {}),
        }

        result = agent_run(
            ctx.seed,
            list(ctx.findings),
            ctx.repo,
            ctx.search_space,
            **call_kwargs,
        )

        return MutationResult(
            variant=result.mutation,
            tool_chain=tuple(
                step["tool"] if isinstance(step, dict) else str(step)
                for step in result.tool_chain
            ),
            stop_reason=result.stop_reason,
            cost_usd=result.cost_usd,
            backend=self.name,
            extra={
                "rationale": result.rationale,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "refusal": result.refusal,
            },
        )


__all__ = ["ToolUseAgentStrategy"]
