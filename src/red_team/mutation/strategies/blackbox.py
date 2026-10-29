"""Blackbox-rewrite baseline wrapped as a :class:`MutationStrategy`.

Adapts :class:`red_team.baselines.blackbox_rewrite.BlackboxRewrite` to
the L2 mutation interface.  The underlying baseline is stateful (it
keeps a ring buffer of ``(round_index, bypass)`` tuples as scalar
feedback); this adapter constructs a fresh baseline on each call.
When the driver needs persistent scalar memory across rounds it should
supply its own instance via ``ctx.extra["baseline"]`` — the adapter
uses it verbatim instead of building a new one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from red_team.baselines.blackbox_rewrite import BlackboxRewrite
from red_team.mutation.base import MutationContext, MutationResult


@dataclass(frozen=True)
class BlackboxStrategy:
    """Adapter around :class:`BlackboxRewrite` — scalar-feedback rewrite."""

    name: str = "blackbox"

    def mutate(self, ctx: MutationContext) -> MutationResult:
        extra: dict[str, Any] = dict(ctx.extra or {})

        # Allow caller-supplied baseline instance to preserve scalar memory
        # across rounds; otherwise construct a fresh one.
        baseline = extra.get("baseline")
        if baseline is None:
            ctor_kwargs: dict[str, Any] = {}
            if "memory_size" in extra:
                ctor_kwargs["memory_size"] = extra["memory_size"]
            if "initial_bypass" in extra:
                ctor_kwargs["initial_bypass"] = extra["initial_bypass"]
            baseline = BlackboxRewrite(**ctor_kwargs)

        call_kwargs: dict[str, Any] = {
            "round_index": int(extra.get("round_index", 0)),
            "model": ctx.target_model,
        }
        if "litellm_url" in extra:
            call_kwargs["litellm_url"] = extra["litellm_url"]
        if "api_key" in extra:
            call_kwargs["api_key"] = extra["api_key"]

        result = baseline.agent_run(
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


__all__ = ["BlackboxStrategy"]
