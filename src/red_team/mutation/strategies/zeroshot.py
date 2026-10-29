"""LLM zero-shot baseline wrapped as a :class:`MutationStrategy`.

Adapts :class:`red_team.baselines.llm_zeroshot.LLMZeroShot` — the T3 row
in the paper tables — to the L2 mutation interface.  ``LLMZeroShot``
itself ignores ``findings`` / ``repo`` / ``search_space`` (zero-shot
means no context), but we still forward them verbatim so the baseline
contract stays uniform across strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from red_team.baselines.llm_zeroshot import LLMZeroShot
from red_team.mutation.base import MutationContext, MutationResult


@dataclass(frozen=True)
class ZeroshotStrategy:
    """Adapter around :class:`LLMZeroShot` — single LLM call, no tools."""

    name: str = "zeroshot"

    def mutate(self, ctx: MutationContext) -> MutationResult:
        baseline = LLMZeroShot()
        extra: dict[str, Any] = dict(ctx.extra or {})

        call_kwargs: dict[str, Any] = {"model": ctx.target_model}
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


__all__ = ["ZeroshotStrategy"]
