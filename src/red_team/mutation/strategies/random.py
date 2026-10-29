"""Random-mutation baseline wrapped as a :class:`MutationStrategy`.

Adapts :class:`red_team.baselines.random_mutation.RandomMutation` to the
L2 mutation interface.  The underlying baseline's ``_build_candidate``
branches on the positional level codes ``S1`` / ``S2`` / ``S3``; this
adapter translates the semantic ``MutationContext.search_space`` into
that legacy vocabulary so the baseline stays untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from red_team.baselines.random_mutation import RandomMutation
from red_team.mutation.base import MutationContext, MutationResult


# Semantic (L2 vocabulary) → legacy positional codes (baseline vocabulary).
# Must exactly mirror :data:`red_team.evolution.search_space._LEGACY_LEVELS`
# inverse; we keep a local copy so this adapter has zero cross-module
# coupling on evolution internals.
_SEMANTIC_TO_LEGACY: dict[str, str] = {
    "evasion": "S1",
    "path_expansion": "S2",
    "surface_expansion": "S3",
}


@dataclass(frozen=True)
class RandomStrategy:
    """Adapter around :class:`RandomMutation` — LLM-free random splicing."""

    name: str = "random"

    def mutate(self, ctx: MutationContext) -> MutationResult:
        baseline = RandomMutation()
        extra: dict[str, Any] = dict(ctx.extra or {})

        call_kwargs: dict[str, Any] = {}
        if "rng_seed" in extra:
            call_kwargs["rng_seed"] = extra["rng_seed"]

        level = _SEMANTIC_TO_LEGACY[ctx.search_space]
        result = baseline.agent_run(
            ctx.seed,
            list(ctx.findings),
            ctx.repo,
            level,
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
                "refusal": result.refusal,
            },
        )


__all__ = ["RandomStrategy"]
