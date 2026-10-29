"""A7 — ``force_topology='single'``.

Maps to run_config.topology_constraint = "single".
"""
from __future__ import annotations

from dataclasses import replace

from core.schemas import SkillVariant, Topology
from red_team.campaign.run_config import RunConfig


def maybe_force_single_topology(
    variant: SkillVariant, run_config: RunConfig
) -> tuple[SkillVariant, bool, list[str] | None, list[str] | None]:
    """Return ``(variant_for_sandbox, forced, chain_original, chain_used)``.

    When the flag is absent the variant is returned untouched with
    ``forced=False``.
    """
    if run_config.topology_constraint != "single":
        return variant, False, None, None

    chain_original = list(variant.chain)
    if len(chain_original) <= 1 and variant.tau == Topology.SINGLE:
        return variant, True, chain_original, list(chain_original)

    new_chain = chain_original[:1] if chain_original else []
    forced = replace(
        variant,
        tau=Topology.SINGLE,
        chain=new_chain,
        c=dict(variant.c) if variant.c else {},
        kappa=list(variant.kappa),
        metadata=dict(variant.metadata or {}),
    )
    return forced, True, chain_original, list(new_chain)
