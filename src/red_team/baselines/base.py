"""Shared Protocol + helpers for red-team baselines (paper §5.1).

This module defines the contract that every red-team method (Random /
LLM zero-shot / Blackbox rewrite / Ours) must satisfy, plus a few helpers
shared across them:

- ``COST_PER_1K``         : USD price table per 1k input/output tokens.
- ``estimate_cost``       : tokens × model → USD, with a conservative
                            1.3x multiplier (Part 2 spec) and fallback to
                            deepseek-v4-flash rates for unknown models.
- ``make_variant_from_new_d``
                          : clone a parent ``SkillVariant`` keeping
                            (o, τ, c, κ, chain, poison_index) and only
                            swapping out the documentation text ``d``.
- ``sample_payload_fragment``
                          : pick a random line from one of
                            ``skill_poisoner.POISON_PAYLOADS``, for use by
                            the Random baseline (T2).

The ``RedTeamBaseline`` Protocol matches the Part 2 ``agent_run`` signature
so the Part 3 driver can dispatch to any baseline via ``--red-team {...}``.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.schemas import PaperFinding, SkillVariant
from red_team.skill_poisoner import POISON_PAYLOADS

if TYPE_CHECKING:  # pragma: no cover - imports only used for typing
    from core.models import RepositorySlice


# --------------------------------------------------------------- agent result

@dataclass(slots=True)
class AgentRunResult:
    """Uniform return type for every red-team agent / baseline ``agent_run``.

    Uniform return type shared between the LLM agent runtime and all
    LLM-free baselines so every caller handles one contract.

    Fields
    ------
    mutation : SkillVariant | None
        The resulting variant, or ``None`` if the attempt failed
        (e.g. the baseline could not satisfy the invariants).
    tool_chain : list[str]
        Ordered list of tool ids the agent invoked.  Baselines that do
        not call any LLM leave this empty.
    rationale : str
        Free-form description of what was done, surfaced in logs.
    cost_usd : float
        Estimated USD cost of any LLM calls made (0.0 for baselines).
    tokens_in : int
        Input tokens consumed (0 for baselines).
    tokens_out : int
        Output tokens produced (0 for baselines).
    refusal : bool
        True iff the underlying model refused to comply.
    stop_reason : str
        Short machine-readable reason for termination, e.g.
        ``"random"``, ``"invariant_violation"``, ``"refusal"``.
    """

    mutation: SkillVariant | None
    tool_chain: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    refusal: bool = False
    stop_reason: str = ""


# ---------------------------------------------------------------------- cost

# USD per 1_000 tokens, separated by input vs output. Keys are canonical
# lower-case model ids (see ``_normalize_model``).  Rates are conservative
# estimates — exact upstream pricing varies by provider/aggregator.
COST_PER_1K: dict[str, dict[str, float]] = {
    # Model ids exposed by the local LiteLLM proxy.
    "deepseek-v4-pro":   {"in": 0.001, "out": 0.002},   # DeepSeek native
    "deepseek-v4-flash": {"in": 0.0005, "out": 0.001},
    "deepseek-v3-0324":  {"in": 0.0007, "out": 0.0014},
    "glm-5":             {"in": 0.003, "out": 0.009},
    "gpt-5.4-mini":      {"in": 0.001, "out": 0.004},
    "gpt-5.1":           {"in": 0.005, "out": 0.020},
    "claude-sonnet-4-5": {"in": 0.003, "out": 0.015},
    "qwen3-coder":       {"in": 0.001, "out": 0.002},
    # Legacy entries (kept for back-compat with old harvest/log rows
    # that reference these model ids — cost estimation only).
    "gpt-5.4":           {"in": 0.015, "out": 0.060},
    "kimi-k2.5":         {"in": 0.008, "out": 0.024},
    # kimi-k2.6 (Moonshot reasoning model, 256K context).  Pricing per
    # platform.moonshot.cn (2026-04): ¥4 / M input, ¥16 / M output ≈
    # $0.0006 / 1k in, $0.0023 / 1k out at 7 CNY/USD.  Reasoning_tokens
    # are billed as output tokens.
    "kimi-k2.6":         {"in": 0.0006, "out": 0.0023},
    "moonshot-v1-128k":  {"in": 0.0017, "out": 0.0017},
    "moonshot-v1-32k":   {"in": 0.0009, "out": 0.0009},
    "claude-sonnet-4-6": {"in": 0.003, "out": 0.015},
    "minimax-m1":        {"in": 0.002, "out": 0.006},
}

# No fallback — unknown model id raises so a typo / forgotten default
# fails loud at the call site instead of silently mis-counting cost.

# Conservative multiplier per Part 2 spec — real wall-clock cost tends to
# exceed per-token estimates because of reasoning/tool tokens.
_COST_SAFETY_MARGIN = 1.3


def _normalize_model(model: str) -> str:
    """Lower-case + strip whitespace for robust lookups."""
    return (model or "").strip().lower()


def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    """Return the estimated USD cost of a single LLM call.

    Looks up per-1000-token rates in :data:`COST_PER_1K`, splits across
    input and output, and applies a 1.3x safety margin (Part 2 spec).

    **Unknown models raise** :class:`KeyError`.  No silent fallback —
    add the model id to ``COST_PER_1K`` (or fix the caller's typo)
    rather than letting cost estimates silently mis-count.
    """
    key = _normalize_model(model)
    if key not in COST_PER_1K:
        raise KeyError(
            f"unknown model id {model!r} for cost estimation; "
            f"add it to COST_PER_1K or fix the caller. "
            f"Known: {sorted(COST_PER_1K)}"
        )
    rates = COST_PER_1K[key]
    cost_in = (tokens_in / 1000.0) * rates["in"]
    cost_out = (tokens_out / 1000.0) * rates["out"]
    return (cost_in + cost_out) * _COST_SAFETY_MARGIN


# ------------------------------------------------------------------- variant

def make_variant_from_new_d(parent: SkillVariant, new_d: str) -> SkillVariant:
    """Clone ``parent`` but replace its SKILL.md body with ``new_d``.

    All of (o, τ, c, κ, chain, poison_index) are preserved. The metadata
    dict is shallow-copied and annotated with the parent's content hash
    so downstream bookkeeping can reconstruct the mutation lineage.
    """
    metadata = dict(parent.metadata) if parent.metadata else {}
    metadata["parent_variant_hash"] = parent.content_hash()
    return SkillVariant(
        o=parent.o,
        tau=parent.tau,
        # Deep-copy c so downstream mutators editing ``child.c["scripts"]``
        # in place do NOT alias and silently mutate ``parent.c`` too —
        # that aliasing makes the S2 "≥1 axis change" invariant compare
        # child.c == parent.c trivially True post-mutation.
        c=copy.deepcopy(parent.c) if parent.c else {},
        kappa=list(parent.kappa),
        d=new_d,
        poison_index=parent.poison_index,
        chain=list(parent.chain),
        metadata=metadata,
    )


# ------------------------------------------------------------------- payload

def sample_payload_fragment(rng: random.Random) -> str:
    """Return a random non-empty line from ``POISON_PAYLOADS``.

    Used by the Random baseline (T2) to splice plausible-looking
    attacker text into a mutated SKILL.md without invoking an LLM.
    """
    if not POISON_PAYLOADS:
        return ""
    key = rng.choice(sorted(POISON_PAYLOADS.keys()))
    body = POISON_PAYLOADS[key]
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return body.strip()
    return rng.choice(lines)


# ------------------------------------------------------------------ protocol

@runtime_checkable
class RedTeamBaseline(Protocol):
    """Protocol every red-team baseline must satisfy.

    The signature mirrors Part 2's ``agent_run`` so the Part 3 driver can
    dispatch to any baseline uniformly via ``--red-team {...}``.
    """

    name: str

    def agent_run(
        self,
        seed: SkillVariant,
        findings: list[PaperFinding],
        repo: "RepositorySlice",
        search_space: str,
        **kwargs: Any,
    ) -> "AgentRunResult": ...


__all__ = [
    "AgentRunResult",
    "COST_PER_1K",
    "RedTeamBaseline",
    "estimate_cost",
    "make_variant_from_new_d",
    "sample_payload_fragment",
]
