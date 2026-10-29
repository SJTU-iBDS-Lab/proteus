"""Runtime configuration for a single evolution run.

Separates high-level experiment flags (ablation conditions like
disable_expansion, tool_use_off) from the concrete parameters that
drive execution.  Internal code reads only RunConfig; the ablation
dict is translated at the run_evolution() entry point via
RunConfig.from_ablation().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_VALID_RED_STRATEGIES = {
    "llm",
    "random_baseline",
    "llm_zeroshot",
    "blackbox_rewrite",
    "pyrit_standalone",
    "skill_inject_bo5",
    "skillject_trace",
}


def _resolve_agent_strategy(a: dict[str, Any]) -> str:
    """Resolve the red-team strategy from the ablation dict.

    Two routes are supported:
      - the legacy ``tool_use_off: true`` flag  → "random_baseline"
      - an explicit ``red_strategy: <name>`` key, accepting any value in
        :data:`_VALID_RED_STRATEGIES`

    ``red_strategy`` takes precedence when both are set.  Unknown values
    raise ``ValueError`` so typos surface immediately.
    """
    explicit = a.get("red_strategy")
    if explicit is not None:
        if explicit not in _VALID_RED_STRATEGIES:
            raise ValueError(
                f"unknown red_strategy={explicit!r}; "
                f"expected one of {sorted(_VALID_RED_STRATEGIES)}"
            )
        return explicit
    if a.get("tool_use_off"):
        return "random_baseline"
    return "llm"


@dataclass
class RunConfig:
    """Concrete runtime parameters for one evolution run.

    All internal modules receive a RunConfig instance.  No internal
    code reads the raw ablation dict after run_evolution() returns
    from from_ablation().

    Defaults match the paper's baseline (no ablations active, memory
    off, both phases, free_mutation search space).
    """

    # Phase control
    phase: str = "both"  # "both" | "main_only" | "expansion_only"
    start_variant_path: str | None = None  # phase-2 start variant path
    search_space: str = "free_mutation"  # "free_mutation" | "evasion" | "path_expansion" | "surface_expansion".

    # Agent strategy
    # one of: "llm" | "random_baseline" | "llm_zeroshot" |
    # "blackbox_rewrite" | "pyrit_standalone" | "skill_inject_bo5" |
    # "skillject_trace"
    # "llm" = full self-evolving agent (default).
    # "random_baseline"   → A4 (tool_use_off) / Random Mutation baseline
    # "llm_zeroshot"      → T3 baseline (one LLM call, no tools/mem/findings)
    # "blackbox_rewrite"  → scalar-feedback rewrite baseline (no findings)
    agent_strategy: str = "llm"
    disabled_tool: str | None = None  # single tool name to disable

    # Expansion control
    expansion_enabled: bool = True
    family_enumeration_enabled: bool = True
    path_expand_exhaustive: bool = False

    # Mutation constraints
    topology_constraint: str | None = None  # None | "single"
    channel_edit_allowed: bool = True

    # Feedback to mutator (full mode for us and scalar for black box ablation)
    feedback_mode: str = "full"  # "full" | "scalar"

    # Audit gating protocol.
    # "upfront" preserves the legacy parent-audit ordering.
    # "lethal" audits a candidate only after the sandbox oracle marks that
    # same candidate lethal, avoiding both unnecessary AIG calls and
    # parent/child outcome mismatch.
    audit_gate: str = "upfront"  # "upfront" | "lethal"

    # Memory subsystem
    memory_enabled: bool = True
    memory_mode: str = "off"  # "off" | "write" | "read_write"
    memory_extractor_model: str = "deepseek-v4-pro"
    memory_evolver_model: str = "deepseek-v4-pro"
    shared_memory_db: str = ""
    cross_seed_retrieval: bool = False
    read_snapshot_rowid: int = 0

    # Scoring / search hints
    axis_diversity_strength: float | None = None
    family_score_weights: tuple[float, float, float] | None = field(default=None)
    family_hint: str | None = None
    s3_hint: str | None = None

    @classmethod
    def from_ablation(cls, ablation: dict[str, Any] | None) -> "RunConfig":
        """Translate a high-level ablation dict into a RunConfig.

        High-level flags (disable_expansion, tool_use_off, etc.) are
        mapped to their corresponding RunConfig fields.  Runtime-config
        keys (phase, memory_mode, shared_memory_db, etc.) are copied
        directly.  Unknown keys are silently ignored.
        """
        a = ablation or {}
        memory_enabled = a.get("memory_enabled", True)
        if memory_enabled is None:
            memory_enabled = True
        return cls(
            phase=str(a.get("phase", "both")).lower(),
            start_variant_path=a.get("start_variant_path") or None,
            search_space=str(a.get("search_space") or "free_mutation"),
            agent_strategy=_resolve_agent_strategy(a),
            disabled_tool=a.get("disable_tool") or None,
            expansion_enabled=not bool(a.get("disable_expansion")),
            family_enumeration_enabled=not bool(a.get("disable_family_enumeration")),
            path_expand_exhaustive=bool(a.get("path_expand_exhaustive")),
            topology_constraint="single"
            if a.get("force_topology") == "single"
            else None,
            channel_edit_allowed=not bool(a.get("forbid_channel_edit")),
            feedback_mode="scalar" if a.get("feedback") == "scalar" else "full",
            audit_gate=str(a.get("audit_gate", "upfront")).lower(),
            memory_enabled=bool(memory_enabled),
            memory_mode=str(a.get("memory_mode", "off")),
            memory_extractor_model=str(
                a.get("memory_extractor_model", "deepseek-v4-pro")
            ),
            memory_evolver_model=str(a.get("memory_evolver_model", "deepseek-v4-pro")),
            shared_memory_db=str(a.get("shared_memory_db", "") or ""),
            cross_seed_retrieval=bool(a.get("cross_seed_retrieval", False)),
            read_snapshot_rowid=int(a.get("read_snapshot_rowid", 0) or 0),
            axis_diversity_strength=a.get("axis_diversity_strength"),
            family_score_weights=a.get("family_score_weights"),
            family_hint=a.get("family_hint"),
            s3_hint=a.get("s3_hint"),
        )


__all__ = ["RunConfig"]
