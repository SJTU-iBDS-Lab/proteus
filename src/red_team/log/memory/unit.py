"""``MemoryUnit`` — single immutable record stored under
``outputs/evolution/<run_id>/memory.sqlite::memory_units``.

Run-scope (Medium plan §5.1): one ``memory.sqlite`` per phase-1 run;
memory does **not** cross run boundaries.  Inside the run, A-Mem
style :mod:`evolver` rewrites ``mechanism`` / ``exploited_assumption`` /
``preconditions`` / ``tags`` and bumps ``evolution_step``; the old
version is kept (``superseded_by`` points at the new one) so paper
supplementary can show the full evolution chain.

Field grouping:

* **Identity**            — deterministic across re-extraction
* **Evolution**           — ``evolution_step`` + ``superseded_by``
* **Structural motif**    — pulled from SkillVariant, never edited
* **Semantic (evolvable)**— rewritten by evolver in place via new rows
* **Outcome (facts)**     — never edited
* **Category**            — closed-form ``positive`` / ``chokepoint``
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from core.schemas.attack_objective import AttackObjective, Channel, Topology

# Allowed string vocabularies; centralised so __post_init__ checks stay
# in sync with the SQLite schema in store.py.
_CATEGORIES: frozenset[str] = frozenset({"positive", "chokepoint"})
_SOURCES: frozenset[str] = frozenset({"seed", "harvested"})


@dataclass(frozen=True, slots=True)
class MemoryUnit:
    """Immutable record of one memory unit.

    ``unit_id`` is deterministic:
    ``sha1(variant_hash + run_id + round + extractor_version + evolution_step)``
    (see plan §5.2).  Two units with the same source HarvestRow but
    different ``evolution_step`` therefore have **different**
    ``unit_id`` and coexist in the store; ``superseded_by`` chains
    them into an evolution lineage.
    """

    # —— Identity ——
    unit_id: str
    source: Literal["seed", "harvested"]
    source_run_id: str | None
    source_seed_id: str | None
    variant_hash: str | None
    extractor_version: str
    authored_by: str
    created_at: str

    # —— Evolution ——
    evolution_step: int             # 0 = freshly extracted; ≥1 = nth evolve
    superseded_by: str | None       # unit_id of the next-step version (None = latest)

    # —— Structural motif (never edited) ——
    o: AttackObjective
    tau: Topology
    kappa: tuple[Channel, ...]
    chain: tuple[str, ...]

    # —— Semantic (evolver may rewrite via new rows) ——
    technique_name: str
    mechanism: str
    exploited_assumption: str
    preconditions: tuple[str, ...]
    tags: tuple[str, ...]

    # —— Outcome (facts, never edited) ——
    bypass: bool
    lethal: bool
    target_model: str
    blue_reason_codes: tuple[str, ...]

    # —— Category (closed-form on bypass × lethal) ——
    category: Literal["positive", "chokepoint"]

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("MemoryUnit.unit_id must be a non-empty string")
        if self.source not in _SOURCES:
            raise ValueError(
                f"MemoryUnit.source={self.source!r} not in {sorted(_SOURCES)}"
            )
        if self.category not in _CATEGORIES:
            raise ValueError(
                f"MemoryUnit.category={self.category!r} not in {sorted(_CATEGORIES)}"
            )
        if not isinstance(self.evolution_step, int) or self.evolution_step < 0:
            raise ValueError(
                f"MemoryUnit.evolution_step={self.evolution_step!r} must be non-negative int"
            )

    # ------------------------------------------------------------------ #
    # Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (enums → ``.value``, tuples → lists)."""
        d = asdict(self)
        d["o"] = self.o.value
        d["tau"] = self.tau.value
        d["kappa"] = [c.value for c in self.kappa]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryUnit":
        return cls(
            unit_id=d["unit_id"],
            source=d["source"],
            source_run_id=d.get("source_run_id"),
            source_seed_id=d.get("source_seed_id"),
            variant_hash=d.get("variant_hash"),
            extractor_version=d["extractor_version"],
            authored_by=d["authored_by"],
            created_at=d["created_at"],
            evolution_step=int(d.get("evolution_step", 0)),
            superseded_by=d.get("superseded_by"),
            o=AttackObjective(d["o"]),
            tau=Topology(d["tau"]),
            kappa=tuple(Channel(v) for v in d.get("kappa", ())),
            chain=tuple(d.get("chain", ())),
            technique_name=d["technique_name"],
            mechanism=d["mechanism"],
            exploited_assumption=d["exploited_assumption"],
            preconditions=tuple(d.get("preconditions", ())),
            tags=tuple(d.get("tags", ())),
            bypass=bool(d["bypass"]),
            lethal=bool(d["lethal"]),
            target_model=d["target_model"],
            blue_reason_codes=tuple(d.get("blue_reason_codes", ())),
            category=d["category"],
        )

    def with_evolution(
        self,
        *,
        new_unit_id: str,
        new_step: int,
        mechanism: str,
        exploited_assumption: str,
        preconditions: tuple[str, ...],
        tags: tuple[str, ...],
        new_authored_by: str,
        new_created_at: str,
    ) -> "MemoryUnit":
        """Return a new MemoryUnit at evolution_step+1 with rewritten semantic fields.

        Facts (``technique_name`` / ``bypass`` / ``lethal`` /
        ``blue_reason_codes`` / ``o`` / ``tau`` / ``kappa`` / ``chain`` /
        ``target_model``) are carried verbatim — evolver must NEVER
        rewrite those (plan §6.4).  ``superseded_by`` of the new unit
        is ``None`` (it is the new latest).
        """
        return MemoryUnit(
            unit_id=new_unit_id,
            source=self.source,
            source_run_id=self.source_run_id,
            source_seed_id=self.source_seed_id,
            variant_hash=self.variant_hash,
            extractor_version=self.extractor_version,
            authored_by=new_authored_by,
            created_at=new_created_at,
            evolution_step=new_step,
            superseded_by=None,
            o=self.o, tau=self.tau, kappa=self.kappa, chain=self.chain,
            technique_name=self.technique_name,
            mechanism=mechanism,
            exploited_assumption=exploited_assumption,
            preconditions=preconditions,
            tags=tags,
            bypass=self.bypass,
            lethal=self.lethal,
            target_model=self.target_model,
            blue_reason_codes=self.blue_reason_codes,
            category=self.category,
        )


__all__ = ["MemoryUnit"]
