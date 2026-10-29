"""LLM-free random-mutation baseline for the red-team search.

This baseline mutates a parent :class:`SkillVariant` by splicing poison
payload fragments together with benign ClawHub SKILL.md snippets, under
the level-specific mutation mask enforced by
:class:`red_team.mutation.search_space.SearchSpace`.

Contract per paper §5.1 / Part 5:

* ``name = "random"`` — dispatched via ``--red-team random``.
* ``agent_run`` returns :class:`AgentRunResult` with ``tool_chain=[]``,
  ``cost_usd=0.0``, and all token counts zero.
* ``rng_seed is not None`` → deterministic output.  ``rng_seed is None``
  seeds from :func:`os.urandom`.
* The returned variant is guaranteed to pass
  ``SearchSpace(level).validate(parent, child)`` on success; we retry up
  to 5 times before giving up with ``stop_reason="invariant_violation"``.

Level behaviour:

* **S1** — mutate only ``d``.  Pick 1–3 operations from
  ``{prepend, append, replace-random-section}``.  Each op splices in a
  random line of a :mod:`red_team.skill_poisoner` payload and a random
  benign one-liner (from ClawHub ``SKILL.md`` files when available).
* **S2** — do the S1 mutation and (with P=0.5) also shove a payload line
  into a randomly chosen ``c["scripts"]`` entry or (if ``entry_script``
  is set) rename the entry script to a random plausible ``.sh``/``.py``
  name plus a payload body.  ``o`` / ``tau`` / ``chain`` / ``kappa`` are
  left untouched so S2's validator (which only forbids ``o`` / ``tau``
  changes) passes.
* **S3** — do the S1 mutation **and** rotate ``o`` to a random OTHER
  objective from :class:`AttackObjective`.  We keep ``chain``, ``kappa``
  and the existing ``c`` unchanged so
  :func:`red_team.mutation.search_space.shared_motif` still reports
  ``≥ 2`` axes (κ overlap + entry_script fallback).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

from core.schemas import AttackObjective, PaperFinding, SkillVariant
from red_team.baselines.base import (
    AgentRunResult,
    make_variant_from_new_d,
    sample_payload_fragment,
)
from red_team.mutation.search_space import InvariantViolation, SearchSpace
from red_team.skill_poisoner import POISON_PAYLOADS  # noqa: F401 (re-export docs)


# Legacy positional → semantic map, kept local so this module has zero
# coupling on evolution internals.  Mirrors
# :data:`red_team.mutation.search_space._LEGACY_LEVELS`; we duplicate it
# instead of importing so the baseline can normalize its ``level``
# argument quietly (no ``DeprecationWarning`` emitted from this module).
_LEGACY_LEVELS: dict[str, str] = {
    "S1": "evasion",
    "S2": "path_expansion",
    "S3": "surface_expansion",
}


def _normalize_level(level: str) -> str:
    """Translate legacy ``S1``/``S2``/``S3`` inputs to semantic names.

    Unlike :func:`red_team.mutation.search_space._normalize_level` we do
    not emit a ``DeprecationWarning`` — the baseline accepts both forms
    as part of its legacy contract, and the warning would surface to
    every caller even when they pass the semantic vocabulary through a
    wrapper.  Downstream ``SearchSpace(level)`` never sees the legacy
    string because this helper rewrites it first.
    """
    return _LEGACY_LEVELS.get(level, level)


# -------------------------------------------------------------- benign corpus

# Fallback one-liners used when the ClawHub SKILL.md tree is not present
# (e.g. in CI without ``data/skills/clawhub`` checked out).  All lines are
# kept short, ASCII-only, and innocuous so validators do not mistake them
# for attacker text.
_FALLBACK_BENIGN_LINES: tuple[str, ...] = (
    "This skill helps summarize long documents into concise bullet points.",
    "Use natural language queries to search through structured datasets.",
    "Provide the target filename and the assistant will format the output.",
    "Supports Markdown and plain-text inputs interchangeably.",
    "Returns structured JSON when called from other skills.",
)


def _clawhub_skill_md_files(root: Path) -> list[Path]:
    """Return every ``SKILL.md`` directly under ``root/*/``.  Missing → []."""
    if not root.is_dir():
        return []
    matches: list[Path] = []
    for entry in root.iterdir():
        candidate = entry / "SKILL.md"
        if candidate.is_file():
            matches.append(candidate)
    return matches


def _sample_benign_line(rng: random.Random) -> str:
    """Pick a non-empty line from ClawHub SKILL.md; fall back to hardcoded."""
    # Anchor the search relative to this repo so tests and runtime both hit
    # the same tree: ``<repo>/data/skills/clawhub/skills/*/SKILL.md``.
    repo_root = Path(__file__).resolve().parents[3]
    skill_root = repo_root / "data" / "skills" / "clawhub" / "skills"
    candidates = _clawhub_skill_md_files(skill_root)

    if candidates:
        path = rng.choice(candidates)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("---")
        ]
        if lines:
            return rng.choice(lines)

    return rng.choice(_FALLBACK_BENIGN_LINES)


# ------------------------------------------------------------------ d ops

def _op_prepend(d: str, fragment: str, benign: str) -> str:
    return f"{benign}\n{fragment}\n\n{d}"


def _op_append(d: str, fragment: str, benign: str) -> str:
    return f"{d}\n\n{fragment}\n{benign}\n"


def _op_replace_section(d: str, fragment: str, benign: str) -> str:
    """Replace the body of a random ``## `` section with payload fragment.

    If ``d`` has no ``## `` headers at all we degrade to append — the
    caller treats that as a legitimate op so we never return ``d``
    unchanged.
    """
    lines = d.splitlines()
    header_indices = [i for i, line in enumerate(lines) if line.startswith("## ")]
    if not header_indices:
        return _op_append(d, fragment, benign)

    # Seed-independent choice lives with the caller; we do not use our own
    # rng here so the operation can be composed with the shared rng upstream.
    # Pick the first header and rewrite its body up to the next ``## `` line.
    # (Deterministic; the randomness comes from *which* op is chosen.)
    start = header_indices[0]
    end = len(lines)
    for idx in header_indices[1:]:
        if idx > start:
            end = idx
            break

    new_body = [lines[start], "", fragment, benign, ""]
    return "\n".join(lines[:start] + new_body + lines[end:])


_D_OPS: dict[str, Any] = {
    "prepend": _op_prepend,
    "append": _op_append,
    "replace-random-section": _op_replace_section,
}


def _apply_d_ops(
    d: str, ops: Iterable[str], rng: random.Random
) -> tuple[str, list[str]]:
    """Apply the chosen ops in order and return ``(new_d, op_descriptions)``."""
    new_d = d
    descriptions: list[str] = []
    for op_name in ops:
        fragment = sample_payload_fragment(rng)
        benign = _sample_benign_line(rng)
        op_fn = _D_OPS[op_name]
        new_d = op_fn(new_d, fragment, benign)
        descriptions.append(op_name)
    return new_d, descriptions


# ------------------------------------------------------------------ c ops

_SCRIPT_NAME_CHOICES: tuple[str, ...] = (
    "run.sh",
    "install.sh",
    "bootstrap.sh",
    "setup.py",
    "main.py",
    "launcher.py",
)


def _maybe_mutate_c(
    child: SkillVariant, rng: random.Random, op_log: list[str]
) -> None:
    """Always mutate ``child.c`` for S2 path_expansion.

    The paper's §4.3 S2 demands attack-surface expansion — a child whose
    (c, κ, chain) are all identical to the parent is not an S2 mutation.
    This function guarantees c changes so ``_validate_path_expansion``
    accepts the result.  Prior versions flipped a coin (P=0.5) and
    frequently produced d-only S2 children that the post-Bug-#3 validator
    now rightly rejects.
    """
    fragment = sample_payload_fragment(rng)
    scripts = child.c.get("scripts")
    if isinstance(scripts, dict) and scripts:
        key = rng.choice(sorted(scripts.keys()))
        existing = str(scripts.get(key, ""))
        scripts[key] = f"{existing}\n# --- appended ---\n{fragment}\n"
        child.c["scripts"] = scripts
        op_log.append(f"script-append:{key}")
        return

    if child.c.get("entry_script") is not None:
        new_name = rng.choice(_SCRIPT_NAME_CHOICES)
        child.c["entry_script"] = new_name
        child.c.setdefault("scripts", {})[new_name] = f"#!/usr/bin/env bash\n{fragment}\n"
        op_log.append(f"entry-rename:{new_name}")
        return

    # Parent has no scripts and no entry_script — seed both so c has a
    # concrete structural delta.  This covers the "benign-skill seed"
    # case where upstream didn't populate c at all.
    new_name = rng.choice(_SCRIPT_NAME_CHOICES)
    child.c["entry_script"] = new_name
    child.c["scripts"] = {new_name: f"#!/usr/bin/env bash\n{fragment}\n"}
    op_log.append(f"entry-init:{new_name}")


# ------------------------------------------------------------------ o ops

def _pick_other_objective(
    current: AttackObjective, rng: random.Random
) -> AttackObjective:
    """Return a random ``AttackObjective`` that is not ``current``."""
    choices = [o for o in AttackObjective if o != current]
    return rng.choice(choices)


# ------------------------------------------------------------------ baseline

class RandomMutation:
    """LLM-free random-mutation baseline."""

    name: str = "random"

    def agent_run(
        self,
        seed: SkillVariant,
        findings: list[PaperFinding],
        repo: "Any",  # RepositorySlice at runtime; kept loose for baselines
        search_space: str,
        *,
        rng_seed: int | None = None,
        **_: Any,
    ) -> AgentRunResult:
        """Return an :class:`AgentRunResult` containing a random mutation.

        ``search_space`` selects the level — either the semantic name
        (``"evasion"`` / ``"path_expansion"`` / ``"surface_expansion"``)
        or the legacy positional code (``"S1"`` / ``"S2"`` / ``"S3"``);
        legacy inputs are normalized quietly so the baseline's internal
        ``SearchSpace`` construction does not emit a ``DeprecationWarning``.
        We delegate invariant enforcement to
        ``SearchSpace(level).validate`` and retry up to 5 times on
        :class:`InvariantViolation` before surfacing
        ``stop_reason="invariant_violation"``.
        """
        level = _normalize_level(search_space)
        rng = _build_rng(rng_seed)
        space = SearchSpace(level)

        last_error: str | None = None
        for attempt in range(5):
            candidate, rationale = self._build_candidate(seed, level, rng)
            try:
                space.validate(seed, candidate)
            except InvariantViolation as exc:
                last_error = str(exc)
                continue
            return AgentRunResult(
                mutation=candidate,
                tool_chain=[],
                rationale=rationale,
                cost_usd=0.0,
                tokens_in=0,
                tokens_out=0,
                refusal=False,
                stop_reason="random",
            )

        return AgentRunResult(
            mutation=None,
            tool_chain=[],
            rationale=f"random_invariant_failed: {last_error or 'unknown'}",
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
            refusal=False,
            stop_reason="invariant_violation",
        )

    # ---------------------------------------------------------- helpers

    def _build_candidate(
        self,
        parent: SkillVariant,
        level: str,
        rng: random.Random,
    ) -> tuple[SkillVariant, str]:
        """Produce one mutation candidate for ``level`` (no validation).

        ``level`` must already be the semantic vocabulary
        (``"evasion"`` / ``"path_expansion"`` / ``"surface_expansion"``);
        the :meth:`agent_run` entry point normalizes legacy aliases
        before calling here.
        """
        # evasion always mutates d; path_expansion and surface_expansion
        # build on that and then mutate more.
        n_ops = rng.randint(1, 3)
        op_names = [
            rng.choice(sorted(_D_OPS.keys())) for _ in range(n_ops)
        ]
        new_d, applied = _apply_d_ops(parent.d, op_names, rng)

        child = make_variant_from_new_d(parent, new_d)
        op_log: list[str] = list(applied)

        if level == "path_expansion":
            _maybe_mutate_c(child, rng, op_log)
        elif level == "surface_expansion":
            new_o = _pick_other_objective(parent.o, rng)
            child.o = new_o
            op_log.append(f"rotate-o:{new_o.value}")
            # Guard: shared_motif needs either κ overlap + file_access
            # overlap, or κ overlap + entry_script equality.  We already
            # keep κ and c intact in make_variant_from_new_d, so the
            # fallback heuristic (entry_script equality) fires when metadata
            # lacks tool_chain_prefix / file_access.  No extra work needed.

        rationale = f"{level}:{','.join(op_log)}"
        return child, rationale


# ----------------------------------------------------------------- rng utils

def _build_rng(seed: int | None) -> random.Random:
    if seed is not None:
        return random.Random(seed)
    # ``os.urandom(8)`` gives 64 bits of entropy; convert to int.
    return random.Random(int.from_bytes(os.urandom(8), "big"))


__all__ = ["RandomMutation"]
