"""Red-team payload primitives.

Holds the catalog of reusable attack binaries the red-team agent can
drop into ``variant.c.scripts`` via the ``get_binary`` tool (see
:mod:`red_team.agent.tools`).  Each binary covers one or more of the
paper's 7 attack objectives; collectively they let the agent attack at
the code layer rather than relying solely on ``d`` (SKILL.md) rewrites.
"""
from __future__ import annotations

from red_team.payloads.binary_catalog import (
    BINARY_CATALOG,
    BinaryBuildError,
    BinaryNotFoundError,
    BinarySpec,
    build_binary,
    build_root,
    get_binary,
    list_binary_specs,
    summarize_catalog,
)


def list_attack_types() -> list[str]:
    """Return the sorted list of ``attack_type`` keys backed by a
    doc-layer poison payload.

    The list is derived from the doc-plugin ``REGISTRY`` so it stays in
    lock-step with whichever payloads are actually available.  The
    historical ``red_team.skill_poisoner.list_attack_types`` re-exports
    this function unchanged.
    """
    from red_team.payloads.doc import REGISTRY

    return sorted(REGISTRY.keys())


__all__ = [
    "BINARY_CATALOG",
    "BinaryBuildError",
    "BinaryNotFoundError",
    "BinarySpec",
    "build_binary",
    "build_root",
    "get_binary",
    "list_attack_types",
    "list_binary_specs",
    "summarize_catalog",
]
