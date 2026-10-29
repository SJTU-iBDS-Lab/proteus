"""Unified payload registry — the single index for doc + bin plugins.

Downstream code (agent tools, attack log, evolution driver) resolves
payload plugins through this module so there is exactly one place that
knows about the doc / bin split. The registry is a plain dict — keys
are stable plugin names (matching the historical ``POISON_PAYLOADS`` /
``BINARY_CATALOG`` keys).

Doc and bin keys must be disjoint; a collision would silently drop one
entry because the merged dict takes the last value. The contract test
in ``tests/unit/red_team/payloads/test_registry.py`` enforces that.
"""
from __future__ import annotations

from core.schemas import AttackObjective
from red_team.payloads.base import Payload
from red_team.payloads.bin import REGISTRY as _BIN_REGISTRY
from red_team.payloads.doc import REGISTRY as _DOC_REGISTRY


REGISTRY: dict[str, Payload] = {**_DOC_REGISTRY, **_BIN_REGISTRY}


def get(name: str) -> Payload:
    """Resolve ``name`` to its plugin; raises ``KeyError`` on miss.

    Callers that want a friendlier error should catch ``KeyError`` and
    surface :data:`REGISTRY` keys.
    """
    return REGISTRY[name]


def by_objective(objective: AttackObjective) -> list[Payload]:
    """Return every plugin whose ``objective`` matches.

    Result is a fresh list — callers may mutate it freely.
    """
    return [p for p in REGISTRY.values() if p.objective is objective]


def by_layer(layer: str) -> list[Payload]:
    """Return every plugin in the given layer (``"doc"`` or ``"bin"``).

    Unknown layers return an empty list rather than raising, so callers
    can ask for both layers without branching.
    """
    return [p for p in REGISTRY.values() if p.layer == layer]


__all__ = ["REGISTRY", "get", "by_objective", "by_layer"]
