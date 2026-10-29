"""Rendered attack payload DTO.

A :class:`PayloadArtifact` is one concrete payload produced by the red
team's mutation layer (``doc`` text appended to a ``SKILL.md``, or the
body of a ``bin`` bash wrapper). Previously these were threaded around
as bare ``dict`` payloads inside ``skill_poisoner.py`` and
``binary_catalog.py``; the new ``red_team/payloads/`` module emits this
DTO so downstream consumers (sandbox runner, attack log, blue-team
reproducer) have a single shape to depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attack_objective import AttackObjective


_VALID_LAYERS = ("doc", "bin")


@dataclass(frozen=True, slots=True)
class PayloadArtifact:
    """One rendered attack payload, either at the doc or binary layer.

    Attributes:
        name: Stable payload id, e.g. ``"data_exfil_bash"``. Must be
            non-empty; used as cache key and log tag.
        layer: ``"doc"`` when ``content`` is text to append to a skill
            ``SKILL.md``; ``"bin"`` when ``content`` is the source of
            an executable bash wrapper.
        objective: The attack objective this payload was generated for.
        content: The rendered payload body. UTF-8 text in both layers.
        metadata: Free-form extras — e.g. ``{"binary_hash": "..."}`` for
            bin-layer artifacts or ``{"render_style": "append"}`` for
            doc-layer. Kept as a plain ``dict`` so the DTO stays
            serialisable; callers are responsible for not mutating it
            after construction.
    """

    name: str
    layer: str
    objective: AttackObjective
    content: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.layer not in _VALID_LAYERS:
            raise ValueError(
                f"invalid layer {self.layer!r}; must be one of {_VALID_LAYERS}"
            )
        if not self.name:
            raise ValueError("name must be non-empty")


__all__ = ["PayloadArtifact"]
