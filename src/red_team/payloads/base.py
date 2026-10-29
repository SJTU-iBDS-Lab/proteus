"""Payload plugin protocol.

Every payload — whether a ``doc`` appendix for a skill's ``SKILL.md`` or
a ``bin`` bash wrapper carrying a base64-embedded ELF — is a pure object
that renders itself into a :class:`~core.schemas.PayloadArtifact`. The
protocol is intentionally tiny so ``registry.REGISTRY`` can store doc
and bin plugins side by side.

Plugins must be pure: ``render`` is a function of its args, it must not
touch the filesystem, network, or any process-level mutable state. File
I/O (staging into ``variant.c.scripts`` or a skill directory) is the
caller's job.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.schemas import AttackObjective, PayloadArtifact


@runtime_checkable
class Payload(Protocol):
    """Single payload plugin — renders to a :class:`PayloadArtifact`.

    Attributes:
        name: Stable plugin id (matches the key under which it is
            registered in ``REGISTRY``).
        layer: ``"doc"`` for SKILL.md-text payloads, ``"bin"`` for
            compiled-binary bash wrappers.
        objective: Attack objective this payload advances.
    """

    name: str
    layer: str
    objective: AttackObjective

    def render(self, **kwargs) -> PayloadArtifact:
        """Render the payload into a :class:`PayloadArtifact`.

        ``kwargs`` are plugin-specific: doc plugins usually ignore them;
        bin plugins forward them as env-var overrides to the underlying
        binary wrapper.
        """
        ...


__all__ = ["Payload"]
