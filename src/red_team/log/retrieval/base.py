"""Retriever protocol — pluggable backend interface for attack-log search.

A retriever takes a textual ``query`` and a ``rows`` list (the parsed
JSONL corpus as dicts) and returns up to ``k`` rows ranked by relevance.
Rows are plain dicts — not DTOs — because the retriever must work on
the on-disk shape without forcing every caller to build an
:class:`~red_team.log.schema.AttackLogRow`.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Minimal search interface for attack-log memory."""

    def search(
        self,
        query: str,
        rows: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Return up to ``k`` rows most relevant to ``query``.

        Implementations must treat ``rows`` as read-only and tolerate
        an empty corpus (return ``[]`` in that case).
        """
        ...


__all__ = ["Retriever"]
