"""``read_attack_log`` tool — BM25 search over the current run's log.

The heavy lifting happens in :mod:`red_team.log.attack_log`.
We keep the plugin thin because the log backend may evolve: lazy import
means the runtime degrades to ``{"error": "attack_log unavailable"}``
instead of crashing when the backend is absent or broken.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_attack_log",
        "description": (
            "BM25 search over the agent's own historical attack-log "
            "entries for this run. Use this to reuse past successful "
            "evasion strategies and avoid repeating failures. Returns "
            "up to k summarized rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text query, e.g. 'autonomy_encouragement "
                        "bypassed' or 'rm -rf retry'."
                    ),
                },
                "k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max number of rows to return.",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass(frozen=True)
class ReadAttackLogTool:
    """Plugin implementation of ``read_attack_log``.

    ``dispatch`` takes ``context`` as the attack-log path (``str`` /
    ``Path`` / ``None``).  ``None`` means round 0 / memory disabled, in
    which case we return an empty ``hits`` list rather than an error so
    the agent loop can continue.
    """

    name: str = "read_attack_log"
    terminal: bool = False
    schema: dict = field(default_factory=lambda: _SCHEMA)

    def dispatch(
        self,
        args: dict,
        context: Path | str | None = None,
    ) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            return {"error": "query is required (string)"}
        k = args.get("k", 5)
        if not isinstance(k, int) or k <= 0:
            return {"error": "k must be a positive integer"}

        if context is None:
            # No log yet (round 0 or memory_enabled=False).  Empty hit
            # list lets the agent proceed instead of surfacing an error.
            return {"query": query, "k": k, "hits": []}

        try:
            # Lazy import so this tool can load independently of the log backend.
            from red_team.agent.attack_log import search  # type: ignore
        except ImportError:
            return {"error": "attack_log unavailable"}

        try:
            hits = search(str(context), query, k=k)
        except Exception as exc:  # search may raise on malformed logs
            return {"error": f"attack_log search failed: {exc}"}

        # Defensively coerce hits into a JSON-safe list of dicts.
        safe_hits: list[dict[str, Any]] = []
        for h in hits or []:
            if isinstance(h, dict):
                safe_hits.append(h)
        return {"query": query, "k": k, "hits": safe_hits}
