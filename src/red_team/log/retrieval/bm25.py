"""BM25 retriever — lexical ranking over attack-log rows.

The retriever builds a fresh BM25 index on each call. Callers that need
cross-call caching (keyed by ``(path, file_mtime)``) should cache the
corpus at the I/O layer in :mod:`red_team.log.attack_log`, not here.
"""
from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_SPLIT_RE.sub(" ", text.lower()).split()


def _row_to_document(row: dict[str, Any]) -> str:
    """Concatenate the text fields used for BM25 scoring.

    Kept in sync with :func:`red_team.log.attack_log.row_to_document`.
    """
    reason = str(row.get("reason") or "")
    codes = row.get("reason_codes_evaded") or []
    codes_str = " ".join(str(c) for c in codes)
    tool_chain = row.get("tool_chain_summary") or []
    tools_str = " ".join(
        str(tc.get("tool", ""))
        for tc in tool_chain
        if isinstance(tc, dict)
    )
    return f"{reason} {codes_str} {tools_str}"


class BM25Retriever:
    """Default lexical retriever using Okapi BM25."""

    def search(
        self,
        query: str,
        rows: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        if not rows or k <= 0:
            return []

        tokenized = [_tokenize(_row_to_document(r)) for r in rows]
        # BM25Okapi requires non-empty documents; substitute a single
        # empty-string token so rows with no text still occupy their slot.
        safe_tokens = [t if t else [""] for t in tokenized]
        index = BM25Okapi(safe_tokens)

        query_tokens = _tokenize(query) or [""]
        scores = index.get_scores(query_tokens)

        order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
        top = order[:k]
        return [rows[i] for i in top]


__all__ = ["BM25Retriever"]
