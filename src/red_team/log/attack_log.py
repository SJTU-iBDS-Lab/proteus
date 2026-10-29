"""Attack log: append-only JSONL writer + BM25-backed retrieval.

This is the real implementation of the red-team attack log. Each line
of the log file is a JSON object describing a single red-team round;
the file lives at the path returned by
:func:`red_team.log.paths.attack_log_path`.

API
---
* :func:`append` — append a single entry (dict *or* :class:`AttackLogRow`)
  as one JSONL line.
* :func:`read_all` — parse the full file into a list of dicts.
* :func:`search` — BM25 search with optional ``seed_id`` / ``filter_bypassed``
  filters. Uses :class:`~red_team.log.retrieval.bm25.BM25Retriever` by default.
* :func:`row_to_document` — the text-blob assembler used by the BM25 index;
  exposed so downstream tooling can rebuild indices on the fly.

Legacy callers continue to import this surface through the
:mod:`red_team.agent.attack_log` shim, which simply re-exports these
symbols.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypedDict

from rank_bm25 import BM25Okapi

from red_team.log.retrieval.bm25 import BM25Retriever
from red_team.log.schema import AttackLogRow


__all__ = [
    "AttackLogEntry",
    "append",
    "read_all",
    "search",
    "row_to_document",
]


class AttackLogEntry(TypedDict, total=False):
    """Canonical schema for a single row of the attack log (legacy TypedDict).

    Kept for backward compatibility with :mod:`red_team.agent.attack_log`.
    New code should prefer :class:`red_team.log.schema.AttackLogRow`.
    """

    round: int
    seed_id: str
    variant_hash: str
    o: str
    tau: str
    bypassed: bool
    lethal: bool
    reason: str
    reason_codes_evaded: list[str]
    tool_chain_summary: list[dict[str, Any]]
    timestamp: str


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_]+")

# Cache keyed by (path_str, mtime_ns) → (rows, tokenized_corpus, BM25 index).
# Kept here (not in the retriever) because it is keyed on on-disk state,
# not on the abstract corpus.
_CORPUS_CACHE: dict[
    tuple[str, int],
    tuple[list[dict[str, Any]], list[list[str]], BM25Okapi],
] = {}


# --------------------------------------------------------------------------- #
# Token / document helpers — shared with the BM25 retriever.
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return _TOKEN_SPLIT_RE.sub(" ", text.lower()).split()


def row_to_document(row: dict[str, Any]) -> str:
    """Concatenate the text fields used for BM25 scoring.

    The blob is ``reason`` + ``reason_codes_evaded`` + the ``tool`` values
    from ``tool_chain_summary``. Missing fields are treated as empty so
    partially populated rows still index cleanly.
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


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def _entry_to_dict(entry: dict[str, Any] | AttackLogRow) -> dict[str, Any]:
    """Normalize the two accepted input shapes to a plain dict."""
    if isinstance(entry, AttackLogRow):
        d = asdict(entry)
        # ``tool_chain`` is a tuple on the DTO; JSON can only serialize lists.
        if isinstance(d.get("tool_chain"), tuple):
            d["tool_chain"] = list(d["tool_chain"])
        return d
    if is_dataclass(entry):  # pragma: no cover — defensive
        return asdict(entry)
    return dict(entry)


def append(path: Path | str, entry: dict[str, Any] | AttackLogRow) -> None:
    """Append ``entry`` as a single JSONL line to ``path``.

    - Creates the parent directory and file if missing.
    - Writes UTF-8, no pretty printing, one trailing newline per line.
    - Append-only: never overwrites existing content.
    - Accepts either a plain ``dict`` (legacy) or an
      :class:`~red_team.log.schema.AttackLogRow` DTO (new typed callers).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _entry_to_dict(entry)
    serialized = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.write("\n")


def read_all(path: Path | str) -> list[dict[str, Any]]:
    """Parse ``path`` into a list of dicts, skipping blank lines.

    Returns ``[]`` if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Cached corpus (path-keyed)
# --------------------------------------------------------------------------- #


def _get_corpus(
    path: Path,
) -> tuple[list[dict[str, Any]], list[list[str]], BM25Okapi] | None:
    """Return cached ``(rows, tokenized_corpus, index)`` or rebuild if stale.

    Returns ``None`` if the file does not exist or contains zero rows.
    """
    if not path.exists():
        return None

    mtime_ns = path.stat().st_mtime_ns
    key = (str(path.resolve()), mtime_ns)
    cached = _CORPUS_CACHE.get(key)
    if cached is not None:
        return cached

    rows = read_all(path)
    if not rows:
        return None

    tokenized = [_tokenize(row_to_document(r)) for r in rows]
    safe_tokenized = [t if t else [""] for t in tokenized]
    index = BM25Okapi(safe_tokenized)
    _CORPUS_CACHE[key] = (rows, safe_tokenized, index)
    return _CORPUS_CACHE[key]


# --------------------------------------------------------------------------- #
# Public search
# --------------------------------------------------------------------------- #


_DEFAULT_RETRIEVER = BM25Retriever()


def search(
    path: Path | str,
    query: str,
    k: int = 5,
    *,
    filter_bypassed: bool | None = None,
    seed_id: str | None = None,
) -> list[dict[str, Any]]:
    """BM25 search over the attack log.

    - Pre-filters rows by ``seed_id`` when provided.
    - Applies ``filter_bypassed`` when provided (``True`` → only
      ``bypassed=True`` rows, ``False`` → only ``bypassed=False``).
    - Returns up to ``k`` rows scored by BM25 against ``query``.
    - If ``filter_bypassed`` is not ``None`` and fewer than 3 rows remain
      after the seed_id+bypassed filters, returns ``[]``. This threshold
      is documented behavior.
    """
    path = Path(path)
    loaded = _get_corpus(path)
    if loaded is None:
        return []
    rows, tokenized_corpus, _index = loaded

    kept: list[tuple[int, dict[str, Any], list[str]]] = []
    for i, row in enumerate(rows):
        if seed_id is not None and row.get("seed_id") != seed_id:
            continue
        if filter_bypassed is True and not bool(row.get("bypassed")):
            continue
        if filter_bypassed is False and bool(row.get("bypassed")):
            continue
        kept.append((i, row, tokenized_corpus[i]))

    if filter_bypassed is not None and len(kept) < 3:
        return []

    if not kept:
        return []

    # Build a fresh BM25 over the filtered subset so scores reflect the
    # conditional corpus (seed-local search is documented behavior).
    sub_corpus = [tokens if tokens else [""] for _, _, tokens in kept]
    sub_index = BM25Okapi(sub_corpus)
    query_tokens = _tokenize(query) or [""]
    scores = sub_index.get_scores(query_tokens)

    order = sorted(range(len(kept)), key=lambda j: (-scores[j], j))
    top = order[: max(0, k)]
    return [kept[j][1] for j in top]
