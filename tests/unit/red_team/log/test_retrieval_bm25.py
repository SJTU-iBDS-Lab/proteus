"""Unit tests for ``src/red_team/log/retrieval/bm25.py``.

The retriever layer is kept pluggable via the ``Retriever`` protocol so
future embedding backends (dense vector search) can be slotted in.
The BM25 implementation is the default; these tests pin the contract:

* Empty corpus → empty result.
* Top-k is respected even when ranked list is larger.
* Query terms that do not appear in any document produce a deterministic
  (possibly empty) tie-broken ordering — we only assert that no extra
  rows are returned and that the result is a subset of the corpus.
* The retriever accepts plain dicts (the shape written to the JSONL)
  without requiring AttackLogRow construction.
"""
from __future__ import annotations

from red_team.log.retrieval.base import Retriever
from red_team.log.retrieval.bm25 import BM25Retriever


def _row(reason: str, codes: list[str], tools: list[str]) -> dict:
    return {
        "reason": reason,
        "reason_codes_evaded": codes,
        "tool_chain_summary": [{"tool": t} for t in tools],
    }


def test_bm25_retriever_is_a_retriever() -> None:
    r = BM25Retriever()
    assert isinstance(r, Retriever)


def test_empty_corpus_returns_empty() -> None:
    r = BM25Retriever()
    assert r.search("anything", [], k=5) == []


def test_single_document_match() -> None:
    r = BM25Retriever()
    corpus = [_row("autonomy encouragement", ["autonomy_encouragement"], ["read_skill"])]
    hits = r.search("autonomy", corpus, k=3)
    assert hits == corpus


def test_top_k_cap() -> None:
    r = BM25Retriever()
    # 4 identical rows; k=2 must yield exactly 2.
    corpus = [
        _row("credential exfiltration via metadata", ["credential_exfil"], ["read_skill"])
        for _ in range(4)
    ]
    hits = r.search("credential", corpus, k=2)
    assert len(hits) == 2


def test_query_absent_from_corpus_returns_subset_only() -> None:
    r = BM25Retriever()
    corpus = [
        _row("autonomy flagged", ["autonomy_encouragement"], ["read_skill"]),
        _row("rm -rf dangerous shell", ["dangerous_command"], ["submit_mutation"]),
    ]
    # "zebra" appears in no document; BM25 scores will be 0; retriever
    # still returns rows (tie-broken by original order) without ever
    # returning anything outside the corpus.
    hits = r.search("zebra", corpus, k=5)
    assert len(hits) <= len(corpus)
    for h in hits:
        assert h in corpus


def test_ranking_prefers_more_relevant_document() -> None:
    r = BM25Retriever()
    corpus = [
        _row("unrelated filler text", ["noise"], ["tool_x"]),
        _row("rm -rf destructive shell invocation", ["dangerous_command", "shell_exec"],
             ["submit_mutation"]),
        _row("autonomy encouragement", ["autonomy_encouragement"], ["read_skill"]),
    ]
    hits = r.search("rm -rf shell", corpus, k=1)
    assert len(hits) == 1
    assert "rm" in hits[0]["reason"]
