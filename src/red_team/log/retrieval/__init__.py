"""Retriever backends for the attack log.

The BM25 implementation is the default; future waves may add dense
(embedding-based) retrievers. All retrievers conform to the
:class:`Retriever` protocol defined in :mod:`red_team.log.retrieval.base`.
"""
from __future__ import annotations

from red_team.log.retrieval.base import Retriever
from red_team.log.retrieval.bm25 import BM25Retriever

__all__ = ["Retriever", "BM25Retriever"]
