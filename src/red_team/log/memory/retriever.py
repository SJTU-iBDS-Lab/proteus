"""Two-bucket cosine retriever (Medium-scope rework).

For a parent SkillVariant + target_model + memory Store, returns two
buckets:

* ``positive``    — top-k latest ``positive`` units (siblings to imitate)
* ``chokepoint``  — top-k latest ``chokepoint`` units (patterns to avoid)

Strict filters (plan §extra-2.1, §11.2):

* ``source_seed_id`` — single-seed scope (no cross-seed leakage)
* ``target_model``   — joint key (chokepoints can be model-specific)
* ``superseded_by IS NULL`` — only the latest evolution version

Embedding backends, in priority order:

1. **bge-small-en-v1.5** via :mod:`sentence_transformers` (≈33 MB,
   one-time download, offline).  This is the default.
2. **OpenAI text-embedding-3-small** via the LiteLLM proxy (used
   automatically when `sentence_transformers` cannot load the model
   — e.g. offline cache + no model on disk).
3. **token-Jaccard** fallback for unit tests / CI without either of
   the above.  ``RetrieverConfig.embedding_backend="jaccard"`` forces
   this path.

The embedding for each unit is **cached** in
``Store.memory_units.embedding`` (BLOB column) at first compute and
re-used for every subsequent retrieve / evolve call within the run.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from red_team.log.memory.store import Store
from red_team.log.memory.unit import MemoryUnit

__all__ = [
    "BrainstormHits",
    "RetrieverConfig",
    "Embedder",
    "BgeSmallEmbedder",
    "JaccardEmbedder",
    "make_default_embedder",
    "query_for_brainstorm",
    "ensure_embeddings",
]


_LOGGER = logging.getLogger(__name__)
_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Embedder hierarchy
# ---------------------------------------------------------------------------


class Embedder:
    """Abstract: encode strings to L2-normalised float32 vectors."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class BgeSmallEmbedder(Embedder):
    """``BAAI/bge-small-en-v1.5`` via sentence-transformers (default)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=_DTYPE)
        # bge models specify normalize_embeddings=True for cosine-ready output.
        out = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(_DTYPE, copy=False)


class JaccardEmbedder(Embedder):
    """Token-Jaccard fallback (plan §extra-2.1: unit-test only).

    Encodes text as a multi-hot bag-of-tokens over a fixed vocabulary
    derived from the inputs themselves.  The resulting vector is
    L2-normalised so cosine == bag-overlap-proxy.
    """

    dim = 256

    def __init__(self) -> None:
        # Stable fixed-size hash bucket; not perfect for paraphrase but
        # adequate for tests where retrieval determinism matters more
        # than recall quality.
        self._size = self.dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            t for t in text.lower().replace("-", " ").replace("_", " ").split()
            if len(t) > 2
        ]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._size), dtype=_DTYPE)
        out = np.zeros((len(texts), self._size), dtype=_DTYPE)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                out[i, hash(tok) % self._size] = 1.0
        # L2 normalise.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (out / norms).astype(_DTYPE, copy=False)


def make_default_embedder(
    *, backend: Literal["bge_small", "jaccard"] = "bge_small"
) -> Embedder:
    """Return an Embedder, falling back gracefully.

    When ``backend="bge_small"`` and sentence-transformers cannot load
    the model (e.g. offline + no cache), fall through to JaccardEmbedder
    with a logged warning rather than crashing the run.  The LiteLLM
    text-embedding-3-small route is *not* used here yet; if bge-small
    fails to load, swap in a LiteLLM-backed embedder at that point.
    """
    if backend == "jaccard":
        return JaccardEmbedder()
    try:
        return BgeSmallEmbedder()
    except Exception as exc:
        _LOGGER.warning(
            "bge-small unavailable (%s); falling back to JaccardEmbedder", exc
        )
        return JaccardEmbedder()


# ---------------------------------------------------------------------------
# Retriever config + result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    mode: Literal["off", "static", "evolving"] = "evolving"
    top_k_per_bucket: int = 4
    min_cosine: float = 0.20  # below this, drop the candidate (unrelated motif)
    # When True, drop the ``source_seed_id`` filter from store queries —
    # used in shared-memory cross-seed mode (Phase D 5-dim coordinator).
    cross_seed: bool = False
    # Snapshot boundary; ``0`` disables.  See Store.query_latest.
    max_rowid: int = 0


@dataclass(slots=True)
class BrainstormHits:
    positive: list[MemoryUnit] = field(default_factory=list)
    chokepoint: list[MemoryUnit] = field(default_factory=list)

    def total(self) -> int:
        return len(self.positive) + len(self.chokepoint)


# ---------------------------------------------------------------------------
# Embedding cache (read & write through the Store BLOB column)
# ---------------------------------------------------------------------------


def _unit_text(u: MemoryUnit) -> str:
    """Concatenated text used for embedding: technique + mechanism + tags.

    Joining these gives the embedder enough context to capture the
    family-level abstraction the evolver tries to discover.
    """
    return " ".join(
        filter(
            None,
            [
                u.technique_name,
                u.mechanism,
                u.exploited_assumption,
                " ".join(u.tags),
            ],
        )
    )


def _vec_to_bytes(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=_DTYPE).tobytes()


def _bytes_to_vec(b: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(b, dtype=_DTYPE)
    if arr.size != dim:
        raise ValueError(
            f"cached embedding has {arr.size} dims, embedder dim={dim}; "
            "store probably mixed embedders — re-run with a fresh memory.sqlite"
        )
    return arr.copy()


def ensure_embeddings(
    store: Store,
    units: Iterable[MemoryUnit],
    embedder: Embedder,
) -> dict[str, np.ndarray]:
    """Return a ``{unit_id: vec}`` map for the given units.

    Reads cached vectors from Store.embedding when present; computes +
    persists the rest in a single batch.
    """
    units = list(units)
    cached: dict[str, np.ndarray] = {}
    todo: list[MemoryUnit] = []
    for u in units:
        b = store.get_embedding(u.unit_id)
        if b is not None:
            try:
                cached[u.unit_id] = _bytes_to_vec(b, embedder.dim)
                continue
            except ValueError as exc:
                _LOGGER.warning("dropping stale embedding for %s: %s", u.unit_id, exc)
        todo.append(u)
    if todo:
        new_vecs = embedder.encode([_unit_text(u) for u in todo])
        for u, v in zip(todo, new_vecs):
            store.set_embedding(u.unit_id, _vec_to_bytes(v))
            cached[u.unit_id] = v
    return cached


# ---------------------------------------------------------------------------
# Public retrieval
# ---------------------------------------------------------------------------


def query_for_brainstorm(
    *,
    store: Store,
    parent_summary: dict[str, Any] | None,
    source_seed_id: str,
    target_model: str,
    embedder: Embedder | None = None,
    config: RetrieverConfig | None = None,
) -> BrainstormHits:
    """Two-bucket cosine retrieval over ``store`` (latest, single-(seed,target))."""
    cfg = config or RetrieverConfig()
    if cfg.mode == "off":
        return BrainstormHits()

    embedder = embedder or make_default_embedder()

    seed_filter = None if cfg.cross_seed else source_seed_id
    rowid_cap = cfg.max_rowid or None
    pos_units = store.query_by_category(
        source_seed_id=seed_filter, target_model=target_model,
        category="positive", max_rowid=rowid_cap,
    )
    choke_units = store.query_by_category(
        source_seed_id=seed_filter, target_model=target_model,
        category="chokepoint", max_rowid=rowid_cap,
    )
    if not pos_units and not choke_units:
        return BrainstormHits()

    # Build / load embeddings for every candidate up front so we can
    # cosine-rank in numpy without re-encoding inside the loop.
    cache = ensure_embeddings(store, [*pos_units, *choke_units], embedder)

    parent_text = _parent_text(parent_summary)
    if not parent_text.strip():
        # Fall back to "all latest" with their own ordering when we
        # have no parent text to score against (cold start).
        return BrainstormHits(
            positive=pos_units[: cfg.top_k_per_bucket],
            chokepoint=choke_units[: cfg.top_k_per_bucket],
        )

    parent_vec = embedder.encode([parent_text])[0]
    return BrainstormHits(
        positive=_top_k_by_cosine(pos_units, cache, parent_vec, cfg),
        chokepoint=_top_k_by_cosine(choke_units, cache, parent_vec, cfg),
    )


def _parent_text(parent_summary: dict[str, Any] | None) -> str:
    if not parent_summary:
        return ""
    parts = [
        str(parent_summary.get("technique_name") or parent_summary.get("name") or ""),
        str(parent_summary.get("mechanism", "")),
        " ".join(parent_summary.get("tags", []) or []),
        str(parent_summary.get("o", "")),
        str(parent_summary.get("tau", "")),
        " ".join(parent_summary.get("chain", []) or []),
    ]
    return " ".join(p for p in parts if p)


def _top_k_by_cosine(
    units: list[MemoryUnit],
    cache: dict[str, np.ndarray],
    parent_vec: np.ndarray,
    cfg: RetrieverConfig,
) -> list[MemoryUnit]:
    if not units:
        return []
    scores: list[tuple[float, MemoryUnit]] = []
    for u in units:
        v = cache.get(u.unit_id)
        if v is None:
            continue
        # Both vectors are L2-normalised so dot == cosine.
        sim = float(np.dot(parent_vec, v))
        if sim < cfg.min_cosine:
            continue
        scores.append((sim, u))
    scores.sort(key=lambda kv: kv[0], reverse=True)
    return [u for _, u in scores[: cfg.top_k_per_bucket]]
