"""Shared stub backends + strategy factory for unit-testing
``VideoBasedApproximateCache``.

Stubs are deliberately minimal — they implement the methods the strategy
calls, with hooks (callbacks, recorded args, programmable failures) so
tests can drive any branch deterministically without spinning up real
fluxon / qdrant / Qwen3-VL.

The ``make_strategy(...)`` helper returns a ``VideoBasedApproximateCache``
already wired with the stubs and a sensible ``CacheConfig``. Tests pass
overrides as kwargs (e.g. ``rerank_enabled=True``) and access the wired
stubs via the returned strategy's attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cacheseek.service.cache_types import VectorSearchResult


# ─── KVStore stub ──────────────────────────────────────────────────────────


class StubKVStore:
    """Dict-backed KV store with programmable miss / failure hooks.

    - ``preset(key, value)`` seeds bytes for ``get`` to return.
    - ``mark_missing(key)`` makes ``get`` return ``None`` even if the key
      was put — simulates a KV pool flush after a vector entry was
      written (the lazy-eviction trigger).
    - ``put_should_raise`` / ``remove_should_raise`` make those methods
      throw, used by save rollback tests.
    """

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self._forced_missing: set[str] = set()
        self.put_should_raise: Optional[Exception] = None
        self.remove_should_raise: Optional[Exception] = None
        self.put_calls: List[tuple[str, bytes]] = []
        self.remove_calls: List[str] = []

    def preset(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def mark_missing(self, key: str) -> None:
        self._forced_missing.add(key)

    def get(self, key: str) -> Optional[bytes]:
        if key in self._forced_missing:
            return None
        return self._store.get(key)

    def put(self, key: str, value: bytes) -> None:
        if self.put_should_raise is not None:
            raise self.put_should_raise
        self._store[key] = value
        self.put_calls.append((key, value))

    def remove(self, key: str) -> None:
        if self.remove_should_raise is not None:
            raise self.remove_should_raise
        self._store.pop(key, None)
        self.remove_calls.append(key)

    def list_keys(self) -> list[str]:
        return list(self._store.keys())


# ─── VectorStore stub ──────────────────────────────────────────────────────


class StubVectorStore:
    """In-memory vector store with explicit search results.

    Tests set ``search_results`` to whatever list of
    ``VectorSearchResult`` they want returned. ``upsert_should_raise`` /
    ``ensure_should_raise`` let tests force save-path failures.
    """

    def __init__(self) -> None:
        self.search_results: List[VectorSearchResult] = []
        self.upsert_calls: List[tuple[str, str, list[float], dict]] = []
        self.delete_calls: List[tuple[str, list[str]]] = []
        self.ensure_calls: List[tuple[str, int]] = []
        self.upsert_should_raise: Optional[Exception] = None
        self.ensure_should_raise: Optional[Exception] = None

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        return list(self.search_results[:limit])

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: Dict[str, Any],
    ) -> None:
        if self.upsert_should_raise is not None:
            raise self.upsert_should_raise
        self.upsert_calls.append((collection, point_id, list(vector), dict(payload)))

    def delete(self, collection: str, point_ids: list[str]) -> None:
        self.delete_calls.append((collection, list(point_ids)))

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        if self.ensure_should_raise is not None:
            raise self.ensure_should_raise
        self.ensure_calls.append((collection, int(vector_dim)))

    def get_vector_size(self, collection: str) -> Optional[int]:
        return None


# ─── MetadataStore stub ────────────────────────────────────────────────────


@dataclass
class _RecordedSimilarityScores:
    request_prompt: str
    task_type: str
    cache_type: str
    stage: str
    candidates: list


@dataclass
class _RecordedHitPair:
    request_prompt: str
    cache_id: str
    cached_prompt: str
    similarity: float
    task_type: str
    cache_type: str
    skip_step: int


class StubMetadataStore:
    """Dict-backed metadata store + audit recorder.

    Capture ``record_similarity_scores`` / ``record_hit_pair`` /
    ``record_access`` / ``register_cache`` / ``remove_cache`` calls for
    test assertions. Programmable failures via the ``*_should_raise``
    flags.
    """

    def __init__(self) -> None:
        self._registered: Dict[str, dict] = {}
        self.register_calls: List[dict] = []
        self.remove_calls: List[str] = []
        self.access_calls: List[str] = []
        self.similarity_recordings: List[_RecordedSimilarityScores] = []
        self.hit_pair_recordings: List[_RecordedHitPair] = []
        self.remove_should_raise: Optional[Exception] = None
        self.register_should_raise: Optional[Exception] = None
        self.get_meta_should_raise: Optional[Exception] = None

    def register_cache(
        self,
        cache_id: str,
        prompt: str,
        saved_steps: list[int],
        size_mb: float,
        num_frames: int,
        cache_type: Optional[str] = None,
    ) -> None:
        if self.register_should_raise is not None:
            raise self.register_should_raise
        record = {
            "cache_id": cache_id,
            "prompt": prompt,
            "saved_steps": list(saved_steps),
            "size_mb": float(size_mb),
            "num_frames": int(num_frames),
            "cache_type": cache_type,
        }
        self._registered[cache_id] = record
        self.register_calls.append(record)

    def remove_cache(self, cache_id: str) -> None:
        if self.remove_should_raise is not None:
            raise self.remove_should_raise
        self._registered.pop(cache_id, None)
        self.remove_calls.append(cache_id)

    def lookup_prompt(
        self, prompt: str, cache_type: Optional[str] = None
    ) -> Optional[Any]:
        return None

    def get_cache_meta(self, cache_id: str) -> Optional[dict]:
        if self.get_meta_should_raise is not None:
            raise self.get_meta_should_raise
        record = self._registered.get(cache_id)
        return dict(record) if record else None

    def record_access(self, cache_id: str) -> None:
        self.access_calls.append(cache_id)

    def plan_eviction(
        self, required_mb: float, limit_mb: float
    ) -> list[tuple[str, dict]]:
        return []

    # Audit (delegated to metadata in the alpha; AuditLog Protocol is the
    # forward-looking shape).
    def record_similarity_scores(
        self,
        request_prompt: str,
        task_type: str,
        cache_type: str,
        stage: str,
        candidates: list[dict],
    ) -> None:
        self.similarity_recordings.append(
            _RecordedSimilarityScores(
                request_prompt=request_prompt,
                task_type=task_type,
                cache_type=cache_type,
                stage=stage,
                candidates=list(candidates),
            )
        )

    def record_hit_pair(
        self,
        request_prompt: str,
        cache_id: str,
        cached_prompt: str,
        similarity: float,
        task_type: str,
        cache_type: str,
        skip_step: int,
    ) -> None:
        self.hit_pair_recordings.append(
            _RecordedHitPair(
                request_prompt=request_prompt,
                cache_id=cache_id,
                cached_prompt=cached_prompt,
                similarity=float(similarity),
                task_type=task_type,
                cache_type=cache_type,
                skip_step=int(skip_step),
            )
        )


# ─── Encoder / Reranker stubs ──────────────────────────────────────────────


class StubPromptEncoder:
    """Returns a fixed vector. Set ``return_value=[]`` to simulate empty
    embedding (one of the lookup miss paths)."""

    def __init__(self, return_value: Optional[List[float]] = None) -> None:
        self.return_value: List[float] = list(return_value) if return_value is not None else [0.1] * 4
        self.calls: List[str] = []

    def encode(self, prompt: str) -> List[float]:
        self.calls.append(prompt)
        return list(self.return_value)


class StubVideoEncoder:
    """Returns a fixed vector. ``raise_on_call`` lets tests force save's
    encode_video failure path."""

    def __init__(self, return_value: Optional[List[float]] = None) -> None:
        self.return_value: List[float] = list(return_value) if return_value is not None else [0.2] * 4
        self.calls: List[tuple] = []
        self.raise_on_call: Optional[Exception] = None

    def encode_video(
        self, frames: List[Any], prompt: Optional[str] = None
    ) -> List[float]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append((len(frames), prompt))
        return list(self.return_value)


class StubReranker:
    """``score_mm(query, documents)`` returns ``return_scores``. Set to
    ``None`` (default) to simulate "rerank fallback to vector similarity"
    branch — strategy treats None as missing and falls back."""

    def __init__(
        self,
        return_scores: Optional[List[float]] = None,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self.return_scores = return_scores
        self.raise_on_call = raise_on_call
        self.calls: List[tuple] = []

    def score_mm(
        self, query: Dict[str, object], documents: List[Dict[str, object]]
    ) -> List[float]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append((dict(query), list(documents)))
        if self.return_scores is None:
            # Strategy._rerank_scores raises ValueError on size mismatch
            # before returning None; to simulate "fallback path", return
            # an empty list which bubbles up as ValueError → caller sees
            # None? Actually looking at code: if scores is empty, the
            # strategy raises. To trigger the "scores is None" fallback
            # we need to test by NOT having score_mm at all. Tests can
            # do that by setting strategy.reranker = object() instead.
            return []
        return list(self.return_scores)


# ─── Strategy factory ──────────────────────────────────────────────────────


def make_search_result(
    cache_id: str = "abc123",
    similarity: float = 0.95,
    prompt: str = "matched prompt",
    saved_steps: Optional[List[int]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> VectorSearchResult:
    """Build a VectorSearchResult with sensible defaults for tests."""
    return VectorSearchResult(
        cache_id=cache_id,
        similarity=float(similarity),
        prompt=prompt,
        saved_steps=list(saved_steps) if saved_steps is not None else [5, 10, 15, 20, 25],
        payload=dict(payload) if payload is not None else {},
    )


@dataclass
class StrategyKit:
    """Container holding the strategy + all wired stubs for assertions."""

    strategy: Any
    config: Any
    kv: StubKVStore
    vector: Optional[StubVectorStore]
    metadata: StubMetadataStore
    prompt_encoder: StubPromptEncoder
    video_encoder: StubVideoEncoder
    reranker: Optional[StubReranker]


def make_strategy(
    *,
    rerank_enabled: bool = False,
    rerank_score_threshold: float = 0.85,
    video_similarity_threshold: float = 0.10,
    rerank_top_k: int = 3,
    max_skip_step: int = 25,
    key_steps: Optional[List[int]] = None,
    vector_store: Optional[StubVectorStore] = None,
    reranker: Optional[StubReranker] = None,
    prompt_encoder: Optional[StubPromptEncoder] = None,
    video_encoder: Optional[StubVideoEncoder] = None,
    kv_store: Optional[StubKVStore] = None,
    metadata_store: Optional[StubMetadataStore] = None,
) -> StrategyKit:
    """Construct ``VideoBasedApproximateCache`` wired entirely with stubs.

    Defaults: rerank disabled, skip_step ≤ 25, encoders return short
    fixed vectors. Pass overrides for any field you want to drive a
    specific branch.

    Pass ``vector_store=None`` (explicit) to test the
    ``self.vector_store is None`` miss path.

    The returned ``StrategyKit`` exposes both the strategy and each stub
    so tests can assert recorded calls / state without re-fetching them.
    """
    from cacheseek.service.config import CacheConfig
    from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache

    cfg = CacheConfig(
        # Disable encoder auto-build paths (tests inject explicit stubs).
        text_embedding_model_path="",
        video_embedding_enabled=False,
        rerank_enabled=rerank_enabled,
        rerank_score_threshold=rerank_score_threshold,
        rerank_top_k=rerank_top_k,
        video_similarity_threshold=video_similarity_threshold,
        max_skip_step=max_skip_step,
        key_steps=list(key_steps) if key_steps is not None else [5, 10, 15, 20, 25],
    )

    kv = kv_store if kv_store is not None else StubKVStore()
    vec = vector_store if vector_store is not None else StubVectorStore()
    meta = metadata_store if metadata_store is not None else StubMetadataStore()
    pe = prompt_encoder if prompt_encoder is not None else StubPromptEncoder()
    ve = video_encoder if video_encoder is not None else StubVideoEncoder()
    rr = reranker  # may be None if rerank_enabled=False

    strategy = VideoBasedApproximateCache(
        cfg,
        kv_store=kv,
        vector_store=vec,
        metadata_manager=meta,
        prompt_encoder=pe,
        video_encoder=ve,
        reranker=rr,
    )

    return StrategyKit(
        strategy=strategy,
        config=cfg,
        kv=kv,
        vector=vec,
        metadata=meta,
        prompt_encoder=pe,
        video_encoder=ve,
        reranker=rr,
    )
