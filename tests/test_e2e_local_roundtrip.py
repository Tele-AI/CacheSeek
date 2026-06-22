"""Local service-level e2e tests.

These tests exercise the CacheSeek lifecycle without external services:
CacheService -> VideoBasedApproximateCache -> KV/vector/metadata -> lookup hit.
The model encoders and vector backend are deterministic in-process doubles so
the test stays fast and can run on machines without GPU, Fluxon, Qdrant, or
Qwen model weights.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from typing import Any

import pytest
import torch

from cacheseek.service.cache_types import VectorSearchResult
from cacheseek.service.config import CacheConfig, CacheMode
from cacheseek.service.lifecycle import CacheService
from cacheseek.backends.metadata import LocalCacheMetadataManager
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import SkipStep
from cacheseek.stores import InMemoryKVStore
from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache
from cacheseek.service.interfaces.vector_store import VectorStore


pytestmark = pytest.mark.e2e


class _DeterministicEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        return self._vector_for(text)

    def encode_video(self, frames: list[Any], prompt: str | None = None) -> list[float]:
        return self._vector_for(prompt or f"frames:{len(frames)}")

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [float(digest[i % len(digest)] + 1) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]


class _InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}
        self._dims: dict[str, int] = {}
        self._lock = threading.RLock()

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        with self._lock:
            self._collections.setdefault(collection, {})
            self._dims[collection] = int(vector_dim)

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            self.ensure_collection(collection, len(vector))
            self._collections[collection][point_id] = (list(vector), dict(payload))

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 1,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        with self._lock:
            items = self._collections.get(collection, {})
            scored: list[VectorSearchResult] = []
            for point_id, (stored_vector, payload) in items.items():
                similarity = self._cosine(vector, stored_vector)
                if score_threshold is not None and similarity < score_threshold:
                    continue
                scored.append(
                    VectorSearchResult(
                        cache_id=point_id,
                        similarity=similarity,
                        prompt=str(payload.get("prompt", "")),
                        saved_steps=list(payload.get("saved_steps", [])),
                        payload=dict(payload),
                    )
                )
            scored.sort(key=lambda item: item.similarity, reverse=True)
            return scored[: max(1, int(limit or 1))]

    def delete(self, collection: str, point_ids: list[str]) -> None:
        with self._lock:
            items = self._collections.get(collection, {})
            for point_id in point_ids:
                items.pop(point_id, None)

    def get_vector_size(self, collection: str) -> int | None:
        with self._lock:
            return self._dims.get(collection)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(numerator / (left_norm * right_norm))


def _run(coro):
    return asyncio.run(coro)


def _build_service(tmp_path):
    cfg = CacheConfig(
        enable_latent_cache=True,
        cache_mode=CacheMode.READ_WRITE,
        latent_cache_dir=str(tmp_path / "cache"),
        vector_dim=8,
        key_steps=[5, 10],
        video_embedding_enabled=True,
        video_vector_collection="local-e2e",
        video_similarity_threshold=0.50,
        rerank_enabled=False,
        rerank_score_threshold=0.90,
        save_async_enabled=True,
        save_queue_size=2,
        save_on_full="block",
        vector_wait_poll_s=0.001,
        vector_wait_warn_s=0.0,
        vector_wait_timeout_s=2.0,
        flush_on_shutdown=True,
    )
    metadata = LocalCacheMetadataManager(tmp_path / "metadata")
    strategy = VideoBasedApproximateCache(
        cfg,
        InMemoryKVStore(),
        _InMemoryVectorStore(),
        metadata,
        prompt_encoder=_DeterministicEncoder(cfg.vector_dim),
        video_encoder=_DeterministicEncoder(cfg.vector_dim),
    )
    service = CacheService(
        [strategy],
        async_save=cfg.save_async_enabled,
        max_queue_size=cfg.save_queue_size,
        on_full=cfg.save_on_full,
        flush_on_shutdown=cfg.flush_on_shutdown,
        vector_wait_poll_s=cfg.vector_wait_poll_s,
        vector_wait_warn_s=cfg.vector_wait_warn_s,
        vector_wait_timeout_s=cfg.vector_wait_timeout_s,
        cache_mode=cfg.cache_mode,
    )
    return service, metadata


def test_local_service_save_then_lookup_hits_without_external_backends(tmp_path) -> None:
    service, metadata = _build_service(tmp_path)
    query = CacheQuery(prompt="local e2e prompt: aurora over a quiet station", task_type="t2v")
    latent_step_5 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    latent_step_10 = latent_step_5 + 10

    try:
        pre_save = _run(service.lookup(query))
        assert pre_save.hit is False

        _run(
            service.save(
                query,
                ModelOutputs(
                    latent_states_dict={5: latent_step_5, 10: latent_step_10},
                    embedding_video_frames=[object()],
                    num_frames=16,
                    final_step=20,
                    saved_steps=[5, 10],
                ),
            )
        )

        post_save = _run(service.lookup(query))

        assert post_save.hit is True
        assert isinstance(post_save.resume_hint, SkipStep)
        assert post_save.resume_hint.k == 5
        # post_save.payload is now a VideoApproxPayload covering only
        # step k=5 (strategy passes a single-step partial_spec).
        assert torch.equal(post_save.payload.get_latent_at_step(5), latent_step_5)

        entry = metadata.lookup_prompt(query.prompt, cache_type="video_approximate_cache")
        assert entry is not None
        assert entry.saved_steps == [5, 10]
    finally:
        service.shutdown()
