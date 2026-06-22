"""VectorStore Protocol — collection-scoped vector index with payload."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from cacheseek.service.cache_types import VectorSearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Contract for vector indexing + similarity search.

    Conformance:
    - Implementations should be thread-safe — ``lookup`` and ``save``
      paths may run concurrently from ``CacheService`` worker threads.
      Backing clients (Qdrant, FAISS) that keep connection state should
      encapsulate it inside the implementation and not expose it on the
      Protocol surface.
    - This Protocol is ``runtime_checkable``: any class that exposes the
      methods below with matching signatures satisfies
      ``isinstance(obj, VectorStore)`` — no inheritance required.
    """

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchResult]: ...

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None: ...

    def delete(self, collection: str, point_ids: list[str]) -> None: ...

    def ensure_collection(self, collection: str, vector_dim: int) -> None: ...

    def get_vector_size(self, collection: str) -> Optional[int]: ...
