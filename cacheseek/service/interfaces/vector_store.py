# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""VectorStore Protocol — collection-scoped vector index with payload."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Return the nearest stored vectors to ``vector`` in ``collection``.

        Results are ordered best-match first. ``similarity`` on each result is a
        higher-is-better score (cosine-style); backends using a distance metric
        must convert it so callers can compare scores uniformly.

        Args:
            collection: Name of the collection (vector index) to query.
            vector: Query embedding. Its length must match the collection's
                configured dimension.
            limit: Maximum number of candidates to return.
            score_threshold: When given, drop results whose ``similarity`` is
                below this value.

        Returns:
            A list of VectorSearchResult, at most ``limit`` long, best match
            first. Empty when the collection is missing or nothing clears the
            threshold.

        Raises:
            ValueError: If ``vector``'s dimension does not match an existing
                collection's dimension (validated once the dimension is known).
        """
        ...

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or replace one vector and its payload in ``collection``.

        Idempotent on ``point_id``: re-upserting the same id overwrites the
        prior vector and payload. Implementations may auto-create the
        collection if it does not yet exist, inferring its dimension from
        ``vector``.

        Args:
            collection: Name of the collection to write to.
            point_id: Stable identifier for the point (typically the cache id).
            vector: Embedding to store. Its length must match the collection's
                dimension.
            payload: Arbitrary metadata stored alongside the vector and
                returned by ``search``.

        Raises:
            ValueError: If ``vector``'s dimension does not match an existing
                collection's dimension (validated once the dimension is known).
        """
        ...

    def delete(self, collection: str, point_ids: list[str]) -> None:
        """Delete the given points from ``collection``.

        Idempotent: deleting unknown ids or from a missing collection is a
        no-op. An empty ``point_ids`` list does nothing.

        Args:
            collection: Name of the collection to delete from.
            point_ids: Identifiers of the points to remove.
        """
        ...

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        """Ensure a collection exists with the given vector dimension.

        Idempotent: if the collection already exists it is left as-is (its
        dimension is not changed); otherwise it is created with ``vector_dim``.

        Args:
            collection: Name of the collection to create or verify.
            vector_dim: Dimensionality of vectors stored in this collection.
        """
        ...

    def get_vector_size(self, collection: str) -> int | None:
        """Return the configured vector dimension of ``collection``.

        Args:
            collection: Name of the collection to inspect.

        Returns:
            The collection's vector dimension, or ``None`` if the collection
            does not exist.
        """
        ...
