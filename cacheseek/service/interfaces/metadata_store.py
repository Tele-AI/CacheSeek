# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""MetadataStore Protocol — cache index + access stats + eviction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cacheseek.service.cache_types import IndexEntry


@runtime_checkable
class MetadataStore(Protocol):
    """Contract for cache metadata: index, access stats, eviction.

    Scope: cache-index lookup + access stats + eviction planning.
    Audit-log responsibilities (``record_hit_pair`` /
    ``record_similarity_scores``) belong on the sibling ``AuditLog``
    Protocol — implementations may delegate or compose, but Strategy
    code only sees this minimal surface.

    Conformance:
    - Implementations must be thread-safe — ``register_cache`` /
      ``record_access`` / ``plan_eviction`` are called from concurrent
      ``CacheService`` workers. Mutable in-memory caches inside backing
      storage (JSON files, SQLite, etc.) should be kept private and
      serialized internally.
    - This Protocol is ``runtime_checkable``: any class that exposes the
      methods below with matching signatures satisfies
      ``isinstance(obj, MetadataStore)`` — no inheritance required.
      Implementations are free to expose additional methods (e.g.
      audit-log writes) and pass an ``AuditLog`` Protocol check too.
    """

    def register_cache(
        self,
        cache_id: str,
        prompt: str,
        saved_steps: list[int],
        size_mb: float,
        num_frames: int,
        cache_type: str | None = None,
    ) -> None:
        """Register (or overwrite) a stored cache in the prompt index and stats.

        Adds ``cache_id`` to the index keyed by its prompt and ``cache_type``,
        and initializes/refreshes its metadata (size, frame count, last-access
        time). Re-registering an existing ``cache_id`` overwrites the index
        entry and refreshes stats; implementations should preserve any prior
        access count rather than resetting it.

        Args:
            cache_id: Stable identifier for the stored cache.
            prompt: The prompt this cache was produced for; used as the
                lookup key by ``lookup_prompt``.
            saved_steps: Denoise step indices snapshotted in the cache.
            size_mb: On-disk / in-store size of the cache in megabytes; feeds
                ``plan_eviction`` capacity accounting.
            num_frames: Number of video frames the cache represents.
            cache_type: Cache category (e.g. ``"approximate_cache"``). When
                ``None``, the implementation's default type is used; index
                lookup is scoped by this type.
        """
        ...

    def remove_cache(self, cache_id: str) -> None:
        """Remove a cache from the index and drop its metadata.

        Idempotent: removing an unknown ``cache_id`` is a no-op. This only
        affects the metadata layer; eviction of the underlying payload from the
        vector/KV stores is the caller's responsibility.

        Args:
            cache_id: Identifier of the cache to forget.
        """
        ...

    def lookup_prompt(
        self,
        prompt: str,
        cache_type: str | None = None,
    ) -> IndexEntry | None:
        """Find the index entry whose prompt exactly matches ``prompt``.

        This is an exact-string match, not a similarity search (similarity
        ranking is the VectorStore's job). When the same prompt was registered
        multiple times, the earliest-registered entry is returned.

        Args:
            prompt: Prompt text to match exactly.
            cache_type: When given, restrict the search to that cache type.
                When ``None``, the default type is searched first, then all
                other types.

        Returns:
            The matching IndexEntry, or ``None`` if no entry matches.
        """
        ...

    def get_cache_meta(self, cache_id: str) -> dict | None:
        """Return a copy of the stored metadata for one cache.

        Args:
            cache_id: Identifier of the cache.

        Returns:
            A dict with stats (``prompt``, ``saved_steps``, ``size_mb``,
            ``num_frames``, ``access_count``, ``last_access_time``,
            ``cache_type``), or ``None`` if the cache is unknown. The returned
            dict is a snapshot copy; mutating it does not affect the store.
        """
        ...

    def record_access(self, cache_id: str) -> None:
        """Record a hit on ``cache_id``, bumping its access stats.

        Increments the access count and updates the last-access timestamp,
        which drives LRU-style ordering in ``plan_eviction``. A no-op for an
        unknown ``cache_id``. Implementations may buffer the write and flush
        lazily; correctness must not depend on immediate persistence.

        Args:
            cache_id: Identifier of the cache that was hit.
        """
        ...

    def plan_eviction(
        self,
        required_mb: float,
        limit_mb: float,
    ) -> list[tuple[str, dict]]:
        """Select caches to evict to fit ``required_mb`` under ``limit_mb``.

        Pure planning: this computes which caches should be removed but does
        not remove them — the caller applies the plan (e.g. via
        ``remove_cache`` plus payload deletion). Candidates are chosen
        least-recently-used first until enough space is freed.

        Args:
            required_mb: Additional space the incoming cache needs.
            limit_mb: Total capacity budget for the store.

        Returns:
            A list of ``(cache_id, meta)`` pairs to evict, oldest-access first.
            Empty when the current usage plus ``required_mb`` already fits
            within ``limit_mb``.
        """
        ...
