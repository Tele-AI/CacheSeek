"""MetadataStore Protocol — cache index + access stats + eviction."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

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
        cache_type: Optional[str] = None,
    ) -> None: ...

    def remove_cache(self, cache_id: str) -> None: ...

    def lookup_prompt(
        self,
        prompt: str,
        cache_type: Optional[str] = None,
    ) -> Optional[IndexEntry]: ...

    def get_cache_meta(self, cache_id: str) -> Optional[dict]: ...

    def record_access(self, cache_id: str) -> None: ...

    def plan_eviction(
        self,
        required_mb: float,
        limit_mb: float,
    ) -> list[tuple[str, dict]]: ...
