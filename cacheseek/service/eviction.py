"""EvictionPolicy Protocol + LRUEviction default.

Eviction decisions live at the metadata layer (``MetadataStore`` provides
the candidate set, ``EvictionPolicy`` picks victims); KV / Vector backends
only execute idempotent deletes.

Alpha default: ``LRUEviction`` (ascending ``last_access_time``).
``FIFOEviction`` is interface-only.

Note (alpha): ``CacheService`` does not currently call any eviction path.
``LocalCacheMetadataManager.plan_eviction`` is implemented but unused;
the policies here are wired-but-dormant. Active eviction is a near-term
roadmap item.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


_Candidates = Sequence[tuple[str, dict[str, Any]]]


@runtime_checkable
class EvictionPolicy(Protocol):
    """Pure-functional victim selection over current metadata candidates.

    Read-only over metadata; no side-effects. ``MetadataStore.plan_eviction``
    produces the candidate list, this policy chooses which entries to drop,
    then the caller orchestrates the actual KV / Vector / Metadata deletes.

    Implementations should be deterministic given a fixed ``candidates``
    sequence — any caching of derived state on ``self`` should not change
    the selection result.
    """

    def select_victims(
        self,
        candidates: _Candidates,
        size_to_free_bytes: int,
    ) -> list[str]:
        """Return cache_ids to evict, in eviction order.

        Args:
            candidates: ``Sequence[(cache_id, metadata)]``. ``metadata``
                must carry at least ``size_mb`` plus the fields the
                policy reads (e.g. ``last_access_time`` for LRU).
            size_to_free_bytes: Bytes the caller needs to free; the
                policy may return fewer cache_ids if a partial eviction
                already satisfies the budget.
        """
        ...


def _accumulate(
    items: _Candidates,
    size_to_free_bytes: int,
) -> list[str]:
    """Walk ``items`` in order, accumulating ``size_mb`` until budget met."""
    if size_to_free_bytes <= 0:
        return []  # budget <= 0 means evict nothing (append-before-check would over-evict by at least one)
    selected: list[str] = []
    freed = 0
    for cache_id, meta in items:
        selected.append(cache_id)
        size_mb = float(meta.get("size_mb", 0.0))
        freed += int(size_mb * 1024 * 1024)
        if freed >= size_to_free_bytes:
            break
    return selected


class LRUEviction:
    """Least-recently-used: sort by ascending ``last_access_time``."""

    def select_victims(
        self,
        candidates: _Candidates,
        size_to_free_bytes: int,
    ) -> list[str]:
        items = sorted(
            candidates,
            key=lambda kv: float(kv[1].get("last_access_time", 0.0)),
        )
        return _accumulate(items, size_to_free_bytes)


class FIFOEviction:
    """First-in-first-out: sort by ascending ``created_at``."""

    def select_victims(
        self,
        candidates: _Candidates,
        size_to_free_bytes: int,
    ) -> list[str]:
        items = sorted(
            candidates,
            key=lambda kv: float(kv[1].get("created_at", 0.0)),
        )
        return _accumulate(items, size_to_free_bytes)


__all__ = ["EvictionPolicy", "LRUEviction", "FIFOEviction"]
