"""cacheseek.service.cache_types — strategy-internal helper types.

- ``VectorSearchResult``: returned by ``Strategy._vector_search`` to
  describe rerank candidates. Strategy-internal helper.
- ``IndexEntry``: returned by metadata-store implementations
  (``cacheseek.service.interfaces.metadata_store.MetadataStore``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class IndexEntry:
    """Index entry (MetadataStore return type)."""

    cache_id: str
    prompt: str
    saved_steps: List[int]
    cache_type: str = "approximate_cache"


@dataclass
class VectorSearchResult:
    """Vector search result."""

    cache_id: str
    similarity: float
    prompt: str
    saved_steps: List[int]
    payload: Dict[str, Any]
