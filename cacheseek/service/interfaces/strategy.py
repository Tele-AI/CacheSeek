"""Strategy Protocol — contract for cache strategies."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult


@runtime_checkable
class Strategy(Protocol):
    """Strategy Protocol.

    A ``Strategy`` answers two questions for ``CacheService``:
    - ``lookup(query, ctx)`` — does this request hit a stored cache?
    - ``save(query, outputs, ctx)`` — persist what this request produced.

    Conformance:
    - Durable state lives in the injected KVStore / VectorStore /
      MetadataStore / AuditLog backends. The Strategy object itself
      should hold only configuration plus the injected backend handles;
      no per-request mutable state should be carried across ``await``
      boundaries on ``self``.
    - Implementations must be thread-safe — ``lookup`` / ``save`` are
      awaited concurrently from many ``CacheService`` workers. Heavy
      work (encoding, rerank) inside the methods may serialize on the
      underlying backend's lock.
    - The Protocol takes neutral types (``CacheQuery`` / ``LookupResult`` /
      ``ModelOutputs``) — framework-specific request / response types are
      translated by ``FrameworkAdapter.{build_query,on_response}`` before
      reaching Strategy.
    - ``runtime_checkable`` allows ABC subclasses (e.g.
      ``VideoBasedApproximateCache``) to satisfy
      ``isinstance(obj, Strategy)`` structurally.
    """

    async def lookup(
        self, query: CacheQuery, ctx: Any = None
    ) -> LookupResult: ...

    async def save(
        self,
        query: CacheQuery,
        outputs: ModelOutputs,
        ctx: Any = None,
    ) -> None: ...
