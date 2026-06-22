# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
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
    ) -> LookupResult:
        """Decide whether ``query`` hits a stored cache.

        Resolves the request against the strategy's backends (encode, vector
        search, optional rerank) and decides whether a usable cache exists. On
        a hit, populates the heavy ``payload`` and the light ``resume_hint``
        (the instruction the FrameworkAdapter applies to the engine). Must
        degrade to a miss rather than raise on a recoverable backend failure.

        Args:
            query: Strategy-agnostic lookup request built by the
                FrameworkAdapter.
            ctx: Optional strategy-specific context (e.g. the engine handle);
                strategies that do not need it ignore it.

        Returns:
            A LookupResult. On a miss, ``hit`` is False with
            ``payload`` / ``resume_hint`` unset and an optional
            ``miss_reason`` tag.
        """
        ...

    async def save(
        self,
        query: CacheQuery,
        outputs: ModelOutputs,
        ctx: Any = None,
    ) -> None:
        """Persist what ``query`` produced for future reuse.

        Encodes/indexes ``outputs`` and writes the durable artifacts (vectors,
        payload, metadata) through the injected backends, applying any
        configured eviction policy. Carries no per-request state on ``self``
        across ``await``. Safe to call concurrently with other ``save`` /
        ``lookup`` calls.

        Args:
            query: The request the outputs were generated for.
            outputs: Strategy-agnostic save-side payload (latents, frames,
                step metadata).
            ctx: Optional strategy-specific context; ignored when unused.
        """
        ...
