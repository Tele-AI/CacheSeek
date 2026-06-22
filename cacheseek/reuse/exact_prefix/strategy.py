"""ExactPrefixStrategy — exact_prefix semantics under the shared Strategy protocol.

Conforms to the existing ``Strategy`` protocol (service/interfaces/strategy.py,
runtime_checkable) with the ``LookupResult``/``ResumeHint`` sealed union; the
adapter dispatches by subclass. This class maps a trie lookup to a
``FastForward`` hint. The approximate ``VideoBasedApproximateCache`` conforms to
the same protocol, so both semantics are peers: CacheService/engine selects a
strategy by pipeline config rather than cascading.

CacheQuery contract (exact_prefix carries strategy-specific fields via ``extra``):
    extra["root_hash"]: bytes   namespace root (H(image, prompt, version))
    extra["actions"]:   list    per-chunk discrete ActionKey sequence

save() contract: exact writeback is chunk-granular streaming (unlike request-level
save). Chunk data is passed via the ``ctx`` dict; the newly created trie node is
written back to ``ctx["node"]`` (the caller uses it to advance its cursor). The
outputs parameter is unused.
"""
from __future__ import annotations

from typing import Any

from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult

from .manager import WorldKVManager


class ExactPrefixStrategy:
    """exact_prefix implementation of the ``Strategy`` protocol (structural
    conformance, no inheritance)."""

    def __init__(self, manager: WorldKVManager) -> None:
        self.mgr = manager

    async def lookup(self, query: CacheQuery, ctx: Any = None) -> LookupResult:
        root = query.extra.get("root_hash")
        actions = query.extra.get("actions")
        if root is None or actions is None:
            return LookupResult.miss("exact_prefix_query_missing_fields")
        m = self.mgr.forest.lookup(root, actions)
        if m.namespace is None:
            return LookupResult.miss("exact_prefix_namespace_miss")
        if m.node is None or m.matched_len == 0:
            return LookupResult.miss("exact_prefix_no_match")
        if m.matched_len < self.mgr.cfg.break_even_k:
            return LookupResult.miss("exact_prefix_below_break_even")
        return LookupResult.hit_fast_forward(k=m.matched_len, node=m.node, namespace=m.namespace)

    async def save(self, query: CacheQuery, outputs: Any = None, ctx: Any = None) -> None:
        assert isinstance(ctx, dict) and "ns" in ctx, "exact_prefix save requires chunk ctx"
        node = self.mgr.ingest(
            ctx["ns"], ctx["parent"], ctx["action"], ctx["node_key"], ctx["depth"],
            ctx["payload"], ctx["latent"], nbytes=ctx.get("nbytes", 0),
        )
        ctx["node"] = node
