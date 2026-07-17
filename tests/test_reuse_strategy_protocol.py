# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Conformance tests: both reuse semantics share one Strategy protocol.

- Structural conformance: ExactPrefixStrategy and VideoBasedApproximateCache
  both satisfy the runtime_checkable ``Strategy`` in
  service/interfaces/strategy.py;
- Functional: the exact path closes a save->lookup loop, the FastForward hint
  carries k/node/namespace, and miss paths yield a discriminable miss_reason.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cacheseek.reuse.exact_prefix import (  # noqa: E402
    InMemoryTierStore,
    NamespaceForest,
    WorldKVConfig,
    WorldKVManager,
    build_action_chain,
    root_hash,
)
from cacheseek.reuse.exact_prefix.keys import canonical_json_bytes, sha256  # noqa: E402
from cacheseek.reuse.exact_prefix.strategy import ExactPrefixStrategy  # noqa: E402
from cacheseek.service.interfaces.strategy import Strategy  # noqa: E402
from cacheseek.service.query import CacheQuery  # noqa: E402
from cacheseek.service.result import FastForward  # noqa: E402


def _stack(break_even_k: int = 1):
    forest = NamespaceForest()
    mgr = WorldKVManager(
        forest, InMemoryTierStore(),
        WorldKVConfig(window_chunks=3, sink_chunks=1, break_even_k=break_even_k),
    )
    return forest, mgr, ExactPrefixStrategy(mgr)


def _root() -> bytes:
    return root_hash(image_fp=sha256(b"img"), prompt_fp=sha256(b"p"), config_blob_hash=sha256(b"cfg"))


def _ingest_chain(strategy, forest, root, actions):
    ns = forest.get_or_create_namespace(root, root)
    chain = build_action_chain(root, [canonical_json_bytes(a) for a in actions])
    q = CacheQuery(prompt="p", extra={"root_hash": root, "actions": actions})
    parent = ns.root
    for i, a in enumerate(actions):
        ctx = {
            "ns": ns, "parent": parent, "action": a, "node_key": chain[i], "depth": i,
            "payload": [(f"k{i}", f"v{i}")], "latent": f"x0@{i}",
        }
        asyncio.run(strategy.save(q, None, ctx))
        parent = ctx["node"]
    return q


def test_both_semantics_satisfy_strategy_protocol():
    assert issubclass(ExactPrefixStrategy, Strategy)
    try:
        from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache
    except Exception:  # skip the approximate half when torch/transformers et al. are unavailable
        import pytest
        pytest.skip("approximate strategy deps unavailable locally")
    assert issubclass(VideoBasedApproximateCache, Strategy)


def test_exact_prefix_save_lookup_roundtrip():
    forest, mgr, strategy = _stack()
    root = _root()
    q = _ingest_chain(strategy, forest, root, ["u", "u", "r"])

    res = asyncio.run(strategy.lookup(q))
    assert res.hit
    hint = res.resume_hint
    assert isinstance(hint, FastForward) and hint.k == 3
    assert hint.node is not None and hint.node.depth == 2
    assert hint.namespace is not None


def test_exact_prefix_miss_reasons():
    forest, mgr, strategy = _stack(break_even_k=5)
    root = _root()
    q = _ingest_chain(strategy, forest, root, ["u", "u"])

    r1 = asyncio.run(strategy.lookup(q))                       # 2 < threshold 5
    assert not r1.hit and r1.miss_reason == "exact_prefix_below_break_even"

    other = CacheQuery(prompt="p", extra={"root_hash": sha256(b"other"), "actions": ["u"]})
    r2 = asyncio.run(strategy.lookup(other))
    assert not r2.hit and r2.miss_reason == "exact_prefix_namespace_miss"

    r3 = asyncio.run(strategy.lookup(CacheQuery(prompt="p")))  # extra missing fields
    assert not r3.hit and r3.miss_reason == "exact_prefix_query_missing_fields"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            if type(e).__name__ == "Skipped":
                print(f"SKIP {fn.__name__}")
            else:
                raise
    print("all reuse strategy protocol tests passed")
