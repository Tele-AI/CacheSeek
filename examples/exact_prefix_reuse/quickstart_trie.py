# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Understand exact_prefix reuse in 60 seconds -- pure Python, no GPU, no weights.

Uses string stand-ins as the KV payload, exercises the real trie/manager code path, and
demonstrates four core behaviors:

    1. A cold session generates 6 chunks and writes them to the cache (action-chain trie)
    2. A new session with the same trajectory -> whole-chain hit, all 6 chunks skipped
    3. A session matching the first 4 steps then forking -> prefix hit of 4, a new branch
       grows at the fork point
    4. Evicting the heavy KV of a middle chunk -> graceful degradation (the hit shrinks to
       the full prefix; the light skeleton latent is still readable)

Run: python examples/exact_prefix_reuse/quickstart_trie.py
For the real-GPU e2e (real LingBot model, byte-exact equivalence assertions) see the
README in this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root: avoid pip install -e

from cacheseek.reuse.exact_prefix import (
    InMemoryTierStore,
    NamespaceForest,
    WorldKVConfig,
    WorldKVManager,
    build_action_chain,
    root_hash,
)
from cacheseek.reuse.exact_prefix.keys import canonical_json_bytes, sha256


class PrintWindow:
    """RollingWindow stand-in: prints what gets seeded into the active window."""

    def seed_layer(self, layer, blobs, depth):
        if layer == 0:
            print(f"      seeding window: chunks {[d for d, _ in blobs]} (resume depth={depth})")

    def set_resume_depth(self, _depth):
        pass


def generate(mgr, forest, root, actions, *, start=0, parent=None, tag=""):
    """Simulate the engine "generating" chunk by chunk and writing back to the cache."""
    ns = forest.get_or_create_namespace(root, root)
    chain = build_action_chain(root, [canonical_json_bytes(a) for a in actions])
    node = parent if parent is not None else ns.root
    for i in range(start, len(actions)):
        payload = [f"{tag}kv@{i}"]                       # real case = per-layer (k,v) tensors
        node = mgr.ingest(ns, node, actions[i], chain[i], i, payload, latent=f"x0@{i}")
    print(f"      recomputed chunks {start}..{len(actions) - 1}")
    return node


def main() -> int:
    forest = NamespaceForest()
    store = InMemoryTierStore()
    mgr = WorldKVManager(forest, store, WorldKVConfig(window_chunks=3, sink_chunks=1, break_even_k=1))
    root = root_hash(image_fp=sha256(b"img"), prompt_fp=sha256(b"prompt"), config_blob_hash=sha256(b"cfg"))

    walk = ["forward", "forward", "left", "left", "forward", "forward"]

    print("\n[1] Cold session: cache empty, everything recomputed")
    res = mgr.try_fast_forward(root, walk, PrintWindow())
    assert res.start_chunk == 0
    generate(mgr, forest, root, walk)

    print("\n[2] Identical-trajectory replay: full-chain hit, 0 chunks need recompute")
    res = mgr.try_fast_forward(root, walk, PrintWindow())
    assert res.start_chunk == 6
    print(f"      fast-forward = {res.start_chunk}/6 chunks (all recompute skipped)")

    fork = walk[:4] + ["right", "right"]
    print("\n[3] Forked trajectory (first 4 steps identical, then turn right): prefix hit of 4, only 2 recomputed")
    res = mgr.try_fast_forward(root, fork, PrintWindow())
    assert res.start_chunk == 4
    generate(mgr, forest, root, fork, start=4, parent=res.node, tag="fork-")
    d3 = forest.lookup(root, walk[:4]).node
    print(f"      trie forks at depth 3, children = {sorted(d3.children)}")

    print("\n[4] Evict chunk 4's heavy KV → hit gracefully shortens; light skeleton latent remains")
    d4 = forest.lookup(root, walk[:5]).node
    mgr.evict_blob(d4)
    res = mgr.try_fast_forward(root, walk, PrintWindow())
    assert res.start_chunk == 4
    print(f"      fast-forward degrades to {res.start_chunk}/6; skeleton latent = {store.get_skeleton(d4.skeleton.latent_locator)!r}")

    print("\n✔ All assertions passed: hit/fork/degradation behavior matches the real e2e (same code path)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
