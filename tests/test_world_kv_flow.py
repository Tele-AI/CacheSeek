# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""End-to-end flow tests for world_kv.

A fake engine drives the full loop: cold request generation -> ingest ->
fast-forward on a same-prefix request -> fork into a new branch -> graceful
degradation after eviction. No torch / real model: KV payloads are string
placeholders (the flow and data flow are independent of the payload type).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cacheseek.reuse.exact_prefix import (  # noqa: E402
    InMemoryTierStore,
    NamespaceForest,
    WorldKVConfig,
    WorldKVManager,
    build_action_chain,
    derive_seed,
    root_hash,
)
from cacheseek.reuse.exact_prefix.keys import canonical_json_bytes, sha256  # noqa: E402

N_LAYERS = 2


class FakeWindow:
    """RollingWindow stub: records what gets seeded into it."""

    def __init__(self) -> None:
        self.seeded: list[tuple[int, list, int]] = []   # (layer, ordered_blobs, depth)
        self.resume_depth: int | None = None

    def seed_layer(self, layer, ordered_blobs, depth):
        self.seeded.append((layer, list(ordered_blobs), depth))

    def set_resume_depth(self, depth):
        self.resume_depth = depth


def make_root(image: str = "imgA", prompt: str = "promptA") -> bytes:
    return root_hash(
        image_fp=sha256(b"img", image.encode()),
        prompt_fp=sha256(b"p", prompt.encode()),
        config_blob_hash=sha256(b"cfg", b"{}"),
    )


def make_stack(break_even_k: int = 1):
    forest = NamespaceForest()
    store = InMemoryTierStore()
    cfg = WorldKVConfig(window_chunks=3, sink_chunks=1, break_even_k=break_even_k)
    return forest, store, WorldKVManager(forest, store, cfg)


def payload_of(nk: bytes) -> list[str]:
    return [f"{nk.hex()[:8]}:L{layer}" for layer in range(N_LAYERS)]


def drive(mgr, forest, root, actions, *, start=0, parent=None):
    """Fake engine: from chunk `start`, "generate" and ingest chunk by chunk.
    Returns the last node."""
    ns = forest.get_or_create_namespace(root, b"cfgblob")
    chain = build_action_chain(root, [canonical_json_bytes(a) for a in actions])
    node = parent if parent is not None else ns.root
    for i in range(start, len(actions)):
        node = mgr.ingest(
            ns, node, actions[i], chain[i], i,
            payload_of(chain[i]), latent=f"x0@{i}",
        )
    return ns, node, chain


# ---------------------------------------------------------------------- tests
def test_cold_then_fast_forward():
    forest, store, mgr = make_stack()
    root = make_root()
    actions = ["u", "u", "r", "r", "d", "d"]

    # Request 0: namespace miss => cold start from scratch
    res0 = mgr.try_fast_forward(root, actions, FakeWindow())
    assert res0.start_chunk == 0 and res0.namespace is None

    # Request A: cold-generate all 6 chunks and ingest
    ns, tip, chain = drive(mgr, forest, root, actions)
    assert tip.depth == 5 and tip.has_kv

    # Request B: same 6 actions => full prefix hit, fast-forward skips 6 chunks
    win = FakeWindow()
    res = mgr.try_fast_forward(root, actions, win)
    assert res.start_chunk == 6
    assert res.node is tip
    assert win.resume_depth == 5
    # Window = sink(d0) + most recent W=3 (d3,d4,d5); (depth, payload) seeded
    # per layer, oldest->newest.
    assert len(win.seeded) == N_LAYERS
    layer0, blobs0, _ = win.seeded[0]
    assert layer0 == 0 and len(blobs0) == 4
    assert [d for d, _ in blobs0] == [0, 3, 4, 5]
    assert [b for _, b in blobs0] == [payload_of(chain[i])[0] for i in (0, 3, 4, 5)]

    # sink is pinned
    d0 = forest.lookup(root, actions[:1]).node
    assert d0.pinned


def test_branch_fork():
    forest, store, mgr = make_stack()
    root = make_root()
    a_main = ["u", "u", "r", "r", "d", "d"]
    drive(mgr, forest, root, a_main)

    # Request C: first 4 actions match, forks at the 5th => fast-forward 4,
    # then continue and attach a new branch.
    a_fork = a_main[:4] + ["x", "x"]
    win = FakeWindow()
    res = mgr.try_fast_forward(root, a_fork, win)
    assert res.start_chunk == 4 and res.node.depth == 3
    drive(mgr, forest, root, a_fork, start=4, parent=res.node)

    # trie forks at the depth-3 node: children = {"d", "x"}
    d3 = forest.lookup(root, a_main[:4]).node
    assert set(d3.children.keys()) == {"d", "x"}
    # Both branches hit the full chain.
    assert mgr.try_fast_forward(root, a_main, FakeWindow()).start_chunk == 6
    assert mgr.try_fast_forward(root, a_fork, FakeWindow()).start_chunk == 6


def test_eviction_graceful_degradation():
    forest, store, mgr = make_stack()
    root = make_root()
    actions = ["u", "u", "r", "r", "d", "d"]
    ns, _, chain = drive(mgr, forest, root, actions)

    d4 = forest.lookup(root, actions[:5]).node
    assert mgr.evict_blob(d4) and d4.blob is None and d4.skeleton is not None
    # Skeleton-hit state: the latent is still present (can return history frames)
    assert store.get_skeleton(d4.skeleton.latent_locator) == "x0@4"

    # lookup truncates at d4 => auto-degrades to reusing the first 4 chunks
    win = FakeWindow()
    res = mgr.try_fast_forward(root, actions, win)
    assert res.start_chunk == 4 and win.resume_depth == 3

    # Continuation refills d4/d5 (get-or-create hits the old nodes, blob republished)
    drive(mgr, forest, root, actions, start=4, parent=res.node)
    assert mgr.try_fast_forward(root, actions, FakeWindow()).start_chunk == 6

    # pinned sink cannot be evicted
    d0 = forest.lookup(root, actions[:1]).node
    assert not mgr.evict_blob(d0)


def test_namespace_isolation_and_seed():
    forest, store, mgr = make_stack()
    actions = ["u", "u", "r"]
    root_a, root_b = make_root("imgA"), make_root("imgB")
    drive(mgr, forest, root_a, actions)

    # Different image = different world: no hit even with identical actions
    assert mgr.try_fast_forward(root_b, actions, FakeWindow()).start_chunk == 0

    # seed = f(node_key): deterministic, and distinct per node / across worlds
    chain_a = build_action_chain(root_a, [canonical_json_bytes(a) for a in actions])
    chain_b = build_action_chain(root_b, [canonical_json_bytes(a) for a in actions])
    assert derive_seed(chain_a[0]) == derive_seed(chain_a[0])
    assert derive_seed(chain_a[0]) != derive_seed(chain_a[1])
    assert derive_seed(chain_a[0]) != derive_seed(chain_b[0])


def test_break_even_gate():
    forest, store, mgr = make_stack(break_even_k=5)
    root = make_root()
    actions = ["u", "u", "r", "r"]          # only a 4-chunk prefix
    drive(mgr, forest, root, actions)
    # matched 4 < break_even_k=5 => no fast-forward, degrades to from-scratch (harmless)
    res = mgr.try_fast_forward(root, actions, FakeWindow())
    assert res.start_chunk == 0 and res.namespace is not None


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all world_kv flow tests passed")
