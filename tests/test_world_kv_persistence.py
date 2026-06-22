"""Cross-process prefix hit for world_kv: NamespaceForest snapshot persistence
and rebuild.

Demonstrates that the previously process-local, volatile gap in the index layer
(NamespaceForest) is closed:
  Process A ingests -> forest.snapshot();
  Process B (a brand-new forest with zero in-memory nodes) rebuilds from the
  snapshot -> lookup hits + materialize fetches from the persistent store.
torch-free: the same InMemoryTierStore instance spans both forests, simulating
"blobs persist, only the index is rebuilt" (same stubs as test_world_kv_flow).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from cacheseek.reuse.exact_prefix import (  # noqa: E402
    InMemoryTierStore,
    NamespaceForest,
    WorldKVConfig,
    WorldKVManager,
    build_action_chain,
    load_forest_snapshot,
    root_hash,
    save_forest_snapshot,
)
from cacheseek.reuse.exact_prefix.keys import canonical_json_bytes, sha256  # noqa: E402

pytestmark = pytest.mark.smoke

N_LAYERS = 2


class FakeWindow:
    """RollingWindow stub: records what gets seeded into it (same as test_world_kv_flow)."""

    def __init__(self) -> None:
        self.seeded: list[tuple[int, list, int]] = []
        self.resume_depth: int | None = None

    def seed_layer(self, layer, ordered_blobs, depth):
        self.seeded.append((layer, list(ordered_blobs), depth))

    def set_resume_depth(self, depth):
        self.resume_depth = depth


def _root() -> bytes:
    return root_hash(
        image_fp=sha256(b"img", b"A"),
        prompt_fp=sha256(b"p", b"A"),
        config_blob_hash=sha256(b"cfg", b"{}"),
    )


def _cfg() -> WorldKVConfig:
    return WorldKVConfig(window_chunks=3, sink_chunks=1, break_even_k=1)


def _payload(nk: bytes) -> list[str]:
    return [f"{nk.hex()[:8]}:L{i}" for i in range(N_LAYERS)]


def _ingest(mgr, forest, root, actions):
    ns = forest.get_or_create_namespace(root, b"cfgblob")
    chain = build_action_chain(root, [canonical_json_bytes(a) for a in actions])
    node = ns.root
    for i in range(len(actions)):
        node = mgr.ingest(
            ns, node, actions[i], chain[i], i, _payload(chain[i]), latent=f"x0@{i}"
        )
    return chain


def test_cross_process_prefix_hit_via_snapshot():
    root = _root()
    actions = ["u", "u", "r", "r", "d", "d"]
    store = InMemoryTierStore()  # "persistent" store: outlives both forest instances

    # Process A: ingest all 6 chunks -> snapshot
    forest_a = NamespaceForest()
    chain = _ingest(WorldKVManager(forest_a, store, _cfg()), forest_a, root, actions)
    assert forest_a.lookup(root, actions).matched_len == 6
    snap = forest_a.snapshot()
    assert snap["version"] == 1 and len(snap["namespaces"]) == 1

    # Process B: brand-new forest (zero in-memory nodes) rebuilt from the
    # snapshot, reusing the same persistent store.
    forest_b = NamespaceForest()
    assert len(forest_b) == 0
    assert forest_b.load_snapshot(snap) == 6
    assert len(forest_b) == 1

    # Cross-"process" prefix hit (after index rebuild)
    m = forest_b.lookup(root, actions)
    assert m.matched_len == 6 and m.namespace is not None
    assert m.node.has_kv and m.node.depth == 5

    # decode-only skeleton also recovers across "processes"
    assert m.node.skeleton is not None
    assert store.get_skeleton(m.node.skeleton.latent_locator) == "x0@5"

    # materialize fetches from the persistent store (rebuilt BlobHandle locator matches)
    win = FakeWindow()
    res = WorldKVManager(forest_b, store, _cfg()).try_fast_forward(root, actions, win)
    assert res.start_chunk == 6 and win.resume_depth == 5
    _, blobs0, _ = win.seeded[0]
    assert [d for d, _ in blobs0] == [0, 3, 4, 5]  # sink(d0) + most recent W=3
    assert [b for _, b in blobs0] == [_payload(chain[i])[0] for i in (0, 3, 4, 5)]

    # sink pin state survives the snapshot
    assert forest_b.lookup(root, actions[:1]).node.pinned


def test_snapshot_file_roundtrip(tmp_path):
    root = _root()
    actions = ["a", "b", "c"]
    store = InMemoryTierStore()
    forest_a = NamespaceForest()
    _ingest(WorldKVManager(forest_a, store, _cfg()), forest_a, root, actions)

    path = tmp_path / "sub" / "forest.json"
    save_forest_snapshot(forest_a, path)  # auto-creates parent dirs + atomic write
    assert path.exists()

    forest_b = load_forest_snapshot(path)
    assert len(forest_b) == 1
    assert forest_b.lookup(root, actions).matched_len == 3

    # Missing file => empty forest (cold start, no error)
    assert len(load_forest_snapshot(tmp_path / "nope.json")) == 0


def test_load_into_existing_forest():
    root = _root()
    actions = ["u", "u", "r"]
    store = InMemoryTierStore()
    forest_a = NamespaceForest()
    _ingest(WorldKVManager(forest_a, store, _cfg()), forest_a, root, actions)

    target = NamespaceForest()
    assert target.load_snapshot(forest_a.snapshot()) == 3
    assert target.lookup(root, actions).matched_len == 3


if __name__ == "__main__":  # the two fixture-free cases run directly
    test_cross_process_prefix_hit_via_snapshot()
    test_load_into_existing_forest()
    print(
        "PASS cross-process snapshot tests (run `pytest -m smoke` for the file-roundtrip case)"
    )
