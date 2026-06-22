# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""The forest: root_hash → Namespace; each Namespace a discrete-action trie.

Read-heavy, write-light: lookup is lock-free; commit takes a per-namespace write lock
only when get-or-create'ing a new child. Never a global lock (one world is exactly the
high-sharing, high-concurrency case, and locking would serialize what most needs to run in parallel).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ActionKey, BlobHandle, NodeKey, RootHash, Skeleton, Tier, TrieNode

SNAPSHOT_VERSION = 1


@dataclass(slots=True)
class Namespace:
    """One "world" = one (image, prompt, version). Owns one action-path trie."""

    root_hash: RootHash
    config_blob_hash: bytes  # version invariant — hashes the real config blob, not the model name
    root: TrieNode  # virtual root (depth=-1); children = first-action nodes
    pinned: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)  # write only; reads are lock-free


@dataclass(slots=True)
class PrefixMatch:
    """Result of descending the action trie: the deepest published node and how
    many action chunks matched."""

    namespace: Namespace | None  # None ⇒ namespace miss (brand-new world, cold start)
    node: TrieNode | None  # deepest matched node (virtual root when K==0)
    matched_len: int  # K — number of matched action chunks


class NamespaceForest:
    """The forest of namespaces (root_hash -> Namespace), each owning one action trie.

    Read-heavy and write-light: ``lookup`` is lock-free; mutating operations take
    a narrow lock (the forest map for namespace get-or-create, the per-namespace
    lock for node get-or-create) so concurrent reuse within one world is not
    serialized.
    """

    def __init__(self) -> None:
        self._namespaces: dict[RootHash, Namespace] = {}
        self._lock = threading.Lock()  # guards only get-or-create on the namespace map

    def __len__(self) -> int:
        """Number of namespaces (= known "worlds"). Empty forest ⇒ 0 (cold-start signal)."""
        return len(self._namespaces)

    def resolve(self, root_hash: RootHash) -> Namespace | None:
        """Return the existing namespace for root_hash, or None if no such world is known."""
        return self._namespaces.get(root_hash)

    def get_or_create_namespace(
        self, root_hash: RootHash, config_blob_hash: bytes
    ) -> Namespace:
        """Get the namespace for root_hash, creating it (with a virtual root node) if absent.

        Thread-safe: the fast path is a lock-free read; on a miss the forest lock
        is taken and the entry re-checked before insertion, so concurrent callers
        share one namespace.
        """
        ns = self._namespaces.get(root_hash)
        if ns is not None:
            return ns
        with self._lock:
            ns = self._namespaces.get(root_hash)
            if ns is None:
                root = TrieNode(
                    node_key=root_hash, action=-1, depth=-1, parent=None, pinned=True
                )
                ns = Namespace(
                    root_hash=root_hash, config_blob_hash=config_blob_hash, root=root
                )
                self._namespaces[root_hash] = ns
            return ns

    def lookup(self, root_hash: RootHash, actions: Sequence[ActionKey]) -> PrefixMatch:
        """Two stages: (1) resolve namespace, (2) descend the action trie exactly (only ready nodes, action verified)."""
        ns = self._namespaces.get(root_hash)
        if ns is None:
            return PrefixMatch(namespace=None, node=None, matched_len=0)
        node, k = ns.root, 0
        for a in actions:
            child = node.children.get(a)
            if child is None or not child.has_kv or child.action != a:
                break
            node, k = child, k + 1
        return PrefixMatch(namespace=ns, node=node, matched_len=k)

    def commit(
        self,
        ns: Namespace,
        parent: TrieNode,
        action: ActionKey,
        node_key: NodeKey,
        depth: int,
    ) -> TrieNode:
        """Get-or-create a child. Concurrency-safe: since VALUE=f(KEY), two requests on the same
        (parent, action) compute identical KV — first wins, later reuses, dedup lossless."""
        existing = parent.children.get(action)
        if existing is not None:
            return existing
        with ns.lock:
            existing = parent.children.get(action)  # re-check under lock
            if existing is not None:
                return existing
            node = TrieNode(
                node_key=node_key, action=action, depth=depth, parent=parent
            )
            parent.children[action] = node
            return node

    # ----------------------------------------------------------- Snapshot persistence (cross-process hits)
    def snapshot(self) -> dict[str, Any]:
        """Serialize the trie topology (KEY + store locator, NOT the tensors) for cross-process recovery.

        Heavy KV blobs are already content-addressed and persisted by KVTierStore (locator =
        node_key.hex()); what is volatile is the index mapping (root, actions) to those locators.
        This method saves it so a new process can rebuild the forest and hit (requires a persistent
        store, e.g. TensorStoreTierStore over LocalDisk/Fluxon). Only ready (published) blobs are
        recorded; unpublished/missing ones are stored skeleton-only (after rebuild, lookup truncates
        there, gracefully degrading to the deepest complete node).
        """
        namespaces: list[dict[str, Any]] = []
        for ns in self._namespaces.values():
            nodes: list[dict[str, Any]] = []
            stack: list[TrieNode] = list(
                ns.root.children.values()
            )  # skip the virtual root; DFS over real nodes
            while stack:
                n = stack.pop()
                nodes.append(
                    {
                        "node_key": n.node_key.hex(),
                        "action": n.action,
                        "depth": n.depth,
                        "parent": n.parent.node_key.hex()
                        if n.parent is not None
                        else None,
                        "pinned": n.pinned,
                        "skeleton": n.skeleton.latent_locator
                        if n.skeleton is not None
                        else None,
                        "blob": (
                            {
                                "tier": n.blob.tier.value,
                                "locator": n.blob.locator,
                                "nbytes": n.blob.nbytes,
                                "n_layers": n.blob.n_layers,
                            }
                            if n.has_kv
                            else None  # persist only published blobs
                        ),
                    }
                )
                stack.extend(n.children.values())
            namespaces.append(
                {
                    "root_hash": ns.root_hash.hex(),
                    "config_blob_hash": ns.config_blob_hash.hex(),
                    "pinned": ns.pinned,
                    "nodes": nodes,
                }
            )
        return {"version": SNAPSHOT_VERSION, "namespaces": namespaces}

    def load_snapshot(self, data: Mapping[str, Any]) -> int:
        """Rebuild snapshot() output into THIS forest (merged per namespace, typically an empty
        forest at process start). Reconstructs BlobHandle(ready=True) / Skeleton so a new process
        can hit on lookup and materialize from the persistent store. Returns the number of nodes loaded."""
        loaded = 0
        for ns_data in data.get("namespaces", []):
            root_h = bytes.fromhex(ns_data["root_hash"])
            cbh = bytes.fromhex(ns_data["config_blob_hash"])
            ns = self.get_or_create_namespace(root_h, cbh)
            ns.pinned = bool(ns_data.get("pinned", ns.pinned))
            by_key: dict[str, TrieNode] = {ns.root.node_key.hex(): ns.root}
            for nd in sorted(
                ns_data.get("nodes", []), key=lambda x: x["depth"]
            ):  # parents before children
                parent_hex = nd.get("parent")
                parent = by_key.get(parent_hex) if parent_hex else ns.root
                if parent is None:  # parent not ready (shouldn't happen with depth sort) → defensive skip
                    continue
                n = TrieNode(
                    node_key=bytes.fromhex(nd["node_key"]),
                    action=nd["action"],
                    depth=int(nd["depth"]),
                    parent=parent,
                    pinned=bool(nd.get("pinned", False)),
                )
                if nd.get("skeleton"):
                    n.skeleton = Skeleton(latent_locator=nd["skeleton"])
                b = nd.get("blob")
                if b:
                    n.blob = BlobHandle(
                        tier=Tier(b["tier"]),
                        locator=b["locator"],
                        nbytes=int(b.get("nbytes", 0)),
                        n_layers=int(b["n_layers"]),
                        ready=True,
                    )
                parent.children[n.action] = n
                by_key[nd["node_key"]] = n
                loaded += 1
        return loaded


def save_forest_snapshot(forest: NamespaceForest, path: str | Path) -> None:
    """Atomically write forest.snapshot() to JSON (tmp + os.replace for crash atomicity, mirroring
    stores/local_file.py). torch-free; can be called between requests or at shutdown."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(forest.snapshot()), encoding="utf-8")
    os.replace(tmp, p)


def load_forest_snapshot(
    path: str | Path, *, into: NamespaceForest | None = None
) -> NamespaceForest:
    """Load a JSON snapshot into a (new or given) forest. Missing file ⇒ empty forest (cold start, no error)."""
    forest = into if into is not None else NamespaceForest()
    p = Path(path)
    if p.exists():
        forest.load_snapshot(json.loads(p.read_text(encoding="utf-8")))
    return forest
