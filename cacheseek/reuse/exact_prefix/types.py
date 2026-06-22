# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Core structural types for world_kv. Pure data — no torch, no I/O.

A cache ROW = KEY + VALUE:
  KEY   = node_key  (content-addresses root..this path)
  VALUE = skeleton (cheap, kept forever) + blob (heavy int4 KV, LRU-evictable)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cacheseek.stores.base import BlobHandle, Tier  # noqa: F401  (re-exported for compatibility)

# A single discrete action (keyboard-style ↑↓←→ … or action code). Discrete ⇒ exact key, no float ulp.
ActionKey = int | str
NodeKey = bytes   # = H(parent.node_key, action) — content-addresses the root..this path
RootHash = bytes  # = H(image_latent, prompt_emb, config_blob_hash) — namespace of one "world"


@dataclass(slots=True)
class Skeleton:
    """The cheap, kept-forever part of a node; structural info lives on TrieNode."""
    latent_locator: str   # x0 output of this chunk (returns history frames without re-denoising/decoding)


@dataclass(slots=True)
class TrieNode:
    """Reusable unit for one chunk ≡ one ROW."""
    node_key: NodeKey
    action: ActionKey                 # verified on hit (cheap collision guard)
    depth: int                        # = seg_index; sets RoPE time-axis origin + write position
    parent: TrieNode | None
    children: dict[ActionKey, TrieNode] = field(default_factory=dict)
    skeleton: Skeleton | None = None
    blob: BlobHandle | None = None     # None ⇒ "skeleton hit" (KV evicted; resume must recompute)
    ref_count: int = 0                 # >0 while materialize/write/offload in flight; eviction must not reclaim
    last_access: float = 0.0
    pinned: bool = False               # sink + hot prefix

    @property
    def has_kv(self) -> bool:
        """True iff this node has a published (ready) heavy KV blob.

        False after blob eviction (skeleton-only) or while an async write is in
        flight; lookup descends only through has_kv nodes so prefix reuse stops
        at the deepest published node.
        """
        return self.blob is not None and self.blob.ready
