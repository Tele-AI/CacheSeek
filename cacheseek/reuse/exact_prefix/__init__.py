"""world_kv — exact-prefix, cross-request persistent KV reuse cache for
autoregressive video world models.

Primary use case: exact replay/continuation to save compute; reuse scope is
cross-request persistent. The structure is a forest of discrete-action-chain
tries, analogous to RadixAttention but with action prefixes instead of token
prefixes and chunk-KV instead of token-KV as the cache unit.

Deliberately independent of `cacheseek.kv_manager`: that implementation is not
relied upon or reused here.

Core invariant: VALUE = pure_function(KEY), otherwise a hit is incorrect.
Consequences:
  - seed is derived as `seed = derive_seed(node_key)`; it does not carry an RNG
    stream.
  - get-or-create dedup is inherently safe (same key always computes the same KV).
  - the version field must hash the real config blob (keys.config_blob_hash).
"""

from .config import ModelGeometry, WorldKVConfig, bytes_per_chunk_kv, calibrate_break_even_k
from .keys import build_action_chain, config_blob_hash, derive_seed, node_key, root_hash
from .manager import FastForwardResult, KVTierStore, RollingWindow, WorldKVManager
from cacheseek.stores.tier import InMemoryTierStore, TensorStoreTierStore
from .trie import (
    Namespace,
    NamespaceForest,
    PrefixMatch,
    load_forest_snapshot,
    save_forest_snapshot,
)
from .types import ActionKey, BlobHandle, NodeKey, RootHash, Skeleton, Tier, TrieNode

__all__ = [
    "ActionKey", "NodeKey", "RootHash", "Tier", "BlobHandle", "Skeleton", "TrieNode",
    "Namespace", "NamespaceForest", "PrefixMatch",
    "save_forest_snapshot", "load_forest_snapshot",
    "WorldKVManager", "WorldKVConfig", "ModelGeometry", "FastForwardResult",
    "KVTierStore", "RollingWindow", "InMemoryTierStore", "TensorStoreTierStore",
    "build_action_chain", "config_blob_hash", "derive_seed", "node_key", "root_hash",
    "bytes_per_chunk_kv", "calibrate_break_even_k",
]
