"""KEY algorithm — model-agnostic hashing.

KEY structure:
    config_blob_hash = H( canonical(config) || weights_fingerprint )       # version invariant
    root_hash        = H( image_fp || prompt_fp || config_blob_hash )      # namespace = one "world"
    node_key         = H( parent_node_key || action_bytes )               # action chain; virtual root = root_hash
    seed(node)       = f( node_key )                                       # deterministic derived seed ⇒ value=f(key)

The model-specific parts (image_fp / prompt_fp / action_bytes / config fields) are supplied by a profile (see lingbot_fast.py).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def sha256(*parts: bytes) -> bytes:
    """Length-prefixed concatenation ⇒ unambiguous (avoids a||b colliding with a'||b')."""
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return h.digest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def config_blob_hash(config: Mapping[str, Any], *, weights_fingerprint: bytes) -> bytes:
    """Version invariant: hash the actual config blob itself + weights fingerprint.

    Must cover every setting that affects computation (schedule / resolution / chunk /
    window / quantization / RoPE / VAE …). Missing one means KV from old weights/config
    is fed to a new request — confidently hitting the wrong thing. Best kept as a tested invariant.
    """
    return sha256(b"cfg", canonical_json_bytes(dict(config)), weights_fingerprint)


def root_hash(*, image_fp: bytes, prompt_fp: bytes, config_blob_hash: bytes) -> bytes:
    """Namespace = one "world". KV can only be shared within the same root (changing image/prompt/version = a new world)."""
    return sha256(b"root", image_fp, prompt_fp, config_blob_hash)


def node_key(parent_node_key: bytes, action_bytes: bytes) -> bytes:
    return sha256(b"node", parent_node_key, action_bytes)


def build_action_chain(root: bytes, action_bytes_seq: Sequence[bytes]) -> list[bytes]:
    """Return the node_key per chunk (excluding the virtual root). chain[i] = H(chain[i-1] or root, action_i).

    node_key chains from root ⇒ it transitively encodes (root + all actions), globally unique across namespaces.
    """
    keys: list[bytes] = []
    prev = root
    for ab in action_bytes_seq:
        prev = node_key(prev, ab)
        keys.append(prev)
    return keys


def derive_seed(node_key_bytes: bytes) -> int:
    """Deterministic seed = f(node_key).

    Does NOT depend on a per-request master_seed, nor only on chunk_idx (this corrects
    the `kv_manager.hash_keys.derive_sub_seed(master_seed, chunk_idx)` approach) —
    otherwise value≠f(key) across requests: the same node would compute different KV in
    different requests/branches, poisoning the cache.

    The engine must use it to drive BOTH the initial latent and every step's randn_like
    for the chunk, or resume will silently diverge from the cached chunk. For a "different
    but internally consistent world", fold the variation/session salt into config_blob
    (⇒ into root ⇒ into node_key) rather than changing the seed source.
    """
    return int.from_bytes(sha256(b"seed", node_key_bytes)[:8], "little")
