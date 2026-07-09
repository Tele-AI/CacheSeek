# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""WorldKVManager — four actions: lookup → materialize(fast-forward) → ingest(write-through) → evict.

Compute-saving path (exact prefix):
    request(image, prompt, config, actions)
      root = H(image_lat, prompt_emb, config_blob_hash)
      res  = mgr.try_fast_forward(root, actions, window)
      if res.start_chunk == 0:  generate normally from scratch        # cold/too short ⇒ no cache, harmless
      else:                     resume from chunk res.start_chunk      # skip denoising the first K chunks
      on each chunk finalize → mgr.ingest(...)  (write-through, extends a trie branch; mount point = res.node)

I/O goes through KVTierStore (see store.py: InMemory first for functionality; Fluxon/async is the perf phase).
This module does not depend on cacheseek.kv_manager.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cacheseek.quant import KVLayerCodec, KVQuantizedLayer, build_kv_codec_from_config

from .config import WorldKVConfig
from .trie import Namespace, NamespaceForest, PrefixMatch
from .types import ActionKey, BlobHandle, NodeKey, RootHash, Skeleton, TrieNode


class KVTierStore(Protocol):
    """Tiered blob store. Functional phase: synchronous is fine; perf phase: truly async + overlapped with compute."""
    def put_async(self, locator: str, payload: Sequence[Any], *, tier: Any, on_ready: Any = None) -> None:
        """Write a heavy KV blob (one element per layer) under locator, on the given tier.

        The write may complete asynchronously; ``on_ready`` (if given) is invoked
        once the blob is durable, which is what publishes it to readers (sets the
        BlobHandle ready). Until then the blob must not be visible to lookups.

        Args:
            locator: Content-addressed key for the blob (node_key.hex() + ":kv").
            payload: Per-layer KV data to persist.
            tier: Target storage tier.
            on_ready: Optional callback fired when the write is durable.
        """
        ...
    def get_layer(self, handle: BlobHandle, layer: int) -> Any:
        """Read back one layer's KV for a previously written blob.

        Args:
            handle: Handle to a ready blob (from a prior put_async).
            layer: Zero-based layer index in [0, handle.n_layers).

        Returns:
            The stored payload element for that layer.
        """
        ...
    def put_skeleton(self, locator: str, latent: Any) -> None:
        """Persist the cheap skeleton latent (kept forever, never LRU-evicted)."""
        ...
    def free(self, handle: BlobHandle) -> None:
        """Release the heavy KV blob backing handle (eviction); the skeleton is untouched."""
        ...


class RollingWindow(Protocol):
    """The engine-held active kv_pool (the rolling self-KV pool on the TeleFuser side).

    materialize seeds it with the sink+W historical KV; generation appends; finalized chunks roll off.
    blobs are given as ``(chunk_depth, layer_payload)`` (oldest→newest); the adapter assembles them
    into the engine's own physical layout (full-length / rolling ring).
    """
    def seed_layer(self, layer: int, blobs: Sequence[tuple[int, Any]], depth: int) -> None:
        """Seed one attention layer's KV pool with the sink + window historical KV.

        Args:
            layer: Zero-based attention layer index to seed.
            blobs: ``(chunk_depth, layer_payload)`` pairs ordered oldest -> newest;
                the adapter assembles them into the engine's physical layout
                (full-length or rolling ring).
            depth: Depth of the resume node, used to place blobs at the correct
                ring positions.
        """
        ...
    def set_resume_depth(self, depth: int) -> None:
        """Set the resume position: RoPE time-axis offset and the next write slot for chunk depth+1."""
        ...                  # sets RoPE offset + write position


@dataclass(slots=True)
class FastForwardResult:
    """Outcome of try_fast_forward: where to resume and the trie handles for the next ingest."""

    start_chunk: int                 # 0 ⇒ compute from scratch (no reuse)
    node: TrieNode | None            # resume mount point (parent for the next ingest); the virtual root may serve too
    namespace: Namespace | None      # None ⇒ namespace miss (caller must get_or_create)


class WorldKVManager:
    """Orchestrates the four cache actions over a trie forest and tiered store:
    lookup -> materialize (fast-forward) -> ingest (write-through) -> evict.

    Holds no per-request mutable state; the trie forest and store carry durable
    state. Does not depend on ``cacheseek.kv_manager``.
    """

    def __init__(self, forest: NamespaceForest, store: KVTierStore, cfg: WorldKVConfig,
                 *, clock: Any = None) -> None:
        """Wire the manager to its forest, store, and config.

        Args:
            forest: The namespace forest holding all action tries.
            store: Tiered blob store for KV blobs and skeleton latents.
            cfg: Window geometry, break-even gate, and quant/commit settings.
            clock: Optional injectable time source (seconds, monotonic) for
                last_access bookkeeping; defaults to a constant for determinism.
        """
        self.forest = forest
        self.store = store
        self.cfg = cfg
        self._kv_codec: KVLayerCodec | None = build_kv_codec_from_config(cfg)
        self._now = clock or (lambda: 0.0)   # injectable clock for testing/determinism

    # ---------------------------------------------------------------- Compute-saving entry
    def try_fast_forward(
        self,
        root_hash: RootHash,
        actions: Sequence[ActionKey],
        window: RollingWindow,
    ) -> FastForwardResult:
        """Find the longest exact prefix; if long enough (≥break_even_k), materialize it into the active window.

        Returns start_chunk=K ⇒ the engine generates from chunk K (denoising of the first K chunks is skipped).
        """
        m: PrefixMatch = self.forest.lookup(root_hash, actions)
        if m.namespace is None:
            return FastForwardResult(start_chunk=0, node=None, namespace=None)
        if m.node is None or m.matched_len == 0 or m.matched_len < self.cfg.break_even_k:
            return FastForwardResult(start_chunk=0, node=m.namespace.root, namespace=m.namespace)
        if not self.materialize(m.node, window):
            # incomplete window (lookup shouldn't return such a node in theory; fall back to from-scratch)
            return FastForwardResult(start_chunk=0, node=m.namespace.root, namespace=m.namespace)
        return FastForwardResult(start_chunk=m.matched_len, node=m.node, namespace=m.namespace)

    def materialize(self, node: TrieNode, window: RollingWindow) -> bool:
        """Gather up the path: sink + the most recent W ancestors' KV → seed into the active ring.

        Done per layer so it can (in the perf phase) overlap with prefetch; moves O(W+sink), not O(full history).
        K already carries the absolute RoPE phase (baked in before storage), not re-applied; pointers computed from depth.
        """
        path = self._window_path(node)               # oldest → newest
        if not path or any(not n.has_kv for n in path):
            return False
        for n in path:
            n.ref_count += 1                         # materialize in flight; eviction must not reclaim
        try:
            n_layers = path[-1].blob.n_layers        # type: ignore[union-attr]
            for layer in range(n_layers):
                blobs = [
                    (n.depth, self._decode_layer_payload(self.store.get_layer(n.blob, layer)))
                    for n in path
                ]
                window.seed_layer(layer, blobs, depth=node.depth)
            window.set_resume_depth(node.depth)
            now = self._now()
            for n in path:
                n.last_access = now
        finally:
            for n in path:
                n.ref_count -= 1
        return True

    # ---------------------------------------------------------------- Write path
    def ingest(
        self,
        ns: Namespace,
        parent: TrieNode,
        action: ActionKey,
        node_key: NodeKey,
        depth: int,
        kv_payload: Sequence[Any],
        latent: Any,
        *,
        n_layers: int | None = None,
    ) -> TrieNode:
        """Write on chunk finalize (write-through; the two pools are decoupled: roll-off is ingest, not evict).

        In the functional phase the store is synchronous; once truly async, a chunk lives in the ring
        for W more chunks before rolling off, so the window's lifetime is long enough for the async put
        to drain ⇒ copy-before-overwrite is not triggered at normal cadence.
        """
        node = self.forest.commit(ns, parent, action, node_key, depth)
        if depth < self.cfg.sink_chunks:
            node.pinned = True                       # sink stays forever (every materialize needs it)
        loc = node_key.hex()
        # skeleton (cheap, visible first)
        self.store.put_skeleton(loc + ":lat", latent)
        node.skeleton = Skeleton(latent_locator=loc + ":lat")
        # blob (heavy; store's write callback sets ready ⇒ only then published to readers)
        stored_payload = self._encode_kv_payload(kv_payload)
        handle = BlobHandle(
            tier=self.cfg.commit_tier,
            locator=loc + ":kv",
            nbytes=self._stored_payload_nbytes(stored_payload),
            n_layers=len(stored_payload) if n_layers is None else n_layers,
            ready=False,
        )
        node.blob = handle

        def _publish() -> None:
            handle.ready = True

        self.store.put_async(handle.locator, stored_payload, tier=self.cfg.commit_tier, on_ready=_publish)
        return node

    # ---------------------------------------------------------------- Eviction
    def evict_blob(self, node: TrieNode) -> bool:
        """Blob LRU: drop the heavy KV but keep the skeleton (latent + structure). Never touch pinned/ref>0.

        After dropping the blob the node degrades to a "skeleton hit" state: lookup truncates there
        (only honors has_kv), and prefix reuse auto-degrades to the deepest complete node —
        graceful degradation, not an error.
        """
        if node.pinned or node.ref_count > 0 or node.blob is None:
            return False
        self.store.free(node.blob)
        node.blob = None                             # structure unchanged, just no heavy KV
        return True

    # ---------------------------------------------------------------- Internal
    def _window_path(self, node: TrieNode) -> list[TrieNode]:
        """``sink + up to W ancestors above node``, oldest→newest, deduplicated.

        lookup's invariant (descend only along has_kv) guarantees the matched path has KV throughout,
        so blocks are usually not missing here; materialize still keeps a guard as a fallback.
        """
        recent: list[TrieNode] = []
        n: TrieNode | None = node
        while n is not None and n.depth >= 0 and len(recent) < self.cfg.window_chunks:
            recent.append(n)
            n = n.parent
        sinks: list[TrieNode] = []
        while n is not None and n.depth >= 0:        # keep walking up to collect sinks (depth < sink_chunks)
            if n.depth < self.cfg.sink_chunks:
                sinks.append(n)
            n = n.parent
        return list(reversed(sinks)) + list(reversed(recent))

    def _encode_kv_payload(self, kv_payload: Sequence[Any]) -> list[Any]:
        """Encode per-layer KV payloads when quantization is enabled."""

        if self._kv_codec is None:
            return list(kv_payload)

        encoded: list[Any] = []
        for layer, payload in enumerate(kv_payload):
            if not isinstance(payload, (tuple, list)) or len(payload) != 2:
                raise TypeError(
                    "quantized WorldKV ingest expects each layer payload to be a "
                    f"(key, value) pair; layer {layer} got {type(payload).__name__}"
                )
            encoded.append(self._kv_codec.encode_layer(payload[0], payload[1]))
        return encoded

    def _decode_layer_payload(self, payload: Any) -> Any:
        """Decode one stored layer payload back to runtime KV form."""

        if self._kv_codec is None:
            return payload
        if not isinstance(payload, KVQuantizedLayer):
            raise TypeError(
                "quantized WorldKV materialize expected KVQuantizedLayer from the store, "
                f"got {type(payload).__name__}"
            )
        return self._kv_codec.decode_layer(payload)

    def _stored_payload_nbytes(self, payload: Sequence[Any]) -> int:
        """Best-effort byte size for accounting after optional quantization."""

        total = 0
        for layer_payload in payload:
            total += self._object_nbytes(layer_payload)
        return total

    def _object_nbytes(self, value: Any) -> int:
        if isinstance(value, KVQuantizedLayer):
            return (
                self._object_nbytes(value.key.qdata)
                + self._object_nbytes(value.key.scale)
                + self._object_nbytes(value.key.offset)
                + self._object_nbytes(value.value.qdata)
                + self._object_nbytes(value.value.scale)
                + self._object_nbytes(value.value.offset)
            )
        if isinstance(value, (tuple, list)):
            return sum(self._object_nbytes(v) for v in value)
        if value is None:
            return 0
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        element_size = getattr(value, "element_size", None)
        numel = getattr(value, "numel", None)
        if callable(element_size) and callable(numel):
            return int(element_size() * numel())
        return 0
