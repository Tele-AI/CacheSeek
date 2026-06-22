# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""KVTierStore implementations.

- InMemoryTierStore: in-process dict. ``put_async`` runs synchronously and calls
  ``on_ready`` immediately. For end-to-end integration and tests; no real data
  movement, no tiering, no async queue.
- TensorStoreTierStore: adapts any duck-typed tensor store (``put_tensor(key, t)``
  / ``get_tensor(key) -> t``, e.g. Fluxon TensorKVStore) into a KVTierStore,
  storing each layer's payload under its own key.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import queue
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from .base import BlobHandle, Tier


class InMemoryTierStore:
    """In-process reference implementation for integration and tests."""

    def __init__(self) -> None:
        self._blobs: dict[str, list[Any]] = {}      # locator -> per-layer payload
        self._skeletons: dict[str, Any] = {}        # locator -> latent

    def put_async(
        self,
        locator: str,
        payload: Sequence[Any],
        *,
        tier: Tier,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Store the per-layer payload synchronously and call ``on_ready`` at once."""
        self._blobs[locator] = list(payload)
        if on_ready is not None:
            on_ready()

    def get_layer(self, handle: BlobHandle, layer: int) -> Any:
        """Return the stored payload for one layer of the blob at ``handle``."""
        return self._blobs[handle.locator][layer]

    def put_skeleton(self, locator: str, latent: Any) -> None:
        """Store the per-node skeleton latent under ``locator``."""
        self._skeletons[locator] = latent

    def get_skeleton(self, locator: str) -> Any:
        """Return the skeleton latent for ``locator``, or None if absent."""
        return self._skeletons.get(locator)

    def free(self, handle: BlobHandle) -> None:
        """Drop the blob at ``handle`` from the in-process dict (idempotent)."""
        self._blobs.pop(handle.locator, None)


class LocalDiskTensorStore:
    """Local-disk tensor backend (``put_tensor/get_tensor`` duck interface, paired
    with TensorStoreTierStore).

    Raw bytes, one file per key (no pickle overhead); the bytes ``put/get`` path
    serves the ``:spec`` sidecar. It goes through the same adapter as the Fluxon
    backend (async writes, spec bookkeeping), so the backend is the only variable
    when comparing against a baseline.
    Note: writes go through the page cache (no fsync), matching ordinary file-server
    semantics; sustained throughput is still disk-bound.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / (hashlib.sha1(key.encode()).hexdigest() + ".bin")

    def put_tensor(self, key: str, t: Any) -> None:
        """Write the tensor's raw little-endian bytes to one file keyed by ``key``."""
        import torch
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        self._path(key).write_bytes(raw)

    def get_tensor(self, key: str, *, shape: Any, dtype: Any) -> Any:
        """Reload a tensor from disk, reinterpreting the raw bytes as ``shape``/``dtype``.

        Returns None on a miss (file absent). ``shape``/``dtype`` are required
        because the on-disk buffer is raw and not self-describing.
        """
        import numpy as np
        import torch
        p = self._path(key)
        if not p.exists():
            return None
        arr = np.fromfile(p, dtype=np.uint8)
        return torch.from_numpy(arr).view(dtype).reshape(tuple(shape))

    def put(self, key: str, val: bytes) -> None:
        """Write raw bytes to the file for ``key`` (serves the ``:spec`` sidecar)."""
        self._path(key).write_bytes(val)

    def get(self, key: str) -> bytes | None:
        """Read the raw bytes for ``key``, or None if the file does not exist."""
        p = self._path(key)
        return p.read_bytes() if p.exists() else None


class TensorStoreTierStore:
    """Adapts any ``put_tensor/get_tensor``-style tensor store (e.g. Fluxon
    TensorKVStore).

    Fluxon's ``get_tensor(key, *, shape, dtype)`` requires those keyword-only args:
    it stores a raw buffer, so reads must supply the view spec. Therefore each
    key's (shape, dtype) is recorded on put — in an in-process dict as the primary
    path, and, if the backend exposes bytes ``put/get``, also written to a
    ``:spec`` sidecar (JSON) so the spec survives across processes.

    True async put: with ``async_put=True``, ``put_async`` only enqueues onto a
    bounded queue and returns (a full queue blocks, providing backpressure). A
    single worker thread drains it and calls ``on_ready`` only after all writes
    complete, publishing ``BlobHandle.ready=True``. Lookups consider only ready
    nodes, so in-flight writes are invisible to readers and correctness holds
    automatically; write latency leaves the chunk-loop critical path.
    Correctness precondition: payloads must already be caller-cloned CPU tensors
    (the binding clones at finalization — copy-before-overwrite — so ring-slot
    reuse cannot corrupt in-flight data).
    ``flush()`` drains the queue; calling it between requests guarantees a definite
    visibility point.
    """

    def __init__(self, tensor_store: Any, *, async_put: bool = False, max_pending_chunks: int = 4) -> None:
        self._ts = tensor_store
        self._specs: dict[str, tuple[tuple[int, ...], str]] = {}   # key -> (shape, dtype name)
        self._async = async_put
        if async_put:
            self._q: queue.Queue = queue.Queue(maxsize=max_pending_chunks)
            self._worker = threading.Thread(target=self._drain, name="worldkv-put", daemon=True)
            self._worker.start()

    @staticmethod
    def _layer_key(locator: str, layer: int) -> str:
        return f"{locator}:L{layer}"

    # ------------------------------------------------------------- spec bookkeeping
    def _put_one(self, key: str, t: Any) -> None:
        spec = (tuple(t.shape), str(t.dtype).replace("torch.", ""))
        self._specs[key] = spec
        self._ts.put_tensor(key, t)
        if hasattr(self._ts, "put"):                       # sidecar: recoverable across processes
            with contextlib.suppress(Exception):
                self._ts.put(key + ":spec", json.dumps(spec).encode())

    def _get_one(self, key: str) -> Any:
        spec = self._specs.get(key)
        if spec is None and hasattr(self._ts, "get"):
            raw = self._ts.get(key + ":spec")
            if raw is not None:
                shape, dt = json.loads(raw)
                spec = (tuple(shape), dt)
                self._specs[key] = spec
        if spec is None:
            return None
        import torch
        return self._ts.get_tensor(key, shape=spec[0], dtype=getattr(torch, spec[1]))

    # ----------------------------------------------------------------- async write
    def _do_put_payload(self, locator: str, payload: Sequence[Any]) -> None:
        for i, p in enumerate(payload):
            if isinstance(p, (tuple, list)) and len(p) == 2:
                # per-layer payload = (k, v) tensor pair -> two keys (put_tensor
                # takes a single tensor)
                self._put_one(self._layer_key(locator, i) + ":k", p[0])
                self._put_one(self._layer_key(locator, i) + ":v", p[1])
            else:
                self._put_one(self._layer_key(locator, i), p)

    def _drain(self) -> None:
        while True:
            locator, payload, on_ready = self._q.get()
            try:
                self._do_put_payload(locator, payload)
                if on_ready is not None:
                    on_ready()                       # publish only after all writes land (ready=True)
            except Exception:                        # ready stays False -> node invisible, safe
                logger.exception(f"[world_kv] async put failed locator={locator}")
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """Drain in-flight writes. Calling between requests gives a definite
        visibility point; otherwise only the latest chunk is briefly not reusable.
        """
        if self._async:
            self._q.join()

    # ---------------------------------------------------------------- KVTierStore
    def put_async(
        self,
        locator: str,
        payload: Sequence[Any],
        *,
        tier: Tier,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Persist a node's per-layer payload, one tensor store key per layer.

        In async mode only enqueues onto the bounded queue and returns (a full
        queue blocks for backpressure); the worker calls ``on_ready`` after all
        writes land. In sync mode writes inline then calls ``on_ready``.
        """
        if self._async:
            self._q.put((locator, payload, on_ready))   # blocks when full = backpressure
            return
        self._do_put_payload(locator, payload)
        if on_ready is not None:
            on_ready()

    def get_layer(self, handle: BlobHandle, layer: int) -> Any:
        """Read back one layer, returning a ``(k, v)`` pair if stored as one, else a single tensor."""
        k = self._get_one(self._layer_key(handle.locator, layer) + ":k")
        if k is not None:
            v = self._get_one(self._layer_key(handle.locator, layer) + ":v")
            return (k, v)
        return self._get_one(self._layer_key(handle.locator, layer))

    def put_skeleton(self, locator: str, latent: Any) -> None:
        """Store the skeleton latent under ``locator`` (single tensor, with spec)."""
        self._put_one(locator, latent)

    def get_skeleton(self, locator: str) -> Any:
        """Return the skeleton latent for ``locator``, or None if its spec is unknown."""
        return self._get_one(locator)

    def free(self, handle: BlobHandle) -> None:   # noqa: ARG002 -- no-op (reclamation not implemented)
        """No-op: reclamation against the backing tensor store is not implemented."""
        return None
