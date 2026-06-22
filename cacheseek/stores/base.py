# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""KVStore Protocol — opaque byte-blob storage keyed by string id.

``TensorKVStore`` is an OPTIONAL capability layered on top: backends that can
store/return tensors without serializing to bytes (e.g. Fluxon via DLPack)
implement it. Callers route through ``adapters/lingbot_fast/tensor_block_io``,
which falls back to a pickle-free raw-bytes path on stores that only satisfy
``KVStore``. ``KVStore`` itself is unchanged — the capability is additive and
non-breaking; existing ``put(bytes)`` / ``get`` callers are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # keep this module torch-free at runtime (lazy-import constraint)
    import torch


@runtime_checkable
class KVStore(Protocol):
    """Contract for opaque byte-blob storage.

    Conformance:
    - Implementations should be thread-safe — ``CacheService`` worker
      threads may call ``put`` / ``get`` / ``remove`` / ``list_keys``
      concurrently from the same process. If an implementation needs
      mutable state (open file handles, connection pools, in-memory
      dicts), keep it private and serialize concurrent access
      internally.
    - This Protocol is ``runtime_checkable``: any class that exposes the
      four methods below with matching signatures satisfies
      ``isinstance(obj, KVStore)`` — no inheritance required.
    """

    def put(self, key: str, value: bytes) -> None:
        """Store a blob under ``key``, overwriting any existing value.

        Args:
            key: Storage id. Reused keys are overwritten last-write-wins.
            value: Opaque payload; the store does not interpret it.
        """
        ...

    def get(self, key: str) -> bytes | None:
        """Fetch the blob stored under ``key``.

        Returns:
            The stored bytes, or None on a miss (so a miss degrades to recompute
            rather than raising).
        """
        ...

    def remove(self, key: str) -> None:
        """Delete the blob under ``key``.

        Removing an absent key is a no-op (idempotent), not an error.
        """
        ...

    def list_keys(self) -> list[str]:
        """Return all keys currently present in the store.

        Returns:
            A snapshot of the live keys; ordering is unspecified.
        """
        ...


@runtime_checkable
class TensorKVStore(Protocol):
    """Optional zero-copy tensor capability on top of ``KVStore``.

    A backend implementing this can ingest/return torch tensors directly
    (Fluxon hands the DLPack pointer to its Rust layer — no Python bytes, no
    pickle). ``isinstance(store, TensorKVStore)`` is the routing check; stores
    without these methods fall back to a pickle-free raw-bytes path.

    Contract:
    - ``shape`` / ``dtype`` on ``get_tensor`` are REQUIRED — raw bytes are not
      self-describing (unlike ``torch.save``). A native implementation may
      validate against or ignore them; the fallback needs them to reconstruct.
    - ``get_tensor`` returns ``None`` on a cache miss, mirroring ``KVStore.get``
      (so a miss degrades to recompute, not a crash).
    - Implementations clone/own the returned tensor (it must outlive any backend
      buffer/holder it was read from).
    """

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Store a tensor under ``key`` without serializing to bytes.

        The backend copies the contents (e.g. to host memory or a shared-memory
        buffer); the caller's tensor is left untouched and stays safe to mutate.

        Args:
            key: Storage id; reused keys are overwritten.
            tensor: The tensor to persist.
        """
        ...

    def get_tensor(
        self,
        key: str,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str | torch.device | None = None,
    ) -> torch.Tensor | None:
        """Read back the tensor stored under ``key``.

        ``shape`` and ``dtype`` are required because the stored buffer is raw and
        not self-describing; they are used to reconstruct the view (the fallback
        path needs them; a native backend may validate against or ignore them).

        Args:
            key: Storage id used at ``put_tensor`` time.
            shape: Logical shape to reconstruct.
            dtype: Element dtype to reinterpret the buffer as.
            device: Optional target device; None leaves placement to the backend.

        Returns:
            A caller-owned tensor (cloned so it outlives any backend buffer), or
            None on a cache miss.
        """
        ...


# ------------------------------------------------------------- storage data types
class Tier(Enum):
    """Hot to cold. Blobs demote one tier at a time rather than being dropped."""
    HBM_STAGING = "hbm_staging"   # transient materialization only; bounded staging
    CPU = "cpu"                   # local pinned host memory
    FLUXON_DRAM = "fluxon_dram"   # distributed DRAM store (same-host shm GET ~12.2 GB/s)
    DISK = "disk"                 # cold persistent fallback


@dataclass(slots=True)
class BlobHandle:
    """Locator for a chunk's KV blob — the heavy, layered part of a node."""
    tier: Tier
    locator: str          # store key / path
    nbytes: int
    n_layers: int
    ready: bool = False   # set True only after all layers are written; readers
                          # consider only ready nodes (visibility)
