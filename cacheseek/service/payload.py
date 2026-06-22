# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Payload Protocol — byte-stream container for a strategy's cache content.

Per-strategy schema; KV/Vector backends never look inside.

Core conventions:
1. ``Payload`` is a Protocol; each strategy defines its own concrete payload class.
2. Per-strategy schemas (no forced uniformity).
3. Partial loading: through multi-KV keys + ``from_kv_loader(partial_spec=...)``.
4. Serialization: torch.save / safetensors for tensors; never pickle on
   untrusted input.
5. Schema versioning: ``schema_version`` string; new versions don't read old
   (no migration).
6. ResumeHint (instruction, light) and Payload (data, heavy) are separate
   ``LookupResult`` fields; lifecycles differ.
7. Small frequently-queried metadata (saved_steps / prompt) is duplicated to
   the vector-store payload for read optimization; eviction state
   (size / access) lives in MetadataStore; detailed payload data only in
   Payload bytes.

Concrete impl example: ``cacheseek.reuse.approximate.payload.VideoApproxPayload``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PartialLoadSpec:
    """Base class for strategy-specific partial-loading instructions.

    Strategy translates a ResumeHint (e.g. SkipStep(k=5)) into a strategy-
    specific PartialLoadSpec (e.g. VideoApproxPartialSpec(steps=[5])) and
    passes it to `Payload.from_kv_loader()`. This avoids loading unneeded
    KV entries (measured ~80% reduction in cross-process reads on wan22:
    48 MB -> 9.7 MB).

    Strategy subclasses override with their own fields:
      VideoApproxPartialSpec(steps=[5])
      ConceptPartialSpec(concept_ids=["c1","c2"])
      KVPrefixPartialSpec(token_range=(0, 200))
    """

    pass


@runtime_checkable
class Payload(Protocol):
    """Strategy-agnostic Protocol for cache payload.

    Core layer (`CacheService`, `MetadataStore`, eviction policies) only sees
    Protocol methods — opaque bytes inside, schema-versioned outside.

    Implementations are strategy-specific dataclasses;they SHOULD be
    @dataclass(frozen=False) only when partial mutability is genuinely
    needed.
    """

    @property
    def cache_id(self) -> str:
        """Stable identifier; same content → same id (typically uuid hex of content hash)."""
        ...

    @property
    def schema_version(self) -> str:
        """e.g. 'video_approx_v1'. Reader checks compatibility on load."""
        ...

    @property
    def estimated_size_bytes(self) -> int:
        """Total payload bytes across all KV entries (used for eviction decisions)."""
        ...

    def to_kv_entries(self) -> Iterator[tuple[str, bytes]]:
        """Yield (kv_key, bytes) pairs for KVStore.put.

        Simple payloads: yield (cache_id, single_blob).
        Multi-part payloads (e.g. video_approximate per-step): yield N entries
        with strategy-specific key naming (e.g. f"{cache_id}_step{N}").
        """
        ...

    @classmethod
    def from_kv_loader(
        cls,
        cache_id: str,
        kv_loader: Callable[[str], bytes | None],
        partial_spec: PartialLoadSpec | None = None,
    ) -> Payload:
        """Reconstruct a Payload by reading KV entries via the kv_loader callback.

        partial_spec=None → load full payload (default).
        partial_spec set → load only the subset specified (saves bandwidth).
        """
        ...
