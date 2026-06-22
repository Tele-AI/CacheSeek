# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""VideoApproxPayload — concrete Payload for VideoBasedApproximateCache.

Encapsulates KV key naming convention ``{cache_id}_step{N}`` and torch.save
serialization. Per-step values are kept as ``Union[Tensor, bytes]``:

- Save side (``to_kv_entries``) accepts Tensors and serializes lazily on
  iteration — peak memory stays at one step's bytes, not the whole set.
- Load side (``from_kv_loader``) returns a Payload holding raw bytes;
  deserialization is deferred to ``get_latent_at_step`` so the caller
  pays only for the steps it actually consumes.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoApproxPartialSpec:
    """Specifies which steps to materialize when loading a payload from KV.

    Always required by ``VideoApproxPayload.from_kv_loader`` — there is
    no implicit "load all" default to avoid silently doing 5× the I/O
    when the caller only needs one step.
    """
    steps: tuple[int, ...]

    @classmethod
    def single(cls, step: int) -> VideoApproxPartialSpec:
        """Convenience: load exactly one step (the common case at lookup hit)."""
        return cls(steps=(int(step),))


@dataclass
class VideoApproxPayload:
    """Diffusion video latent payload for VideoBasedApproximateCache.

    Schema v1:
    - One independent KV key per step k: ``{cache_id}_step{k}``
    - The bytes value is ``torch.save(tensor, BytesIO).getvalue()``
    - Load uses ``weights_only=True`` (blocks RCE)

    The ``step_to_latents`` field carries either Tensors (save side, before
    serialization) or raw bytes (load side, before deserialization). The
    asymmetry lets us serialize / deserialize lazily — see module docstring.

    Shapes:
        step_to_latents[k]: (B, C, T, H, W) — one diffusion latent per key step k
            B     batch; 1 for a cached donor          — stable
            C     VAE latent channels                  — stable per model
            T     latent frames                        — stable per model profile
            H, W  latent spatial dims                  — vary with output resolution
        At a fixed model profile only H, W vary across requests; a reuse candidate
        must match on the stable dims for its latents to be substitutable.
    """
    cache_id: str
    step_to_latents: dict[int, torch.Tensor | bytes]
    schema_version: str = "video_approx_v1"

    @property
    def estimated_size_bytes(self) -> int:
        """Sum of per-step sizes — in-memory tensor footprint for Tensor
        entries, byte length for bytes entries.

        Used by ``MetadataStore.register_cache`` for eviction accounting,
        so we want a number that reflects "how much storage / memory
        does this payload hold" without forcing serialization of Tensors.
        """
        total = 0
        for value in self.step_to_latents.values():
            if isinstance(value, torch.Tensor):
                nelement = getattr(value, "nelement", None)
                element_size = getattr(value, "element_size", None)
                if callable(nelement) and callable(element_size):
                    total += int(nelement()) * int(element_size())
            elif isinstance(value, (bytes, bytearray, memoryview)):
                total += len(value)
        return total

    def to_kv_entries(self) -> Iterator[tuple[str, bytes]]:
        """Yield ``(kv_key, bytes)`` pairs for KVStore.put.

        Lazy-serializes Tensors on demand so the caller can ``put`` one
        entry at a time without holding all serialized bytes in memory.
        """
        for step, value in self.step_to_latents.items():
            if isinstance(value, torch.Tensor):
                blob = self.serialize_tensor(value)
            elif isinstance(value, (bytes, bytearray, memoryview)):
                blob = bytes(value)
            else:
                raise TypeError(
                    "VideoApproxPayload.step_to_latents value must be "
                    f"Tensor or bytes; got step={step} type={type(value).__name__}"
                )
            yield f"{self.cache_id}_step{int(step)}", blob

    @classmethod
    def from_kv_loader(
        cls,
        cache_id: str,
        kv_loader: Callable[[str], bytes | None],
        partial_spec: VideoApproxPartialSpec,
    ) -> VideoApproxPayload:
        """Reconstruct a payload by loading the steps in ``partial_spec``.

        ``partial_spec`` is required (no default) so callers cannot
        accidentally do 5× reads when they only want one step. Loaded
        bytes are stored as-is; ``get_latent_at_step`` deserializes on
        demand.
        """
        if partial_spec is None:
            raise TypeError(
                "VideoApproxPayload.from_kv_loader requires partial_spec; "
                "use VideoApproxPartialSpec.single(step) for one step or "
                "VideoApproxPartialSpec(steps=(...)) for multiple."
            )
        step_to_latents: dict[int, torch.Tensor | bytes] = {}
        for step in partial_spec.steps:
            data = kv_loader(f"{cache_id}_step{int(step)}")
            if data is not None:
                step_to_latents[int(step)] = data
        return cls(
            cache_id=cache_id,
            step_to_latents=step_to_latents,
        )

    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> bytes:
        """``torch.save`` tensor to bytes."""
        buf = io.BytesIO()
        torch.save(tensor, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_tensor(data: bytes) -> torch.Tensor:
        """``torch.load`` with ``weights_only=True`` (blocks RCE on untrusted bytes)."""
        return torch.load(
            io.BytesIO(data), weights_only=True, map_location="cpu"
        )

    def get_latent_at_step(self, step: int) -> torch.Tensor:
        """Return the tensor for ``step``, deserializing if it's still bytes.

        Raises ``KeyError`` if the step wasn't loaded into this payload.
        """
        key = int(step)
        if key not in self.step_to_latents:
            raise KeyError(
                f"step {step} not in payload "
                f"(have {sorted(self.step_to_latents.keys())})"
            )
        value = self.step_to_latents[key]
        if isinstance(value, torch.Tensor):
            return value
        return self.deserialize_tensor(bytes(value))
