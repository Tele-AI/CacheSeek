"""Encoder Protocols — prompt → vector and video → vector contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Only used for type annotations; avoids importing PIL at module load.
    from PIL import Image


@runtime_checkable
class PromptEncoder(Protocol):
    """Contract for text-prompt → vector encoding.

    Conformance:
    - A heavyweight model handle (Qwen3VL weights on GPU) is acceptable
      as private state on the encoder. ``encode(prompt)`` must be a pure
      function of ``prompt`` for a given encoder configuration — no
      per-call mutable state.
    - Implementations must be thread-safe — ``lookup`` and ``save``
      paths may call ``encode()`` concurrently. Backends that hold a
      single CUDA context typically serialize internally with a lock.
    - This Protocol is ``runtime_checkable``: any class exposing
      ``encode`` with the signature below satisfies ``isinstance``.
    """

    def encode(self, prompt: str) -> list[float]: ...


@runtime_checkable
class VideoEncoder(Protocol):
    """Contract for video-frames → vector encoding.

    Conformance:
    - The model handle may be cached privately, but
      ``encode_video(frames, prompt)`` must be deterministic for a fixed
      encoder configuration.
    - Implementations must be thread-safe — video encoding is invoked
      from ``CacheService.save`` paths that may run in parallel.
    - This Protocol is ``runtime_checkable``: ``prompt`` is exposed as an
      optional kwarg so multimodal encoders (e.g. ``Qwen3VLEncoder``)
      can accept it as a query hint without breaking the contract.
    """

    def encode_video(
        self,
        frames: list["Image.Image"],
        prompt: Optional[str] = None,
    ) -> list[float]: ...
