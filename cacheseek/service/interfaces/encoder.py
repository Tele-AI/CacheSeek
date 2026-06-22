# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Encoder Protocols — prompt → vector and video → vector contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

    def encode(self, prompt: str) -> list[float]:
        """Encode a text prompt into a fixed-dimension embedding vector.

        Deterministic for a given encoder configuration: the same ``prompt``
        always yields the same vector. The returned vector's dimension is
        constant for the encoder and must match the dimension of the
        VectorStore collection it is searched against. L2-normalized output is
        expected so that cosine and dot-product similarity coincide.

        Args:
            prompt: Text to embed.

        Returns:
            The embedding as a list of floats.
        """
        ...


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
        frames: list[Image.Image],
        prompt: str | None = None,
    ) -> list[float]:
        """Encode a sequence of video frames into a fixed-dimension embedding.

        Deterministic for a given encoder configuration. The returned vector's
        dimension is constant for the encoder and must match the VectorStore
        collection it is stored in. L2-normalized output is expected.

        Args:
            frames: Video frames to embed, in temporal order.
            prompt: Optional query hint. Multimodal encoders may condition the
                embedding on it; unimodal encoders ignore it. Passing it must
                never break the contract.

        Returns:
            The embedding as a list of floats.
        """
        ...
