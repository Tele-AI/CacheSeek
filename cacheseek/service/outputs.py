# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""ModelOutputs — response-side input to CacheService.save().

Built by FrameworkAdapter.on_response(request, raw_outputs) from framework-
specific output types, normalized into a strategy-agnostic shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelOutputs:
    """Strategy-agnostic save-side payload.

    Strategy.save(query, outputs, ctx) reads from this; FrameworkAdapter
    populates it from framework-specific output types (TeleFuser
    `latent_payload` dict, SGLang-Diffusion response, etc.).

    Fields:
        latent_states_dict: {step: torch.Tensor} — diffusion step-snapshot
            latents (the data going into KV via VideoApproxPayload).
        embedding_video_frames: List[PIL.Image] — frames for video encoder
            to embed; required by VideoBasedApproximateCache.save.
        num_frames: total video frames generated.
        final_step: final denoise step the pipeline ran.
        saved_steps: list of step indices snapshotted (e.g. [5,10,15,20,25]).
        extra: strategy-specific extension fields.
    """

    latent_states_dict: dict[int, Any] = field(default_factory=dict)
    embedding_video_frames: list[Any] | None = None
    num_frames: int = 0
    final_step: int = 0
    saved_steps: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["ModelOutputs"]
