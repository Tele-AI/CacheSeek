# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""CacheQuery — request-side input to CacheService.lookup().

Built by FrameworkAdapter.build_query(request) from the framework-specific
request type, normalized into a strategy-agnostic shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CacheQuery:
    """Strategy-agnostic lookup query.

    Strategy.lookup(query, ctx) reads from this; FrameworkAdapter.build_query
    constructs it from framework-specific request types (TeleFuser TaskRequest,
    SGLang-Diffusion request, etc.).

    Fields:
        prompt: The text prompt (primary signature for video/image diffusion).
        seed: RNG seed (when present, reproducible inference enables exact-hash strategies).
        task_type: Pipeline task identifier ("t2v", "i2v", "t2i", ...).
        model_profile: Optional ModelProfile for the target model. Strategies use
            this to decide which ResumeHint subtypes to produce.
        hint_spec: Free-form strategy hints from the user (e.g. "force_miss",
            "no_save"); rarely used in alpha.
        extra: Strategy-specific extension fields (e.g. negative_prompt, image refs).
    """

    prompt: str
    seed: int | None = None
    task_type: str = "t2v"
    model_profile: Any | None = None
    hint_spec: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
