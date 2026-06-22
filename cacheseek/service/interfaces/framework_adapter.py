# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""FrameworkAdapter Protocol — bridge inference framework hooks to cacheseek types."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Contract for inference-framework integration.

    Each inference framework (TeleFuser / SGLang-Diffusion / vLLM-Omni /
    future) implements this Protocol to translate its native request /
    engine-context / response objects into cacheseek's neutral
    ``CacheQuery`` / ``LookupResult`` / ``ModelOutputs`` shapes. Currently
    only TeleFuser is implemented; the other adapters wait on community
    PRs.

    Conformance:
    - Adapters are pure converters. They translate framework-specific
      objects into cacheseek types and apply lookup results back into
      the framework engine context. No per-request mutable state should
      be carried across ``await`` boundaries on ``self`` — the only
      state on the adapter should be construction-time defaults read
      from ``CacheConfig`` (e.g. ``key_steps``).
    - Implementations must be thread-safe — adapter methods are invoked
      from concurrent inference workers.
    """

    def build_query(self, request: Any) -> CacheQuery:  # noqa: F821
        """Translate a framework-native request into a neutral CacheQuery.

        Extracts the cache-relevant signals (prompt, seed, task type, model
        profile, etc.) from the framework's own request object and normalizes
        them into the strategy-agnostic CacheQuery that
        ``CacheService.lookup`` consumes. Pure conversion: no engine state is
        read or mutated.

        Args:
            request: The framework-specific request object (e.g. a TeleFuser
                ``TaskRequest``).

        Returns:
            A CacheQuery built from ``request``.
        """
        ...

    def apply_resume(
        self,
        result: LookupResult,  # noqa: F821
        engine_ctx: Any,
    ) -> dict[str, Any]:
        """Translate ``LookupResult`` into a framework-shaped dict.

        Returns a dict the caller passes back into the inference engine
        (e.g. wan22 expects ``{hit, skip_step, cached_latent, saved_steps}``).
        ``engine_ctx`` is framework-specific context the adapter MAY read
        to make framework-specific decisions; current TeleFuser impl does
        not use it.
        """
        ...

    def on_response(self, request: Any, outputs: Any) -> ModelOutputs:  # noqa: F821
        """Translate a framework-native response into neutral ModelOutputs.

        Extracts the cacheable artifacts (step-snapshot latents, frames, step
        counts, etc.) from the framework's raw output and normalizes them into
        the ModelOutputs that ``CacheService.save`` persists. Pure conversion.

        Args:
            request: The originating framework-specific request, available for
                fields the response alone does not carry.
            outputs: The framework-specific raw output object.

        Returns:
            A ModelOutputs built from ``request`` and ``outputs``.
        """
        ...
