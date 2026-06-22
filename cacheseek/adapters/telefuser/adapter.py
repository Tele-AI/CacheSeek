"""TeleFuserCacheAdapter — FrameworkAdapter Protocol impl for TeleFuser.

Framework-side lifecycle hooks:
    build_query(task_request) → CacheQuery     # request entering
    apply_resume(LookupResult, engine_ctx)     # injecting hit into engine
        — ``engine_ctx`` is reserved for future engine state injection
          (e.g. WMA snapshot restore); the TeleFuser alpha path does not
          read it. Kept on the public signature for Protocol conformance.
    on_response(request, raw_outputs) → ModelOutputs  # save-side packing

ResumeHint dispatch uses straight ``if/elif`` on ``isinstance(hint, ...)``
instead of ``functools.singledispatchmethod``. With only two implemented
subtypes (``SkipStep`` + ``NoOp``) the two forms are behaviourally
equivalent and the explicit form is easier to read; switching to
``singledispatchmethod`` becomes worthwhile once a third subtype lands
(``LoadStateSnapshot`` for WMA / ``ReturnCachedOutput`` for exact-hash).

This adapter is the **only** TeleFuser-specific piece of cacheseek.
Other framework adapters (SGLang-Diffusion, vLLM-Omni) live in their own
subpackages.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import (
    LookupResult,
    NoOp,
    SkipStep,
    UnsupportedResumeHint,
)


class TeleFuserCacheAdapter:
    """FrameworkAdapter Protocol impl for TeleFuser.

    No *mutable* instance fields. ``__init__`` only captures
    **immutable, construction-time defaults** read from the
    ``CacheConfig`` (e.g. ``key_steps``); these are never mutated after
    construction, so the adapter is thread-safe and module-level
    singleton-friendly.
    """

    FRAMEWORK_NAME = "telefuser"

    # Supported ResumeHint subclasses (rest raise UnsupportedResumeHint).
    # ReturnCachedOutput / LoadStateSnapshot reserved but not implemented in alpha.
    SUPPORTED_HINTS = [SkipStep, NoOp]

    # Fallback used when the adapter is constructed without an injected
    # CacheConfig.key_steps; never read once a real config flows in.
    _BUILTIN_FALLBACK_SAVED_STEPS = (5, 10, 15, 20, 25)

    def __init__(
        self,
        *,
        default_saved_steps: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Args:
            default_saved_steps: ``CacheConfig.key_steps`` snapshot. Used by
                ``_miss_dict`` and ``_extract_saved_steps`` when the
                ``LookupResult.payload`` doesn't carry per-entry steps.
                ``None`` falls back to ``_BUILTIN_FALLBACK_SAVED_STEPS``.
        """
        if default_saved_steps is None:
            self._default_saved_steps: tuple[int, ...] = self._BUILTIN_FALLBACK_SAVED_STEPS
        else:
            self._default_saved_steps = tuple(int(s) for s in default_saved_steps)

    def build_query(self, request: Any) -> CacheQuery:
        """Translate TeleFuser TaskRequest → CacheQuery.

        TaskRequest has `.task` / `.prompt` / `.seed` attrs (from telefuser.service_types).
        """
        prompt = getattr(request, "prompt", "") or ""
        task_type = getattr(request, "task", "t2v") or "t2v"
        seed = getattr(request, "seed", None)
        logger.debug(
            "TeleFuserCacheAdapter.build_query task_type={} prompt_len={} seed={}",
            task_type,
            len(prompt),
            seed,
        )
        return CacheQuery(
            prompt=prompt,
            seed=seed,
            task_type=task_type,
            # model_profile / hint_spec / extra not used in TeleFuser alpha path
        )

    def apply_resume(
        self,
        result: LookupResult,
        engine_ctx: Any,
    ) -> dict[str, Any]:
        """Inject cache hit into TeleFuser pipeline.

        TeleFuser's pipeline expects a ``latent_data`` dict with fields:
          ``{hit, skip_step, cached_latent, saved_steps}``

        Dispatches with ``isinstance`` on the ``ResumeHint`` subtype.
        For ``NoOp`` / miss / unknown hint the standard miss-shape dict
        is returned (graceful degrade).

        ``engine_ctx`` is reserved for future engine state injection
        (e.g. ``LoadStateSnapshot`` for WMA). The TeleFuser alpha path
        does not read it; it stays on the signature so callers wired
        against the Protocol keep working when new ResumeHint subtypes
        start consuming it.
        """
        del engine_ctx  # reserved; see docstring
        if not result.hit:
            logger.debug(
                "TeleFuserCacheAdapter.apply_resume miss matched_cache_id={} score={}",
                result.matched_cache_id,
                result.matched_score,
            )
            return self._miss_dict(result)
        hint = result.resume_hint
        if hint is None:
            logger.warning(
                "TeleFuserCacheAdapter.apply_resume: hit but resume_hint=None,"
                " degrading to miss"
            )
            return self._miss_dict(result)
        logger.debug(
            "TeleFuserCacheAdapter.apply_resume dispatch hint={} cache_id={} score={} sim={}",
            type(hint).__name__,
            result.matched_cache_id,
            result.matched_score,
            result.matched_similarity,
        )
        if isinstance(hint, SkipStep):
            return self._handle_skip_step(hint, result)
        if isinstance(hint, NoOp):
            return self._miss_dict(result)
        raise UnsupportedResumeHint(
            hint=hint,
            framework=self.FRAMEWORK_NAME,
            supported=self.SUPPORTED_HINTS,
        )

    def on_response(
        self,
        request: Any,
        raw_outputs: Any,
    ) -> ModelOutputs:
        """Pack TeleFuser pipeline output into ModelOutputs for save().

        TeleFuser pipeline returns a ``latent_payload`` dict containing:
          - latent_states_dict: {step: tensor}
          - embedding_video_frames: list[PIL.Image]
          - num_frames / final_step / saved_steps
        """
        if isinstance(raw_outputs, dict):
            outputs = ModelOutputs(
                latent_states_dict=raw_outputs.get("latent_states_dict", {}) or {},
                embedding_video_frames=raw_outputs.get("embedding_video_frames"),
                num_frames=int(raw_outputs.get("num_frames", 0) or 0),
                final_step=int(raw_outputs.get("final_step", 0) or 0),
                saved_steps=list(raw_outputs.get("saved_steps", []) or []),
            )
            logger.debug(
                "TeleFuserCacheAdapter.on_response saved_steps={} num_frames={} final_step={} embedding_frames={}",
                outputs.saved_steps,
                outputs.num_frames,
                outputs.final_step,
                len(outputs.embedding_video_frames) if outputs.embedding_video_frames else 0,
            )
            return outputs
        logger.warning(
            "TeleFuserCacheAdapter.on_response unexpected raw_outputs type={}, returning empty ModelOutputs",
            type(raw_outputs).__name__,
        )
        return ModelOutputs()

    def _handle_skip_step(
        self,
        hint: SkipStep,
        result: LookupResult,
    ) -> dict[str, Any]:
        """SkipStep: produce TeleFuser latent_data with hit=True / skip_step=k."""
        cached_latent = self._extract_latent(result, hint.k)
        saved_steps = self._extract_saved_steps(result)
        logger.debug(
            "TeleFuserCacheAdapter._handle_skip_step k={} latent_present={} saved_steps={}",
            int(hint.k),
            cached_latent is not None,
            saved_steps,
        )
        return {
            "hit": True,
            "skip_step": int(hint.k),
            "cached_latent": cached_latent,
            "saved_steps": saved_steps,
        }

    def _miss_dict(self, result: LookupResult) -> dict[str, Any]:
        """Miss path latent_data dict.

        ``saved_steps`` MUST stay non-empty even on miss — the wan22
        pipeline reads this list to decide which denoise steps to
        snapshot for the future ``save`` write-back. Returning ``[]``
        here makes the pipeline skip snapshotting,
        ``latent_states_dict`` ends up empty, and ``service.save``
        early-exits with ``save skip: no latent_states or saved_steps``,
        which breaks the miss → save → hit chain entirely.

        The value comes from ``self._default_saved_steps``, which the
        factory snapshots from ``CacheConfig.key_steps`` at construction
        time.
        """
        return {
            "hit": False,
            "skip_step": 0,
            "cached_latent": None,
            "saved_steps": list(self._default_saved_steps),
        }

    @staticmethod
    def _extract_latent(result: LookupResult, k: int) -> Any:
        """Get the cached latent for step k from ``result.payload``.

        ``result.payload`` is a ``VideoApproxPayload`` produced by the
        strategy's lookup path. ``get_latent_at_step`` deserializes on
        demand and raises ``KeyError`` if the step wasn't part of the
        partial_spec the strategy passed to ``from_kv_loader`` —
        treating that as a miss surfaces the under-specified spec
        instead of silently corrupting the resume hint.
        """
        payload = result.payload
        if payload is None:
            return None
        get_at = getattr(payload, "get_latent_at_step", None)
        if not callable(get_at):
            logger.warning(
                "TeleFuserCacheAdapter._extract_latent: payload of type={} "
                "does not implement get_latent_at_step; treating as miss",
                type(payload).__name__,
            )
            return None
        try:
            return get_at(k)
        except KeyError:
            logger.warning(
                "TeleFuserCacheAdapter._extract_latent: payload missing step={};"
                " strategy may have under-specified partial_spec",
                k,
            )
            return None

    def _extract_saved_steps(self, result: LookupResult) -> list[int]:
        """Get saved_steps from payload metadata or fall back to injected default.

        Fallback chain:
          1. ``result.payload.saved_steps`` (list attr — set on some payloads)
          2. ``result.payload.step_to_latents`` (dict keys — VideoApproxPayload)
          3. ``self._default_saved_steps`` (injected by cache_factory from CacheConfig.key_steps)
        """
        payload = result.payload
        if payload is not None:
            steps = getattr(payload, "saved_steps", None)
            if steps:
                return list(steps)
            step_to_latents = getattr(payload, "step_to_latents", None)
            if isinstance(step_to_latents, dict) and step_to_latents:
                return sorted(step_to_latents.keys())
        return list(self._default_saved_steps)


__all__ = ["TeleFuserCacheAdapter"]
