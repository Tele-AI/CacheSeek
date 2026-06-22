"""LookupResult + ResumeHint sealed union.

Sealed dataclass union pattern (emulating a Java/Kotlin/Rust enum in Python):
- All ResumeHint subclasses listed in the `ResumeHintT` Union;
- All are @dataclass(frozen=True);
- FrameworkAdapter dispatch via `functools.singledispatchmethod` (open-closed).

ResumeHint is the strategy -> adapter "instruction" (light); Payload is the
"data" (heavy). They are two separate LookupResult fields with different
lifecycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from cacheseek.service.payload import Payload


class ResumeHint:
    """Sealed base for resume instructions.

    Concrete subclasses must be @dataclass(frozen=True). The strategy picks the
    subclass; FrameworkAdapter consumes it via `singledispatchmethod` dispatch.

    Adding a new subclass is a breaking change — every FrameworkAdapter that
    intends to support it must register a `@apply_resume.register` handler.

    Implemented:
      - SkipStep + NoOp (wan22 + LTX-2.3)
    Reserved (no impl in alpha):
      - LoadStateSnapshot (latent dynamics WMA)
      - ReturnCachedOutput (exact-hash strategy)
    """


@dataclass(frozen=True)
class SkipStep(ResumeHint):
    """Diffusion / video DiT: start denoising from step k, using cached latent
    for the first k steps.

    Applies to any step-based diffusion pipeline (wan22 / LTX-2.3 / OpenSora /
    Hunyuan / SDXL / Flux, etc.).
    """

    k: int


@dataclass(frozen=True)
class LoadStateSnapshot(ResumeHint):
    """Latent dynamics WMA: reset to a snapshot state and continue the rollout.

    Reserved interface (no impl); a hook for Dreamer / RSSM / any dynamical
    system.
    """

    state_id: str
    rollout_horizon: Optional[int] = None


@dataclass(frozen=True)
class ReturnCachedOutput(ResumeHint):
    """Exact-hash strategy: identical prompt hit; return the prior output
    directly and skip inference.

    Reserved interface (no impl); a FrameworkAdapter receiving this hint should
    short-circuit inference.
    """

    output_uri: str
    media_type: str  # "video" / "image" / "text"


@dataclass(frozen=True)
class ResumeKVChain(ResumeHint):
    """AR-diffusion: resume from a matched self-KV chunk chain."""

    chain_id: str
    matched_groups: int
    last_global_end_index: int
    cross_attn_group_id: Optional[str] = None
    plucker_block_ids: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class ReuseCrossAttnKV(ResumeHint):
    """LingBot Phase B: reuse deterministic cross-attention K/V."""

    group_id: str
    block_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReuseEmbedding(ResumeHint):
    """LingBot Phase B: reuse deterministic input-side embeddings."""

    embedding_kind: str
    block_keys: tuple[str, ...]
    slot_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class FastForward(ResumeHint):
    """Chunked AR (world model): reuse cached output for the first k chunks and
    continue from chunk k.

    An exact_prefix hint. node/namespace are trie-hit handles (light object
    references); the engine adapter (LingBotWorldKVBinding) interprets them:
    materialize KV into the window, decode-only latent stash, and RNG
    burn/draw alignment.
    """

    k: int
    node: Any = None
    namespace: Any = None


@dataclass(frozen=True)
class NoOp(ResumeHint):
    """Strategy deliberately declines to use a hit (debug / shadow mode / A-B benchmark)."""

    reason: str = ""


# Public union type for type-checking exhaustive dispatch (with assert_never).
ResumeHintT = Union[
    SkipStep,
    FastForward,
    LoadStateSnapshot,
    ReturnCachedOutput,
    ResumeKVChain,
    ReuseCrossAttnKV,
    ReuseEmbedding,
    NoOp,
]


class UnsupportedResumeHint(Exception):
    """Raised when FrameworkAdapter receives a ResumeHint subtype it doesn't
    handle. Strategy should not produce hints unsupported by the target adapter
    — when it happens, it's a programming bug, surfaced via this exception so
    the calling layer can fall back to cache miss (graceful degrade)."""

    def __init__(
        self,
        hint: ResumeHint,
        framework: str,
        supported: list[type[ResumeHint]],
    ):
        self.hint = hint
        self.framework = framework
        self.supported = supported
        super().__init__(
            f"FrameworkAdapter '{framework}' does not support "
            f"{type(hint).__name__}; supported: {[t.__name__ for t in supported]}"
        )


@dataclass(frozen=True)
class LookupResult:
    """CacheService.lookup() return value.

    Strategy populates payload + resume_hint;FrameworkAdapter consumes them
    via apply_resume(). Payload is the heavy data (latent bytes / KV cache),
    ResumeHint is the light instruction (k / state_id).

    For cache miss, hit=False, payload / resume_hint / matched_* are None,
    and miss_reason carries an optional debug tag (e.g. "vector_search_no_match",
    "rerank_below_threshold").
    """

    hit: bool
    payload: Optional["Payload"] = None
    resume_hint: Optional[ResumeHint] = None
    matched_cache_id: Optional[str] = None
    matched_score: Optional[float] = None
    matched_similarity: Optional[float] = None
    cached_prompt: Optional[str] = None
    miss_reason: Optional[str] = None

    @classmethod
    def miss(cls, reason: str = "") -> "LookupResult":
        """Construct a cache-miss result, optionally tagging the reason."""
        return cls(hit=False, miss_reason=reason or None)

    @classmethod
    def hit_skip_step(
        cls,
        payload: "Payload",
        k: int,
        cache_id: str,
        score: float,
        similarity: float,
        cached_prompt: str = "",
    ) -> "LookupResult":
        """Construct a hit result with SkipStep instruction (the alpha path)."""
        return cls(
            hit=True,
            payload=payload,
            resume_hint=SkipStep(k=k),
            matched_cache_id=cache_id,
            matched_score=score,
            matched_similarity=similarity,
            cached_prompt=cached_prompt,
        )

    @classmethod
    def hit_fast_forward(cls, k: int, node: Any = None, namespace: Any = None) -> "LookupResult":
        """Construct a hit result with FastForward instruction (exact-prefix path)."""
        return cls(hit=True, resume_hint=FastForward(k=k, node=node, namespace=namespace))
