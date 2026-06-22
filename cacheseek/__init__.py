"""cacheseek — cross-request KV-cache middleware for world-model long-horizon reasoning.

Top-level API re-exports (lazy):
- Lifecycle types: ``CacheService`` (orchestrator), ``TeleFuserCacheAdapter``,
  ``CacheQuery``, ``LookupResult``, ResumeHint sealed union
  (``SkipStep`` / ``NoOp``), ``ModelOutputs``, ``CacheConfig``.
- LingBot-Fast Phase B: ``LingBotFastRuntimeCacheController`` for wiring
  cross-attention cache reuse into the TeleFuser runtime.

Heavy modules (encoders, fluxon, qdrant, telefuser adapter) load lazily
on first attribute access so ``import cacheseek`` is cheap.

    from cacheseek import CacheService, TeleFuserCacheAdapter, CacheConfig
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0a1"

__all__ = [
    "CacheService",
    "TeleFuserCacheAdapter",
    "CacheQuery",
    "LookupResult",
    "SkipStep",
    "ResumeKVChain",
    "ReuseCrossAttnKV",
    "ReuseEmbedding",
            "NoOp",
    "UnsupportedResumeHint",
    "ModelOutputs",
    "CacheConfig",
    "__version__",
]


# Lazy-load map: attribute name → (module path, attribute on module).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CacheService": ("cacheseek.service.lifecycle", "CacheService"),
    "TeleFuserCacheAdapter": (
        "cacheseek.adapters.telefuser.adapter",
        "TeleFuserCacheAdapter",
    ),
    "CacheQuery": ("cacheseek.service.query", "CacheQuery"),
    "LookupResult": ("cacheseek.service.result", "LookupResult"),
    "ResumeHint": ("cacheseek.service.result", "ResumeHint"),
    "ResumeHintT": ("cacheseek.service.result", "ResumeHintT"),
    "FastForward": ("cacheseek.service.result", "FastForward"),
    "LoadStateSnapshot": ("cacheseek.service.result", "LoadStateSnapshot"),
    "ReturnCachedOutput": ("cacheseek.service.result", "ReturnCachedOutput"),
    "Payload": ("cacheseek.service.payload", "Payload"),
    "PartialLoadSpec": ("cacheseek.service.payload", "PartialLoadSpec"),
    "SkipStep": ("cacheseek.service.result", "SkipStep"),
    "ResumeKVChain": ("cacheseek.service.result", "ResumeKVChain"),
    "ReuseCrossAttnKV": ("cacheseek.service.result", "ReuseCrossAttnKV"),
    "ReuseEmbedding": ("cacheseek.service.result", "ReuseEmbedding"),
    "NoOp": ("cacheseek.service.result", "NoOp"),
    "UnsupportedResumeHint": ("cacheseek.service.result", "UnsupportedResumeHint"),
    "ModelOutputs": ("cacheseek.service.outputs", "ModelOutputs"),
    "CacheConfig": ("cacheseek.service.config", "CacheConfig"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
