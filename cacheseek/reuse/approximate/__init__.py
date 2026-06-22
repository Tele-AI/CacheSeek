"""reuse.approximate — approximate-recall semantics: embedding ANN hit ->
donor latent -> skip_step resumption.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseCacheStrategy", "VideoBasedApproximateCache", "get_strategy_class",
    "VideoApproxPayload", "VideoApproxPartialSpec",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BaseCacheStrategy": ("cacheseek.reuse.approximate.strategy", "BaseCacheStrategy"),
    "VideoBasedApproximateCache": ("cacheseek.reuse.approximate.strategy", "VideoBasedApproximateCache"),
    "get_strategy_class": ("cacheseek.reuse.approximate.strategy", "get_strategy_class"),
    "VideoApproxPayload": ("cacheseek.reuse.approximate.payload", "VideoApproxPayload"),
    "VideoApproxPartialSpec": ("cacheseek.reuse.approximate.payload", "VideoApproxPartialSpec"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
