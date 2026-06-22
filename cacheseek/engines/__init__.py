"""engines — facade for the engine-integration boundary.

  telefuser_lingbot   exact_prefix LingBot binding (hook-level, duck-typed)
  telefuser_service   approximate service-level adapter / factory
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["LingBotWorldKVBinding", "TeleFuserCacheAdapter", "CacheServiceFactory"]

_LAZY: dict[str, tuple[str, str]] = {
    "LingBotWorldKVBinding": ("cacheseek.reuse.exact_prefix.telefuser_lingbot", "LingBotWorldKVBinding"),
    "TeleFuserCacheAdapter": ("cacheseek.adapters.telefuser.adapter", "TeleFuserCacheAdapter"),
    "CacheServiceFactory": ("cacheseek.adapters.telefuser.cache_factory", "CacheServiceFactory"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
