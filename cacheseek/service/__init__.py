"""service — orchestration layer: CacheService lifecycle and eviction policies.

Note: exact and approximate currently serve different pipelines (LingBot world
model / wan22 t2v). Orchestration selects semantics per pipeline rather than
cascading; composition only matters once a single pipeline gains a second
semantic (e.g. a world model adding approximate recall).
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["CacheService", "EvictionPolicy", "LRUEviction", "FIFOEviction"]

_LAZY: dict[str, tuple[str, str]] = {
    "CacheService": ("cacheseek.service.lifecycle", "CacheService"),
    "EvictionPolicy": ("cacheseek.service.eviction", "EvictionPolicy"),
    "LRUEviction": ("cacheseek.service.eviction", "LRUEviction"),
    "FIFOEviction": ("cacheseek.service.eviction", "FIFOEviction"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
