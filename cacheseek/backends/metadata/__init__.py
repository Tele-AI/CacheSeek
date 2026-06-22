"""Metadata backends — cache index + access stats + eviction.

Alpha default: ``LocalCacheMetadataManager`` (in-memory dict + JSON
persistence). Future roadmap: Redis / SQLite / Qdrant-payload backed.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["LocalCacheMetadataManager"]

_LAZY: dict[str, tuple[str, str]] = {
    "LocalCacheMetadataManager": (
        "cacheseek.backends.metadata.local",
        "LocalCacheMetadataManager",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
