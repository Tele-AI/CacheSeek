# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Vector backends — collection-scoped vector index with payload.

Implementations are lazy-loaded so ``faiss`` / ``qdrant-client`` are not
imported at package-load time. Users only pay the cost of the backend
they actually instantiate.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from cacheseek.service.interfaces.vector_store import VectorStore

__all__ = ["VectorStore", "FAISSVectorStore", "QdrantVectorStore"]

_LAZY: dict[str, tuple[str, str]] = {
    "FAISSVectorStore": ("cacheseek.backends.vector.faiss", "FAISSVectorStore"),
    "QdrantVectorStore": ("cacheseek.backends.vector.qdrant", "QdrantVectorStore"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
