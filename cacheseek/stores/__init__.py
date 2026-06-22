# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""stores — unified storage layer.

Capability tiers:
  - Bytes contract: ``KVStore`` (put/get/remove/list_keys over bytes).
  - Tensor contract: ``TensorKVStore`` (put_tensor/get_tensor, optional zero-copy).
  - Tier adapter: ``TensorStoreTierStore`` (spec bookkeeping + async write queue
    + per-layer (k, v) splitting).
Backends: memory / local_file / fluxon (the bytes trio) plus InMemoryTierStore /
LocalDiskTensorStore.

Lazy loading: importing this package does not pull in heavy deps (fluxon_py / torch).
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BlobHandle, KVStore, TensorKVStore, Tier

__all__ = [
    "KVStore", "TensorKVStore", "Tier", "BlobHandle",
    "InMemoryKVStore", "LocalFileKVStore", "FluxonKVStore",
    "InMemoryTierStore", "LocalDiskTensorStore", "TensorStoreTierStore",
]

_LAZY: dict[str, tuple[str, str]] = {
    "InMemoryKVStore": ("cacheseek.stores.memory", "InMemoryKVStore"),
    "LocalFileKVStore": ("cacheseek.stores.local_file", "LocalFileKVStore"),
    "FluxonKVStore": ("cacheseek.stores.fluxon", "FluxonKVStore"),
    "InMemoryTierStore": ("cacheseek.stores.tier", "InMemoryTierStore"),
    "LocalDiskTensorStore": ("cacheseek.stores.tier", "LocalDiskTensorStore"),
    "TensorStoreTierStore": ("cacheseek.stores.tier", "TensorStoreTierStore"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
