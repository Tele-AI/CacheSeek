"""Encoder backends — prompt / video → vector and reranker scoring.

Implementations are lazy-loaded so ``torch`` / ``transformers`` are only
imported when an encoder or reranker is actually instantiated.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["Qwen3VLEncoder", "Qwen3VLReranker"]

_LAZY: dict[str, tuple[str, str]] = {
    "Qwen3VLEncoder": ("cacheseek.backends.encoder.qwen3vl", "Qwen3VLEncoder"),
    "Qwen3VLReranker": ("cacheseek.backends.encoder.qwen3vl", "Qwen3VLReranker"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
