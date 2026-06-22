"""Audit log backends — append-only event stream.

Alpha default: ``JSONLAuditLog`` (one JSON object per line, written via
POSIX append-atomic ``open(mode="a")``). Lazy-loaded.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["JSONLAuditLog"]

_LAZY: dict[str, tuple[str, str]] = {
    "JSONLAuditLog": ("cacheseek.backends.audit.jsonl", "JSONLAuditLog"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
