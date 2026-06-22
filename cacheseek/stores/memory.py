from __future__ import annotations

from typing import Dict, Optional


class InMemoryKVStore:
    """In-memory KV store backed by a plain dict."""

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self) -> list[str]:
        return list(self._store.keys())
