# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""In-memory KVStore: an ephemeral dict-backed store for tests and single-process use."""

from __future__ import annotations


class InMemoryKVStore:
    """In-memory KV store backed by a plain dict."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self) -> list[str]:
        return list(self._store.keys())
