from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class LocalFileKVStore:
    """On-disk KV store that persists one file per key.

    Thread-safe within a single process via an internal RLock guarding
    the in-memory index dict and disk index file. For cross-process
    safety, callers should additionally hold a FileLock on the cache
    directory around multi-resource transactions.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root_dir / "kv_index.json"
        self._lock = threading.RLock()
        self._index: Dict[str, str] = self._load_index()

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            filename = self._index.get(key)
            if not filename:
                filename = self._hash_key(key)
                self._index[key] = filename
            file_path = self.root_dir / filename
            tmp_path = self.root_dir / f"{filename}.tmp"
            tmp_path.write_bytes(value)
            os.replace(tmp_path, file_path)
            self._save_index()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            filename = self._index.get(key)
            if not filename:
                return None
            file_path = self.root_dir / filename
        # Read bytes outside the lock — file content is immutable once
        # written via put()'s atomic os.replace, so concurrent reads are safe.
        # A concurrent remove() can still unlink the file between the index
        # lookup and read_bytes(); treat that race as a miss rather than
        # surfacing FileNotFoundError to the caller.
        try:
            return file_path.read_bytes()
        except FileNotFoundError:
            return None

    def remove(self, key: str) -> None:
        with self._lock:
            filename = self._index.pop(key, None)
            if filename:
                file_path = self.root_dir / filename
                if file_path.exists():
                    file_path.unlink()
                self._save_index()

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._index.keys())

    def _hash_key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{digest}.bin"

    def _load_index(self) -> Dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            raw = self._index_path.read_text()
        except OSError as exc:
            logger.exception(
                "LocalFileKVStore failed to read index path={} err={}",
                self._index_path,
                exc,
            )
            raise RuntimeError(f"LocalFileKVStore failed to read index path={self._index_path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.exception(
                "LocalFileKVStore index is not valid JSON path={} err={}",
                self._index_path,
                exc,
            )
            raise ValueError(f"LocalFileKVStore index is not valid JSON path={self._index_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"LocalFileKVStore index must be a JSON object path={self._index_path} got_type={type(data).__name__}"
            )
        invalid_items = [
            (key, value) for key, value in data.items() if not isinstance(key, str) or not isinstance(value, str)
        ]
        if invalid_items:
            key, value = invalid_items[0]
            raise ValueError(
                "LocalFileKVStore index contains non-string entry "
                f"path={self._index_path} key_type={type(key).__name__} "
                f"value_type={type(value).__name__}"
            )
        return data

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._index, ensure_ascii=True))
        os.replace(tmp, self._index_path)
