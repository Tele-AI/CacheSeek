# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Local filesystem MetadataStore: a JSON cache index plus size/access accounting."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from loguru import logger

from cacheseek.service.cache_types import IndexEntry


class LocalCacheMetadataManager:
    """File-backed metadata store + audit log for the local cache.

    Fulfils both the ``MetadataStore`` (cache index, access stats, eviction
    planning) and ``AuditLog`` (``record_hit_pair`` / ``record_similarity_scores``)
    Protocols. State persists as two JSON documents under ``metadata_cache_dir``:
    ``prompt_index.json`` (a ``{cache_type: {cache_id: IndexEntry}}`` index keyed
    for fast prompt lookup) and ``cache_meta.json`` (per-cache size / access
    stats for eviction). Audit events are appended to sibling ``.jsonl`` files.

    All mutating operations are serialized under a reentrant lock, so the
    instance is safe to share across ``CacheService`` worker threads. Index and
    meta writes are atomic (temp file + ``os.replace``). When
    ``flush_on_record_access`` is False, ``record_access`` only marks meta dirty
    in memory and defers the disk write until ``flush`` / ``shutdown``.
    """

    def __init__(
        self,
        metadata_cache_dir: str | Path,
        *,
        flush_on_record_access: bool = True,
    ) -> None:
        self._default_cache_type = "approximate_cache"
        self.metadata_cache_dir = Path(metadata_cache_dir)
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.metadata_cache_dir / "prompt_index.json"
        self._meta_path = self.metadata_cache_dir / "cache_meta.json"
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, IndexEntry]] = self._load_index()
        self._meta: dict[str, dict[str, object]] = self._load_meta()
        self._flush_on_record_access = bool(flush_on_record_access)
        self._meta_dirty = False

    def register_cache(
        self,
        cache_id: str,
        prompt: str,
        saved_steps: list[int],
        size_mb: float,
        num_frames: int,
        cache_type: str | None = None,
    ) -> None:
        """Index a cache entry and (re)write its size/access metadata; persists both files.

        ``saved_steps`` are deduped and sorted; ``cache_type`` defaults to
        ``"approximate_cache"``. Re-registering an existing ``cache_id`` preserves its
        prior ``access_count`` and refreshes ``last_access_time``.
        """
        steps = sorted(set(int(s) for s in saved_steps))
        # Normalize so None never collides with the string "None" after JSON round-trip.
        normalized_cache_type = self._normalize_cache_type(cache_type)
        with self._lock:
            index = self._index.setdefault(normalized_cache_type, {})
            index[cache_id] = IndexEntry(
                cache_id=cache_id,
                prompt=prompt,
                saved_steps=steps,
                cache_type=normalized_cache_type,
            )
            self._meta[cache_id] = {
                "prompt": prompt,
                "saved_steps": steps,
                "size_mb": float(size_mb),
                "num_frames": int(num_frames),
                "access_count": int(self._meta.get(cache_id, {}).get("access_count", 0)),
                "last_access_time": float(time.time()),
                "cache_type": normalized_cache_type,
            }
            self._save_index()
            self._save_meta()

    def remove_cache(self, cache_id: str) -> None:
        """Drop a cache from both the index and the meta map; persists both files.

        Uses the recorded ``cache_type`` to find the index bucket in O(1);
        falls back to a full scan across all buckets if it is missing. No-op if
        ``cache_id`` is unknown.
        """
        with self._lock:
            meta = self._meta.pop(cache_id, None)
            cache_type = meta.get("cache_type") if meta else None
            if cache_type:
                self._index.get(str(cache_type), {}).pop(cache_id, None)
            else:
                # cache_type unknown: fall back to a full scan (rare path).
                logger.debug(
                    "LocalCacheMetadataManager.remove_cache fallback scan (cache_type missing) cache_id={}",
                    cache_id,
                )
                for mapping in self._index.values():
                    mapping.pop(cache_id, None)
            self._save_index()
            self._save_meta()

    def lookup_prompt(self, prompt: str, cache_type: str | None = None) -> IndexEntry | None:
        """Return the earliest-indexed entry whose prompt exactly equals ``prompt``.

        When ``cache_type`` is given, only that bucket is scanned; otherwise the
        default bucket (``"approximate_cache"``) is tried first, then the rest.
        Returns None if no exact match exists.
        """
        # dict.values() iterates in insertion order (Python 3.7+), so when the
        # same prompt was saved multiple times this returns the earliest-inserted
        # entry; calling it in a loop (as purge_by_prompt does) clears the full
        # history.
        def _scan(mapping: dict[str, IndexEntry]) -> IndexEntry | None:
            for entry in mapping.values():
                if entry.prompt == prompt:
                    return entry
            return None

        with self._lock:
            if cache_type:
                return _scan(self._index.get(self._normalize_cache_type(cache_type), {}))
            # Default to text cache first, then scan others.
            entry = _scan(self._index.get(self._default_cache_type, {}))
            if entry is not None:
                return entry
            for mapping in self._index.values():
                entry = _scan(mapping)
                if entry is not None:
                    return entry
            return None

    def get_cache_meta(self, cache_id: str) -> dict | None:
        """Return a copy of the cache's metadata dict, or None if unknown."""
        with self._lock:
            meta = self._meta.get(cache_id)
            if meta is None:
                return None
            return dict(meta)

    def record_access(self, cache_id: str) -> None:
        """Bump the cache's access count and last-access time; no-op if unknown.

        The ``cache_id`` is normalized (dashes stripped) before lookup. The meta
        write is flushed immediately unless ``flush_on_record_access`` is False,
        in which case the update is buffered and persisted on the next ``flush``.
        """
        normalized = self._normalize_cache_id(cache_id)
        with self._lock:
            meta = self._meta.get(normalized)
            if meta is None:
                return
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            meta["last_access_time"] = float(time.time())
            if self._flush_on_record_access:
                self._save_meta()
            else:
                self._meta_dirty = True

    def flush(self) -> None:
        """Persist any in-memory meta updates that were deferred from record_access."""
        with self._lock:
            if self._meta_dirty:
                self._save_meta()

    def shutdown(self) -> None:
        """Flush any deferred meta updates before the process exits."""
        self.flush()

    def plan_eviction(self, required_mb: float, limit_mb: float) -> list[tuple[str, dict[str, object]]]:
        """Select least-recently-accessed caches to evict to fit a new write.

        Computes current total size from the meta map; if adding ``required_mb``
        stays within ``limit_mb``, returns an empty list. Otherwise returns
        ``(cache_id, meta)`` pairs ordered oldest-access-first, accumulating just
        enough to free the deficit. This is a plan only — it does not mutate
        state or delete anything.
        """
        with self._lock:
            current_mb = sum(float(v.get("size_mb", 0.0)) for v in self._meta.values())
            if current_mb + required_mb <= limit_mb:
                return []
            need = current_mb + required_mb - limit_mb
            items = sorted(
                self._meta.items(),
                key=lambda kv: float(kv[1].get("last_access_time", 0.0)),
            )
            selected: list[tuple[str, dict[str, object]]] = []
            freed = 0.0
            for cache_id, meta in items:
                selected.append((cache_id, meta))
                freed += float(meta.get("size_mb", 0.0))
                if freed >= need:
                    break
            return selected

    def record_hit_pair(
        self,
        request_prompt: str,
        cache_id: str,
        cached_prompt: str,
        similarity: float,
        task_type: str,
        cache_type: str,
        skip_step: int,
    ) -> None:
        """Append one cache-hit record to ``hit_pairs.jsonl`` (AuditLog surface).

        Writes a single JSON line pairing the incoming request prompt with the
        matched cached prompt and the hit's similarity / task / cache type and
        skip step. The row schema mirrors ``HitPairEvent``.
        """
        payload = {
            "timestamp": float(time.time()),
            "request_prompt": str(request_prompt or ""),
            "cache_id": str(cache_id),
            "cached_prompt": str(cached_prompt or ""),
            "similarity": float(similarity),
            "task_type": str(task_type or ""),
            "cache_type": str(cache_type or ""),
            "skip_step": int(skip_step),
        }
        log_path = self.metadata_cache_dir / "hit_pairs.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def record_similarity_scores(
        self,
        request_prompt: str,
        task_type: str,
        cache_type: str,
        stage: str,
        candidates: list[dict],
    ) -> None:
        """Append a candidate-ranking snapshot to ``similarity_scores.jsonl``.

        Records the scored ``candidates`` for one ranking ``stage`` (e.g.
        ``"vector_search"`` or ``"rerank"``) against a request. The row schema
        mirrors ``SimilarityScoreEvent``.
        """
        payload = {
            "timestamp": float(time.time()),
            "request_prompt": str(request_prompt or ""),
            "task_type": str(task_type or ""),
            "cache_type": str(cache_type or ""),
            "stage": str(stage or ""),
            "candidates": candidates,
        }
        log_path = self.metadata_cache_dir / "similarity_scores.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _load_index(self) -> dict[str, dict[str, IndexEntry]]:
        if not self._index_path.exists():
            return {}
        raw = self._read_json_object(self._index_path, "prompt index")

        # Schema: {cache_type: {cache_id: entry_dict}}
        result: dict[str, dict[str, IndexEntry]] = {}
        for cache_type, entries in raw.items():
            if not isinstance(entries, dict) or not entries:
                continue
            ct_str = str(cache_type)
            mapping: dict[str, IndexEntry] = {}
            for cache_id, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                mapping[str(cache_id)] = IndexEntry(
                    cache_id=str(cache_id),
                    prompt=str(entry.get("prompt", "")),
                    saved_steps=[int(x) for x in entry.get("saved_steps", [])],
                    cache_type=str(entry.get("cache_type") or ct_str or self._default_cache_type),
                )
            if mapping:
                result[ct_str] = mapping
        return result

    def _load_meta(self) -> dict[str, dict[str, object]]:
        if not self._meta_path.exists():
            return {}
        raw = self._read_json_object(self._meta_path, "cache metadata")
        return raw

    def _save_index(self) -> None:
        # Schema: {cache_type: {cache_id: entry_dict}}
        data: dict[str, dict[str, dict[str, object]]] = {}
        for cache_type, mapping in self._index.items():
            data[str(cache_type)] = {
                cache_id: {
                    "prompt": entry.prompt,
                    "saved_steps": entry.saved_steps,
                    "cache_type": entry.cache_type or cache_type,
                }
                for cache_id, entry in mapping.items()
            }
        self._atomic_write_json(self._index_path, data)

    def _save_meta(self) -> None:
        self._atomic_write_json(self._meta_path, self._meta)
        self._meta_dirty = False

    @staticmethod
    def _atomic_write_json(path: Path, data: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=True))
        os.replace(tmp, path)

    def _normalize_cache_id(self, cache_id: str) -> str:
        return (cache_id or "").replace("-", "")

    def _normalize_cache_type(self, cache_type: str | None) -> str:
        cache_type = str(cache_type or "").strip()
        return cache_type or self._default_cache_type

    def _read_json_object(self, path: Path, label: str) -> dict[str, object]:
        try:
            raw = path.read_text()
        except OSError as exc:
            logger.exception(
                "LocalCacheMetadataManager failed to read {} path={} err={}",
                label,
                path,
                exc,
            )
            raise RuntimeError(f"LocalCacheMetadataManager failed to read {label} path={path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.exception(
                "LocalCacheMetadataManager {} is not valid JSON path={} err={}",
                label,
                path,
                exc,
            )
            raise ValueError(f"LocalCacheMetadataManager {label} is not valid JSON path={path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"LocalCacheMetadataManager {label} must be a JSON object path={path} got_type={type(data).__name__}"
            )
        return data
