from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from cacheseek.service.cache_types import IndexEntry


class LocalCacheMetadataManager:
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
        self._index: Dict[str, Dict[str, IndexEntry]] = self._load_index()
        self._meta: Dict[str, Dict[str, object]] = self._load_meta()
        self._flush_on_record_access = bool(flush_on_record_access)
        self._meta_dirty = False

    def register_cache(
        self,
        cache_id: str,
        prompt: str,
        saved_steps: List[int],
        size_mb: float,
        num_frames: int,
        cache_type: Optional[str] = None,
    ) -> None:
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

    def lookup_prompt(self, prompt: str, cache_type: Optional[str] = None) -> Optional[IndexEntry]:
        # dict.values() iterates in insertion order (Python 3.7+), so when the
        # same prompt was saved multiple times this returns the earliest-inserted
        # entry; calling it in a loop (as purge_by_prompt does) clears the full
        # history.
        def _scan(mapping: Dict[str, IndexEntry]) -> Optional[IndexEntry]:
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

    def get_cache_meta(self, cache_id: str) -> Optional[dict]:
        with self._lock:
            meta = self._meta.get(cache_id)
            if meta is None:
                return None
            return dict(meta)

    def record_access(self, cache_id: str) -> None:
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
        self.flush()

    def plan_eviction(self, required_mb: float, limit_mb: float) -> List[Tuple[str, Dict[str, object]]]:
        with self._lock:
            current_mb = sum(float(v.get("size_mb", 0.0)) for v in self._meta.values())
            if current_mb + required_mb <= limit_mb:
                return []
            need = current_mb + required_mb - limit_mb
            items = sorted(
                self._meta.items(),
                key=lambda kv: float(kv[1].get("last_access_time", 0.0)),
            )
            selected: List[Tuple[str, Dict[str, object]]] = []
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
        candidates: List[dict],
    ) -> None:
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

    def _load_index(self) -> Dict[str, Dict[str, IndexEntry]]:
        if not self._index_path.exists():
            return {}
        raw = self._read_json_object(self._index_path, "prompt index")

        # Schema: {cache_type: {cache_id: entry_dict}}
        result: Dict[str, Dict[str, IndexEntry]] = {}
        for cache_type, entries in raw.items():
            if not isinstance(entries, dict) or not entries:
                continue
            ct_str = str(cache_type)
            mapping: Dict[str, IndexEntry] = {}
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

    def _load_meta(self) -> Dict[str, Dict[str, object]]:
        if not self._meta_path.exists():
            return {}
        raw = self._read_json_object(self._meta_path, "cache metadata")
        return raw

    def _save_index(self) -> None:
        # Schema: {cache_type: {cache_id: entry_dict}}
        data: Dict[str, Dict[str, Dict[str, object]]] = {}
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

    def _normalize_cache_type(self, cache_type: Optional[str]) -> str:
        cache_type = str(cache_type or "").strip()
        return cache_type or self._default_cache_type

    def _read_json_object(self, path: Path, label: str) -> Dict[str, object]:
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
