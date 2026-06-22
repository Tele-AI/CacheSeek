# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""FAISS-backed VectorStore (embedded, on-disk index)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from cacheseek.service.cache_types import VectorSearchResult


class FAISSVectorStore:
    """FAISS-backed VectorStore with on-disk persistence per collection.

    Fulfils the ``VectorStore`` Protocol. Each collection is a flat FAISS index
    wrapped in an ``IndexIDMap2`` plus a sidecar JSON holding the stable
    string-``cache_id`` -> int-id map, payloads, and a monotonic ``next_id``.
    Both are stored under ``index_dir`` as ``{collection}.faiss`` /
    ``{collection}.json``; indices are cached in memory after first load.

    ``index_type`` selects the metric: ``"cosine"`` (default; vectors are
    L2-normalized and an inner-product index returns the cosine score directly),
    ``"ip"``/``"innerproduct"`` (raw inner product), or ``"l2"`` (Euclidean,
    converted to a similarity via ``1 / (1 + d)``). All operations are
    serialized under a reentrant lock for thread safety. ``faiss`` is imported
    lazily so installs that do not use this backend need not ship the wheel.
    """

    def __init__(
        self,
        index_dir: Path,
        vector_dim: int,
        index_type: str = "cosine",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.vector_dim = vector_dim
        self.index_type = index_type
        self._lock = threading.RLock()
        self._indices: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 1,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Return up to ``limit`` nearest payloads for ``vector`` in ``collection``.

        Returns an empty list if the collection does not exist. The query vector
        is normalized for cosine indices; L2 distances are converted to a
        similarity score. Results below ``score_threshold`` (when given) are
        dropped, as are FAISS slots with no live ``id_map`` entry.

        Raises:
            ValueError: If ``vector`` length does not match the store's
                configured ``vector_dim``.
        """
        with self._lock:
            index = self._load_index(collection)
            if index is None:
                return []
            import numpy as np

            meta = self._metadata.get(collection, {})
            id_map = meta.get("id_map", {})
            payload_map = meta.get("payload", {})

            vec = np.asarray([vector], dtype="float32")
            if vec.shape[1] != self.vector_dim:
                raise ValueError(
                    "FAISSVectorStore.search vector dimension mismatch "
                    f"collection={collection} got={vec.shape[1]} expected={self.vector_dim}"
                )
            if self.index_type.lower() == "cosine":
                faiss = self._import_faiss()
                faiss.normalize_L2(vec)

            distances, ids = index.search(vec, limit)
            results: list[VectorSearchResult] = []
            for dist, idx in zip(distances[0], ids[0], strict=False):
                if idx < 0:
                    continue
                point_id = self._find_point_id(id_map, int(idx))
                if point_id is None:
                    continue
                payload = payload_map.get(point_id, {})
                if self.index_type.lower() == "l2":
                    similarity = 1.0 / (1.0 + float(dist))
                else:
                    similarity = float(dist)
                if score_threshold is not None and similarity < score_threshold:
                    continue
                results.append(
                    VectorSearchResult(
                        cache_id=str(point_id),
                        similarity=similarity,
                        prompt=str(payload.get("prompt", "")),
                        saved_steps=list(payload.get("saved_steps", [])),
                        payload=payload,
                    )
                )
            return results

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or replace ``point_id`` with ``vector`` and ``payload``; persists.

        Auto-creates the collection if missing. Re-upserting an existing
        ``point_id`` removes the old vector first (keeping its int id), so the
        operation is idempotent. The vector is normalized for cosine indices.

        Raises:
            ValueError: If ``vector`` length does not match the store's
                configured ``vector_dim``.
            RuntimeError: If the collection can be neither loaded nor created.
        """
        with self._lock:
            index = self._load_index(collection)
            if index is None:
                self.ensure_collection(collection, self.vector_dim)
                index = self._load_index(collection)
            if index is None:
                raise RuntimeError(
                    f"FAISSVectorStore.upsert could not load or create collection collection={collection}"
                )
            import numpy as np

            meta = self._metadata.setdefault(collection, {"id_map": {}, "payload": {}, "next_id": 1})
            id_map = meta["id_map"]
            payload_map = meta["payload"]
            vec = np.asarray([vector], dtype="float32")
            if vec.shape[1] != self.vector_dim:
                raise ValueError(
                    "FAISSVectorStore.upsert vector dimension mismatch "
                    f"collection={collection} got={vec.shape[1]} expected={self.vector_dim}"
                )
            if self.index_type.lower() == "cosine":
                faiss = self._import_faiss()
                faiss.normalize_L2(vec)

            existing = id_map.get(point_id)
            if existing is not None:
                self._remove_ids(index, [int(existing)])
            else:
                existing = int(meta.get("next_id", 1))
                meta["next_id"] = existing + 1
                id_map[point_id] = existing

            index.add_with_ids(vec, self._as_faiss_ids([existing]))
            payload_map[point_id] = payload
            self._save_index(collection, index)

    def delete(self, collection: str, point_ids: list[str]) -> None:
        """Remove the given point ids and their payloads from ``collection``; persists.

        No-op for a missing collection or for ids not present. Always rewrites
        the index/meta files (even when nothing matched).
        """
        with self._lock:
            index = self._load_index(collection)
            if index is None:
                return
            meta = self._metadata.get(collection, {})
            id_map = meta.get("id_map", {})
            payload_map = meta.get("payload", {})
            to_remove = []
            for pid in point_ids:
                idx = id_map.pop(pid, None)
                if idx is not None:
                    to_remove.append(int(idx))
                payload_map.pop(pid, None)
            if to_remove:
                self._remove_ids(index, to_remove)
            self._save_index(collection, index)

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        """Create the collection's index if absent; no-op if it already exists.

        The index family is chosen from ``index_type`` (``IndexFlatL2`` for
        ``"l2"``, ``IndexFlatIP`` for inner-product / cosine) and wrapped in an
        ``IndexIDMap2``. Initializes empty metadata and persists to disk.

        Raises:
            ValueError: If ``index_type`` is not a supported value.
        """
        with self._lock:
            index = self._load_index(collection)
            if index is not None:
                return
            faiss = self._import_faiss()
            if self.index_type.lower() == "l2":
                base = faiss.IndexFlatL2(vector_dim)
            elif self.index_type.lower() in ("ip", "innerproduct") or self.index_type.lower() == "cosine":
                base = faiss.IndexFlatIP(vector_dim)
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
            index = faiss.IndexIDMap2(base)
            self._indices[collection] = index
            self._metadata[collection] = {"id_map": {}, "payload": {}, "next_id": 1}
            self._save_index(collection, index)

    def get_vector_size(self, collection: str) -> int | None:
        """Return the collection's stored vector dimension, or None if it is missing."""
        with self._lock:
            index = self._load_index(collection)
            if index is None:
                return None
            return int(index.d)

    def _load_index(self, collection: str) -> Any | None:
        if collection in self._indices:
            return self._indices[collection]
        index_path, meta_path = self._get_paths(collection)
        if not index_path.exists():
            return None
        faiss = self._import_faiss()
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            logger.exception(
                "FAISSVectorStore failed to read index collection={} path={} err={}",
                collection,
                index_path,
                exc,
            )
            raise RuntimeError(
                "FAISSVectorStore failed to read index "
                f"collection={collection} path={index_path} "
                f"err_type={type(exc).__name__} err={exc}"
            ) from exc
        self._indices[collection] = index
        empty_meta = {"id_map": {}, "payload": {}, "next_id": 1}
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text())
            except OSError as exc:
                logger.exception(
                    "FAISSVectorStore failed to read metadata collection={} path={} err={}",
                    collection,
                    meta_path,
                    exc,
                )
                raise RuntimeError(
                    "FAISSVectorStore failed to read metadata "
                    f"collection={collection} path={meta_path} "
                    f"err_type={type(exc).__name__} err={exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                logger.exception(
                    "FAISSVectorStore metadata is not valid JSON collection={} path={} err={}",
                    collection,
                    meta_path,
                    exc,
                )
                raise ValueError(
                    f"FAISSVectorStore metadata is not valid JSON collection={collection} path={meta_path}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    "FAISSVectorStore metadata must be a JSON object "
                    f"collection={collection} path={meta_path} "
                    f"got_type={type(raw).__name__}"
                )
            id_map = raw.get("id_map", {})
            if not isinstance(id_map, dict):
                raise ValueError(
                    "FAISSVectorStore metadata.id_map must be a dict "
                    f"collection={collection} path={meta_path} "
                    f"got_type={type(id_map).__name__}"
                )
            if len(id_map) != index.ntotal:
                # Don't hard-fail and brick the whole collection on a torn-write
                # id_map / index count mismatch — the read path tolerates both
                # directions. An orphan vector in the index (no id_map entry) is
                # skipped by _find_point_id returning None during search; a dead
                # id_map entry pointing at a nonexistent int id simply never
                # matches. Degrading to a warning and continuing is safer than
                # discarding the entire store (including all payloads).
                logger.warning(
                    "FAISSVectorStore metadata/index count mismatch (tolerated) "
                    "collection={} path={} id_map={} ntotal={}",
                    collection,
                    str(meta_path),
                    len(id_map),
                    index.ntotal,
                )
            self._metadata[collection] = raw
        else:
            self._metadata[collection] = empty_meta
        return index

    def _save_index(self, collection: str, index: Any) -> None:
        import os

        index_path, meta_path = self._get_paths(collection)
        faiss = self._import_faiss()
        tmp_index = index_path.with_suffix(".faiss.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")
        try:
            faiss.write_index(index, str(tmp_index))
        except Exception as exc:
            logger.exception(
                "FAISSVectorStore._save_index failed writing index collection={} path={} err={}",
                collection,
                str(tmp_index),
                exc,
            )
            raise RuntimeError(f"FAISSVectorStore failed to write index for collection={collection}") from exc
        meta = self._metadata.get(collection, {"id_map": {}, "payload": {}, "next_id": 1})
        try:
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=True))
        except Exception as exc:
            logger.exception(
                "FAISSVectorStore._save_index failed writing metadata collection={} path={} err={}",
                collection,
                str(tmp_meta),
                exc,
            )
            raise RuntimeError(f"FAISSVectorStore failed to write metadata for collection={collection}") from exc
        # Publish the index before the meta: if a crash lands between the two
        # os.replace calls, letting the index lead the meta only yields an
        # orphan vector not covered by id_map (skipped harmlessly at search
        # time). The reverse order would leave the meta pointing at a vector
        # that doesn't exist (a riskier phantom read). Together with the
        # relaxed reload check above, a torn write no longer bricks the
        # collection.
        os.replace(tmp_index, index_path)
        os.replace(tmp_meta, meta_path)

    def _get_paths(self, collection: str) -> tuple[Path, Path]:
        index_path = self.index_dir / f"{collection}.faiss"
        meta_path = self.index_dir / f"{collection}.json"
        return index_path, meta_path

    def _import_faiss(self):
        try:
            import faiss  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError("faiss is not installed; FAISSVectorStore is unavailable.") from exc
        return faiss

    def _as_faiss_ids(self, ids: list[int]):
        import numpy as np

        return np.asarray(ids, dtype="int64")

    def _remove_ids(self, index: Any, ids: list[int]) -> None:
        faiss = self._import_faiss()
        selector = faiss.IDSelectorBatch(len(ids), self._as_faiss_ids(ids))
        index.remove_ids(selector)

    def _find_point_id(self, id_map: dict[str, Any], idx: int) -> str | None:
        for key, value in id_map.items():
            if int(value) == idx:
                return str(key)
        return None
