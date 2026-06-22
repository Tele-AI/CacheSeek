from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from cacheseek.service.cache_types import VectorSearchResult


class QdrantVectorStore:
    """Qdrant-backed VectorStore.

    Drop-in replacement for ``FAISSVectorStore`` that speaks the same contract
    declared in :mod:`cacheseek.service.interfaces.vector_store`.

    Connection modes
    ----------------
    * **In-memory** (default when ``url`` is empty): an ephemeral Qdrant
      instance is spun up via ``qdrant_client.QdrantClient(":memory:")``.
      Useful for tests and local smoke-checks without infra.
    * **Remote**: a real Qdrant server reached via ``url`` (e.g.
      ``http://host:6333``) with optional ``api_key``.

    Environment fallbacks: ``QDRANT_URL`` and ``QDRANT_API_KEY`` are
    consulted when the constructor args are empty.

    Distance metric is ``COSINE`` by default because the Qwen3-VL embeddings
    produced by :mod:`encoders` are L2-normalized, and upstream consumers in
    :mod:`strategies` already treat the returned ``similarity`` field as a
    direct cosine-style score (higher == better).  FAISS here uses plain L2
    and converts via ``1 / (1 + d)``; Qdrant returns the cosine score natively
    so no conversion is needed.

    The ``qdrant-client`` dependency is lazily imported inside each method —
    matches how :mod:`encoders` defers its Qwen3VL model import so installs
    that only use FAISS don't need to ship an extra wheel.
    """

    def __init__(
        self,
        url: str = "",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
        distance: str = "COSINE",
    ) -> None:
        # Resolution order: explicit arg → env var → empty (in-memory)
        self.url = (url or os.environ.get("QDRANT_URL", "") or "").strip()
        env_api_key = os.environ.get("QDRANT_API_KEY", "") or None
        self.api_key = api_key if api_key else env_api_key
        self.prefer_grpc = bool(prefer_grpc)
        self.timeout = int(timeout)
        self.distance = (distance or "COSINE").upper()

        self._client: Any = None
        # Track per-collection dim to validate upsert / search inputs fast,
        # cheaper than round-tripping to Qdrant.
        self._dims: Dict[str, int] = {}
        # Stable string-cache-id <-> point UUID map. Qdrant point IDs must be
        # int or UUID. We keep both directions so delete() can translate the
        # caller's cache_id back to the underlying point id.
        self._id_map: Dict[str, Dict[str, str]] = {}

        logger.debug(
            "QdrantVectorStore init url={} in_memory={} distance={}",
            self.url or "<in-memory>",
            not bool(self.url),
            self.distance,
        )

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        client = self._get_client()
        known_dim = self._dims.get(collection)
        if known_dim is not None and len(vector) != known_dim:
            raise ValueError(
                "QdrantVectorStore.search vector dimension mismatch "
                f"collection={collection} got={len(vector)} expected={known_dim}"
            )
        # qdrant-client >=1.10 deprecated `search()` in favor of
        # `query_points()`. Prefer the newer API, fall back for older servers.
        try:
            if hasattr(client, "query_points"):
                resp = client.query_points(
                    collection_name=collection,
                    query=list(vector),
                    limit=int(max(1, limit)),
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(resp, "points", None)
                if points is None and isinstance(resp, list):
                    points = resp
            else:
                points = client.search(
                    collection_name=collection,
                    query_vector=list(vector),
                    limit=int(max(1, limit)),
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception as exc:
            # Treat "collection not found" as empty result to match FAISS
            # behavior (returns [] when index missing). Other errors bubble up.
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "status_code=404" in msg:
                logger.debug(
                    "QdrantVectorStore.search collection missing collection={} err={}",
                    collection,
                    exc,
                )
                return []
            logger.exception(
                "QdrantVectorStore.search failed collection={} err={}",
                collection,
                exc,
            )
            raise RuntimeError(
                "QdrantVectorStore.search failed "
                f"collection={collection} err_type={type(exc).__name__} err={exc}"
            ) from exc

        results: List[VectorSearchResult] = []
        id_to_cache = self._id_map.setdefault(collection, {})
        for p in points or []:
            payload = dict(getattr(p, "payload", None) or {})
            # Prefer the caller-supplied cache_id embedded in payload; fall
            # back to the reverse map, then the raw point id.
            cache_id = payload.get("cache_id")
            if cache_id is None:
                cache_id = id_to_cache.get(str(getattr(p, "id", "")))
            if cache_id is None:
                cache_id = str(getattr(p, "id", ""))
            similarity = float(getattr(p, "score", 0.0))
            if score_threshold is not None and similarity < float(score_threshold):
                continue
            results.append(
                VectorSearchResult(
                    cache_id=str(cache_id),
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
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        client = self._get_client()
        qmodels = self._qdrant_models()

        known_dim = self._dims.get(collection)
        if known_dim is None:
            # Collection may have been created out-of-band; best-effort auto-create
            # matches FAISS's upsert path (which calls ensure_collection when missing).
            self.ensure_collection(collection, len(vector))
            known_dim = self._dims.get(collection, len(vector))
        if len(vector) != known_dim:
            logger.exception(
                "QdrantVectorStore.upsert vector dimension mismatch "
                f"collection={collection} got={len(vector)} expected={known_dim}"
            )
            raise ValueError(
                "QdrantVectorStore.upsert vector dimension mismatch "
                f"collection={collection} got={len(vector)} expected={known_dim}"
            )

        qid = self._to_point_id(point_id)
        # Mirror cache_id into payload so search() can recover the original
        # string even after a process restart (the in-memory _id_map is lost).
        merged_payload = dict(payload or {})
        merged_payload.setdefault("cache_id", str(point_id))

        try:
            client.upsert(
                collection_name=collection,
                points=[
                    qmodels.PointStruct(
                        id=qid,
                        vector=list(vector),
                        payload=merged_payload,
                    )
                ],
            )
        except Exception as exc:
            logger.exception(
                "QdrantVectorStore.upsert failed collection={} point_id={} err={}",
                collection,
                point_id,
                exc,
            )
            raise RuntimeError(
                "QdrantVectorStore.upsert failed "
                f"collection={collection} point_id={point_id} "
                f"err_type={type(exc).__name__} err={exc}"
            ) from exc

        self._id_map.setdefault(collection, {})[str(qid)] = str(point_id)

    def delete(self, collection: str, point_ids: List[str]) -> None:
        if not point_ids:
            return
        client = self._get_client()
        qmodels = self._qdrant_models()
        qids = [self._to_point_id(pid) for pid in point_ids]
        try:
            client.delete(
                collection_name=collection,
                points_selector=qmodels.PointIdsList(points=qids),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "status_code=404" in msg:
                logger.debug(
                    "QdrantVectorStore.delete collection missing collection={} err={}",
                    collection,
                    exc,
                )
                return
            logger.exception(
                "QdrantVectorStore.delete failed collection={} ids={} err={}",
                collection,
                point_ids,
                exc,
            )
            raise RuntimeError(
                "QdrantVectorStore.delete failed "
                f"collection={collection} ids={point_ids} "
                f"err_type={type(exc).__name__} err={exc}"
            ) from exc

        id_map = self._id_map.get(collection)
        if id_map:
            for pid, qid in zip(point_ids, qids):
                id_map.pop(str(qid), None)
                id_map.pop(str(pid), None)

    def ensure_collection(self, collection: str, vector_dim: int) -> None:
        client = self._get_client()
        qmodels = self._qdrant_models()
        vector_dim = int(vector_dim)

        exists = False
        try:
            exists = bool(client.collection_exists(collection_name=collection))
        except AttributeError:
            # Older qdrant-client: fall back to get_collection, trap not-found.
            try:
                client.get_collection(collection_name=collection)
                exists = True
            except Exception:
                exists = False
        except Exception as exc:
            logger.exception(
                "QdrantVectorStore.ensure_collection exists check failed "
                "collection={} err={}",
                collection,
                exc,
            )
            raise RuntimeError(
                "QdrantVectorStore.ensure_collection exists check failed "
                f"collection={collection} err_type={type(exc).__name__} err={exc}"
            ) from exc

        if exists:
            # Sync cached dim so validation in upsert/search has a source.
            self._dims[collection] = self._probe_vector_size(collection) or vector_dim
            return

        distance = self._distance_enum(qmodels)
        try:
            client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_dim,
                    distance=distance,
                ),
            )
        except Exception as exc:
            logger.exception(
                "QdrantVectorStore.ensure_collection create failed "
                "collection={} dim={} err={}",
                collection,
                vector_dim,
                exc,
            )
            raise RuntimeError(
                "QdrantVectorStore.ensure_collection create failed "
                f"collection={collection} dim={vector_dim} "
                f"err_type={type(exc).__name__} err={exc}"
            ) from exc
        self._dims[collection] = vector_dim
        self._id_map.setdefault(collection, {})

    def delete_collection(self, collection: str) -> None:
        """Idempotent — missing collection is not an error."""
        client = self._get_client()
        try:
            client.delete_collection(collection_name=collection)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "doesn't exist" in msg or "status_code=404" in msg:
                logger.debug(
                    "QdrantVectorStore.delete_collection missing collection={} err={}",
                    collection,
                    exc,
                )
            else:
                logger.exception(
                    "QdrantVectorStore.delete_collection failed collection={} err={}",
                    collection,
                    exc,
                )
                raise RuntimeError(
                    "QdrantVectorStore.delete_collection failed "
                    f"collection={collection} err_type={type(exc).__name__} err={exc}"
                ) from exc
        self._dims.pop(collection, None)
        self._id_map.pop(collection, None)

    def get_vector_size(self, collection: str) -> Optional[int]:
        cached = self._dims.get(collection)
        if cached is not None:
            return int(cached)
        return self._probe_vector_size(collection)

    @property
    def client(self) -> Any:
        """Expose underlying qdrant-client; mirrors the ``client`` attribute
        that ``ConnectionManager.health_check`` probes for reachability."""
        return self._get_client()

    def close(self) -> None:
        if self._client is None:
            return
        for method in ("close", "shutdown"):
            fn = getattr(self._client, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    logger.exception(
                        "QdrantVectorStore.close {} failed err={}", method, exc
                    )
                break
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "qdrant-client is not installed; QdrantVectorStore is unavailable."
                " Install it with `pip install qdrant-client`."
            ) from exc

        if not self.url:
            # In-memory Qdrant: no server, no persistence — the
            # "user has no infra" default for tests and smoke-checks.
            self._client = QdrantClient(location=":memory:")
            logger.debug("QdrantVectorStore in-memory client ready")
        else:
            kwargs: Dict[str, Any] = {
                "url": self.url,
                "prefer_grpc": self.prefer_grpc,
                "timeout": self.timeout,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            try:
                self._client = QdrantClient(**kwargs)
            except Exception as exc:
                logger.exception(
                    "QdrantVectorStore client init failed url={} err={}",
                    self.url,
                    exc,
                )
                raise RuntimeError(
                    "QdrantVectorStore client init failed "
                    f"url={self.url} err_type={type(exc).__name__} err={exc}"
                ) from exc
            logger.debug("QdrantVectorStore remote client ready url={}", self.url)
        return self._client

    def _qdrant_models(self) -> Any:
        try:
            from qdrant_client.http import models as qmodels  # type: ignore
        except (ImportError, ModuleNotFoundError):
            # Newer qdrant-client exposes models directly on the package root.
            try:
                from qdrant_client import models as qmodels  # type: ignore
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "qdrant-client models module not importable"
                ) from exc
        return qmodels

    def _distance_enum(self, qmodels: Any) -> Any:
        distance_cls = getattr(qmodels, "Distance", None)
        if distance_cls is None:
            raise RuntimeError(
                "qdrant-client models.Distance not available — version too old?"
            )
        name = self.distance.upper()
        value = getattr(distance_cls, name, None)
        if value is None:
            raise ValueError(
                f"Unsupported Qdrant distance metric '{self.distance}'. "
                "Supported: COSINE | EUCLID | DOT."
            )
        return value

    def _probe_vector_size(self, collection: str) -> Optional[int]:
        client = self._get_client()
        try:
            info = client.get_collection(collection_name=collection)
        except Exception:
            return None
        # qdrant-client exposes vector params at slightly different paths
        # depending on version — be defensive.
        try:
            cfg = info.config.params.vectors
        except AttributeError:
            return None
        size = getattr(cfg, "size", None)
        if size is None and isinstance(cfg, dict):
            # Named-vectors mode: pick first.
            for v in cfg.values():
                size = getattr(v, "size", None)
                if size is not None:
                    break
        if size is None:
            return None
        self._dims[collection] = int(size)
        return int(size)

    def _to_point_id(self, cache_id: str) -> str:
        """Convert the caller's string cache_id to a Qdrant-acceptable UUID.

        Qdrant rejects arbitrary strings; only unsigned int or UUID strings
        are valid point IDs. We derive a deterministic UUID5 from the
        cache_id so the same cache_id always maps to the same point, keeping
        upsert() idempotent without needing a server-side lookup.
        """
        cid = str(cache_id)
        try:
            return str(uuid.UUID(cid))
        except (ValueError, AttributeError) as exc:
            logger.trace("cache_id {!r} is not a UUID, deriving deterministic id ({})", cid, exc)
        # Non-cryptographic: hash is only used to derive a stable UUID-shaped
        # point id from an arbitrary string. SHA-256 (truncated) is plenty.
        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()
        return str(uuid.UUID(digest[:32]))
