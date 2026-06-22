from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from cacheseek.service.cache_types import VectorSearchResult
from cacheseek.service.config import CacheConfig
from cacheseek.reuse.approximate.payload import (
    VideoApproxPartialSpec,
    VideoApproxPayload,
)
from cacheseek.service.interfaces.encoder import PromptEncoder, VideoEncoder
from cacheseek.stores.base import KVStore
from cacheseek.service.interfaces.metadata_store import MetadataStore
from cacheseek.service.interfaces.vector_store import VectorStore
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult


class BaseCacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    def __init__(
        self,
        config: CacheConfig,
        kv_store: KVStore,
        metadata_manager: MetadataStore,
    ):
        self.config = config
        self.kv_store = kv_store
        self.metadata_manager = metadata_manager

    @abstractmethod
    async def lookup(
        self, query: CacheQuery, ctx: Any = None
    ) -> LookupResult:
        """Strategy Protocol: query in, LookupResult out (with payload + resume_hint)."""
        pass

    @abstractmethod
    async def save(
        self, query: CacheQuery, outputs: ModelOutputs, ctx: Any = None
    ) -> None:
        """Strategy Protocol: query + outputs in, no return."""
        pass

    def _audit_emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Best-effort: forward event to attached ``AuditLog``.

        ``cache_factory`` injects an ``AuditLog`` impl as
        ``self._cacheseek_audit_log`` when constructing the strategy; if
        absent (tests / minimal wiring), this is a no-op. Errors are
        logged but never re-raised — audit must not crash the lookup /
        save paths.
        """
        audit_log = getattr(self, "_cacheseek_audit_log", None)
        if audit_log is None:
            return
        try:
            audit_log.record(event_type, payload)
        except Exception as exc:
            logger.exception(
                "Strategy._audit_emit failed event_type={} err={}",
                event_type,
                exc,
            )

    def shutdown(self) -> None:
        """Cascade-close attached backend handles.

        Cleans up in this order:
        1. ``metadata_manager`` (if it exposes ``shutdown`` / ``close``)
        2. attached ``ConnectionManager`` — ``cache_factory`` injects it
           as ``self._cacheseek_conn_mgr``; the manager owns KV + vector
           handles.
        3. fallback close on KV / vector store directly when no
           ``ConnectionManager`` is attached (e.g. tests that pass raw
           stores).
        """
        for name in ("metadata_manager",):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            for method_name in ("shutdown", "close"):
                if hasattr(obj, method_name):
                    try:
                        getattr(obj, method_name)()
                    except Exception as exc:
                        logger.exception(
                            "Strategy.shutdown {}.{} failed err={}",
                            name,
                            method_name,
                            exc,
                        )
                    break

        conn_mgr = getattr(self, "_cacheseek_conn_mgr", None)
        if conn_mgr is not None:
            try:
                conn_mgr.shutdown()
            except Exception as exc:
                logger.exception("Strategy.shutdown ConnectionManager failed err={}", exc)
        else:
            for name in ("kv_store", "vector_store"):
                obj = getattr(self, name, None)
                if obj is None:
                    continue
                for method_name in ("shutdown", "close"):
                    if hasattr(obj, method_name):
                        try:
                            getattr(obj, method_name)()
                        except Exception as exc:
                            logger.exception(
                                "Strategy.shutdown {}.{} failed err={}",
                                name,
                                method_name,
                                exc,
                            )
                        break

    def _load_payload(
        self, cache_id: str, partial_spec: VideoApproxPartialSpec
    ) -> Optional[VideoApproxPayload]:
        """Load a VideoApproxPayload covering the requested steps.

        Returns ``None`` when no requested step is present in the KV
        store (caller treats this as a stale-vector miss and lazy-evicts
        the entry). Returns the payload otherwise — bytes are stored
        un-deserialized; ``get_latent_at_step`` does the conversion.
        """
        try:
            payload = VideoApproxPayload.from_kv_loader(
                cache_id, self.kv_store.get, partial_spec=partial_spec
            )
        except Exception as exc:
            logger.exception(
                "Cache payload load failed cache_id={} steps={} err={}",
                cache_id,
                partial_spec.steps,
                exc,
            )
            raise RuntimeError(
                f"Cache payload load failed cache_id={cache_id} "
                f"steps={partial_spec.steps} type={type(exc).__name__} err={exc}"
            ) from exc
        if not payload.step_to_latents:
            return None
        return payload

    def _latent_size_bytes(self, cache_id: str, step: int, latent: torch.Tensor) -> int:
        nelement = getattr(latent, "nelement", None)
        element_size = getattr(latent, "element_size", None)
        if not callable(nelement) or not callable(element_size):
            raise TypeError(
                "Latent tensor does not expose size methods "
                f"cache_id={cache_id} step={int(step)} type={type(latent).__name__}"
            )
        return int(nelement()) * int(element_size())

    def _generate_cache_id(self) -> str:
        return uuid.uuid4().hex

    def _normalize_cache_id(self, cache_id: str) -> str:
        return (cache_id or "").replace("-", "")

    def _normalize_search_results(self, results: List[VectorSearchResult]) -> None:
        for r in results:
            r.cache_id = self._normalize_cache_id(r.cache_id)

    def _candidate_text(self, result: VectorSearchResult) -> str:
        text = result.prompt or ""
        if not text and isinstance(result.payload, dict):
            text = result.payload.get("prompt") or ""
        return text


class VideoBasedApproximateCache(BaseCacheStrategy):
    def __init__(
        self,
        config,
        kv_store: KVStore,
        vector_store: Optional[VectorStore],
        metadata_manager: MetadataStore,
        *,
        prompt_encoder: Optional[PromptEncoder] = None,
        video_encoder: Optional["VideoEncoder"] = None,
        reranker: Optional[object] = None,
    ):
        super().__init__(config, kv_store, metadata_manager)
        self.vector_store = vector_store

        # Encoders / reranker are injected (built by ConnectionManager from
        # config). The strategy depends only on the PromptEncoder / VideoEncoder
        # protocols — it never imports or constructs a concrete backend.
        # lookup / save paths already degrade gracefully when any of these is None.
        self.prompt_encoder = prompt_encoder
        self.video_encoder = video_encoder
        self.reranker = reranker

    async def lookup(
        self, query: CacheQuery, ctx: Any = None
    ) -> LookupResult:
        prompt = query.prompt or ""
        task_type = query.task_type or "t2v"
        logger.debug(f"VideoBasedApproximateCache.lookup start task_type={task_type} prompt_len={len(prompt)}")
        if not prompt:
            logger.debug("VideoBasedApproximateCache.lookup miss: empty prompt")
            return LookupResult.miss()
        if self.vector_store is None:
            logger.debug("VideoBasedApproximateCache.lookup miss: vector_store unavailable")
            return LookupResult.miss()

        if self.prompt_encoder is None:
            logger.warning("VideoBasedApproximateCache.lookup miss: prompt encoder unavailable")
            return LookupResult.miss()

        query_vec = self.prompt_encoder.encode(prompt)
        if not query_vec:
            logger.debug("VideoBasedApproximateCache.lookup miss: prompt embedding unavailable")
            return LookupResult.miss()

        hit_score = None
        rerank_enabled = bool(getattr(self.config, "rerank_enabled", False))
        top_k = int(getattr(self.config, "rerank_top_k", 1) or 1) if rerank_enabled else 1
        results = self._vector_search(query_vec, top_k=top_k)
        # Normalize once, before either branch reads cache_id — otherwise the
        # no-rerank branch returns a hyphenated id that fails KV lookups.
        self._normalize_search_results(results)
        if not results:
            logger.debug("VideoBasedApproximateCache.lookup miss: no vector result")
            return LookupResult.miss()

        if rerank_enabled:
            try:
                self.metadata_manager.record_similarity_scores(
                    request_prompt=prompt,
                    task_type=task_type,
                    cache_type="video_approximate_cache",
                    stage="vector_search",
                    candidates=[
                        {
                            "cache_id": item.cache_id,
                            "similarity": float(item.similarity),
                            "prompt": item.prompt,
                            "saved_steps": item.saved_steps,
                        }
                        for item in results
                    ],
                )
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache record_similarity_scores failed stage=vector_search err_type={} err={}",
                    type(exc).__name__,
                    exc,
                )
            scores = self._rerank_scores(prompt, results, "VideoBasedApproximateCache")
            if scores is None:
                logger.debug("VideoBasedApproximateCache.lookup rerank skip: fallback to vector similarity")
                result = results[0]
                threshold = getattr(self.config, "video_similarity_threshold", 0.10)
                if result.similarity < threshold:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: similarity below threshold "
                        f"sim={result.similarity:.4f} threshold={threshold:.4f}"
                    )
                    return LookupResult.miss()
                hit_score = result.similarity
                skip_step = self._determine_skip_step(result.saved_steps)
                if skip_step <= 0:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                        f"sim={result.similarity:.4f} saved_steps={result.saved_steps}"
                    )
                    return LookupResult.miss()
            else:
                if len(scores) != len(results):
                    logger.warning(
                        "VideoBasedApproximateCache.lookup rerank invalid scores size={}",
                        len(scores or []),
                    )
                    return LookupResult.miss()
                try:
                    self.metadata_manager.record_similarity_scores(
                        request_prompt=prompt,
                        task_type=task_type,
                        cache_type="video_approximate_cache",
                        stage="rerank",
                        candidates=[
                            {
                                "cache_id": item.cache_id,
                                "similarity": float(item.similarity),
                                "rerank_score": float(scores[idx]),
                                "prompt": item.prompt,
                                "saved_steps": item.saved_steps,
                            }
                            for idx, item in enumerate(results)
                        ],
                    )
                except Exception as exc:
                    logger.exception(
                        "VideoBasedApproximateCache record_similarity_scores failed stage=rerank err_type={} err={}",
                        type(exc).__name__,
                        exc,
                    )
                best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
                rerank_score = float(scores[best_idx])
                result = results[best_idx]
                logger.debug(
                    "VideoBasedApproximateCache.lookup rerank select cache_id={} score={:.4f} sim={:.4f}",
                    result.cache_id,
                    rerank_score,
                    result.similarity,
                )
                rerank_threshold = float(getattr(self.config, "rerank_score_threshold", 0.80) or 0.80)
                if rerank_score < rerank_threshold:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: rerank score below threshold "
                        f"score={rerank_score:.4f} threshold={rerank_threshold:.4f}"
                    )
                    return LookupResult.miss()
                hit_score = rerank_score
                # Staircase: deeper skip is gated on the rerank score's tier.
                # Only this branch has a real rerank score — the fallback /
                # no-rerank paths use vector similarity and keep the legacy
                # max_skip_step-only logic.
                skip_step = self._determine_skip_step(
                    result.saved_steps, rerank_score=rerank_score
                )
                if skip_step <= 0:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                        f"score={rerank_score:.4f} saved_steps={result.saved_steps}"
                    )
                    return LookupResult.miss()
        else:
            try:
                self.metadata_manager.record_similarity_scores(
                    request_prompt=prompt,
                    task_type=task_type,
                    cache_type="video_approximate_cache",
                    stage="vector_search",
                    candidates=[
                        {
                            "cache_id": item.cache_id,
                            "similarity": float(item.similarity),
                            "prompt": item.prompt,
                            "saved_steps": item.saved_steps,
                        }
                        for item in results
                    ],
                )
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache record_similarity_scores failed stage=vector_search err_type={} err={}",
                    type(exc).__name__,
                    exc,
                )
            result = results[0]

            threshold = getattr(self.config, "video_similarity_threshold", 0.10)
            if result.similarity < threshold:
                logger.debug(
                    "VideoBasedApproximateCache.lookup miss: similarity below threshold "
                    f"sim={result.similarity:.4f} threshold={threshold:.4f}"
                )
                return LookupResult.miss()

            hit_score = result.similarity
            skip_step = self._determine_skip_step(result.saved_steps)
            if skip_step <= 0:
                logger.debug(
                    "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                    f"sim={result.similarity:.4f} saved_steps={result.saved_steps}"
                )
                return LookupResult.miss()

        payload = self._load_payload(
            result.cache_id, partial_spec=VideoApproxPartialSpec.single(skip_step)
        )
        if payload is None:
            meta = None
            try:
                meta = self.metadata_manager.get_cache_meta(result.cache_id)
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache lookup meta check failed cache_id={} err_type={} err={}",
                    result.cache_id,
                    type(exc).__name__,
                    exc,
                )
            meta_hint = ""
            if meta:
                meta_hint = (
                    f" meta_prompt={meta.get('prompt')} "
                    f"meta_steps={meta.get('saved_steps')} "
                    f"meta_type={meta.get('cache_type')}"
                )
            logger.warning(
                "VideoBasedApproximateCache.lookup miss: hit by threshold but KV missing "
                f"cache_id={result.cache_id} step={skip_step} sim={result.similarity:.4f} "
                f"meta_exists={bool(meta)}{meta_hint}"
            )
            # Lazy eviction — vector hit + KV miss = stale vector entry
            # (typical cause: fluxon DRAM pool flush after process restart while
            # FAISS file index persisted). Self-heal so the same cache_id can't
            # be matched on subsequent lookups.
            self._audit_emit("kv_missing_after_vector_hit", {
                "cache_id": result.cache_id,
                "skip_step": int(skip_step),
                "similarity": float(result.similarity),
                "rerank_score": float(hit_score) if hit_score is not None else 0.0,
            })
            self._lazy_evict_stale_entry(
                result.cache_id, reason="kv_missing_after_vector_hit"
            )
            return LookupResult.miss()
        self.metadata_manager.record_access(result.cache_id)
        try:
            self.metadata_manager.record_hit_pair(
                request_prompt=prompt,
                cache_id=result.cache_id,
                cached_prompt=result.prompt,
                similarity=float(hit_score if hit_score is not None else result.similarity),
                task_type=task_type,
                cache_type="video_approximate_cache",
                skip_step=skip_step,
            )
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache record_hit_pair failed cache_id={} err_type={} err={}",
                result.cache_id,
                type(exc).__name__,
                exc,
            )
        logger.debug(
            "VideoBasedApproximateCache.lookup hit "
            f"cache_id={result.cache_id} step={skip_step} sim={result.similarity:.4f}"
        )
        self._audit_emit("lookup_hit", {
            "cache_id": result.cache_id,
            "task_type": task_type,
            "skip_step": int(skip_step),
            "similarity": float(result.similarity),
            "rerank_score": float(hit_score) if hit_score is not None else float(result.similarity),
        })
        return LookupResult.hit_skip_step(
            payload=payload,
            k=int(skip_step),
            cache_id=result.cache_id,
            score=float(hit_score) if hit_score is not None else float(result.similarity),
            similarity=float(result.similarity),
            cached_prompt=result.prompt or "",
        )

    async def save(
        self,
        query: CacheQuery,
        outputs: ModelOutputs,
        ctx: Any = None,
    ) -> None:
        prompt = query.prompt or ""
        task_type = query.task_type or "t2v"
        latent_states_dict = outputs.latent_states_dict
        num_frames = int(outputs.num_frames or 0)
        saved_steps = list(outputs.saved_steps or [])
        embedding_video_frames = outputs.embedding_video_frames
        logger.debug(
            "VideoBasedApproximateCache.save start "
            f"task_type={task_type} prompt_len={len(prompt)} saved_steps={saved_steps}"
        )
        if not prompt:
            logger.debug("VideoBasedApproximateCache.save skip: empty prompt")
            return
        if not latent_states_dict or not saved_steps:
            logger.debug("VideoBasedApproximateCache.save skip: no latent_states or saved_steps")
            return

        cache_id = self._generate_cache_id()
        requested_steps = sorted(set(int(s) for s in saved_steps))
        # Filter out None latents — pipeline may report a step in
        # ``saved_steps`` even if the snapshot didn't actually land.
        valid_latents: Dict[int, torch.Tensor] = {
            step: latent_states_dict[step]
            for step in requested_steps
            if latent_states_dict.get(step) is not None
        }
        if not valid_latents:
            logger.debug("VideoBasedApproximateCache.save skip: no latent saved")
            return

        payload_to_save = VideoApproxPayload(
            cache_id=cache_id,
            step_to_latents=dict(valid_latents),
        )

        saved_steps: List[int] = []
        collection = getattr(self.config, "video_vector_collection", "video")
        vector_written = False
        metadata_attempted = False

        try:
            for kv_key, kv_bytes in payload_to_save.to_kv_entries():
                self.kv_store.put(kv_key, kv_bytes)
                # Recover step number from the well-known key format —
                # to_kv_entries iterates step_to_latents in insertion order,
                # so we could also enumerate, but parsing the key is
                # robust to future reorderings.
                step_part = kv_key.rsplit("_step", 1)[-1]
                saved_steps.append(int(step_part))
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache.save latent persistence failed cache_id={} err={}",
                cache_id,
                exc,
            )
            if saved_steps:
                try:
                    self._remove_saved_latents(cache_id, saved_steps)
                except Exception as cleanup_exc:
                    raise RuntimeError(
                        "VideoBasedApproximateCache.save failed during latent persistence "
                        f"cache_id={cache_id} err={exc}; cleanup_err={cleanup_exc}"
                    ) from exc
            raise RuntimeError(
                f"VideoBasedApproximateCache.save failed during latent persistence cache_id={cache_id} err={exc}"
            ) from exc

        if not saved_steps:
            logger.debug("VideoBasedApproximateCache.save skip: no latent saved")
            return

        # In-memory tensor footprint for size accounting — serialized
        # bytes would require eager serialization, which we avoid.
        total_bytes = payload_to_save.estimated_size_bytes
        size_mb = float(total_bytes) / (1024 * 1024) if total_bytes > 0 else 0.0
        logger.debug(
            "VideoBasedApproximateCache.save stored "
            f"cache_id={cache_id} steps={saved_steps} size_mb={size_mb:.4f} frames={num_frames}"
        )
        self._audit_emit("save_stored", {
            "cache_id": cache_id,
            "task_type": task_type,
            "saved_steps": saved_steps,
            "size_mb": size_mb,
            "num_frames": int(num_frames),
        })

        if self.vector_store is None:
            logger.warning("VideoBasedApproximateCache.save skip: vector_store unavailable")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        if not embedding_video_frames:
            logger.debug("VideoBasedApproximateCache.save skip: no video frames provided")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        if self.video_encoder is None:
            logger.warning("VideoBasedApproximateCache.save skip: video encoder unavailable")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        try:
            frames = self._load_frames_for_embedding(
                embedding_video_frames=embedding_video_frames,
            )
            if not frames:
                logger.debug("VideoBasedApproximateCache.save skip: sampled frames empty")
                self._remove_saved_latents(cache_id, saved_steps)
                return
            logger.debug(
                "VideoBasedApproximateCache.save frames decoded "
                f"count={len(frames)} size={getattr(frames[0], 'size', None)}"
            )
            video_vec = self.video_encoder.encode_video(frames, prompt=prompt)
            if not video_vec:
                logger.debug("VideoBasedApproximateCache.save skip: video embedding unavailable")
                self._remove_saved_latents(cache_id, saved_steps)
                return

            vector_dim = len(video_vec)
            self.vector_store.ensure_collection(collection, vector_dim)
            logger.debug(f"VideoBasedApproximateCache.save ensure collection={collection} dim={vector_dim}")

            payload = {
                "prompt": prompt,
                "saved_steps": saved_steps,
                "task_type": task_type,
            }
            self.vector_store.upsert(
                collection,
                cache_id,
                video_vec,
                payload,
            )
            vector_written = True
            metadata_attempted = True
            self.metadata_manager.register_cache(
                cache_id,
                prompt,
                saved_steps,
                size_mb,
                num_frames,
                cache_type="video_approximate_cache",
            )
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache.save failed cache_id={} collection={} err={}",
                cache_id,
                collection,
                exc,
            )
            try:
                self._rollback_cache_entry(
                    cache_id=cache_id,
                    saved_steps=saved_steps,
                    collection=collection,
                    remove_vector=vector_written,
                    remove_metadata=metadata_attempted,
                )
            except Exception as rollback_exc:
                raise RuntimeError(
                    "VideoBasedApproximateCache.save failed "
                    f"cache_id={cache_id} collection={collection} err={exc}; "
                    f"rollback_err={rollback_exc}"
                ) from exc
            raise RuntimeError(
                f"VideoBasedApproximateCache.save failed cache_id={cache_id} collection={collection} err={exc}"
            ) from exc
        logger.debug(f"VideoBasedApproximateCache.vector_store upsert collection={collection} cache_id={cache_id}")

    def _lazy_evict_stale_entry(self, cache_id: str, reason: str) -> None:
        """Self-heal vector / metadata when KV is inconsistent.

        Triggered when ``lookup`` matches a candidate via vector + rerank
        but ``KVStore.get`` returns ``None`` for the latent — the vector
        store carries a stale entry. Removing the vector + metadata
        records prevents the same ``cache_id`` being matched on
        subsequent requests until a fresh ``save`` writes new latents.

        Errors are logged but never re-raised — eviction is best-effort
        and must not crash the lookup path.
        """
        collection = getattr(self.config, "video_vector_collection", "video")
        if self.vector_store is not None:
            try:
                self.vector_store.delete(collection, [cache_id])
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache lazy eviction vector.delete failed "
                    "collection={} cache_id={} reason={} err={}",
                    collection,
                    cache_id,
                    reason,
                    exc,
                )
        try:
            self.metadata_manager.remove_cache(cache_id)
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache lazy eviction metadata.remove failed "
                "cache_id={} reason={} err={}",
                cache_id,
                reason,
                exc,
            )
        logger.info(
            "VideoBasedApproximateCache lazy-evicted stale entry "
            "cache_id={} reason={}",
            cache_id,
            reason,
        )

    def _remove_saved_latents(self, cache_id: str, saved_steps: List[int]) -> None:
        errors: List[str] = []
        for step in saved_steps:
            try:
                self.kv_store.remove(f"{cache_id}_step{int(step)}")
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache latent cleanup failed cache_id={} step={} err={}",
                    cache_id,
                    int(step),
                    exc,
                )
                errors.append(
                    f"kv remove failed cache_id={cache_id} step={int(step)} type={type(exc).__name__} err={exc}"
                )
        if errors:
            raise RuntimeError("VideoBasedApproximateCache latent cleanup failed: " + "; ".join(errors))

    def _rollback_cache_entry(
        self,
        cache_id: str,
        saved_steps: List[int],
        collection: str,
        remove_vector: bool,
        remove_metadata: bool,
    ) -> None:
        errors: List[str] = []
        if remove_vector and self.vector_store is not None:
            try:
                self.vector_store.delete(collection, [cache_id])
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache vector rollback failed collection={} cache_id={} err={}",
                    collection,
                    cache_id,
                    exc,
                )
                errors.append(
                    "vector rollback failed "
                    f"collection={collection} cache_id={cache_id} "
                    f"type={type(exc).__name__} err={exc}"
                )
        if remove_metadata:
            try:
                self.metadata_manager.remove_cache(cache_id)
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache metadata rollback failed cache_id={} err={}",
                    cache_id,
                    exc,
                )
                errors.append(f"metadata rollback failed cache_id={cache_id} type={type(exc).__name__} err={exc}")
        try:
            self._remove_saved_latents(cache_id, saved_steps)
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            raise RuntimeError(f"VideoBasedApproximateCache rollback failed cache_id={cache_id}: {'; '.join(errors)}")

    def _vector_search(self, query_vec: List[float], top_k: int = 1) -> List[VectorSearchResult]:
        if self.vector_store is None:
            return []
        collection = getattr(self.config, "video_vector_collection", "video")
        top_k = max(1, int(top_k or 1))
        res = self.vector_store.search(collection, query_vec, limit=top_k)
        if not res:
            return []
        res.sort(key=lambda item: item.similarity, reverse=True)
        return res[:top_k]

    def _load_frames_for_embedding(
        self,
        *,
        embedding_video_frames: Optional[List[Any]],
    ) -> List[Any]:
        if embedding_video_frames:
            return list(embedding_video_frames)
        return []

    def _sample_indices(self, total: int, max_frames: int) -> List[int]:
        if total <= 0:
            return []
        max_frames = max(1, int(max_frames or 1))
        if total <= max_frames:
            return list(range(total))
        step = float(total) / float(max_frames)
        return [min(int(i * step), total - 1) for i in range(max_frames)]

    def _select_k_by_score(self, rerank_score: float) -> Optional[int]:
        """Staircase upper bound on skip-step K given a rerank score.

        Implements the online rule ``K*(s) = max{K : τ_K ≤ s}``: each tier K
        carries a minimum rerank threshold τ_K calibrated to keep donor drift
        bounded, so higher-scoring donors are allowed to skip more steps.
        Returns the largest qualifying K, or ``None`` when the table is
        empty / disabled / no tier qualifies (caller falls back to the legacy
        ``max(saved_steps ≤ max_skip_step)`` logic; a score below every τ_K
        means "too risky to reuse this deep").
        """
        if not bool(getattr(self.config, "staircase_skip_enabled", True)):
            return None
        tau_table = getattr(self.config, "skip_step_tau_table", None)
        if not tau_table:
            return None
        qualifying = [
            int(k) for k, tau in tau_table.items() if rerank_score >= float(tau)
        ]
        return max(qualifying) if qualifying else None

    def _determine_skip_step(
        self, saved_steps: List[int], rerank_score: Optional[float] = None
    ) -> int:
        """Resolve *which snapshotted step to resume from* for a hit.

        Two regimes (caller has already gated on the relevant similarity /
        rerank threshold):

        - **Staircase** (``rerank_score`` supplied + ``staircase_skip_enabled``
          + a τ_K table): cap K by the score-derived tier
          ``K*(score) = max{K : τ_K ≤ score}`` (see ``_select_k_by_score``).
          High rerank -> deeper skip, low -> shallow.
        - **Legacy** (no score / staircase off / empty table): cap K only by
          ``max_skip_step``.

        In both regimes the result is snapped to the largest *actually
        snapshotted* step in ``(0, cap]`` — skip can only land on a step the
        donor saved. Returns 0 when ``saved_steps`` is empty or no element
        falls in range — caller treats 0 as "miss".
        """
        if not saved_steps:
            return 0
        cap = int(getattr(self.config, "max_skip_step", 5) or 5)
        if rerank_score is not None:
            tier_k = self._select_k_by_score(float(rerank_score))
            if tier_k is not None:
                # Staircase tier and max_skip_step are both upper bounds.
                cap = min(cap, tier_k)
        valid = [int(s) for s in saved_steps if 0 < int(s) <= cap]
        return max(valid) if valid else 0

    def _build_rerank_documents(
        self,
        results: List[VectorSearchResult],
    ) -> List[Dict[str, object]]:
        documents: List[Dict[str, object]] = []
        for item in results:
            text = self._candidate_text(item)
            doc: Dict[str, object] = {}
            if text:
                doc["text"] = text
            documents.append(doc)
        return documents

    def _rerank_scores(
        self,
        query: str,
        results: List[VectorSearchResult],
        source: str,
    ) -> Optional[List[float]]:
        reranker = getattr(self, "reranker", None)
        if reranker is None:
            logger.warning(f"{source} rerank skip: reranker unavailable")
            return None
        if not hasattr(reranker, "score_mm"):
            logger.warning(f"{source} rerank skip: text reranker unavailable")
            return None
        documents = self._build_rerank_documents(results)
        has_text_docs = any("text" in doc and doc["text"] for doc in documents)
        if not has_text_docs:
            logger.debug(f"{source} rerank skip: no text candidates available")
            return None

        try:
            logger.debug(f"{source} rerank mode=text candidates={len(results)}")
            scores = reranker.score_mm({"text": query}, documents)
        except Exception as exc:
            logger.exception(f"{source} text rerank failed: {exc}")
            raise RuntimeError(f"{source} text rerank failed err_type={type(exc).__name__} err={exc}") from exc
        if not scores or len(scores) != len(results):
            raise ValueError(f"{source} rerank invalid scores size={len(scores or [])} expected={len(results)}")
        score_pairs = []
        for idx, item in enumerate(results):
            try:
                score_value = float(scores[idx])
                score_pairs.append(f"{item.cache_id}:{score_value:.4f}/{item.similarity:.4f}")
            except (IndexError, TypeError, ValueError) as exc:
                logger.exception(
                    "{} rerank score formatting failed cache_id={} idx={} err_type={} err={}",
                    source,
                    item.cache_id,
                    idx,
                    type(exc).__name__,
                    exc,
                )
                raise RuntimeError(
                    f"{source} rerank score formatting failed "
                    f"cache_id={item.cache_id} idx={idx} "
                    f"err_type={type(exc).__name__} err={exc}"
                ) from exc
        logger.debug(f"{source} rerank scores={score_pairs}")
        return [float(value) for value in scores]


_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(name: str, cls: type) -> None:
    _STRATEGY_REGISTRY[name] = cls


def get_strategy_class(name: str) -> Optional[type]:
    return _STRATEGY_REGISTRY.get(name)


register_strategy("video_approximate", VideoBasedApproximateCache)
