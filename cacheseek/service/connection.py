from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from cacheseek.stores.base import KVStore
from cacheseek.service.interfaces.vector_store import VectorStore
from cacheseek.service.interfaces.encoder import PromptEncoder, VideoEncoder


class ConnectionManager:
    def __init__(
        self,
        config: Any,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._lock = threading.Lock()

        self._vector_store: Optional[VectorStore] = None
        self._vector_store_created = False
        self._kv_store: Optional[KVStore] = None
        self._kv_store_created = False

        self._encoders_created = False
        self._prompt_encoder: Optional[PromptEncoder] = None
        self._video_encoder: Optional[VideoEncoder] = None
        self._reranker: Optional[object] = None

    @property
    def vector_store(self) -> Optional[VectorStore]:
        """Lazily create the VectorStore connection (Qdrant / FAISS)."""
        if not self._vector_store_created:
            with self._lock:
                if not self._vector_store_created:
                    self._vector_store = self._create_vector_store()
                    self._vector_store_created = True
        return self._vector_store

    @property
    def kv_store(self) -> Optional[KVStore]:
        """Lazily create the KVStore connection (Fluxon / LocalFile)."""
        if not self._kv_store_created:
            with self._lock:
                if not self._kv_store_created:
                    self._kv_store = self._create_kv_store()
                    self._kv_store_created = True
        return self._kv_store

    @property
    def prompt_encoder(self) -> Optional[PromptEncoder]:
        """Lazily create the prompt encoder (backend assembled from config)."""
        self._ensure_encoders()
        return self._prompt_encoder

    @property
    def video_encoder(self) -> Optional[VideoEncoder]:
        """Lazily create the video encoder (may share the prompt encoder instance to save GPU memory)."""
        self._ensure_encoders()
        return self._video_encoder

    @property
    def reranker(self) -> Optional[object]:
        """Lazily create the reranker (built when rerank_enabled; degrades gracefully to None if weights/deps are missing)."""
        self._ensure_encoders()
        return self._reranker

    def _ensure_encoders(self) -> None:
        if self._encoders_created:
            return
        with self._lock:
            if self._encoders_created:
                return
            pe, ve, rr = self._create_encoders()
            self._prompt_encoder = pe
            self._video_encoder = ve
            self._reranker = rr
            self._encoders_created = True

    def _create_encoders(self):
        """config -> (prompt_encoder, video_encoder, reranker).

        The concrete backend (Qwen3VL) is imported lazily here. Encoders are a
        third class of swappable backend (alongside KV/vector stores), all
        instantiated by ConnectionManager from config; upper-layer strategies
        depend only on the ``PromptEncoder`` / ``VideoEncoder`` protocols.
        """
        from cacheseek.backends.encoder.qwen3vl import Qwen3VLEncoder, Qwen3VLReranker

        config = self._config
        enable_video_embedding = bool(getattr(config, "video_embedding_enabled", False))
        text_model_path = getattr(config, "text_embedding_model_path", None) or None
        use_text_embedding = bool(text_model_path) or enable_video_embedding

        def _build_prompt_encoder() -> Qwen3VLEncoder:
            model_path = (
                text_model_path
                or getattr(config, "video_embedding_model_path", None)
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            device_id = getattr(config, "text_embedding_device_id", None)
            encoder = Qwen3VLEncoder(
                model_path=model_path,
                instruction=getattr(
                    config, "text_embedding_instruction", "Represent the user's input"
                ),
                max_frames=int(getattr(config, "video_embedding_max_frames", 16)),
                fps=float(getattr(config, "video_embedding_fps", 1.0)),
                device_id=device_id,
                torch_dtype=getattr(config, "text_embedding_torch_dtype", None),
                attn_implementation=getattr(config, "text_embedding_attn_impl", None),
            )
            logger.info(
                "ConnectionManager prompt encoder enabled model_path={} device_id={}",
                model_path,
                device_id,
            )
            return encoder

        def _build_video_encoder() -> Qwen3VLEncoder:
            model_path = (
                getattr(config, "video_embedding_model_path", None)
                or text_model_path
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            device_id = getattr(config, "video_embedding_device_id", None)
            encoder = Qwen3VLEncoder(
                model_path=model_path,
                instruction=getattr(
                    config, "video_embedding_instruction", "Represent the user's input"
                ),
                max_frames=int(getattr(config, "video_embedding_max_frames", 16)),
                fps=float(getattr(config, "video_embedding_fps", 1.0)),
                device_id=device_id,
                torch_dtype=getattr(config, "video_embedding_torch_dtype", None),
                attn_implementation=getattr(config, "video_embedding_attn_impl", None),
            )
            logger.info(
                "ConnectionManager video encoder enabled model_path={} device_id={}",
                model_path,
                device_id,
            )
            return encoder

        prompt_encoder: Optional[PromptEncoder] = None
        video_encoder: Optional[VideoEncoder] = None

        if use_text_embedding:
            prompt_encoder = _build_prompt_encoder()
        if enable_video_embedding:
            # Reuse prompt_encoder when text/video configs target the same
            # model+device — save ~5GB GPU mem and one cold load.
            video_model_path = (
                getattr(config, "video_embedding_model_path", None)
                or getattr(config, "text_embedding_model_path", None)
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            video_device_id = getattr(config, "video_embedding_device_id", None)
            if (
                prompt_encoder is not None
                and getattr(prompt_encoder, "model_path", None) == video_model_path
                and getattr(prompt_encoder, "device_id", None) == video_device_id
            ):
                video_encoder = prompt_encoder
                logger.info(
                    "ConnectionManager video_encoder shares prompt_encoder instance "
                    "(same model_path={} device_id={}, save ~5GB)",
                    video_model_path,
                    video_device_id,
                )
            else:
                video_encoder = _build_video_encoder()

        reranker: Optional[object] = None
        if getattr(config, "rerank_enabled", False):
            # rerank is on by default (false-hit gate), but degrade gracefully
            # to vector-similarity-only when the reranker is unavailable —
            # missing weights/deps must not fail the whole cache assembly;
            # the lookup path already handles scores=None.
            try:
                reranker = Qwen3VLReranker(
                    model_path=getattr(config, "rerank_model_path", None)
                    or "Qwen/Qwen3-VL-Reranker-2B",
                    device_id=getattr(config, "rerank_device_id", None),
                    batch_size=int(getattr(config, "rerank_batch_size", 2) or 2),
                    torch_dtype=getattr(config, "rerank_torch_dtype", None),
                )
            except Exception as exc:
                logger.warning(
                    "ConnectionManager reranker unavailable ({}: {}); "
                    "falling back to vector-similarity only (rerank effectively off)",
                    type(exc).__name__,
                    exc,
                )
                reranker = None
            backend_reranker = getattr(reranker, "_reranker", None)
            actual_reranker_device = getattr(getattr(backend_reranker, "model", None), "device", None)
            if actual_reranker_device is None:
                actual_reranker_device = getattr(backend_reranker, "device", "unknown")
            logger.debug(
                "ConnectionManager reranker enabled model_path={} device_id={} actual_device={}",
                getattr(config, "rerank_model_path", ""),
                getattr(config, "rerank_device_id", None),
                actual_reranker_device,
            )

        return prompt_encoder, video_encoder, reranker

    def health_check(self) -> dict:
        result: dict = {}

        vs = self._vector_store
        if vs is None and not self._vector_store_created:
            result["vector_store"] = {"status": "not_initialized"}
        elif vs is None:
            result["vector_store"] = {"status": "disabled"}
        else:
            vs_status: dict = {
                "status": "connected",
                "type": type(vs).__name__,
            }
            if hasattr(vs, "client"):
                try:
                    vs.client.get_collections()
                    vs_status["reachable"] = True
                except Exception as exc:
                    logger.exception(
                        "ConnectionManager.health_check vector_store reachability failed: {}",
                        exc,
                    )
                    vs_status["reachable"] = False
                    vs_status["error"] = str(exc)
            result["vector_store"] = vs_status

        kvs = self._kv_store
        if kvs is None and not self._kv_store_created:
            result["kv_store"] = {"status": "not_initialized"}
        elif kvs is None:
            result["kv_store"] = {"status": "disabled"}
        else:
            result["kv_store"] = {
                "status": "connected",
                "type": type(kvs).__name__,
            }

        return result

    def shutdown(self) -> None:
        with self._lock:
            for name, store in [
                ("vector_store", self._vector_store),
                ("kv_store", self._kv_store),
            ]:
                if store is None:
                    continue
                for method_name in ("shutdown", "close"):
                    if hasattr(store, method_name):
                        try:
                            getattr(store, method_name)()
                        except Exception as exc:
                            logger.exception(
                                "ConnectionManager.{}.{} failed: {}",
                                name,
                                method_name,
                                exc,
                            )
                        break
            self._vector_store = None
            self._vector_store_created = False
            self._kv_store = None
            self._kv_store_created = False

    def _create_vector_store(self) -> Optional[VectorStore]:
        from cacheseek.backends.vector.qdrant import QdrantVectorStore

        config = self._config
        store_type = (getattr(config, "vector_store_type", "") or "").lower()

        if store_type == "faiss":
            return self._build_faiss_store()

        if store_type == "qdrant":
            qdrant_url = getattr(config, "qdrant_url", None)
            if not qdrant_url:
                logger.debug("Qdrant vector store selected without qdrant_url; using in-memory Qdrant")
            return QdrantVectorStore(
                url=qdrant_url or "",
                api_key=getattr(config, "qdrant_api_key", None),
            )

        if store_type:
            logger.debug(
                "Unknown vector_store_type '{}'; vector store disabled",
                store_type,
            )
        else:
            logger.debug("vector_store_type not set; vector store disabled")
        return None

    def _build_faiss_store(self) -> "VectorStore":
        from cacheseek.backends.vector.faiss import FAISSVectorStore

        config = self._config
        cache_dir = self._storage_dir.parent if self._storage_dir else Path(".")
        index_dir = getattr(config, "faiss_index_dir", None) or str(cache_dir / "faiss")
        vector_dim = int(getattr(config, "vector_dim", 2048))
        # cosine (IndexFlatIP + normalize_L2): Qwen3-VL embeddings are already
        # L2-normalized and thus on the same scale as Qdrant's native cosine,
        # so video_similarity_threshold lines up. (The old "L2" metric, after
        # 1/(1+d) rescaling, gave unrelated prompts ~0.41 and almost always
        # false-hit with rerank off.)
        return FAISSVectorStore(
            Path(index_dir), vector_dim=vector_dim, index_type="cosine"
        )

    def _create_kv_store(self) -> KVStore:
        from cacheseek.stores.fluxon import FluxonKVStore
        from cacheseek.stores.local_file import LocalFileKVStore

        config = self._config
        store_type = (getattr(config, "kv_store_type", "") or "").lower()

        if store_type == "fluxon":
            config_path = getattr(config, "fluxon_config_path", None)
            try:
                return FluxonKVStore(config_path=config_path)
            except Exception as exc:
                logger.exception(
                    "FluxonKVStore init failed config_path={}: {}",
                    config_path,
                    exc,
                )
                raise RuntimeError(
                    f"FluxonKVStore init failed config_path={config_path} err_type={type(exc).__name__} err={exc}"
                ) from exc

        if store_type and store_type not in {"local", "local_file"}:
            logger.debug(
                "Unknown kv_store_type '{}'; falling back to LocalFileKVStore",
                store_type,
            )
        storage_dir = self._storage_dir or Path("./storage")
        return LocalFileKVStore(storage_dir)
