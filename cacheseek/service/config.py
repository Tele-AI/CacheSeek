# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Cache-service configuration: the CacheConfig dataclass and CacheMode enum."""

from dataclasses import dataclass, field
from enum import Enum


class CacheMode(Enum):
    """How `CacheService` is allowed to touch the cache."""

    READ_WRITE = "read_write"  # read and write cache (default)
    READ_ONLY = "read_only"  # read cache only
    WRITE_ONLY = "write_only"  # write cache only


@dataclass
class CacheConfig:
    """Cache configuration shared across stages/pipelines.

    Fields are documented with PEP-257 attribute docstrings (the triple-quoted
    string after each field) so the documentation travels with the default. YAML
    keys map 1:1 onto these fields; omitted keys take the defaults below.
    """

    # Basic cache
    enable_latent_cache: bool = False
    """Master switch for the cross-request latent cache. When False, lookup always
    misses and save is a no-op."""
    cache_mode: CacheMode = CacheMode.READ_WRITE
    """Cache access mode: read+write (default), read-only, or write-only."""
    latent_cache_dir: str = "./latent_cache"
    """Root directory for cached latents, metadata, and logs."""
    max_cache_size_gb: int = 10
    """Soft cap on total cache size, in GB (advisory; active eviction is not yet
    wired into the save path)."""
    cache_log_enabled: bool = True
    """Whether to write the cache-service log."""
    cache_log_dir: str | None = None
    """Directory for cache logs. Defaults to ``{latent_cache_dir}/logs``."""
    cache_log_level: str = "DEBUG"
    """Loguru level for the cache-service log."""
    cache_log_rotation: str = "100 MB"
    """Loguru rotation policy for the cache log (e.g. ``"100 MB"``)."""
    cache_log_retention: str = "7 days"
    """Loguru retention policy for rotated cache logs."""

    # KV store (for latent and other key-value caches)
    kv_store_type: str = "local_file"
    """KV/tensor store backend: ``"local_file"`` (single machine) or ``"fluxon"``
    (distributed)."""
    fluxon_config_path: str | None = ""
    """Path to the Fluxon client config; required when ``kv_store_type="fluxon"``."""

    # Vector store (for embedding retrieval)
    vector_store_type: str = "faiss"
    """Vector store backend: ``"faiss"`` (embedded) or ``"qdrant"`` (service)."""
    qdrant_url: str | None = ""
    """Qdrant server URL; required when ``vector_store_type="qdrant"``."""
    qdrant_api_key: str | None = None
    """Optional API key for an authenticated Qdrant server."""
    faiss_index_dir: str | None = None
    """Directory for the FAISS index. Defaults to ``{latent_cache_dir}/faiss``."""
    vector_dim: int = 2048
    """Embedding vector dimension. Must match the embedding model's output dim
    (FAISS needs it at index-init time)."""
    cache_strategy_type: str = "video_approximate"
    """Strategy key resolved against the strategy registry."""

    # Similarity & lookup strategy
    key_steps: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    """Denoise steps to checkpoint — the steps eligible for cross-request reuse."""
    max_skip_step: int = 5
    """Upper bound on how many denoise steps a hit may skip. At lookup, the largest
    checkpointed step ``<= max_skip_step`` is chosen."""
    lookup_mode: str = "video"
    """Retrieval mode, e.g. ``"video"``."""

    staircase_skip_enabled: bool = False
    """Tier the skip depth by the donor's rerank score instead of a flat
    ``max_skip_step``.

    We fit a logistic curve for P(donor_drift | rerank) on Wan2.2-T2V-A14B, then
    invert it under drift <= 20% to recover, per skip bucket K, the minimum rerank
    threshold tau_K, giving the online rule ``K*(s) = max{K : tau_K <= s}`` (higher
    score allows skipping more steps).

    - When True and a rerank score is available: bucket by ``skip_step_tau_table``
      (still bounded by ``max_skip_step`` and limited to steps actually in
      ``saved_steps``).
    - Otherwise (rerank off / no score / table off): fall back to the old rule of
      "largest ``saved_step <= max_skip_step``".
    """
    skip_step_tau_table: dict[int, float] = field(
        default_factory=lambda: {3: 0.63, 7: 0.85, 11: 0.85, 14: 1.01}
    )
    """Skip-bucket table ``{K: min rerank score}`` for staircase skipping. Default
    targets the 0.20-SLO bucket (donor-drift <= 20%). K=7 and K=11 tie at 0.85 (the
    high-rerank bucket has n~30 with overlapping Wilson CIs, so the ordering is
    small-sample noise); K=14 is set to 1.01 (> the observed rerank ceiling 0.926),
    i.e. disabled — it only injects the donor into steps AdaTaylor already skips,
    trading quality for zero speedup."""

    # Prompt / text embedding model
    text_embedding_model_path: str = ""
    """Path or HF id of the prompt (text) embedding model. Empty disables text
    embedding."""
    text_embedding_instruction: str = "Represent the user's input"
    """Instruction prefix passed to the text embedding model."""
    text_embedding_device_id: int | None = None
    """CUDA device index for the text encoder; None lets the backend choose."""
    text_embedding_torch_dtype: str | None = None
    """Torch dtype override for the text encoder (e.g. ``"bfloat16"``)."""
    text_embedding_attn_impl: str | None = None
    """Attention implementation override for the text encoder (e.g.
    ``"flash_attention_2"``)."""

    # Video embedding model
    video_embedding_enabled: bool = True
    """Whether to embed sampled video frames. When False the lifecycle runs but
    always misses — useful as an install/wiring check without GPU weights."""
    video_embedding_model_path: str = "Qwen/Qwen3-VL-Embedding-2B"
    """Path or HF id of the Qwen3-VL video embedding model."""
    video_embedding_instruction: str = "Represent the user's input"
    """Instruction prefix passed to the video embedding model."""
    video_embedding_fps: float = 1.0
    """Frame sampling rate (frames per second) for video embedding."""
    video_embedding_max_frames: int = 16
    """Maximum number of frames sampled per video for embedding."""
    video_embedding_max_length: int = 8192
    """Maximum token length for the video embedding model."""
    video_embedding_min_pixels: int = 4096
    """Minimum per-frame pixel budget for the vision processor."""
    video_embedding_max_pixels: int = 1843200
    """Maximum per-frame pixel budget for the vision processor."""
    video_embedding_total_pixels: int = 7864320
    """Total pixel budget across all sampled frames."""
    video_embedding_device_id: int | None = None
    """CUDA device index for the video encoder; None lets the backend choose."""
    video_embedding_torch_dtype: str | None = None
    """Torch dtype override for the video encoder."""
    video_embedding_attn_impl: str | None = None
    """Attention implementation override for the video encoder."""

    # Video vector search & rerank
    video_similarity_threshold: float | None = 0.10
    """Minimum cosine similarity for a vector-search candidate to be kept. None
    disables the floor."""
    video_vector_collection: str = "video"
    """Vector store collection name for video embeddings."""
    rerank_enabled: bool = True
    """Whether to rerank vector-search candidates before deciding a hit.

    Default on, per real-hardware A/B: with rerank off, unrelated prompts false-hit
    at sim 0.41 (``video_similarity_threshold=0.10`` is too loose); the 0.80 rerank
    threshold correctly rejects them.
    """
    rerank_model_path: str = "Qwen/Qwen3-VL-Reranker-2B"
    """Path or HF id of the Qwen3-VL reranker."""
    rerank_top_k: int = 5
    """Number of top vector-search candidates passed to the reranker."""
    rerank_batch_size: int = 2
    """Reranker batch size."""
    rerank_device_id: int | None = None
    """CUDA device index for the reranker; None lets the backend choose."""
    rerank_torch_dtype: str | None = None
    """Torch dtype override for the reranker."""
    rerank_score_threshold: float = 0.80
    """Minimum rerank score for a candidate to count as a hit."""

    # Async save / write-behind
    save_async_enabled: bool = True
    """Run save on a background worker (write-behind) instead of inline."""
    save_queue_size: int = 2
    """Maximum pending async-save jobs before ``save_on_full`` applies."""
    save_on_full: str = "drop"
    """Policy when the save queue is full: ``"drop"``, ``"sync"`` (block), or
    ``"downgrade"``."""
    save_queue_warn_threshold: int = 8
    """Save-queue depth at which a warning is logged."""
    vector_wait_warn_s: float = 2.0
    """Lookup waits for in-flight vector upserts to settle; warn after this many
    seconds."""
    vector_wait_poll_s: float = 0.05
    """Poll interval (seconds) while waiting for vector upserts to settle."""
    vector_wait_timeout_s: float = 120.0
    """Hard timeout (seconds) for the vector-upsert wait barrier."""
    flush_on_shutdown: bool = True
    """Drain the async-save queue on shutdown."""
