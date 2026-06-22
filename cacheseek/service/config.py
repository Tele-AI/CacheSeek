from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CacheMode(Enum):
    READ_WRITE = "read_write"  # read and write cache (default)
    READ_ONLY = "read_only"  # read cache only
    WRITE_ONLY = "write_only"  # write cache only


@dataclass
class CacheConfig:
    """Cache configuration shared across stages/pipelines."""

    # Basic cache
    enable_latent_cache: bool = False
    cache_mode: CacheMode = CacheMode.READ_WRITE  # read_write | read_only | write_only
    latent_cache_dir: str = "./latent_cache"
    max_cache_size_gb: int = 10
    cache_log_enabled: bool = True
    cache_log_dir: Optional[str] = None  # default: {latent_cache_dir}/logs
    cache_log_level: str = "DEBUG"
    cache_log_rotation: str = "100 MB"
    cache_log_retention: str = "7 days"

    # KV store (for latent and other key-value caches)
    kv_store_type: str = "local_file"  # "local_file" | "fluxon"
    fluxon_config_path: Optional[str] = ""

    # Vector store (for embedding retrieval)
    vector_store_type: str = "faiss"  # "qdrant" | "faiss"
    qdrant_url: Optional[str] = ""
    qdrant_api_key: Optional[str] = None
    faiss_index_dir: Optional[str] = None
    vector_dim: int = 2048  # vector dimension (required by FAISS init; must match the embedding model output dim)
    cache_strategy_type: str = "video_approximate"  # strategy key in STRATEGY_REGISTRY

    # Similarity & lookup strategy
    key_steps: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])  # steps eligible for cache reuse
    max_skip_step: int = 5  # max step actually skipped at lookup; pick the largest saved_step <= this bound
    lookup_mode: str = "video"  # lookup mode, e.g. "video"

    # Staircase skip-step: rerank score bucket decides how many steps to skip.
    # We fit a logistic curve for P(donor_drift | rerank) on Wan2.2-T2V-A14B,
    # then invert it under drift <= 20% to recover, per skip bucket K, the
    # minimum rerank threshold tau_K, yielding the online rule
    # K*(s) = max{K : tau_K <= s}. Higher score allows skipping more steps.
    #   - staircase_skip_enabled=True with a rerank score available: bucket by
    #     this table (still bounded by max_skip_step and limited to steps that
    #     were actually snapshotted in saved_steps).
    #   - Otherwise (rerank off / no score / table off): fall back to the old
    #     rule of "largest saved_step <= max_skip_step".
    # Table is {K: min rerank score}. Default targets the 0.20-SLO bucket
    # (donor-drift <= 20%); see the conservative 0.10 / aggressive 0.30
    # buckets in the experiment notes. K=7 and K=11 tie at 0.85 (the high-rerank
    # bucket has n~30 with fully overlapping Wilson CIs, so the ordering is
    # small-sample noise). K=14 is set to 1.01 (> the observed rerank ceiling
    # 0.926) = disabled: it only injects donor into step12-14, which AdaTaylor
    # already skips, trading quality for zero speedup; the high end lands on K=11.
    staircase_skip_enabled: bool = False
    skip_step_tau_table: Dict[int, float] = field(
        default_factory=lambda: {3: 0.63, 7: 0.85, 11: 0.85, 14: 1.01}
    )

    # Prompt / text embedding model
    text_embedding_model_path: str = ""
    text_embedding_instruction: str = "Represent the user's input"
    text_embedding_device_id: Optional[int] = None
    text_embedding_torch_dtype: Optional[str] = None
    text_embedding_attn_impl: Optional[str] = None

    # Video embedding model
    video_embedding_enabled: bool = True
    video_embedding_model_path: str = "Qwen/Qwen3-VL-Embedding-2B"
    video_embedding_instruction: str = "Represent the user's input"
    video_embedding_fps: float = 1.0
    video_embedding_max_frames: int = 16
    video_embedding_max_length: int = 8192
    video_embedding_min_pixels: int = 4096
    video_embedding_max_pixels: int = 1843200
    video_embedding_total_pixels: int = 7864320
    video_embedding_device_id: Optional[int] = None
    video_embedding_torch_dtype: Optional[str] = None
    video_embedding_attn_impl: Optional[str] = None

    # Video vector search & rerank
    video_similarity_threshold: Optional[float] = 0.10
    video_vector_collection: str = "video"
    # Default on (per real-hardware A/B): with rerank off, unrelated prompts
    # false-hit at sim 0.41 (video_similarity_threshold=0.10 is too loose);
    # the 0.80 threshold correctly rejects them.
    rerank_enabled: bool = True
    rerank_model_path: str = "Qwen/Qwen3-VL-Reranker-2B"
    rerank_top_k: int = 5
    rerank_batch_size: int = 2
    rerank_device_id: Optional[int] = None
    rerank_torch_dtype: Optional[str] = None
    rerank_score_threshold: float = 0.80

    # Async save / write-behind
    save_async_enabled: bool = True
    save_queue_size: int = 2
    save_on_full: str = "drop"  # drop | sync | downgrade
    save_queue_warn_threshold: int = 8
    vector_wait_warn_s: float = 2.0
    vector_wait_poll_s: float = 0.05
    vector_wait_timeout_s: float = 120.0
    flush_on_shutdown: bool = True
