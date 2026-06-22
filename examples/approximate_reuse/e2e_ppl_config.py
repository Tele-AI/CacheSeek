# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Example CACHE_CONFIG for the approximate e2e (paths injected via env vars for cross-machine reuse).

Required env vars:
    QWEN_EMBED_PATH    Qwen3-VL-Embedding-2B weights directory
Optional:
    QWEN_RERANK_PATH   Qwen3-VL-Reranker-2B weights directory (default = guessed as a sibling
                       of the embedding directory)
    APPROX_KV_STORE    "local_file" (default, no dependencies) | "fluxon"
    FLUXON_CONFIG      client external_config.yaml when kv=fluxon
    APPROX_E2E_DIR     working directory (faiss index / latents / logs), default /tmp/approx_e2e
"""
import os

_WORK = os.environ.get("APPROX_E2E_DIR", "/tmp/approx_e2e")

CACHE_CONFIG = dict(
    enable_latent_cache=True,
    cache_mode="read_write",
    latent_cache_dir=f"{_WORK}/latents",
    kv_store_type=os.environ.get("APPROX_KV_STORE", "local_file"),
    fluxon_config_path=os.environ.get("FLUXON_CONFIG", ""),
    vector_store_type="faiss",
    faiss_index_dir=f"{_WORK}/faiss",
    vector_dim=2048,
    text_embedding_model_path=os.environ["QWEN_EMBED_PATH"],
    video_embedding_model_path=os.environ["QWEN_EMBED_PATH"],
    # rerank on by default (0.80): real-model A/B shows it is an effective gate against false
    # hits -- with rerank off an unrelated prompt falsely hits at sim 0.41 (threshold 0.10);
    # with it on the prompt is correctly rejected.
    rerank_enabled=True,
    rerank_model_path=os.environ.get("QWEN_RERANK_PATH", os.environ["QWEN_EMBED_PATH"].replace("Embedding", "Reranker")),
    rerank_score_threshold=0.80,
    key_steps=[5, 10],
    max_skip_step=5,
)
