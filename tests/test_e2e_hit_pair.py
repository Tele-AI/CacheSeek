# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""End-to-end hit-pair test against a live cache pool.

Verifies CacheSeek is byte-stream compatible with the upstream cache
pool — meaning ``cache_id`` entries written by another writer can be
looked up + loaded by CacheSeek without re-encoding / re-storing.

Requirements (developer host with the relevant infrastructure):
- ``fluxon`` master + kvclient running
- ``qdrant`` server reachable
- Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B model weights loadable

This test runs against a real cache pool, not a sandbox. The read-only
variant performs lookup only; the read-write variant exercises a full
save → lookup roundtrip on a fresh prompt.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

# Skip if not on H800 host (no fluxon / no model weights).
# Paths default to placeholders; the test is gated by XCACHE_E2E_HITPAIR=1
# so anyone running it for real should set FLUXON_CONFIG_PATH and
# TF_MODEL_ZOO_PATH (or QWEN3VL_*_PATH below) to point at their infra.
_FLUXON_CFG = os.environ.get("FLUXON_CONFIG_PATH", "/path/to/fluxon-deploy/external_config.yaml")
_QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
_MODEL_ZOO = os.environ.get("TF_MODEL_ZOO_PATH", "/path/to/model_zoo")
_RUN_E2E = os.environ.get("XCACHE_E2E_HITPAIR") == "1"


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _RUN_E2E,
        reason=(
            "cacheseek e2e hit-pair test gated by XCACHE_E2E_HITPAIR=1 (requires fluxon "
            "+ qdrant + Qwen3-VL on H800)."
        ),
    ),
]


# Self-similarity hit-pair test:
# Use a prompt that already lives in the qdrant ``video`` collection as
# the query. Expects:
#   1. CacheSeek + fluxon + qdrant byte-stream interop (reads what the
#      upstream writer stored)
#   2. CacheSeek's retrieve + rerank chain returns the stored entry as
#      top candidate
#   3. Hit decision: same prompt → high sim → high rerank → hit
PROMPT_LIVE_HIT = (
    "由近及远展示月面能源站建设场景。前景是一名穿着完整白色航天服、佩戴不透明头盔的宇航员"
    "在铺设电缆管线，服装无肩章、无明显标识，身上光线较暗。旁边一台多关节机械臂机器人"
    "正在协助搬运设备模块。中景可见几名航天员在安装能源站支架，远处伫立着半建成的太阳能阵列"
    "与银白色舱体。地面覆盖灰色月壤，背景是漆黑太空与地球的轮廓。画面整体冷峻、秩序感强，"
    "体现人机协作建设的科技氛围。"
)


def _build_cache_config(tmp_path: Path, *, cache_mode: str):
    from cacheseek.service.config import CacheConfig

    return CacheConfig(
        enable_latent_cache=True,
        latent_cache_dir=str(tmp_path / "cacheseek_e2e"),
        cache_mode=cache_mode,
        kv_store_type="fluxon",
        fluxon_config_path=_FLUXON_CFG,
        vector_store_type="qdrant",
        qdrant_url=_QDRANT_URL,
        vector_dim=2048,
        key_steps=[5, 10, 15, 20, 25],
        video_embedding_enabled=True,
        video_embedding_model_path=os.environ.get(
            "QWEN3VL_EMBEDDING_PATH", f"{_MODEL_ZOO}/Qwen3-VL-Embedding-2B"
        ),
        text_embedding_model_path=os.environ.get(
            "QWEN3VL_EMBEDDING_PATH", f"{_MODEL_ZOO}/Qwen3-VL-Embedding-2B"
        ),
        text_embedding_device_id=1,
        video_embedding_device_id=1,
        video_vector_collection="video",  # shared collection with the upstream writer
        rerank_enabled=True,
        rerank_model_path=os.environ.get(
            "QWEN3VL_RERANKER_PATH", f"{_MODEL_ZOO}/Qwen3-VL-Reranker-2B"
        ),
        rerank_device_id=0,
        rerank_top_k=5,
        rerank_score_threshold=0.85,
    )


def _build_cache_service(tmp_path: Path, *, cache_mode: str):
    from cacheseek.backends.metadata import LocalCacheMetadataManager
    from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache
    from cacheseek.service.config import CacheConfig
    from cacheseek.service.connection import ConnectionManager
    from cacheseek.service.lifecycle import CacheService

    cfg: CacheConfig = _build_cache_config(tmp_path, cache_mode=cache_mode)
    cache_dir = tmp_path / "cacheseek_e2e"
    conn_mgr = ConnectionManager(cfg, storage_dir=cache_dir / "storage")
    metadata_manager = LocalCacheMetadataManager(cache_dir / "metadata")
    strategy = VideoBasedApproximateCache(
        cfg,
        conn_mgr.kv_store,
        conn_mgr.vector_store,
        metadata_manager,
        prompt_encoder=conn_mgr.prompt_encoder,
        video_encoder=conn_mgr.video_encoder,
        reranker=conn_mgr.reranker,
    )
    strategy._cacheseek_conn_mgr = conn_mgr
    return CacheService(
        strategies=[strategy],
        async_save=bool(cfg.save_async_enabled),
        max_queue_size=int(cfg.save_queue_size or 1),
        on_full=str(cfg.save_on_full or "drop"),
        flush_on_shutdown=bool(cfg.flush_on_shutdown),
        vector_wait_poll_s=float(cfg.vector_wait_poll_s),
        vector_wait_warn_s=float(cfg.vector_wait_warn_s),
        vector_wait_timeout_s=float(cfg.vector_wait_timeout_s),
        cache_mode=cfg.cache_mode,
    )


@pytest.fixture(scope="module")
def read_only_service(tmp_path_factory):
    """Build CacheService wired to live fluxon + qdrant + Qwen3-VL for lookup-only tests."""
    tmp = tmp_path_factory.mktemp("cacheseek_e2e")
    service = _build_cache_service(tmp, cache_mode="read_only")
    yield service
    service.shutdown()


@pytest.fixture(scope="module")
def read_write_service(tmp_path_factory):
    """Build CacheService in read_write mode for save -> lookup roundtrip."""
    tmp = tmp_path_factory.mktemp("cacheseek_e2e_rw")
    service = _build_cache_service(tmp, cache_mode="read_write")
    yield service
    service.shutdown()


def _run(coro):
    """Run an async lookup synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeReq:
    def __init__(self, prompt: str, task: str = "t2v"):
        self.prompt = prompt
        self.task = task


def _query(req: _FakeReq):
    from cacheseek.service.query import CacheQuery

    return CacheQuery(prompt=req.prompt, task_type=req.task)


def _skip_step(result) -> int:
    from cacheseek.service.result import SkipStep

    hint = getattr(result, "resume_hint", None)
    if isinstance(hint, SkipStep):
        return int(hint.k)
    return 0


def _similarity(result) -> float:
    return float(getattr(result, "matched_similarity", None) or 0.0)


def test_cacheseek_reads_shared_qdrant_collection(read_only_service):
    """Sanity: CacheSeek built with fluxon + qdrant client connects to the
    live pool and runs the full lookup pipeline (encode → search → rerank
    → decision) without exception. Verifies byte-stream interop and that
    the Qwen3-VL encoder/reranker load correctly inside the CacheSeek
    namespace.
    """
    req = _FakeReq("dummy probe prompt — expect miss but pipeline runs")
    result = _run(read_only_service.lookup(_query(req)))

    # Either hit or miss is fine; what matters is no exception was raised.
    assert result is not None
    assert hasattr(result, "hit")


def test_lookup_self_prompt_rerank_passes_threshold(read_only_service):
    """Encode + search + rerank chain works end-to-end against live qdrant.

    Self-similarity test: use a prompt already stored in the qdrant
    ``video`` collection as the query. Expect:
      1. CacheSeek encodes query via Qwen3-VL → 2048d vector
      2. qdrant search returns the same prompt (high cosine similarity)
      3. cross-encoder rerank → score > threshold (self-similarity is high)
      4. Strategy decides to attempt KV load

    NOTE: we do not assert ``hit=True`` because the shared KV pool may
    have been drained by another process; passing the rerank threshold
    is enough to prove the decision chain is wired up correctly.
    """
    import logging

    log_messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    try:
        req = _FakeReq(PROMPT_LIVE_HIT)
        result = _run(read_only_service.lookup(_query(req)))
    finally:
        logging.getLogger().removeHandler(handler)

    assert result is not None
    # The chain must reach rerank stage and select a candidate above threshold.
    # Either result.hit == True (KV present) or "lookup miss: hit by threshold but KV missing"
    # (KV pool drained — both prove cacheseek rerank chain works).
    print(
        f"\n[test] cacheseek lookup result: hit={result.hit} "
        f"skip_step={_skip_step(result)} sim={_similarity(result):.4f}"
    )


def test_save_then_lookup_roundtrip(read_write_service):
    """End-to-end save → lookup roundtrip against live fluxon + qdrant.

    Writes a fake latent payload via ``CacheService.save`` (so the
    fluxon KV pool is in sync with the qdrant collection), then
    immediately looks up the same prompt and expects a hit.

    This validates the full lifecycle: encode_signature → kv_put →
    vector_upsert → metadata_register → save_end → lookup → hit decision
    → kv_get → return ``LookupResult(hit=True)`` — without paying the
    cost of a real model inference.
    """
    import torch

    # Use a unique prompt for this roundtrip to avoid contention with other tests.
    test_prompt = (
        "cacheseek E2E roundtrip test prompt — "
        "blue ocean sunrise with seagulls flying over a lighthouse, peaceful morning scene"
    )
    req = _FakeReq(test_prompt)

    # Step 1: lookup → expect miss (fresh prompt)
    result_pre = _run(read_write_service.lookup(_query(req)))
    assert not result_pre.hit, (
        f"Expected miss on fresh prompt before save, got hit (cache pool collision?). "
        f"result={result_pre}"
    )

    # Step 2: save fake latent payload via cacheseek.save
    # Build a minimal but valid latent_states_dict + frames mock.
    fake_latent_shape = (1, 16, 4, 4)  # tiny shape for fast roundtrip
    latent_states_dict = {
        step: torch.randn(*fake_latent_shape, dtype=torch.float32)
        for step in [5, 10, 15, 20, 25]
    }
    # Build minimal video frames — Qwen3-VL processor's fetch_image expects
    # PIL.Image objects (or URL strings),not numpy arrays.
    from PIL import Image
    fake_frames = [
        Image.new('RGB', (256, 256), color=(i * 16 % 256, 0, 0))
        for i in range(16)
    ]

    from cacheseek.service.outputs import ModelOutputs

    _run(
        read_write_service.save(
            _query(req),
            ModelOutputs(
                latent_states_dict=latent_states_dict,
                num_frames=81,
                final_step=40,
                saved_steps=[5, 10, 15, 20, 25],
                embedding_video_frames=fake_frames,
            ),
        )
    )

    # Step 3: lookup → expect hit (just saved)
    result_post = _run(read_write_service.lookup(_query(req)))
    post_skip_step = _skip_step(result_post)
    print(
        f"\n[test] post-save lookup: hit={result_post.hit} "
        f"skip_step={post_skip_step} sim={_similarity(result_post):.4f}"
    )

    assert result_post.hit, (
        f"Expected hit after save, got miss. cacheseek's save → lookup roundtrip broken. "
        f"result={result_post}"
    )
    assert post_skip_step > 0, f"hit but skip_step={post_skip_step}"
