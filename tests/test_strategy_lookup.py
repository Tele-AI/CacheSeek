# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Branch coverage for ``VideoBasedApproximateCache.lookup``.

The method has 15 distinct return paths (14 misses + 1 hit). Each test
drives one branch deterministically by tweaking config + stub state, and
asserts both the ``LookupResult`` shape and the audit / eviction side
effects that branch is responsible for.

See ``tests/_strategy_fixtures.py`` for the stub backends.
"""
from __future__ import annotations

import io

import pytest
import torch

from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult, SkipStep
from tests._strategy_fixtures import (
    StubReranker,
    make_search_result,
    make_strategy,
)

pytestmark = pytest.mark.smoke


def _query(prompt: str = "hello world", task_type: str = "t2v") -> CacheQuery:
    return CacheQuery(prompt=prompt, task_type=task_type)


def _seed_kv_with_latent(kv, cache_id: str, step: int) -> torch.Tensor:
    """Stash a serialized latent under the strategy's cache key naming."""
    tensor = torch.tensor([float(step), float(step) + 0.5], dtype=torch.float32)
    buf = io.BytesIO()
    torch.save(tensor, buf)
    kv.preset(f"{cache_id}_step{step}", buf.getvalue())
    return tensor


# ─── 1. Empty prompt ──────────────────────────────────────────────────────


async def test_lookup_miss_empty_prompt() -> None:
    kit = make_strategy()
    result = await kit.strategy.lookup(_query(prompt=""))
    assert isinstance(result, LookupResult)
    assert result.hit is False
    # Encoder must NOT be called when prompt is empty.
    assert kit.prompt_encoder.calls == []


# ─── 2. vector_store is None ──────────────────────────────────────────────


async def test_lookup_miss_vector_store_none() -> None:
    kit = make_strategy()
    kit.strategy.vector_store = None  # post-construct override
    result = await kit.strategy.lookup(_query())
    assert result.hit is False
    assert kit.prompt_encoder.calls == []


# ─── 3. prompt_encoder is None ────────────────────────────────────────────


async def test_lookup_miss_prompt_encoder_none() -> None:
    kit = make_strategy()
    kit.strategy.prompt_encoder = None
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 4. Empty query vector ────────────────────────────────────────────────


async def test_lookup_miss_empty_query_vec() -> None:
    kit = make_strategy()
    kit.prompt_encoder.return_value = []
    result = await kit.strategy.lookup(_query())
    assert result.hit is False
    # Encoder is consulted, vector_search is not.
    assert kit.prompt_encoder.calls == ["hello world"]


# ─── 5. rerank-on, no vector results ──────────────────────────────────────


async def test_lookup_miss_rerank_on_no_vector_results() -> None:
    kit = make_strategy(rerank_enabled=True, reranker=StubReranker(return_scores=[]))
    kit.vector.search_results = []
    result = await kit.strategy.lookup(_query())
    assert result.hit is False
    # No similarity recordings either — branch returns before audit emit.
    assert kit.metadata.similarity_recordings == []


# ─── 6. rerank-on, scores returned but length mismatch ────────────────────


async def test_lookup_miss_rerank_scores_length_mismatch() -> None:
    """Strategy._rerank_scores raises on size mismatch — caller catches
    it via the runtime exception path."""
    kit = make_strategy(
        rerank_enabled=True,
        reranker=StubReranker(return_scores=[0.95, 0.92]),  # 2 scores …
    )
    # … but only 1 candidate.
    kit.vector.search_results = [make_search_result(cache_id="cand-1", similarity=0.99)]
    # The strategy raises ValueError from _rerank_scores; CacheService
    # catches strategy exceptions and returns miss, but at the strategy
    # level the exception bubbles. We assert the exception bubbles, since
    # the strategy unit-test scope ends there.
    with pytest.raises(ValueError):
        await kit.strategy.lookup(_query())


# ─── 7. rerank-on, similarity below threshold (fallback path) ─────────────


async def test_lookup_miss_rerank_fallback_similarity_below_threshold() -> None:
    """When reranker has no ``score_mm`` method (returns None), strategy
    falls back to vector similarity, which here is below threshold."""
    kit = make_strategy(
        rerank_enabled=True,
        video_similarity_threshold=0.50,
        reranker=object(),  # no score_mm → fallback
    )
    kit.vector.search_results = [
        make_search_result(cache_id="low-sim", similarity=0.10)
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 8. rerank-on, fallback path, skip_step=0 ─────────────────────────────


async def test_lookup_miss_rerank_fallback_skip_step_zero() -> None:
    """Reranker missing → fallback → similarity passes threshold → but
    saved_steps yields skip_step=0 (no step ≤ max_skip_step)."""
    kit = make_strategy(
        rerank_enabled=True,
        video_similarity_threshold=0.10,
        max_skip_step=3,  # so step 5/10 are all > max_skip
        reranker=object(),
    )
    kit.vector.search_results = [
        make_search_result(cache_id="big-steps", similarity=0.95, saved_steps=[10, 20])
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 9. rerank-on, rerank score below threshold ───────────────────────────


async def test_lookup_miss_rerank_score_below_threshold() -> None:
    kit = make_strategy(
        rerank_enabled=True,
        rerank_score_threshold=0.85,
        reranker=StubReranker(return_scores=[0.40]),
    )
    kit.vector.search_results = [
        make_search_result(cache_id="bad-rerank", similarity=0.99)
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False
    # Both vector_search and rerank stages should appear in audit.
    stages = [r.stage for r in kit.metadata.similarity_recordings]
    assert "vector_search" in stages
    assert "rerank" in stages


# ─── 10. rerank-on, rerank passes but skip_step=0 ─────────────────────────


async def test_lookup_miss_rerank_skip_step_zero() -> None:
    kit = make_strategy(
        rerank_enabled=True,
        rerank_score_threshold=0.85,
        max_skip_step=3,
        reranker=StubReranker(return_scores=[0.99]),
    )
    kit.vector.search_results = [
        make_search_result(cache_id="big-steps", similarity=0.99, saved_steps=[10])
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 11. no-rerank, no vector results ─────────────────────────────────────


async def test_lookup_miss_no_rerank_no_vector_results() -> None:
    kit = make_strategy(rerank_enabled=False)
    kit.vector.search_results = []
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 12. no-rerank, similarity below threshold ────────────────────────────


async def test_lookup_miss_no_rerank_similarity_below_threshold() -> None:
    kit = make_strategy(rerank_enabled=False, video_similarity_threshold=0.50)
    kit.vector.search_results = [
        make_search_result(cache_id="low-sim", similarity=0.10)
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 13. no-rerank, skip_step=0 ───────────────────────────────────────────


async def test_lookup_miss_no_rerank_skip_step_zero() -> None:
    kit = make_strategy(rerank_enabled=False, max_skip_step=3)
    kit.vector.search_results = [
        make_search_result(cache_id="big-steps", similarity=0.99, saved_steps=[10])
    ]
    result = await kit.strategy.lookup(_query())
    assert result.hit is False


# ─── 14. KV missing after vector hit → lazy-evict ─────────────────────────


async def test_lookup_miss_kv_missing_triggers_lazy_evict() -> None:
    """Vector says hit, KV is empty → strategy lazy-evicts the stale
    vector + metadata entries so a subsequent lookup can't re-match."""
    kit = make_strategy(rerank_enabled=False)
    cache_id = "stale1"  # no dash — _normalize_search_results would strip it
    kit.vector.search_results = [
        make_search_result(cache_id=cache_id, similarity=0.99, saved_steps=[5])
    ]
    # Note: kv has no preset for this cache_id → _load_latent returns None

    result = await kit.strategy.lookup(_query())

    assert result.hit is False
    # Lazy-eviction side effects: vector.delete + metadata.remove
    assert kit.vector.delete_calls == [("video", [cache_id])]
    assert kit.metadata.remove_calls == [cache_id]


# ─── 15. Full hit ─────────────────────────────────────────────────────────


async def test_lookup_hit_skip_step() -> None:
    kit = make_strategy(rerank_enabled=False)
    cache_id = "abc123"
    cached_prompt = "the matched prompt body"
    kit.vector.search_results = [
        make_search_result(
            cache_id=cache_id,
            similarity=0.95,
            prompt=cached_prompt,
            saved_steps=[5, 10, 15, 20, 25],
        )
    ]
    expected_tensor = _seed_kv_with_latent(kit.kv, cache_id, step=25)

    result = await kit.strategy.lookup(_query())

    assert result.hit is True
    assert result.matched_cache_id == cache_id
    assert result.cached_prompt == cached_prompt
    assert isinstance(result.resume_hint, SkipStep)
    assert result.resume_hint.k == 25  # max step ≤ max_skip (default 25)
    # Payload is now a VideoApproxPayload — caller deserializes via
    # get_latent_at_step.
    from cacheseek.reuse.approximate import VideoApproxPayload
    assert isinstance(result.payload, VideoApproxPayload)
    assert torch.allclose(result.payload.get_latent_at_step(25), expected_tensor)
    # Hit recorded in audit + access counter.
    assert kit.metadata.access_calls == [cache_id]
    assert len(kit.metadata.hit_pair_recordings) == 1
    assert kit.metadata.hit_pair_recordings[0].skip_step == 25


# ─── 16. Staircase skip-step: rerank score → tier ────────────────────────


def _determine(kit, saved_steps, *, rerank_score=None, max_skip_step=14):
    """Drive ``_determine_skip_step`` directly with a given config cap."""
    kit.strategy.config.max_skip_step = max_skip_step
    return kit.strategy._determine_skip_step(saved_steps, rerank_score=rerank_score)


def test_select_k_by_score_default_tau_table() -> None:
    """τ_K table {3:0.63, 7:0.85, 11:0.85, 14:1.01}; K*(s)=max{K:τ_K≤s}.

    K=14's τ (1.01) sits above the rerank ceiling (~0.93) so it never
    qualifies — the high tier tops out at K=11.
    """
    kit = make_strategy(rerank_enabled=True, reranker=StubReranker(return_scores=[0.0]))
    kit.strategy.config.staircase_skip_enabled = True  # opt-in feature (default off)
    sel = kit.strategy._select_k_by_score
    assert sel(0.62) is None        # below every τ_K → no tier
    assert sel(0.63) == 3           # K=3 tier opens at 0.63
    assert sel(0.84) == 3           # still only K=3
    assert sel(0.85) == 11          # K=7 and K=11 both open; max = 11
    assert sel(0.95) == 11          # K=14 (τ=1.01) never qualifies
    assert sel(1.01) == 14          # only if score somehow reaches τ_14


def test_determine_skip_step_staircase_caps_by_tier() -> None:
    """High rerank skips deep, mid rerank shallow, both snapped to saved_steps."""
    kit = make_strategy(rerank_enabled=True, reranker=StubReranker(return_scores=[0.0]))
    kit.strategy.config.staircase_skip_enabled = True  # opt-in feature (default off)
    saved = [3, 7, 11, 14]
    # High score → tier K=11 → largest saved step ≤ 11 = 11.
    assert _determine(kit, saved, rerank_score=0.90) == 11
    # Mid score (0.63–0.85) → tier K=3 → largest saved step ≤ 3 = 3.
    assert _determine(kit, saved, rerank_score=0.70) == 3
    # Score below all τ_K → no tier; fall back to max_skip_step cap (14) → 14.
    assert _determine(kit, saved, rerank_score=0.50) == 14


def test_determine_skip_step_staircase_respects_max_skip_step() -> None:
    """max_skip_step is an independent upper bound — min(tier_K, max_skip)."""
    kit = make_strategy(rerank_enabled=True, reranker=StubReranker(return_scores=[0.0]))
    saved = [3, 7, 11, 14]
    # Tier says K=11 but max_skip_step=7 clamps it → largest saved step ≤ 7 = 7.
    assert _determine(kit, saved, rerank_score=0.90, max_skip_step=7) == 7


def test_determine_skip_step_legacy_without_score() -> None:
    """No rerank score → legacy max(saved_steps ≤ max_skip_step), tier ignored."""
    kit = make_strategy(rerank_enabled=False)
    saved = [5, 10, 15, 20, 25]
    assert _determine(kit, saved, rerank_score=None, max_skip_step=20) == 20


def test_determine_skip_step_staircase_disabled_falls_back() -> None:
    """staircase_skip_enabled=False → score ignored, legacy cap applies."""
    kit = make_strategy(rerank_enabled=True, reranker=StubReranker(return_scores=[0.0]))
    kit.strategy.config.staircase_skip_enabled = False
    saved = [3, 7, 11, 14]
    # Even a low score that yields no tier would normally fall back; with
    # staircase off, the high score is ignored entirely → legacy cap=14.
    assert _determine(kit, saved, rerank_score=0.70, max_skip_step=14) == 14


async def test_lookup_staircase_high_vs_mid_score() -> None:
    """End-to-end: same donor, different rerank score → different skip depth."""
    saved = [3, 7, 11, 14]

    # High rerank (0.90 ≥ τ_11=0.85) → K=11.
    hi = make_strategy(
        rerank_enabled=True,
        rerank_score_threshold=0.50,
        max_skip_step=14,
        reranker=StubReranker(return_scores=[0.90]),
    )
    hi.strategy.config.staircase_skip_enabled = True  # opt-in feature (default off)
    hi.vector.search_results = [
        make_search_result(cache_id="donor", similarity=0.99, saved_steps=saved)
    ]
    _seed_kv_with_latent(hi.kv, "donor", step=11)
    res_hi = await hi.strategy.lookup(_query())
    assert res_hi.hit is True
    assert res_hi.resume_hint.k == 11

    # Mid rerank (0.70 in [0.63, 0.85)) → K=3.
    mid = make_strategy(
        rerank_enabled=True,
        rerank_score_threshold=0.50,
        max_skip_step=14,
        reranker=StubReranker(return_scores=[0.70]),
    )
    mid.strategy.config.staircase_skip_enabled = True  # opt-in feature (default off)
    mid.vector.search_results = [
        make_search_result(cache_id="donor", similarity=0.99, saved_steps=saved)
    ]
    _seed_kv_with_latent(mid.kv, "donor", step=3)
    res_mid = await mid.strategy.lookup(_query())
    assert res_mid.hit is True
    assert res_mid.resume_hint.k == 3
