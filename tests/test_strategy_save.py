# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Branch coverage for ``VideoBasedApproximateCache.save`` + private helpers.

Save has 9 distinct branches across the latent-persistence path and the
vector-upsert path; helpers (``_determine_skip_step``,
``_normalize_search_results``, ``_candidate_text``) have their own
small unit tests because at least one of them was a real bug source
(``_determine_skip_step`` previously hardcoded ``return 5``).
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from cacheseek.service.cache_types import VectorSearchResult
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from tests._strategy_fixtures import make_strategy

pytestmark = pytest.mark.smoke


def _query(prompt: str = "save target prompt") -> CacheQuery:
    return CacheQuery(prompt=prompt, task_type="t2v")


def _frames(n: int = 3) -> list:
    return [Image.new("RGB", (16, 16), color=(i * 30, 0, 0)) for i in range(n)]


def _outputs(
    *,
    saved_steps: list[int] | None = None,
    latent_states_dict: dict | None = None,
    embedding_video_frames: list | None = None,
    num_frames: int = 16,
    final_step: int = 40,
) -> ModelOutputs:
    if saved_steps is None:
        saved_steps = [5, 10, 15, 20, 25]
    if latent_states_dict is None:
        latent_states_dict = {
            step: torch.tensor([float(step)], dtype=torch.float32)
            for step in saved_steps
        }
    if embedding_video_frames is None:
        embedding_video_frames = _frames(3)
    return ModelOutputs(
        latent_states_dict=latent_states_dict,
        embedding_video_frames=embedding_video_frames,
        num_frames=num_frames,
        final_step=final_step,
        saved_steps=list(saved_steps),
    )


# ─── 1. Empty prompt → silent return ──────────────────────────────────────


async def test_save_skip_empty_prompt() -> None:
    kit = make_strategy()
    await kit.strategy.save(_query(prompt=""), _outputs())
    # Nothing written to KV / vector / metadata.
    assert kit.kv.put_calls == []
    assert kit.vector.upsert_calls == []
    assert kit.metadata.register_calls == []


# ─── 2. Empty latent_states_dict → silent return ──────────────────────────


async def test_save_skip_empty_latents() -> None:
    kit = make_strategy()
    await kit.strategy.save(_query(), _outputs(latent_states_dict={}))
    assert kit.kv.put_calls == []
    assert kit.vector.upsert_calls == []


# ─── 3. Empty saved_steps → silent return ─────────────────────────────────


async def test_save_skip_empty_saved_steps() -> None:
    kit = make_strategy()
    await kit.strategy.save(_query(), _outputs(saved_steps=[]))
    assert kit.kv.put_calls == []


# ─── 4. Partial latents (some steps None) → still saves valid ones ────────


async def test_save_handles_partial_latents() -> None:
    """saved_steps lists steps the *pipeline* claimed it snapshotted, but
    the actual ``latent_states_dict`` may have ``None`` for some — those
    are silently skipped."""
    kit = make_strategy()
    latents = {
        5: torch.tensor([5.0]),
        10: None,  # missing
        15: torch.tensor([15.0]),
    }
    await kit.strategy.save(
        _query(),
        _outputs(saved_steps=[5, 10, 15], latent_states_dict=latents),
    )
    saved_keys = sorted(k for k, _ in kit.kv.put_calls)
    # step 10 dropped (was None)
    assert any(k.endswith("_step5") for k in saved_keys)
    assert any(k.endswith("_step15") for k in saved_keys)
    assert all("_step10" not in k for k in saved_keys)


# ─── 5. vector_store=None → cleanup latents, no vector upsert ─────────────


async def test_save_no_vector_store_cleans_up_latents() -> None:
    kit = make_strategy()
    kit.strategy.vector_store = None  # post-construct override
    await kit.strategy.save(_query(), _outputs())
    # Latents were written then immediately removed (compensating action).
    assert kit.kv.put_calls != []
    assert kit.kv.remove_calls != []
    assert kit.vector.upsert_calls == []
    assert kit.metadata.register_calls == []


# ─── 6. embedding_video_frames empty → cleanup latents ────────────────────


async def test_save_no_frames_cleans_up_latents() -> None:
    kit = make_strategy()
    await kit.strategy.save(_query(), _outputs(embedding_video_frames=[]))
    assert kit.kv.put_calls != []
    assert kit.kv.remove_calls != []
    assert kit.vector.upsert_calls == []


# ─── 7. video_encoder=None → cleanup latents ──────────────────────────────


async def test_save_no_video_encoder_cleans_up_latents() -> None:
    kit = make_strategy()
    kit.strategy.video_encoder = None
    await kit.strategy.save(_query(), _outputs())
    assert kit.kv.put_calls != []
    assert kit.kv.remove_calls != []
    assert kit.vector.upsert_calls == []


# ─── 8. encode_video raises → rollback ────────────────────────────────────


async def test_save_encode_failure_rolls_back() -> None:
    kit = make_strategy()
    kit.video_encoder.raise_on_call = RuntimeError("encoder boom")
    with pytest.raises(RuntimeError, match="encoder boom"):
        await kit.strategy.save(_query(), _outputs())
    # Rollback should have removed any latents put before the failure.
    assert kit.kv.remove_calls != []
    # Vector / metadata never reached the success branch.
    assert kit.vector.upsert_calls == []
    assert kit.metadata.register_calls == []


# ─── 9. vector.upsert raises → rollback removes vector + KV + metadata ───


async def test_save_vector_upsert_failure_rolls_back() -> None:
    kit = make_strategy()
    kit.vector.upsert_should_raise = RuntimeError("qdrant boom")
    with pytest.raises(RuntimeError, match="qdrant boom"):
        await kit.strategy.save(_query(), _outputs())
    # Latents were put, then removed on rollback.
    assert kit.kv.put_calls != []
    assert kit.kv.remove_calls != []
    # vector_written=False at exception time → no vector.delete needed,
    # but metadata wasn't yet attempted either.
    assert kit.metadata.register_calls == []


# ─── 10. Happy path → KV + vector + metadata all written ──────────────────


async def test_save_happy_path() -> None:
    kit = make_strategy()
    outs = _outputs(saved_steps=[5, 10, 15])
    await kit.strategy.save(_query(prompt="happy save"), outs)

    # KV: one put per step.
    saved_steps = sorted(int(k.split("_step")[-1]) for k, _ in kit.kv.put_calls)
    assert saved_steps == [5, 10, 15]
    # Vector: one upsert with the video embedding.
    assert len(kit.vector.upsert_calls) == 1
    coll, point_id, vec, payload = kit.vector.upsert_calls[0]
    assert coll == "video"
    assert payload["prompt"] == "happy save"
    assert payload["saved_steps"] == [5, 10, 15]
    # Metadata: registered with the same cache_id used for KV.
    assert len(kit.metadata.register_calls) == 1
    assert kit.metadata.register_calls[0]["cache_id"] == point_id


# ─── Helper unit tests ────────────────────────────────────────────────────


def test_determine_skip_step_empty() -> None:
    kit = make_strategy()
    assert kit.strategy._determine_skip_step([]) == 0


def test_determine_skip_step_picks_max_within_cap() -> None:
    kit = make_strategy(max_skip_step=20)
    # 25 is over cap, 20 is exactly cap, 15 < 20.
    assert kit.strategy._determine_skip_step([5, 15, 20, 25]) == 20


def test_determine_skip_step_all_over_cap() -> None:
    kit = make_strategy(max_skip_step=3)
    assert kit.strategy._determine_skip_step([5, 10, 25]) == 0


def test_determine_skip_step_exact_cap() -> None:
    kit = make_strategy(max_skip_step=10)
    assert kit.strategy._determine_skip_step([10]) == 10


def test_determine_skip_step_zero_excluded() -> None:
    """Per the spec, only ``0 < s ≤ max_skip``; step 0 doesn't count."""
    kit = make_strategy(max_skip_step=5)
    assert kit.strategy._determine_skip_step([0, 3, 5]) == 5


def test_normalize_search_results_strips_dashes() -> None:
    kit = make_strategy()
    results = [
        VectorSearchResult(cache_id="ab-cd", similarity=0.9, prompt="", saved_steps=[5], payload={}),
        VectorSearchResult(cache_id="no-dash-here", similarity=0.8, prompt="", saved_steps=[5], payload={}),
    ]
    kit.strategy._normalize_search_results(results)
    assert results[0].cache_id == "abcd"
    assert results[1].cache_id == "nodashhere"


def test_candidate_text_prefers_prompt() -> None:
    kit = make_strategy()
    r = VectorSearchResult(
        cache_id="x", similarity=0.9, prompt="primary prompt", saved_steps=[], payload={"prompt": "fallback"}
    )
    assert kit.strategy._candidate_text(r) == "primary prompt"


def test_candidate_text_falls_back_to_payload_prompt() -> None:
    kit = make_strategy()
    r = VectorSearchResult(
        cache_id="x", similarity=0.9, prompt="", saved_steps=[], payload={"prompt": "from payload"}
    )
    assert kit.strategy._candidate_text(r) == "from payload"


def test_candidate_text_empty_when_neither() -> None:
    kit = make_strategy()
    r = VectorSearchResult(cache_id="x", similarity=0.9, prompt="", saved_steps=[], payload={})
    assert kit.strategy._candidate_text(r) == ""
