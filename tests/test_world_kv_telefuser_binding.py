# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""TeleFuser binding integration tests for the rolling window (sink=3 frames,
local_attn=7 frames including sink).

The fake engine ports the TeleFuser DiT rolling-eviction arithmetic
(lingbot_world_fast_dit.py:117-145) verbatim, and verifies three things:
  1. The warm-restored ring (physical layout + both pointers) is bit-equal to
     the ring of a cold run at the same position, for K=1..4 covering both
     before rolling (K<=2) and after rolling (K>=3);
  2. After fast-forward burns the skipped sampling, a forked continuation is
     bit-equal to a cold run with the same seed (the RNG-alignment crux);
  3. Seed isolation (a different seed means a different world).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cacheseek.reuse.exact_prefix import (  # noqa: E402
    InMemoryTierStore,
    NamespaceForest,
    WorldKVManager,
)
from cacheseek.reuse.exact_prefix.telefuser_lingbot import (  # noqa: E402
    LingBotWorldKVBinding,
    make_world_kv_config,
)

# Tiny geometry: chunk=3 frames, frame_tokens=2; window L=7 frames including
# sink S=3 frames => rolling starts at the 3rd chunk.
N_CHUNKS, CHUNK_FRAMES, FRAME_TOKENS, N_LAYERS, HEADS, HEAD_DIM = 4, 3, 2, 2, 2, 4
LOCAL_ATTN, SINK = 7, 3
CT = CHUNK_FRAMES * FRAME_TOKENS                      # tokens per chunk
KV_TOKENS = LOCAL_ATTN * FRAME_TOKENS                 # ring physical capacity (tokens)
SINK_TOKENS = SINK * FRAME_TOKENS
TIMESTEPS = torch.tensor([999, 679, 358, 0])          # len=4 => 3 samplings per chunk


class FakeImage:
    mode, size = "RGB", (8, 8)

    def tobytes(self) -> bytes:
        return b"\x01" * 64


def make_session(seed: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        prompt="a world", image=FakeImage(), seed=seed, frame_num=49,
        chunk_size=CHUNK_FRAMES, sample_shift=5.0, control_mode="act",
        max_attention_size=None, max_sequence_length=512,
    )


def make_runtime(seed: int, actions: list[int]) -> SimpleNamespace:
    """Mirror create_runtime: sample the full noise tensor once then split; the
    KV buffer holds only L frames (rolling)."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn((1, 2, N_CHUNKS * CHUNK_FRAMES, 2, 2), generator=gen, dtype=torch.float32)
    return SimpleNamespace(
        noise_chunks=list(noise.split(CHUNK_FRAMES, dim=2)),
        control_chunks=[torch.full((1, 4), float(a)) for a in actions],
        self_kv_cache=[
            {
                "k": torch.zeros((1, KV_TOKENS, HEADS, HEAD_DIM)),
                "v": torch.zeros((1, KV_TOKENS, HEADS, HEAD_DIM)),
                "global_end_index": torch.tensor([0]),
                "local_end_index": torch.tensor([0]),
            }
            for _ in range(N_LAYERS)
        ],
        timesteps=TIMESTEPS,
        generator=gen,
        chunk_size=CHUNK_FRAMES,
        frame_tokens=FRAME_TOKENS,
        kv_local_attn_size=LOCAL_ATTN,
        kv_sink_size=SINK,
        world_kv_cached_latents={},
    )


def kv_write(kv: dict, k_new: torch.Tensor, v_new: torch.Tensor, current_start: int) -> None:
    """Line-by-line port of the DiT rolling-write arithmetic
    (lingbot_world_fast_dit.py:108-145)."""
    num_new = k_new.shape[1]
    current_end = current_start + num_new
    global_end = int(kv["global_end_index"][0])
    local_end = int(kv["local_end_index"][0])
    if current_end > global_end and num_new + local_end > KV_TOKENS:
        evicted = num_new + local_end - KV_TOKENS
        rolled = local_end - evicted - SINK_TOKENS
        kv["k"][:, SINK_TOKENS : SINK_TOKENS + rolled] = kv["k"][
            :, SINK_TOKENS + evicted : SINK_TOKENS + evicted + rolled
        ].clone()
        kv["v"][:, SINK_TOKENS : SINK_TOKENS + rolled] = kv["v"][
            :, SINK_TOKENS + evicted : SINK_TOKENS + evicted + rolled
        ].clone()
        local_end = local_end + current_end - global_end - evicted
    else:
        local_end = local_end + current_end - global_end
    local_start = local_end - num_new
    kv["k"][:, local_start:local_end] = k_new
    kv["v"][:, local_start:local_end] = v_new
    kv["global_end_index"][0] = current_end
    kv["local_end_index"][0] = local_end


def fake_denoise(runtime: SimpleNamespace, idx: int) -> torch.Tensor:
    """Mirror denoise_chunk's RNG consumption (3 bf16 samplings) plus the
    clean-KV rolling write.

    KV varies per token (base+tok), so a frame-slice misalignment shows up
    directly in the values.
    """
    shape = tuple(runtime.noise_chunks[idx].shape)
    noise = None
    for _ in range(len(runtime.timesteps) - 1):
        noise = torch.randn(shape, generator=runtime.generator, dtype=torch.bfloat16)
    denoised = runtime.noise_chunks[idx].to(torch.bfloat16) + noise
    base = denoised.float().mean()
    tok = torch.arange(CT, dtype=torch.float32).view(1, CT, 1, 1)
    for layer, kv in enumerate(runtime.self_kv_cache):
        k_new = (base + layer * 10 + idx + tok / 100).expand(1, CT, HEADS, HEAD_DIM).contiguous()
        kv_write(kv, k_new, k_new + 1000.0, current_start=idx * CT)
    return denoised


def snapshot(runtime: SimpleNamespace) -> list[dict]:
    return [
        {
            "k": kv["k"].clone(), "v": kv["v"].clone(),
            "ge": int(kv["global_end_index"][0]), "le": int(kv["local_end_index"][0]),
        }
        for kv in runtime.self_kv_cache
    ]


def run_session(binding, runtime, session, *, snapshots: list | None = None):
    """Mirror the pipeline's chunk loop: decode-only fast path + finalize hook."""
    binding.on_runtime_created(runtime, session)
    out: list[torch.Tensor | None] = []
    for idx in range(N_CHUNKS):
        if snapshots is not None:
            snapshots.append(snapshot(runtime))             # ring state BEFORE chunk idx
        cached = runtime.world_kv_cached_latents.pop(idx, None)
        if cached is not None:
            out.append(None)                                # decode-only: no denoise, no RNG burn
            continue
        denoised = fake_denoise(runtime, idx)
        binding.on_chunk_finalized(runtime, idx, denoised)
        out.append(denoised)
    return out


def make_stack():
    forest = NamespaceForest()
    cfg = make_world_kv_config(local_attn_size=LOCAL_ATTN, sink_size=SINK, chunk_size=CHUNK_FRAMES)
    return forest, WorldKVManager(forest, InMemoryTierStore(), cfg)


# ---------------------------------------------------------------------- tests
def test_warm_ring_equals_cold_ring_all_resume_points():
    """K=1..4: the warm-restored ring is bit-equal to a cold run's ring just
    before chunk K (both before and after rolling)."""
    actions = [1, 2, 3, 4]
    forest, mgr = make_stack()
    snaps: list = []
    rt_donor = make_runtime(42, actions)
    run_session(LingBotWorldKVBinding(mgr, forest), rt_donor, make_session(42), snapshots=snaps)
    snaps.append(snapshot(rt_donor))                        # snaps[K] = ring state just before chunk K

    for K in (1, 2, 3, 4):
        fork = actions[:K] + [99] * (N_CHUNKS - K)          # first K shared, then forked => hits exactly K
        rt = make_runtime(42, fork)
        b = LingBotWorldKVBinding(mgr, forest, ingest_enabled=False)
        b.on_runtime_created(rt, make_session(42))
        assert b.last_fast_forward == K, f"K={K}: matched {b.last_fast_forward}"
        for layer, (kv, r) in enumerate(zip(rt.self_kv_cache, snaps[K], strict=False)):
            le = r["le"]
            assert int(kv["local_end_index"][0]) == le, f"K={K} layer{layer} local_end"
            assert int(kv["global_end_index"][0]) == r["ge"], f"K={K} layer{layer} global_end"
            assert torch.equal(kv["k"][:, :le], r["k"][:, :le]), f"K={K} layer{layer} k ring"
            assert torch.equal(kv["v"][:, :le], r["v"][:, :le]), f"K={K} layer{layer} v ring"


def test_branch_resume_rng_aligned():
    """Crux: after burning the skipped chunks' sampling, the warm continuation
    is bit-equal to a cold run with the same seed."""
    forest, mgr = make_stack()
    run_session(LingBotWorldKVBinding(mgr, forest), make_runtime(42, [1, 2, 3, 4]), make_session(42))

    fork = [1, 2, 9, 9]                                     # shared 2-chunk prefix
    rt_warm = make_runtime(42, fork)
    out_warm = run_session(LingBotWorldKVBinding(mgr, forest), rt_warm, make_session(42))
    assert out_warm[0] is None and out_warm[1] is None      # prefix is decode-only

    forest_c, mgr_c = make_stack()                          # empty cache = cold-run reference
    rt_cold = make_runtime(42, fork)
    out_cold = run_session(LingBotWorldKVBinding(mgr_c, forest_c), rt_cold, make_session(42))

    for i in (2, 3):
        assert torch.equal(out_warm[i], out_cold[i]), f"chunk {i} diverged: RNG misaligned"
    for lw, lc in zip(rt_warm.self_kv_cache, rt_cold.self_kv_cache, strict=False):
        le = int(lc["local_end_index"][0])
        assert torch.equal(lw["k"][:, :le], lc["k"][:, :le])  # ring also matches after continuation

    # After branch write-back, both paths hit the full chain.
    for acts in ([1, 2, 3, 4], fork):
        b = LingBotWorldKVBinding(mgr, forest)
        b.on_runtime_created(make_runtime(42, acts), make_session(42))
        assert b.last_fast_forward == N_CHUNKS


def test_namespace_isolation_by_seed():
    forest, mgr = make_stack()
    run_session(LingBotWorldKVBinding(mgr, forest), make_runtime(42, [1, 2, 3, 4]), make_session(42))
    b = LingBotWorldKVBinding(mgr, forest)
    b.on_runtime_created(make_runtime(7, [1, 2, 3, 4]), make_session(7))
    assert b.last_fast_forward == 0


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all telefuser binding (rolling) tests passed")
