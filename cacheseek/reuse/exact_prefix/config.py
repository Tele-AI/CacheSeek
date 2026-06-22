# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""WorldKVConfig + model geometry + break_even_k calibration.

break_even_k is not a magic constant; it is computed from three quantities:
    reusing a length-K prefix pays off ⟺  K·R  >  fixed + min(K,W)·F
      R = time to recompute one chunk (denoising micro-bench, MUST be measured)
      F = time to fetch one chunk of int4 KV (= bytes/chunk ÷ bandwidth; measured at 3-4GB/s)
      W = window chunks; fixed = per-fast-forward overhead (lookup/setup/first-block non-overlappable fetch)
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Tier

# Quantization scheme. Default NONE (bf16 lossless): the impact of KV quantization on
# LingBot quality has not yet been implemented/validated, so int4 is not assumed. Quantization
# is an opt-in to enable only after validation passes, not a fixed design choice.
QUANT_BYTES_PER_ELEM = {
    "none": 2.0,       # bf16/fp16, lossless original values (default)
    "int8": 1.0,
    "int4": 0.5,       # enable only after a quality A/B passes
}


@dataclass(slots=True)
class ModelGeometry:
    """Per-model KV geometry used to size one chunk's cache and the break-even cost model.

    Fields are model/checkpoint constants; chunk_tokens folds in the active
    resolution (frame_seqlen depends on H, W).
    """

    n_layers: int
    n_kv_heads: int          # under GQA much smaller than attn heads; confirm whether the ckpt uses GQA
    head_dim: int
    chunk_tokens: int        # = num_frame_per_chunk(latent) × frame_seqlen

    def kv_elems_per_chunk(self) -> int:
        """Number of KV elements stored for one chunk: layers x (k,v) x tokens x heads x head_dim."""
        # per-layer × (k,v) × token × head × head_dim
        return self.n_layers * 2 * self.chunk_tokens * self.n_kv_heads * self.head_dim


def bytes_per_chunk_kv(
    geo: ModelGeometry,
    *,
    quant: str = "none",       # default lossless; quantization not assumed
    group_size: int = 64,
) -> int:
    """KV bytes for one chunk. quant="none" is bf16 lossless (default).

    Only quantization schemes incur the per-group scale+min overhead; none does not.
    """
    elems = geo.kv_elems_per_chunk()
    per_elem = QUANT_BYTES_PER_ELEM[quant]
    payload = elems * per_elem
    if quant == "none":
        return int(payload)
    groups = elems / group_size
    overhead = groups * 2 * 2          # per group: scale+min, each fp16 (2B)
    return int(payload + overhead)


def fetch_per_chunk_s(
    geo: ModelGeometry,
    bandwidth_bytes_per_s: float,
    *,
    quant: str = "none",
    group_size: int = 64,
    overlap_factor: float = 1.0,       # <1: fraction overlappable with compute (per-layer prefetch)
) -> float:
    """Wall-clock seconds to fetch one chunk's KV = bytes / bandwidth, scaled by overlap_factor.

    F in the break-even inequality. overlap_factor < 1 models the fraction of the
    fetch that cannot be hidden behind compute (per-layer prefetch).
    """
    return bytes_per_chunk_kv(geo, quant=quant, group_size=group_size) / bandwidth_bytes_per_s * overlap_factor


def calibrate_break_even_k(
    *,
    fetch_per_chunk_s: float,
    recompute_per_chunk_s: float,
    window_chunks: int,
    fixed_overhead_s: float = 0.0,
    max_k: int = 256,
) -> int:
    """Smallest K at which reusing a length-K prefix pays off (fetch capped by W).

    Returns the smallest K satisfying `K·R > fixed + min(K,W)·F`; if it never pays off
    within the window (F≥R and fixed≥0), returns max_k+1 (i.e. don't cache for this workload/bandwidth).
    """
    R, F, W = recompute_per_chunk_s, fetch_per_chunk_s, window_chunks
    for k in range(1, max_k + 1):
        if k * R > fixed_overhead_s + min(k, W) * F:
            return k
    return max_k + 1


@dataclass(slots=True)
class WorldKVConfig:
    """Runtime knobs for WorldKVManager: rolling-window geometry, the break-even
    gate, and KV quantization/commit tier.

    Construct directly when the geometry is known, or via ``from_geometry`` to
    derive ``break_even_k`` from the cost model.
    """

    window_chunks: int        # W = local_attn_size (in chunks)
    sink_chunks: int          # pinned window head
    break_even_k: int         # see above; below it, don't fast-forward (falls back to normal generation, harmless)
    quant: str = "none"       # default bf16 lossless; enable quantization (int8/int4) only after a quality A/B passes
    group_size: int = 64
    commit_tier: Tier = Tier.FLUXON_DRAM

    @classmethod
    def from_geometry(
        cls,
        geo: ModelGeometry,
        *,
        window_chunks: int,
        sink_chunks: int,
        bandwidth_bytes_per_s: float,
        recompute_per_chunk_s: float,    # required: denoising micro-bench
        fixed_overhead_s: float = 0.0,
        quant: str = "none",             # default lossless; quantization not assumed
        group_size: int = 64,
        overlap_factor: float = 1.0,
        commit_tier: Tier = Tier.FLUXON_DRAM,
    ) -> WorldKVConfig:
        """Build a config, computing break_even_k from the measured cost model.

        Derives the per-chunk fetch time F from geometry and bandwidth, then
        calibrates the smallest profitable prefix length K against the required
        recompute-per-chunk micro-benchmark.

        Args:
            geo: Model KV geometry sizing one chunk's cache.
            window_chunks: Rolling-window width W (= local_attn_size in chunks).
            sink_chunks: Pinned window-head chunks kept forever.
            bandwidth_bytes_per_s: Measured store->engine fetch bandwidth.
            recompute_per_chunk_s: Measured denoising time to recompute one chunk (R).
            fixed_overhead_s: Per-fast-forward fixed overhead (lookup/setup/first-block fetch).
            quant: KV quantization scheme; "none" is bf16 lossless (default).
            group_size: Quantization group size (ignored when quant="none").
            overlap_factor: Fraction of fetch not overlappable with compute.
            commit_tier: Storage tier new blobs are committed to.

        Returns:
            A WorldKVConfig whose break_even_k is the smallest profitable K (or
            above max_k, meaning caching never pays off for this workload).
        """
        F = fetch_per_chunk_s(
            geo, bandwidth_bytes_per_s, quant=quant, group_size=group_size, overlap_factor=overlap_factor
        )
        k = calibrate_break_even_k(
            fetch_per_chunk_s=F,
            recompute_per_chunk_s=recompute_per_chunk_s,
            window_chunks=window_chunks,
            fixed_overhead_s=fixed_overhead_s,
        )
        return cls(
            window_chunks=window_chunks,
            sink_chunks=sink_chunks,
            break_even_k=k,
            quant=quant,
            group_size=group_size,
            commit_tier=commit_tier,
        )
