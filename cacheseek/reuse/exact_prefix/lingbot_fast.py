"""KEY materialization and geometry for LingBot-World-Fast.

Backbone = Wan2.1-14B (MHA), standard geometry: n_layers=40, num_heads=40,
head_dim=128. frame_seqlen = (H/16)*(W/16) (vae_stride spatial 8 x patch 2 = 16);
chunk_tokens = num_frame_per_chunk * frame_seqlen. These numbers should be
verified once against the real checkpoint's config.json (dit_original_ckpt ships
with the weights, not in this repo).
"""
from __future__ import annotations

import hashlib
from typing import Any

from .config import ModelGeometry
from .keys import canonical_json_bytes, sha256

# Wan2.1-14B standard geometry (TODO: verify against ckpt config.json, esp. GQA)
WAN_14B_N_LAYERS = 40
WAN_14B_N_KV_HEADS = 40        # MHA => = num_heads; reduce if actually GQA
WAN_14B_HEAD_DIM = 128
SPATIAL_DOWNSAMPLE = 16        # vae_stride[1,2]=8 x patch_size[1,2]=2


def frame_seqlen(height: int, width: int) -> int:
    return (height // SPATIAL_DOWNSAMPLE) * (width // SPATIAL_DOWNSAMPLE)


def lingbot_geometry(height: int, width: int, num_frame_per_chunk: int = 3) -> ModelGeometry:
    return ModelGeometry(
        n_layers=WAN_14B_N_LAYERS,
        n_kv_heads=WAN_14B_N_KV_HEADS,
        head_dim=WAN_14B_HEAD_DIM,
        chunk_tokens=num_frame_per_chunk * frame_seqlen(height, width),
    )


# ----------------------------------------------------------------------- KEY fingerprints
def image_fingerprint(image_input: Any) -> bytes:
    """Fingerprint the canonical model input (recommended: the first-frame pixel
    tensor at VAE input resolution), not the raw file bytes (jpeg/png encoding
    differences would cause false mismatches)."""
    return _tensor_digest(image_input)


def prompt_fingerprint(prompt_or_embedding: Any) -> bytes:
    """Same prompt -> same text embedding (deterministic). Hash the normalized
    string if given a string; hash the tensor if given a tensor."""
    if isinstance(prompt_or_embedding, str):
        return sha256(b"prompt", prompt_or_embedding.strip().encode("utf-8"))
    return _tensor_digest(prompt_or_embedding)


def canonical_action(raw_action: Any) -> bytes:
    """Discrete action -> canonical bytes (exact by construction).

    Continuous poses (camera) must either be quantized to a fixed grid before
    being passed in, or this exact trie does not apply (use approximate retrieval
    + a metric index instead).
    """
    if isinstance(raw_action, (int, str)):
        return canonical_json_bytes(raw_action)
    return canonical_json_bytes(_jsonable(raw_action))


# Every setting that affects computation must go into config_blob_hash.
# Omitting one => returning an incorrect hit.
CONFIG_BLOB_KEYS = (
    "denoising_step_list", "timesteps_index", "infer_steps", "sample_shift",
    "num_frame_per_chunk", "local_attn_size", "sink_size",
    "target_height", "target_width", "vae_stride", "patch_size",
    "kv_quant", "causal_rope_type",
)


def config_blob_fields(engine_config: dict) -> dict:
    return {k: engine_config[k] for k in CONFIG_BLOB_KEYS if k in engine_config}


def _tensor_digest(t: Any) -> bytes:
    try:
        import numpy as np
        arr = t.detach().to("cpu").contiguous().numpy() if hasattr(t, "detach") else np.asarray(t)
        return hashlib.sha256(arr.tobytes()).digest()
    except Exception:
        return hashlib.sha256(repr(t).encode("utf-8")).digest()


def _jsonable(x: Any) -> Any:
    return x.tolist() if hasattr(x, "tolist") else x


# ----------------------------------------------------------------------- break_even_k demo
def _demo() -> None:
    """Print break_even_k sensitivity to (quantization, recompute time R, overlap)
    using real geometry and measured bandwidth.

    Defaults to quant='none' (bf16 lossless); KV quantization is not yet
    implemented/verified. The int4 row is shown for contrast, since whether the
    KV is compressed determines whether this cache is viable.
    """
    from .config import bytes_per_chunk_kv, calibrate_break_even_k, fetch_per_chunk_s

    for (h, w, tag) in [(480, 832, "480p"), (720, 1280, "720p")]:
        geo = lingbot_geometry(h, w)
        for quant in ("none", "int4"):
            b = bytes_per_chunk_kv(geo, quant=quant)
            tag2 = "bf16 lossless (default)" if quant == "none" else "int4 (unverified)"
            print(f"\n=== {tag} · {tag2}  chunk_tokens={geo.chunk_tokens}  KV/chunk ≈ {b/1e9:.2f} GB ===")
            F = fetch_per_chunk_s(geo, 3.5e9, quant=quant)
            print(f"  @ 3.5GB/s → fetch one chunk F ≈ {F*1e3:.0f} ms")
            for R_ms in (300, 600, 1000):
                for ov in (1.0, 0.5):
                    k = calibrate_break_even_k(
                        fetch_per_chunk_s=F * ov, recompute_per_chunk_s=R_ms / 1e3,
                        window_chunks=7, fixed_overhead_s=0.05,
                    )
                    k_str = "never worth it" if k > 256 else str(k)
                    print(f"    R={R_ms:>4}ms, overlap={ov} → break_even_k = {k_str}")


if __name__ == "__main__":
    _demo()
