# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""TeleFuser LingBot-World-Fast integration (duck-typed; does not import telefuser).

TeleFuser needs only two hooks:
  - end of ``create_runtime``:  binding.on_runtime_created(runtime, session_config)
  - after ``generate_next_chunk`` clean-KV rewrite:
    binding.on_chunk_finalized(runtime, idx, denoised)
Plus a decode-only fast path: chunks present in ``runtime.world_kv_cached_latents``
skip denoise + rewrite and decode directly (KV is already seeded, so history
frames are not recomputed).

Three key facts aligned with TeleFuser (verified against pipeline.py /
denoising.py / lingbot_world_fast_dit.py):
  1. The KV window is declared by the runtime: ``kv_local_attn_size`` (latent
     frames, including sink; -1 = full length) and ``kv_sink_size``. The rolling
     eviction logic lives in the DiT. _RingKVWindow materializes per frame to
     reproduce the physical ring layout a cold run would have at chunk K:
     pre-roll is fully contiguous; post-roll = [sink S frames][most recent L-S
     frames].
  2. RNG: the full noise tensor is drawn once at the start; denoise then draws
     len(timesteps)-1 times per chunk (shape=chunk latent, dtype=bf16). So
     fast-forward must burn the draws for skipped chunks, otherwise the RNG
     stream is misaligned from chunk K onward and exact replay silently breaks.
  3. crossattn_cache is not cached: the first dit call at chunk K auto-inits it
     from prompt_emb.

Semantics: bit-exact replay. root includes seed/frame_num and every other
session config field that affects computation; the action key is the byte digest
of each chunk's control tensor (same control input => same key; approximate
matching is future work).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import numpy as np
import torch

from cacheseek.service.query import CacheQuery

from .config import WorldKVConfig
from .keys import canonical_json_bytes, config_blob_hash, root_hash, sha256
from .manager import WorldKVManager
from .strategy import ExactPrefixStrategy
from .trie import NamespaceForest, TrieNode, load_forest_snapshot, save_forest_snapshot
from .types import ActionKey


def make_full_kv_config(*, break_even_k: int = 1) -> WorldKVConfig:
    """Full-length KV (local_attn_size=-1) => materialization needs the entire
    prefix."""
    return WorldKVConfig(
        window_chunks=1_000_000, sink_chunks=1, break_even_k=break_even_k
    )


def make_rolling_config(
    *,
    local_attn_size: int = 7,  # latent frames, incl. sink (matches TeleFuser pipeline config)
    sink_size: int = 3,  # latent frames
    chunk_size: int = 3,  # latent frames per chunk
    break_even_k: int = 1,
) -> WorldKVConfig:
    """Rolling window: materialization needs only the sink chunk plus the chunks
    covering the most recent (L-S) frames -- O(W) rather than O(K)."""
    recent_frames = max(local_attn_size - sink_size, 1)
    return WorldKVConfig(
        window_chunks=-(-recent_frames // chunk_size),  # ceil
        sink_chunks=-(-sink_size // chunk_size),
        break_even_k=break_even_k,
    )


# Session fields that affect computation; all go into root (frame_num affects the
# full noise tensor shape, so it must be included).
SESSION_KEY_FIELDS = (
    "seed",
    "frame_num",
    "chunk_size",
    "sample_shift",
    "control_mode",
    "max_attention_size",
    "max_sequence_length",
)


def session_root_hash(session_config: Any, *, model_fingerprint: bytes) -> bytes:
    """Compute the namespace root_hash for a TeleFuser session.

    Combines an image fingerprint (mode, size, raw pixel bytes), a normalized
    prompt fingerprint, and a config_blob_hash over the computation-affecting
    SESSION_KEY_FIELDS plus the model weights fingerprint. Two sessions share a
    namespace (and thus may reuse KV) iff all three match.

    Args:
        session_config: TeleFuser session config (duck-typed; needs image, prompt,
            and the SESSION_KEY_FIELDS attributes).
        model_fingerprint: Weights/version fingerprint folded into config_blob_hash.

    Returns:
        The root_hash identifying this world.
    """
    image = session_config.image
    image_fp = sha256(
        b"img",
        image.mode.encode(),
        canonical_json_bytes(list(image.size)),
        hashlib.sha256(image.tobytes()).digest(),
    )
    prompt_fp = sha256(b"prompt", session_config.prompt.strip().encode("utf-8"))
    blob = {f: getattr(session_config, f, None) for f in SESSION_KEY_FIELDS}
    cfg_hash = config_blob_hash(blob, weights_fingerprint=model_fingerprint)
    return root_hash(image_fp=image_fp, prompt_fp=prompt_fp, config_blob_hash=cfg_hash)


ACTION_KEY_EPS = (
    1e-6  # control-tensor quantization (pre-hash). Well above 1e-10 float noise, well below bf16 resolution.
)


def chunk_action_keys(runtime: Any) -> list[ActionKey]:
    """One discrete ActionKey per chunk = byte digest of the control-tensor slice
    quantized to 1e-6.

    Why quantization is required: the LingBot conditioning pipeline applies a
    whole-trajectory max-normalization to framewise translation, so any change at
    the tail shifts the float value of max_norm at the ulp level, contaminating
    every prefix value with ~1e-10 noise. A bit-exact hash would then always miss
    on a legitimate prefix fork. Quantizing to 1e-6 absorbs the ulp noise while
    preserving the real fork signal (observed fork difference ~1e-1); 1e-10 is far
    below bf16 resolution, so the conditioning the model sees is identical -- this
    is still exact at model precision, so the VALUE=pure_function(KEY) invariant holds.
    """
    n = len(runtime.noise_chunks)
    keys: list[ActionKey] = []
    for i in range(n):
        if runtime.control_chunks is not None and i < len(runtime.control_chunks):
            arr = (
                runtime.control_chunks[i]
                .detach()
                .to("cpu", torch.float32)
                .contiguous()
                .numpy()
            )
            q = np.round(arr / ACTION_KEY_EPS).astype(np.int64)
            keys.append(hashlib.sha256(q.tobytes()).hexdigest()[:32])
        else:
            keys.append("noctrl")
    return keys


class _RingKVWindow:
    """RollingWindow adapter: assemble cached KV per frame into the physical ring
    layout a cold run would have at chunk K.

    Let c = frames per chunk, ft = tokens per frame, L = buffer frames (derived
    from buffer shape), S = sink frames (runtime.kv_sink_size), K = depth+1,
    F = K*c (logical frames generated so far):
      - F <= L (not yet rolled / full-length mode): frames 0..F-1 laid out
        contiguously; local_end = global_end = F*ft.
      - F > L (rolled, steady state): physical = [sink frames 0..S-1][most recent
        L-S frames F-(L-S)..F-1]; local_end = L*ft; global_end = F*ft (the two
        diverge, matching the DiT eviction arithmetic).
    Frame g's KV is taken from chunk g//c's blob by the token slice
    [g%c*ft : (g%c+1)*ft].
    """

    def __init__(self, runtime: Any) -> None:
        self._rt = runtime
        self._local_end_tokens = 0

    def _frames_to_seed(self, layer_kv: dict, depth: int) -> list[int]:
        rt = self._rt
        ft, c = rt.frame_tokens, rt.chunk_size
        buf_frames = layer_kv["k"].shape[1] // ft  # L (= lat_f in full-length mode)
        total_frames = (depth + 1) * c  # F
        sink = int(getattr(rt, "kv_sink_size", 0))
        if total_frames <= buf_frames:
            return list(range(total_frames))
        return list(range(sink)) + list(
            range(total_frames - (buf_frames - sink), total_frames)
        )

    def seed_layer(self, layer: int, blobs: list[tuple[int, Any]], depth: int) -> None:
        """Write cached KV into the runtime's self_kv_cache for one layer, reproducing
        the physical ring layout (per-frame token slices) a cold run would have at chunk K.

        Raises:
            KeyError: if a frame's source chunk is missing from blobs (the manager
                window did not provide it; rolling config likely mismatches runtime
                KV geometry).
        """
        rt = self._rt
        ft, c = rt.frame_tokens, rt.chunk_size
        kv = rt.self_kv_cache[layer]
        device, dtype = kv["k"].device, kv["k"].dtype
        by_depth = dict(blobs)
        frames = self._frames_to_seed(kv, depth)
        for pos, g in enumerate(frames):  # g = global frame index; pos = physical frame slot
            if g // c not in by_depth:
                raise KeyError(
                    f"world_kv ring needs chunk {g // c} but manager window did not provide it; "
                    f"check make_rolling_config matches runtime kv geometry"
                )
            k_c, v_c = by_depth[g // c]
            off = (g % c) * ft
            kv["k"][:, pos * ft : (pos + 1) * ft] = k_c[:, off : off + ft].to(
                device=device, dtype=dtype
            )
            kv["v"][:, pos * ft : (pos + 1) * ft] = v_c[:, off : off + ft].to(
                device=device, dtype=dtype
            )
        self._local_end_tokens = len(frames) * ft

    def set_resume_depth(self, depth: int) -> None:
        """Set each layer's global_end_index (logical, F*ft) and local_end_index
        (physical buffer fill from the last seed) so the DiT resumes at chunk depth+1."""
        rt = self._rt
        global_end = (depth + 1) * rt.chunk_size * rt.frame_tokens
        for kv in rt.self_kv_cache:
            kv["global_end_index"] = global_end
            kv["local_end_index"] = self._local_end_tokens


class LingBotWorldKVBinding:
    """One binding instance per session (holds that session's trie cursor)."""

    def __init__(
        self,
        manager: WorldKVManager,
        forest: NamespaceForest,
        *,
        model_fingerprint: bytes = b"lingbot-world-fast",
        ingest_enabled: bool = True,
        snapshot_path: str | None = None,
        snapshot_on_finalize: bool = False,
    ) -> None:
        """Create a per-session binding holding this session's trie cursor.

        If snapshot_path is set and the forest is empty at startup, loads the index
        snapshot to enable cross-process prefix hits (only meaningful with a
        persistent store).

        Args:
            manager: The WorldKVManager driving lookup/materialize/ingest.
            forest: The shared namespace forest.
            model_fingerprint: Weights/version fingerprint folded into root_hash.
            ingest_enabled: Whether finalized chunks are written back to the cache.
            snapshot_path: Optional path for the forest index snapshot.
            snapshot_on_finalize: Whether to flush the snapshot after each finalize.
        """
        self.mgr = manager
        self.forest = forest
        self.strategy = ExactPrefixStrategy(
            manager
        )  # lookup/writeback go through the shared Strategy protocol
        self.model_fingerprint = model_fingerprint
        self.ingest_enabled = ingest_enabled
        # Optional cross-process hits: if the forest is empty at startup, load the
        # index from a snapshot; write it back between sessions / after finalize.
        # Only meaningful with a persistent store (TensorStoreTierStore over
        # LocalDisk/Fluxon); InMemory lives only within the process. See
        # docs/design_exact_prefix_reuse/04-physical-view.md.
        self.snapshot_path = snapshot_path
        self.snapshot_on_finalize = snapshot_on_finalize
        if snapshot_path is not None and len(forest) == 0:
            load_forest_snapshot(snapshot_path, into=forest)
        self._ns = None
        self._parent: TrieNode | None = None
        self._actions: list[ActionKey] = []
        self._chain: list[bytes] = []
        self._query: CacheQuery | None = None
        self.last_fast_forward = 0  # observability: how many chunks this session skipped

    def flush_snapshot(self) -> None:
        """Atomically write the current forest index topology back to
        snapshot_path; no-op if unset. Typically called at session end / before
        process exit so the next process can get cross-process prefix hits."""
        if self.snapshot_path is not None:
            save_forest_snapshot(self.forest, self.snapshot_path)

    # ---------------------------------------------------------- hook 1: after runtime is built
    def on_runtime_created(self, runtime: Any, session_config: Any) -> None:
        """TeleFuser hook 1: after the runtime is built, attempt prefix fast-forward.

        Resolves the namespace, derives per-chunk action keys and their node-key
        chain, then runs the shared Strategy lookup. On a hit it materializes the
        cached KV into the runtime window, stashes the skipped chunks' latents for
        the decode-only fast path (``runtime.world_kv_cached_latents``), and burns
        the RNG draws for the skipped chunks so the noise stream stays aligned for
        bit-exact replay from chunk K. Falls back to a cold run (parent = root,
        last_fast_forward = 0) on miss or an incomplete window.
        """
        from .keys import build_action_chain

        root = session_root_hash(
            session_config, model_fingerprint=self.model_fingerprint
        )
        self._ns = self.forest.get_or_create_namespace(root, root)
        self._actions = chunk_action_keys(runtime)
        self._chain = build_action_chain(
            root, [canonical_json_bytes(a) for a in self._actions]
        )
        self._query = CacheQuery(
            prompt=getattr(session_config, "prompt", ""),
            seed=getattr(session_config, "seed", None),
            task_type="lingbot_world_fast",
            extra={"root_hash": root, "actions": self._actions},
        )

        # Lookup goes through the shared Strategy protocol (lookup = pure search +
        # break-even gate); materialization/latent/RNG are engine-adapter
        # responsibilities (interpreting the FastForward hint) and stay in this binding.
        res = asyncio.run(self.strategy.lookup(self._query))
        if not res.hit:
            self._parent = self._ns.root
            self.last_fast_forward = 0
            return
        hint = res.resume_hint
        if not self.mgr.materialize(hint.node, _RingKVWindow(runtime)):
            self._parent = self._ns.root  # incomplete window -> fall back to cold run
            self.last_fast_forward = 0
            return
        self._parent = hint.node
        k = hint.k
        self.last_fast_forward = k

        # 1. Skipped chunks -> decode-only: take the latent from the skeleton, no
        #    denoise / no rewrite.
        cached: dict[int, torch.Tensor] = {}
        path: list[TrieNode] = []
        n = hint.node
        while n is not None and n.depth >= 0:
            path.append(n)
            n = n.parent
        for node in reversed(path):  # chunk 0..K-1
            latent = self.mgr.store.get_skeleton(node.skeleton.latent_locator)
            cached[node.depth] = latent
        runtime.world_kv_cached_latents = cached

        # 2. Burn the generator draws for skipped chunks (len(timesteps)-1 per
        #    chunk, shape=chunk latent, dtype=bf16, one-to-one with denoise_chunk's
        #    torch.randn). Without this, the RNG stream is misaligned from chunk K
        #    onward and exact replay silently breaks.
        draws_per_chunk = max(len(runtime.timesteps) - 1, 0)
        for i in range(k):
            shape = tuple(runtime.noise_chunks[i].shape)
            for _ in range(draws_per_chunk):
                torch.randn(
                    shape,
                    generator=runtime.generator,
                    device=runtime.generator.device
                    if runtime.generator is not None
                    else "cpu",
                    dtype=torch.bfloat16,
                )

        # 3. The engine runs denoise from chunk K onward; chunks 0..K-1 take the
        #    decode-only fast path (implemented on the pipeline side).

    # ---------------------------------------------------------- hook 2: after chunk finalized
    def on_chunk_finalized(
        self, runtime: Any, idx: int, denoised: torch.Tensor
    ) -> None:
        """Called after the clean-KV rewrite: slice out this chunk's clean KV +
        latent and ingest them."""
        if not self.ingest_enabled or self._ns is None:
            return
        ct = runtime.chunk_size * runtime.frame_tokens
        payload = []
        for kv in runtime.self_kv_cache:
            # In rolling mode this chunk's clean KV sits at the physical tail
            # [local_end-ct : local_end] (logical position idx*ct holds only in
            # full-length mode); local_end was just advanced by the clean rewrite.
            e = int(kv["local_end_index"])
            s = e - ct
            payload.append(
                (
                    kv["k"][:, s:e].detach().to("cpu").clone(),
                    kv["v"][:, s:e].detach().to("cpu").clone(),
                )
            )
        latent = denoised.detach().to("cpu").clone()
        # Writeback goes through the shared Strategy protocol; chunk data is passed
        # via ctx (exact save is chunk-granular streaming).
        ctx = {
            "ns": self._ns,
            "parent": self._parent if self._parent is not None else self._ns.root,
            "action": self._actions[idx],
            "node_key": self._chain[idx],
            "depth": idx,
            "payload": payload,
            "latent": latent,
            "nbytes": sum(k.nbytes + v.nbytes for k, v in payload),
        }
        asyncio.run(self.strategy.save(self._query, None, ctx))
        self._parent = ctx["node"]
        if self.snapshot_on_finalize:
            self.flush_snapshot()  # optional: write the index back on finalize for cross-process hits
