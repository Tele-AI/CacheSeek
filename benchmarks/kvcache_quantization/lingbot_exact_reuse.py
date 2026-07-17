# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
# ruff: noqa: E402  (imports intentionally follow the sys.path/editable-finder shim)
"""Quality/performance benchmark for LingBot exact-prefix KV quantization.

The benchmark deliberately separates *measurement* from *profiling*:

* The benchmark suite always runs without cProfile or ``torch.profiler`` so
  request latency is not polluted by profiler overhead.
* Optional profiler passes are executed afterwards on fresh cache managers.
  Those passes exist only for bottleneck diagnosis and are never used in the
  reported speed comparison.

The unprofiled benchmark contains two cache-reuse branches plus one cold
reference:

  main_cache_seed_quant
      Main trajectory, empty cache, full denoise, writes the selected KIVI
      representation.

  fork_cache_reuse_quant
      Forked trajectory, reuses the selected quantized prefix KV, then denoises
      only the suffix.

  main_cache_seed_none
      Internal seed run for the automatically included unquantized baseline.

  fork_cache_reuse_none
      Forked trajectory, reuses an FP/unquantized prefix KV.  This baseline is
      always measured, so ``--quant none`` is intentionally not a valid CLI
      choice.

  fork_cold_reference
      Same forked trajectory with an empty cache and full denoising.

Generated images are indexed by the runtime's *absolute chunk index*.  Quality
metrics compare only common absolute chunks, preventing a fast-forwarded suffix
from being accidentally compared with frame zero of the cold reference.  The
manifest records per-phase latency, aligned quality metrics, cache occupancy,
profiling metadata, and the checks required for a valid prefix-fork run.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import math
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

_WORLDKV_ROOTS = [r for r in os.environ.get("WORLDKV_REPO_ROOTS", "").split(":") if r]
_WORLDKV_REAL_ROOTS = [os.path.realpath(r) for r in _WORLDKV_ROOTS]
if _WORLDKV_ROOTS:
    sys.meta_path = [
        f for f in sys.meta_path
        if not any(
            tag in getattr(f, "__module__", "")
            for tag in ("__editable___telefuser", "__editable___cacheseek")
        )
    ]
    for _r in reversed(_WORLDKV_ROOTS):
        if _r not in sys.path:
            sys.path.insert(0, _r)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

try:
    import telefuser  # noqa: F401
except ImportError:
    _tf = os.environ.get("TELEFUSER", "")
    assert _tf, "cannot import telefuser: set TELEFUSER=/path/to/telefuser-internal"
    sys.path.insert(0, _tf)

from telefuser.pipelines.lingbot_world_fast import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
    LingBotWorldFastSessionConfig,
)

from cacheseek.reuse.exact_prefix import NamespaceForest, WorldKVManager
from cacheseek.reuse.exact_prefix.telefuser_lingbot import (
    LingBotWorldKVBinding,
    make_world_kv_config,
)
from cacheseek.stores import InMemoryTierStore

if _WORLDKV_ROOTS:
    import cacheseek as _cs
    import telefuser as _tf

    def _is_under_worldkv_root(path: str) -> bool:
        real_path = os.path.realpath(path)
        return any(
            real_path == root or real_path.startswith(root + os.sep)
            for root in _WORLDKV_REAL_ROOTS
        )

    for _mod in (_tf, _cs):
        _src = inspect.getsourcefile(_mod) or ""
        assert _is_under_worldkv_root(_src), (
            f"{_mod.__name__} resolved to {_src} (editable finder not stripped?)"
        )
        print(f"[worldkv-shim] {_mod.__name__} -> {_src}", flush=True)

DEFAULT_PROMPT = "a quiet stone courtyard, ancient walls, soft afternoon light"
DEFAULT_SEED = 42
ORIG_H, ORIG_W = 720, 1280

# The LingBot control path maps one reusable KV chunk to three latent frames,
# with a temporal expansion factor of four and two initial conditioning frames.
# Keep these values named because the fork position must remain consistent with
# the binding's chunk geometry.
KV_CHUNK_SIZE = 3
TEMPORAL_DOWNSAMPLE = 4
INITIAL_FRAME_OFFSET = 2

RUN_MAIN_CACHE_SEED_QUANT = "main_cache_seed_quant"
RUN_FORK_CACHE_REUSE_QUANT = "fork_cache_reuse_quant"
RUN_MAIN_CACHE_SEED_NONE = "main_cache_seed_none"
RUN_FORK_CACHE_REUSE_NONE = "fork_cache_reuse_none"
RUN_FORK_COLD_REFERENCE = "fork_cold_reference"
RUN_DESCRIPTIONS = {
    RUN_MAIN_CACHE_SEED_QUANT: (
        "main trajectory cold run that populates the selected quantized KV cache"
    ),
    RUN_FORK_CACHE_REUSE_QUANT: (
        "forked trajectory reusing the selected quantized exact-prefix KV"
    ),
    RUN_MAIN_CACHE_SEED_NONE: (
        "main trajectory cold run that populates the unquantized baseline cache"
    ),
    RUN_FORK_CACHE_REUSE_NONE: (
        "forked trajectory reusing an unquantized exact-prefix KV baseline"
    ),
    RUN_FORK_COLD_REFERENCE: "forked trajectory cold reference with an empty cache",
}

PROFILE_RUN_TARGETS = (
    RUN_MAIN_CACHE_SEED_QUANT,
    RUN_FORK_CACHE_REUSE_QUANT,
    RUN_MAIN_CACHE_SEED_NONE,
    RUN_FORK_CACHE_REUSE_NONE,
    RUN_FORK_COLD_REFERENCE,
)
PROFILE_SCOPES = ("full_request", "create_runtime", "generation", "chunk")

T = TypeVar("T")

PROFILE_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "manager_encode_payload",
        "cacheseek/reuse/exact_prefix/manager.py",
        "_encode_kv_payload",
    ),
    (
        "manager_decode_payload",
        "cacheseek/reuse/exact_prefix/manager.py",
        "_decode_layer_payload",
    ),
    ("kivi_encode_layer", "cacheseek/quant/kivi.py", "encode_layer"),
    ("kivi_decode_layer", "cacheseek/quant/kivi.py", "decode_layer"),
    ("kivi_encode_tensor", "cacheseek/quant/kivi.py", "_encode_tensor"),
    ("kivi_decode_tensor", "cacheseek/quant/kivi.py", "_decode_tensor"),
    ("kivi_quantize_grouped", "cacheseek/quant/kivi.py", "_quantize_grouped"),
    ("kernel_quantize_grouped", "cacheseek/quant/kernel.py", "quantize_grouped"),
    ("kernel_dequantize_grouped", "cacheseek/quant/kernel.py", "dequantize_grouped"),
    ("kernel_pack_int4", "cacheseek/quant/kernel.py", "pack_int4_to_int32"),
    ("kernel_unpack_int4", "cacheseek/quant/kernel.py", "unpack_int4_from_int32"),
)


def _profile_path(path: str) -> str:
    return path.replace(os.sep, "/")


def _profile_target_label(filename: str, funcname: str) -> str | None:
    normalized = _profile_path(filename)
    for label, suffix, target_func in PROFILE_TARGETS:
        if normalized.endswith(suffix) and funcname == target_func:
            return label
    return None


def write_cprofile_outputs(
    profiler: cProfile.Profile,
    out_dir: Path,
    tag: str,
    *,
    sort_by: str,
    top_n: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"{tag}.prof"
    summary_path = out_dir / f"{tag}.txt"
    profiler.dump_stats(str(stats_path))

    stats = pstats.Stats(profiler)
    targets: list[dict[str, Any]] = []
    for (filename, line, funcname), raw in stats.stats.items():
        primitive_calls, total_calls, inline_s, cumulative_s, _callers = raw
        label = _profile_target_label(filename, funcname)
        if label is None:
            continue
        targets.append(
            {
                "label": label,
                "file": _profile_path(filename),
                "line": int(line),
                "function": funcname,
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "inline_s": float(inline_s),
                "cumulative_s": float(cumulative_s),
            }
        )
    targets.sort(key=lambda row: row["cumulative_s"], reverse=True)

    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# cProfile summary for request {tag}\n")
        fh.write(f"# sort={sort_by} top_n={top_n}\n\n")
        pstats.Stats(profiler, stream=fh).strip_dirs().sort_stats(sort_by).print_stats(top_n)

    return {
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "targets": targets,
    }


class RequestProfileController:
    """Profile exactly one semantic request stage.

    A controller is attached only to the separate diagnostic pass.  The normal
    benchmark passes use no controller, which keeps their timing free from
    profiler instrumentation.  ``run()`` can be nested: when a whole request or
    the generation loop is under ``torch.profiler``, nested stages are emitted as
    readable ``world_kv:*`` annotations instead of starting another profiler.
    """

    def __init__(
        self,
        *,
        kind: str,
        tag: str,
        scope: str,
        chunk_index: int,
        cprofile_dir: Path | None,
        torch_profile_dir: Path | None,
        sort_by: str,
        top_n: int,
        export_chrome_trace: bool,
        torch_detail: bool,
    ) -> None:
        self.kind = kind
        self.tag = tag
        self.scope = scope
        self.chunk_index = int(chunk_index)
        self.cprofile_dir = cprofile_dir
        self.torch_profile_dir = torch_profile_dir
        self.sort_by = sort_by
        self.top_n = max(int(top_n), 1)
        self.export_chrome_trace = bool(export_chrome_trace)
        self.torch_detail = bool(torch_detail)

        self._captured = False
        self._torch_active = False
        self._cprofiler = cProfile.Profile() if kind == "cprofile" else None
        self._torch_profiler: Any | None = None
        self._torch_activities: list[Any] = []

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    def _matches(self, stage: str, chunk_index: int | None) -> bool:
        """Return whether this stage is the single requested capture window."""

        if self._captured:
            return False
        if self.scope == "chunk":
            return stage == "chunk" and chunk_index == self.chunk_index
        return stage == self.scope

    @staticmethod
    def _stage_label(stage: str, chunk_index: int | None) -> str:
        if stage == "chunk" and chunk_index is not None:
            return f"world_kv:chunk:{chunk_index}"
        return f"world_kv:{stage}"

    def run(
        self,
        stage: str,
        fn: Callable[[], T],
        *,
        chunk_index: int | None = None,
    ) -> T:
        """Run ``fn`` and profile it only when it matches the configured scope."""

        label = self._stage_label(stage, chunk_index)

        # A parent torch-profiler scope is already collecting.  Emit nested
        # semantic ranges so Perfetto can collapse/search the KV-specific stages.
        if self._torch_active:
            with torch.profiler.record_function(label):
                return fn()

        if not self._matches(stage, chunk_index):
            return fn()

        self._captured = True
        if self.kind == "cprofile":
            assert self._cprofiler is not None
            self._cprofiler.enable()
            try:
                return fn()
            finally:
                self._cprofiler.disable()

        if self.kind == "torch":
            self._torch_activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                self._torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

            trace_handler = None
            run_trace_dir = self._torch_run_dir()
            if not self.export_chrome_trace:
                trace_handler = torch.profiler.tensorboard_trace_handler(str(run_trace_dir))

            # Shape, memory and Python-stack collection create a much denser
            # trace.  They are opt-in because the default use case is locating
            # KV lookup/decode/copy gaps rather than inspecting every operator.
            with torch.profiler.profile(
                activities=self._torch_activities,
                record_shapes=self.torch_detail,
                profile_memory=self.torch_detail,
                with_stack=self.torch_detail,
                on_trace_ready=trace_handler,
            ) as profiler:
                self._torch_profiler = profiler
                self._torch_active = True
                try:
                    with torch.profiler.record_function(label):
                        return fn()
                finally:
                    self._torch_active = False

        return fn()

    def _torch_run_dir(self) -> Path:
        assert self.torch_profile_dir is not None
        run_dir = self.torch_profile_dir / self.tag
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def finalize(self) -> dict[str, Any]:
        """Write profiler artifacts after the diagnostic request has completed."""

        base = {
            "kind": self.kind,
            "target": self.tag,
            "scope": self.scope,
            "chunk_index": self.chunk_index if self.scope == "chunk" else None,
            "captured": self._captured,
            "timing_valid_for_speedup": False,
        }
        if not self._captured:
            base["warning"] = (
                "The requested profile scope was not reached; for a chunk scope, "
                "the chunk may have been skipped by exact-prefix fast-forward."
            )
            return base

        artifact_tag = f"{self.tag}.{self.scope}"
        if self.scope == "chunk":
            artifact_tag += f"_{self.chunk_index}"

        if self.kind == "cprofile":
            assert self._cprofiler is not None
            assert self.cprofile_dir is not None
            outputs = write_cprofile_outputs(
                self._cprofiler,
                self.cprofile_dir,
                artifact_tag,
                sort_by=self.sort_by,
                top_n=self.top_n,
            )
            return {**base, **outputs}

        if self.kind == "torch":
            assert self._torch_profiler is not None
            run_dir = self._torch_run_dir()
            chrome_trace_path = None
            if self.export_chrome_trace:
                chrome_trace_path = run_dir / f"{artifact_tag}.chrome_trace.json"
                self._torch_profiler.export_chrome_trace(str(chrome_trace_path))
            return {
                **base,
                "trace_dir": str(run_dir),
                "chrome_trace_path": (
                    str(chrome_trace_path) if chrome_trace_path is not None else None
                ),
                "activities": [activity.name for activity in self._torch_activities],
                "detail": self.torch_detail,
            }

        return base


def run_profiled_request(
    controller: RequestProfileController,
    fn: Callable[[RequestProfileController], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one diagnostic request and return its result plus artifacts."""

    result = controller.run("full_request", lambda: fn(controller))
    return result, controller.finalize()


def parse_group_axis(axis: str) -> str | int:
    axis = axis.strip()
    if axis.lstrip("-").isdigit():
        return int(axis)
    return axis


def load_image(image_path: str) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    y, x = np.mgrid[0:ORIG_H, 0:ORIG_W]
    arr = np.stack(
        [
            (x / ORIG_W * 255).astype(np.uint8),
            (y / ORIG_H * 255).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    arr[180:300, 320:520] = (200, 60, 60)
    return Image.fromarray(arr)


def make_trajectory(frame_num: int, action_path: str) -> tuple[np.ndarray, np.ndarray]:
    if action_path:
        root = Path(action_path)
        poses = np.load(root / "poses.npy").astype(np.float32)[:frame_num]
        intrinsics = np.load(root / "intrinsics.npy").astype(np.float32)
        if intrinsics.ndim > 1:
            intrinsics = intrinsics[:frame_num]
        assert len(poses) >= frame_num, (
            f"poses.npy has only {len(poses)} frames < frame_num={frame_num}"
        )
        return poses, intrinsics

    poses = np.tile(np.eye(4, dtype=np.float32), (frame_num, 1, 1))
    for i in range(frame_num):
        poses[i, 2, 3] = 0.05 * i
    intrinsics = np.tile(
        np.array([400.0, 400.0, ORIG_W / 2, ORIG_H / 2], dtype=np.float32),
        (frame_num, 1),
    )
    return poses, intrinsics


def _apply_fork(poses: np.ndarray, fork_from: int) -> np.ndarray:
    """Apply a pure yaw rotation after fork_from frames.

    The prefix remains byte-identical at the control-key level. Avoid changing
    translation because LingBot normalizes camera translation over the whole
    trajectory, which would contaminate the prefix and correctly miss the cache.
    """

    forked = poses.copy()
    for i in range(fork_from, len(forked)):
        j = i - fork_from
        yaw = 0.03 * j
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=forked.dtype)
        forked[i, :3, :3] = rot @ forked[i, :3, :3]
    return forked


def make_trajectories(
    frame_num: int,
    prefix_chunks: int,
    action_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build byte-identical prefix controls followed by a yaw-only fork."""

    poses, intrinsics = make_trajectory(frame_num, action_path)
    fork_from = (
        prefix_chunks * KV_CHUNK_SIZE * TEMPORAL_DOWNSAMPLE
        + INITIAL_FRAME_OFFSET
    )
    if fork_from >= frame_num:
        raise ValueError(
            f"fork_from={fork_from} must be smaller than frame_num={frame_num}; "
            "increase --frame-num or reduce --prefix-chunks"
        )
    return poses, _apply_fork(poses, fork_from), intrinsics


def cache_stats(forest: NamespaceForest) -> dict[str, int]:
    """Summarize cache blobs reachable from the current namespace forest."""

    ready_nodes = 0
    ready_bytes = 0
    total_nodes = 0
    for namespace in forest.snapshot().get("namespaces", []):
        for node in namespace.get("nodes", []):
            total_nodes += 1
            blob = node.get("blob")
            if blob is not None:
                ready_nodes += 1
                ready_bytes += int(blob.get("nbytes", 0))
    return {
        "ready_blob_nodes": ready_nodes,
        "ready_blob_bytes": ready_bytes,
        "total_nodes": total_nodes,
    }


def _cuda_synchronize() -> None:
    """Synchronize only at coarse phase boundaries used for wall-clock timing."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_profile_stage(
    controller: RequestProfileController | None,
    stage: str,
    fn: Callable[[], T],
    *,
    chunk_index: int | None = None,
) -> T:
    """Dispatch a stage through the optional diagnostic profiler controller."""

    if controller is None:
        return fn()
    return controller.run(stage, fn, chunk_index=chunk_index)


def run_request(
    pipeline: Any,
    binding: LingBotWorldKVBinding,
    forest: NamespaceForest,
    *,
    prompt: str,
    seed: int,
    frame_num: int,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    image_path: str,
    tag: str,
    profile_controller: RequestProfileController | None = None,
) -> dict[str, Any]:
    """Execute one request and retain generated frames by absolute chunk index.

    ``create_runtime_s`` includes exact-prefix lookup/materialization and any
    runtime initialization.  ``generation_s`` covers only generated suffix
    chunks.  ``flush_s`` is kept separate because asynchronous stores may defer
    persistence beyond the online generation path.  CUDA is synchronized only
    at these coarse boundaries; per-chunk timing remains host-observed so the
    benchmark does not introduce a synchronization after every chunk.
    """

    cache_before = cache_stats(forest)
    session = LingBotWorldFastSessionConfig(
        prompt=prompt,
        image=load_image(image_path),
        control_mode="cam",
        frame_num=frame_num,
        seed=seed,
        sample_shift=10,    # Set 10 for better quality
        poses=poses,
        intrinsics=intrinsics,
        world_kv_binding=binding,
    )

    _cuda_synchronize()
    request_start = time.perf_counter()

    create_start = time.perf_counter()
    runtime = _run_profile_stage(
        profile_controller,
        "create_runtime",
        lambda: pipeline.create_runtime(session),
    )
    _cuda_synchronize()
    create_runtime_s = time.perf_counter() - create_start

    # A reuse request may begin at ``prefix_chunks`` rather than zero.  Store
    # this initial absolute index before generation mutates the runtime cursor.
    initial_chunk_index = int(runtime.current_chunk_index)
    frames: list[Image.Image] = []
    frames_by_chunk: dict[int, list[Image.Image]] = {}
    chunk_host_times_s: dict[int, float] = {}

    def _generate_all_chunks() -> None:
        while runtime.active:
            # Capture the cursor before ``generate_next_chunk`` advances it.
            # This is the absolute chunk id used to align reuse and cold output.
            chunk_index = int(runtime.current_chunk_index)
            chunk_start = time.perf_counter()
            chunk_frames = list(
                _run_profile_stage(
                    profile_controller,
                    "chunk",
                    lambda: pipeline.generate_next_chunk(runtime),
                    chunk_index=chunk_index,
                )
            )
            chunk_host_times_s[chunk_index] = time.perf_counter() - chunk_start
            frames_by_chunk[chunk_index] = chunk_frames
            frames.extend(chunk_frames)

    generation_start = time.perf_counter()
    _run_profile_stage(
        profile_controller,
        "generation",
        _generate_all_chunks,
    )
    _cuda_synchronize()
    generation_s = time.perf_counter() - generation_start

    store = binding.mgr.store

    def _flush_store() -> None:
        if hasattr(store, "flush"):
            store.flush()

    flush_start = time.perf_counter()
    _run_profile_stage(profile_controller, "store_flush", _flush_store)
    flush_s = time.perf_counter() - flush_start
    total_with_flush_s = time.perf_counter() - request_start
    cache_after = cache_stats(forest)

    return {
        "tag": tag,
        "frames": frames,
        "frames_by_chunk": frames_by_chunk,
        "fast_forward_k": binding.last_fast_forward,
        "initial_chunk_index": initial_chunk_index,
        "final_chunk_index": int(runtime.current_chunk_index),
        "generated_chunk_indices": sorted(frames_by_chunk),
        "generated_chunk_frame_counts": {
            str(chunk_index): len(chunk_frames)
            for chunk_index, chunk_frames in sorted(frames_by_chunk.items())
        },
        "n_generated_chunks": len(frames_by_chunk),
        "n_frames": len(frames),
        "create_runtime_s": create_runtime_s,
        "generation_s": generation_s,
        "flush_s": flush_s,
        "total_without_flush_s": create_runtime_s + generation_s,
        "total_with_flush_s": total_with_flush_s,
        "timing_valid_for_speedup": profile_controller is None,
        "chunk_host_times_s": {
            str(chunk_index): duration
            for chunk_index, duration in sorted(chunk_host_times_s.items())
        },
        "cache": {
            "before": cache_before,
            "after": cache_after,
            "delta_ready_blob_bytes": (
                cache_after["ready_blob_bytes"] - cache_before["ready_blob_bytes"]
            ),
        },
        "frame_sha256_by_chunk": {
            str(chunk_index): [
                hashlib.sha256(np.asarray(frame).tobytes()).hexdigest()
                for frame in chunk_frames
            ]
            for chunk_index, chunk_frames in sorted(frames_by_chunk.items())
        },
    }


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def save_video(frames: list[Image.Image], path: Path, fps: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        return {"path": str(save_gif(frames, path, fps)), "format": "gif"}
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames], axis=0)
    try:
        try:
            import imageio.v3 as iio

            iio.imwrite(path, arr, fps=fps, codec="libx264", quality=8)
        except TypeError:
            import imageio.v3 as iio

            iio.imwrite(path, arr, fps=fps)
        return {"path": str(path), "format": path.suffix.lstrip(".").lower()}
    except Exception as exc:  # pragma: no cover - optional codec dependent
        fallback = path.with_suffix(".gif")
        save_gif(frames, fallback, fps)
        return {
            "path": str(fallback),
            "format": "gif",
            "requested_path": str(path),
            "warning": f"MP4 save failed; wrote GIF fallback: {type(exc).__name__}: {exc}",
        }


def save_frames_by_chunk(
    frames_by_chunk: dict[int, list[Image.Image]],
    root: Path,
) -> None:
    """Save frames with absolute chunk ids encoded in their file names."""

    root.mkdir(parents=True, exist_ok=True)
    for chunk_index, chunk_frames in sorted(frames_by_chunk.items()):
        for frame_index, frame in enumerate(chunk_frames):
            frame.save(
                root / f"chunk_{chunk_index:04d}_frame_{frame_index:04d}.jpg",
                quality=92,
            )


def frame_metrics(a: list[Image.Image], b: list[Image.Image]) -> dict[str, Any]:
    """Compare two frame lists without silently hiding a length mismatch.

    Metrics are computed over the common prefix only, but both original lengths
    and ``frame_count_match`` are returned.  Callers therefore cannot mistake a
    truncated comparison for a complete one.  ``global_psnr`` is derived from
    the aggregate MSE, while ``finite_psnr_*`` excludes exactly equal frames
    whose mathematical PSNR is infinite.
    """

    n = min(len(a), len(b))
    base = {
        "frames_a": len(a),
        "frames_b": len(b),
        "frame_count_match": len(a) == len(b),
        "n_compared_frames": n,
    }
    if n == 0:
        return {
            **base,
            "byte_equal_frames": 0,
            "first_different_frame": None,
            "mae_mean": None,
            "mae_max": None,
            "mse_mean": None,
            "mse_max": None,
            "global_psnr": None,
            "finite_psnr_mean": None,
            "finite_psnr_min": None,
        }

    maes: list[float] = []
    mses: list[float] = []
    finite_psnrs: list[float] = []
    equal = 0
    first_diff: int | None = None
    squared_error_sum = 0.0
    pixel_count = 0

    for i in range(n):
        aa = np.asarray(a[i].convert("RGB"), dtype=np.float32)
        bb = np.asarray(b[i].convert("RGB"), dtype=np.float32)
        if aa.shape != bb.shape:
            raise ValueError(
                f"frame shape mismatch at index {i}: {aa.shape} != {bb.shape}"
            )

        diff = aa - bb
        absolute = np.abs(diff)
        squared = diff * diff
        mae = float(np.mean(absolute))
        mse = float(np.mean(squared))
        maes.append(mae)
        mses.append(mse)
        squared_error_sum += float(np.sum(squared, dtype=np.float64))
        pixel_count += int(squared.size)

        if mse == 0:
            equal += 1
        else:
            finite_psnrs.append(float(20 * math.log10(255.0 / math.sqrt(mse))))
            if first_diff is None:
                first_diff = i

    global_mse = squared_error_sum / pixel_count
    global_psnr = (
        None
        if global_mse == 0
        else float(20 * math.log10(255.0 / math.sqrt(global_mse)))
    )
    return {
        **base,
        "byte_equal_frames": equal,
        "first_different_frame": first_diff,
        "mae_mean": float(np.mean(maes)),
        "mae_max": float(np.max(maes)),
        "mse_mean": float(np.mean(mses)),
        "mse_max": float(np.max(mses)),
        "global_psnr": global_psnr,
        "finite_psnr_mean": (
            None if not finite_psnrs else float(np.mean(finite_psnrs))
        ),
        "finite_psnr_min": (
            None if not finite_psnrs else float(np.min(finite_psnrs))
        ),
    }


def aligned_chunk_metrics(
    a_by_chunk: dict[int, list[Image.Image]],
    b_by_chunk: dict[int, list[Image.Image]],
) -> dict[str, Any]:
    """Compare only matching absolute chunks from two generation requests.

    Exact-prefix reuse skips already materialized chunks.  Consequently, list
    position zero in a reuse result is not necessarily absolute chunk zero.  By
    intersecting the runtime chunk ids first, this function compares like with
    like and also exposes missing chunks and per-chunk frame-count mismatches.
    """

    chunks_a = sorted(a_by_chunk)
    chunks_b = sorted(b_by_chunk)
    common_chunks = sorted(set(chunks_a) & set(chunks_b))
    only_a = sorted(set(chunks_a) - set(chunks_b))
    only_b = sorted(set(chunks_b) - set(chunks_a))

    per_chunk = {
        str(chunk_index): frame_metrics(
            a_by_chunk[chunk_index],
            b_by_chunk[chunk_index],
        )
        for chunk_index in common_chunks
    }
    frame_count_match_per_chunk = all(
        metrics["frame_count_match"] for metrics in per_chunk.values()
    )

    # Flatten only common chunks, preserving absolute chunk order, to provide a
    # single overall quality number in addition to the diagnostic per-chunk view.
    aligned_a: list[Image.Image] = []
    aligned_b: list[Image.Image] = []
    for chunk_index in common_chunks:
        # Truncate independently inside each absolute chunk.  Flattening the
        # complete unequal lists would shift every later chunk by one frame and
        # recreate the same temporal-alignment bug this function is meant to
        # prevent.
        n = min(
            len(a_by_chunk[chunk_index]),
            len(b_by_chunk[chunk_index]),
        )
        aligned_a.extend(a_by_chunk[chunk_index][:n])
        aligned_b.extend(b_by_chunk[chunk_index][:n])
    return {
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "common_chunks": common_chunks,
        "chunks_only_in_a": only_a,
        "chunks_only_in_b": only_b,
        "has_common_chunks": bool(common_chunks),
        "frame_count_match_per_chunk": frame_count_match_per_chunk,
        "overall": frame_metrics(aligned_a, aligned_b),
        "per_chunk": per_chunk,
    }


def strip_frames(result: dict[str, Any]) -> dict[str, Any]:
    """Remove PIL objects while retaining all JSON-serializable request data."""

    return {
        key: value
        for key, value in result.items()
        if key not in {"frames", "frames_by_chunk"}
    }



def make_store(args: argparse.Namespace, store_namespace: str) -> Any:
    """Create an isolated store for one benchmark branch.

    In-memory managers are naturally isolated.  Local-disk managers receive a
    unique child directory so an earlier script invocation cannot accidentally
    make a nominally cold reference observe stale physical objects.  Fluxon is
    externally configured; lookup isolation still comes from a fresh forest and
    the WorldKV namespace/config fingerprint.
    """

    if args.store == "fluxon":
        from cacheseek.stores import TensorStoreTierStore
        from cacheseek.stores.fluxon import FluxonKVStore

        assert args.fluxon_config, "--store fluxon requires --fluxon-config"
        return TensorStoreTierStore(FluxonKVStore(config_path=args.fluxon_config), async_put=True)
    if args.store == "localdisk":
        from cacheseek.stores import TensorStoreTierStore
        from cacheseek.stores.tier import LocalDiskTensorStore

        disk_root = Path(args.disk_root) / store_namespace
        return TensorStoreTierStore(LocalDiskTensorStore(str(disk_root)), async_put=True)
    return InMemoryTierStore()


def build_manager(
    args: argparse.Namespace,
    cfg: LingBotWorldFastPipelineConfig,
    key_group_axis: str | int,
    value_group_axis: str | int,
    *,
    quant: str,
    store_namespace: str,
) -> tuple[NamespaceForest, WorldKVManager]:
    """Build one exact-prefix manager for a fixed quantization branch."""

    forest = NamespaceForest()
    mgr = WorldKVManager(
        forest,
        make_store(args, store_namespace),
        make_world_kv_config(
            local_attn_size=cfg.local_attn_size,
            sink_size=cfg.sink_size,
            chunk_size=KV_CHUNK_SIZE,
            quant=quant,
            group_size=args.group_size,
            kv_layout=args.kv_layout,
            key_group_axis=key_group_axis,
            value_group_axis=value_group_axis,
            scale_dtype=args.scale_dtype,
            offset_dtype=args.offset_dtype,
        ),
    )
    return forest, mgr


def timing_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare one faster-expected request against a reference request.

    ``speedup`` is reported as ``reference / candidate``; values above one mean
    the candidate is faster.  The profiler never feeds this function because all
    entries supplied by the benchmark suite are collected without instrumentation.
    """

    phases = (
        "create_runtime_s",
        "generation_s",
        "flush_s",
        "total_without_flush_s",
        "total_with_flush_s",
    )
    comparisons: dict[str, Any] = {}
    for phase in phases:
        candidate_s = float(candidate[phase])
        reference_s = float(reference[phase])
        comparisons[phase] = {
            "candidate_s": candidate_s,
            "reference_s": reference_s,
            "saved_s": reference_s - candidate_s,
            "speedup": None if candidate_s <= 0 else reference_s / candidate_s,
        }
    return {
        "candidate": candidate["tag"],
        "reference": reference["tag"],
        "timing_valid_for_speedup": bool(
            candidate["timing_valid_for_speedup"]
            and reference["timing_valid_for_speedup"]
        ),
        "phases": comparisons,
    }


def main() -> int:
    """Run the unprofiled benchmark, then optional isolated diagnostic passes."""

    ap = argparse.ArgumentParser(
        description=(
            "Benchmark selected KIVI exact-prefix reuse, an automatic unquantized "
            "reuse baseline, and a cold fork reference."
        )
    )
    ap.add_argument(
        "--quant",
        choices=["kivi_int4", "kivi_int8"],
        required=True,
        help=(
            "Selected quantized mode. The unquantized 'none' branch is always "
            "run automatically as the reuse baseline."
        ),
    )
    ap.add_argument("--frame-num", type=int, default=37)
    ap.add_argument("--prefix-chunks", type=int, default=2)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--video-format", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--image-path", type=str, default="")
    ap.add_argument("--action-path", type=str, default="")
    ap.add_argument(
        "--store",
        type=str,
        default="inmem",
        choices=["inmem", "localdisk", "fluxon"],
    )
    ap.add_argument("--fluxon-config", type=str, default="")
    ap.add_argument("--disk-root", type=str, default="")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--kv-layout", type=str, default="B,T,H,D")
    ap.add_argument("--key-group-axis", type=str, default="T")
    ap.add_argument("--value-group-axis", type=str, default="D")
    ap.add_argument("--scale-dtype", type=str, default="float32")
    ap.add_argument("--offset-dtype", type=str, default="float32")
    ap.add_argument("--aux-device", type=str, default="")

    # cProfile and torch.profiler are diagnostic-only.  Both run after the
    # benchmark on fresh managers, so the benchmark's latency remains usable.
    ap.add_argument(
        "--profile",
        choices=["none", "cprofile"],
        default="none",
        help="Run a separate cProfile diagnostic pass after the benchmark.",
    )
    ap.add_argument(
        "--profile-target",
        choices=[*PROFILE_RUN_TARGETS, "all"],
        default=RUN_FORK_CACHE_REUSE_QUANT,
        help="Semantic request to rerun under cProfile.",
    )
    ap.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="Directory for .prof/.txt artifacts; defaults to <out-dir>/profiles.",
    )
    ap.add_argument(
        "--profile-sort",
        choices=["cumulative", "time", "calls"],
        default="cumulative",
        help="Sort key for generated cProfile text summaries.",
    )
    ap.add_argument("--profile-top-n", type=int, default=80)
    ap.add_argument(
        "--profile-scope",
        choices=PROFILE_SCOPES,
        default="create_runtime",
        help=(
            "Capture only one stage. create_runtime is the recommended scope for "
            "exact-prefix lookup/materialize/decode analysis."
        ),
    )
    ap.add_argument(
        "--profile-chunk-index",
        type=int,
        default=-1,
        help=(
            "Absolute runtime chunk to capture when --profile-scope=chunk. "
            "Default: first generated chunk for the selected target."
        ),
    )
    ap.add_argument(
        "--torch-profile",
        choices=["none", *PROFILE_RUN_TARGETS, "all"],
        default="none",
        help="Run a separate torch.profiler diagnostic pass after the benchmark.",
    )
    ap.add_argument(
        "--torch-profile-dir",
        type=str,
        default="",
        help="Trace directory; defaults to <out-dir>/torch_traces.",
    )
    ap.add_argument(
        "--torch-profile-chrome-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Export Chrome/Perfetto JSON. Disable to use the TensorBoard trace "
            "handler instead."
        ),
    )
    ap.add_argument(
        "--torch-profile-detail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect shapes, memory events and Python stacks. Disabled by default "
            "to keep Chrome/Perfetto traces readable."
        ),
    )
    args = ap.parse_args()

    if args.profile != "none" and args.torch_profile != "none":
        ap.error(
            "cProfile and torch.profiler must be run separately; enabling both "
            "would distort the diagnostic trace and duplicate expensive passes"
        )
    if args.frame_num <= 0:
        ap.error("--frame-num must be positive")
    if args.prefix_chunks < 0:
        ap.error("--prefix-chunks must be non-negative")
    if args.group_size <= 0:
        ap.error("--group-size must be positive")

    if not args.out_dir:
        args.out_dir = f"/tmp/worldkv_quant_benchmark_{args.quant}"
    if not args.disk_root:
        args.disk_root = f"/tmp/worldkv_quant_benchmark_diskstore_{args.quant}"

    ckpt = os.environ.get("LINGBOT_WORLD_CHECKPOINT_DIR", "")
    assert ckpt, "Set LINGBOT_WORLD_CHECKPOINT_DIR"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = Path(args.profile_dir) if args.profile_dir else out / "profiles"
    torch_profile_dir = (
        Path(args.torch_profile_dir)
        if args.torch_profile_dir
        else out / "torch_traces"
    )

    # Initialize the heavyweight pipeline once.  Cache managers remain separate
    # per branch, which isolates cache state without reloading model weights.
    pipeline = LingBotWorldFastPipeline(device="cuda")
    cfg = LingBotWorldFastPipelineConfig(checkpoint_dir=ckpt)
    if args.aux_device:
        from telefuser.core.config import ModelRuntimeConfig

        _t, _i = (args.aux_device.split(":") + ["0"])[:2]
        aux = dict(device_type=_t, device_id=int(_i))
        cfg.text_encoding_config = ModelRuntimeConfig(**aux)
        cfg.vae_config = ModelRuntimeConfig(**aux)
        print(
            f"[2-gpu] text-encoder + VAE -> {args.aux_device}; DiT -> cuda:0",
            flush=True,
        )
    pipeline.init(None, cfg)

    poses_main, poses_fork, intrinsics = make_trajectories(
        args.frame_num,
        args.prefix_chunks,
        args.action_path,
    )
    fork_from_frame = (
        args.prefix_chunks * KV_CHUNK_SIZE * TEMPORAL_DOWNSAMPLE
        + INITIAL_FRAME_OFFSET
    )
    key_group_axis = parse_group_axis(args.key_group_axis)
    value_group_axis = parse_group_axis(args.value_group_axis)

    # The run id creates fresh local-disk child directories without deleting any
    # user-provided root.  Managers that must share cache state reuse one manager.
    run_id = f"pid{os.getpid()}_{time.time_ns()}"

    def _build_branch(
        *,
        quant: str,
        branch: str,
    ) -> tuple[NamespaceForest, WorldKVManager]:
        return build_manager(
            args,
            cfg,
            key_group_axis,
            value_group_axis,
            quant=quant,
            store_namespace=f"{run_id}_{branch}",
        )

    def _execute(
        *,
        tag: str,
        mgr: WorldKVManager,
        forest: NamespaceForest,
        poses: np.ndarray,
        controller: RequestProfileController | None = None,
    ) -> dict[str, Any]:
        # Bindings are per request so ``last_fast_forward`` cannot leak from a
        # previous trajectory while the manager/forest cache remains shared.
        return run_request(
            pipeline,
            LingBotWorldKVBinding(mgr, forest),
            forest,
            prompt=args.prompt,
            seed=args.seed,
            frame_num=args.frame_num,
            poses=poses,
            intrinsics=intrinsics,
            image_path=args.image_path,
            tag=tag,
            profile_controller=controller,
        )

    # ------------------------------------------------------------------
    # Unprofiled benchmark suite
    # ------------------------------------------------------------------
    # Selected KIVI branch: seed the main path, then reuse its exact prefix on
    # the forked path.
    forest_quant, mgr_quant = _build_branch(
        quant=args.quant,
        branch="benchmark_quant",
    )
    main_cache_seed_quant = _execute(
        tag=RUN_MAIN_CACHE_SEED_QUANT,
        mgr=mgr_quant,
        forest=forest_quant,
        poses=poses_main,
    )
    fork_cache_reuse_quant = _execute(
        tag=RUN_FORK_CACHE_REUSE_QUANT,
        mgr=mgr_quant,
        forest=forest_quant,
        poses=poses_fork,
    )

    # Automatic unquantized baseline.  It has its own cache so selected KIVI
    # payloads can never be mistaken for the FP/no-quant representation.
    forest_none, mgr_none = _build_branch(
        quant="none",
        branch="benchmark_none",
    )
    main_cache_seed_none = _execute(
        tag=RUN_MAIN_CACHE_SEED_NONE,
        mgr=mgr_none,
        forest=forest_none,
        poses=poses_main,
    )
    fork_cache_reuse_none = _execute(
        tag=RUN_FORK_CACHE_REUSE_NONE,
        mgr=mgr_none,
        forest=forest_none,
        poses=poses_fork,
    )

    # A fresh manager and empty forest guarantee no exact-prefix match.  The
    # manager uses quant=none because this request does not consume cached KV;
    # any cache written during the cold run is irrelevant to the comparison.
    forest_ref, mgr_ref = _build_branch(
        quant="none",
        branch="benchmark_cold_reference",
    )
    fork_cold_reference = _execute(
        tag=RUN_FORK_COLD_REFERENCE,
        mgr=mgr_ref,
        forest=forest_ref,
        poses=poses_fork,
    )

    suffix = "." + args.video_format
    videos = {
        RUN_FORK_CACHE_REUSE_QUANT: save_video(
            fork_cache_reuse_quant["frames"],
            out / (RUN_FORK_CACHE_REUSE_QUANT + suffix),
            args.fps,
        ),
        RUN_FORK_CACHE_REUSE_NONE: save_video(
            fork_cache_reuse_none["frames"],
            out / (RUN_FORK_CACHE_REUSE_NONE + suffix),
            args.fps,
        ),
        RUN_FORK_COLD_REFERENCE: save_video(
            fork_cold_reference["frames"],
            out / (RUN_FORK_COLD_REFERENCE + suffix),
            args.fps,
        ),
    }
    if args.save_frames:
        save_frames_by_chunk(
            fork_cache_reuse_quant["frames_by_chunk"],
            out / "frames" / RUN_FORK_CACHE_REUSE_QUANT,
        )
        save_frames_by_chunk(
            fork_cache_reuse_none["frames_by_chunk"],
            out / "frames" / RUN_FORK_CACHE_REUSE_NONE,
        )
        save_frames_by_chunk(
            fork_cold_reference["frames_by_chunk"],
            out / "frames" / RUN_FORK_COLD_REFERENCE,
        )

    # Quality comparisons answer three different questions:
    #   selected vs cold: end-to-end effect of quantized reuse;
    #   none vs cold: baseline effect of exact reuse without quantization;
    #   selected vs none: incremental effect attributable to KIVI quantization.
    quant_vs_cold = aligned_chunk_metrics(
        fork_cache_reuse_quant["frames_by_chunk"],
        fork_cold_reference["frames_by_chunk"],
    )
    none_vs_cold = aligned_chunk_metrics(
        fork_cache_reuse_none["frames_by_chunk"],
        fork_cold_reference["frames_by_chunk"],
    )
    quant_vs_none = aligned_chunk_metrics(
        fork_cache_reuse_quant["frames_by_chunk"],
        fork_cache_reuse_none["frames_by_chunk"],
    )
    metrics = {
        "quant_reuse_vs_cold_reference": quant_vs_cold,
        "unquantized_reuse_vs_cold_reference": none_vs_cold,
        "quant_reuse_vs_unquantized_reuse": quant_vs_none,
    }

    performance = {
        "quant_reuse_vs_cold_reference": timing_comparison(
            fork_cache_reuse_quant,
            fork_cold_reference,
        ),
        "unquantized_reuse_vs_cold_reference": timing_comparison(
            fork_cache_reuse_none,
            fork_cold_reference,
        ),
        "quant_reuse_vs_unquantized_reuse": timing_comparison(
            fork_cache_reuse_quant,
            fork_cache_reuse_none,
        ),
    }

    checks = {
        "trajectory_prefix_is_identical": bool(
            np.array_equal(
                poses_main[:fork_from_frame],
                poses_fork[:fork_from_frame],
            )
        ),
        "trajectory_suffix_is_forked": bool(
            not np.array_equal(
                poses_main[fork_from_frame:],
                poses_fork[fork_from_frame:],
            )
        ),
        "main_cache_seed_quant_is_cold": (
            main_cache_seed_quant["fast_forward_k"] == 0
        ),
        "fork_cache_reuse_quant_prefix_hit": (
            fork_cache_reuse_quant["fast_forward_k"] == args.prefix_chunks
        ),
        "main_cache_seed_none_is_cold": (
            main_cache_seed_none["fast_forward_k"] == 0
        ),
        "fork_cache_reuse_none_prefix_hit": (
            fork_cache_reuse_none["fast_forward_k"] == args.prefix_chunks
        ),
        "fork_cold_reference_is_cold": (
            fork_cold_reference["fast_forward_k"] == 0
        ),
        "quant_vs_cold_has_common_chunks": quant_vs_cold["has_common_chunks"],
        "quant_vs_cold_frame_counts_match": (
            quant_vs_cold["frame_count_match_per_chunk"]
        ),
        "none_vs_cold_has_common_chunks": none_vs_cold["has_common_chunks"],
        "none_vs_cold_frame_counts_match": (
            none_vs_cold["frame_count_match_per_chunk"]
        ),
        "quant_vs_none_has_common_chunks": quant_vs_none["has_common_chunks"],
        "quant_vs_none_frame_counts_match": (
            quant_vs_none["frame_count_match_per_chunk"]
        ),
        "benchmark_timings_are_unprofiled": all(
            result["timing_valid_for_speedup"]
            for result in (
                main_cache_seed_quant,
                fork_cache_reuse_quant,
                main_cache_seed_none,
                fork_cache_reuse_none,
                fork_cold_reference,
            )
        ),
    }

    # ------------------------------------------------------------------
    # Optional diagnostic passes
    # ------------------------------------------------------------------
    # Every selected target gets a fresh manager.  Reuse targets receive an
    # unprofiled seed request first; only the requested target/scope is captured.
    profile_kind = "none"
    requested_profile_target = "none"
    if args.profile == "cprofile":
        profile_kind = "cprofile"
        requested_profile_target = args.profile_target
    elif args.torch_profile != "none":
        profile_kind = "torch"
        requested_profile_target = args.torch_profile

    profile_targets = (
        list(PROFILE_RUN_TARGETS)
        if requested_profile_target == "all"
        else (
            []
            if requested_profile_target == "none"
            else [requested_profile_target]
        )
    )
    profile_requests: dict[str, Any] = {}

    def _default_profile_chunk_index(target: str) -> int:
        if args.profile_chunk_index >= 0:
            return args.profile_chunk_index
        if target in {RUN_FORK_CACHE_REUSE_QUANT, RUN_FORK_CACHE_REUSE_NONE}:
            return args.prefix_chunks
        return 0

    for profile_target in profile_targets:
        target_quant = (
            args.quant
            if profile_target
            in {RUN_MAIN_CACHE_SEED_QUANT, RUN_FORK_CACHE_REUSE_QUANT}
            else "none"
        )
        profile_forest, profile_mgr = _build_branch(
            quant=target_quant,
            branch=f"profile_{profile_kind}_{profile_target}",
        )

        # Seed only when profiling a reuse request.  The seed remains outside
        # every profiler scope, including full_request.
        if profile_target == RUN_FORK_CACHE_REUSE_QUANT:
            _execute(
                tag=f"profile_setup_{RUN_MAIN_CACHE_SEED_QUANT}",
                mgr=profile_mgr,
                forest=profile_forest,
                poses=poses_main,
            )
        elif profile_target == RUN_FORK_CACHE_REUSE_NONE:
            _execute(
                tag=f"profile_setup_{RUN_MAIN_CACHE_SEED_NONE}",
                mgr=profile_mgr,
                forest=profile_forest,
                poses=poses_main,
            )

        target_poses = (
            poses_main
            if profile_target
            in {RUN_MAIN_CACHE_SEED_QUANT, RUN_MAIN_CACHE_SEED_NONE}
            else poses_fork
        )
        controller = RequestProfileController(
            kind=profile_kind,
            tag=profile_target,
            scope=args.profile_scope,
            chunk_index=_default_profile_chunk_index(profile_target),
            cprofile_dir=profile_dir if profile_kind == "cprofile" else None,
            torch_profile_dir=(
                torch_profile_dir if profile_kind == "torch" else None
            ),
            sort_by=args.profile_sort,
            top_n=args.profile_top_n,
            export_chrome_trace=args.torch_profile_chrome_trace,
            torch_detail=args.torch_profile_detail,
        )
        profiled_result, profile_artifacts = run_profiled_request(
            controller,
            lambda active_controller, target=profile_target, poses=target_poses: _execute(
                tag=target,
                mgr=profile_mgr,
                forest=profile_forest,
                poses=poses,
                controller=active_controller,
            ),
        )
        profile_requests[profile_target] = {
            "artifacts": profile_artifacts,
            "request_summary": {
                "fast_forward_k": profiled_result["fast_forward_k"],
                "initial_chunk_index": profiled_result["initial_chunk_index"],
                "generated_chunk_indices": profiled_result[
                    "generated_chunk_indices"
                ],
                "timing_valid_for_speedup": profiled_result[
                    "timing_valid_for_speedup"
                ],
            },
        }

    cprofile_manifest = {
        "enabled": profile_kind == "cprofile",
        "target": args.profile_target if args.profile == "cprofile" else "none",
        "scope": args.profile_scope,
        "output_dir": str(profile_dir) if profile_kind == "cprofile" else None,
        "sort_by": args.profile_sort,
        "top_n": max(int(args.profile_top_n), 1),
        "requests": profile_requests if profile_kind == "cprofile" else {},
        "separate_from_benchmark": True,
    }
    torch_profile_manifest = {
        "enabled": profile_kind == "torch",
        "target": args.torch_profile,
        "scope": args.profile_scope,
        "output_dir": (
            str(torch_profile_dir) if profile_kind == "torch" else None
        ),
        "chrome_trace": bool(args.torch_profile_chrome_trace),
        "detail": bool(args.torch_profile_detail),
        "requests": profile_requests if profile_kind == "torch" else {},
        "separate_from_benchmark": True,
    }

    results = {
        RUN_MAIN_CACHE_SEED_QUANT: strip_frames(main_cache_seed_quant),
        RUN_FORK_CACHE_REUSE_QUANT: strip_frames(fork_cache_reuse_quant),
        RUN_MAIN_CACHE_SEED_NONE: strip_frames(main_cache_seed_none),
        RUN_FORK_CACHE_REUSE_NONE: strip_frames(fork_cache_reuse_none),
        RUN_FORK_COLD_REFERENCE: strip_frames(fork_cold_reference),
    }
    manifest = {
        "selected_quant": args.quant,
        "automatic_baseline_quant": "none",
        "frame_num": args.frame_num,
        "prefix_chunks": args.prefix_chunks,
        "fork_from_frame": fork_from_frame,
        "prompt": args.prompt,
        "seed": args.seed,
        "store": args.store,
        "run_descriptions": RUN_DESCRIPTIONS,
        "benchmark": {
            "profiled": False,
            "timing_valid_for_speedup": True,
            "repeated_trials": False,
        },
        "pipeline": {
            "checkpoint_dir": ckpt,
            "local_attn_size": cfg.local_attn_size,
            "sink_size": cfg.sink_size,
            "kv_chunk_size": KV_CHUNK_SIZE,
        },
        "world_kv_quant": {
            "selected_quant": args.quant,
            "baseline_quant": "none",
            "group_size": args.group_size,
            "kv_layout": args.kv_layout,
            "key_group_axis": key_group_axis,
            "value_group_axis": value_group_axis,
            "scale_dtype": args.scale_dtype,
            "offset_dtype": args.offset_dtype,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cuda_unavailable"
            ),
        },
        "videos": videos,
        "profile": cprofile_manifest,
        "torch_profile": torch_profile_manifest,
        "results": results,
        "quality_metrics": metrics,
        "performance": performance,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "all_checks_pass": manifest["all_checks_pass"],
                "selected_quant": args.quant,
                "timing_s": {
                    tag: {
                        "create_runtime": result["create_runtime_s"],
                        "generation": result["generation_s"],
                        "flush": result["flush_s"],
                        "total_without_flush": result["total_without_flush_s"],
                        "total_with_flush": result["total_with_flush_s"],
                    }
                    for tag, result in results.items()
                },
                "speedup": {
                    name: comparison["phases"]["total_without_flush_s"][
                        "speedup"
                    ]
                    for name, comparison in performance.items()
                },
                "profile": cprofile_manifest,
                "torch_profile": torch_profile_manifest,
                "checks": checks,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if manifest["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())