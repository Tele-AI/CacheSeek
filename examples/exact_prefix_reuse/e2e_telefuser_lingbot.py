# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
# ruff: noqa: E402  (imports intentionally follow the sys.path/editable-finder shim)
"""exact_prefix x TeleFuser e2e: replay/fork consistency of exact-prefix fast-forward.

Single process, four requests (no service; forest/store shared in-process):
  A  cold full run                              -> assert K=0
  B  request identical to A                     -> assert K=all chunks (decode-only),
                                                    video byte-exact to A
  C  shares first prefix_chunks with A, then forks the camera trajectory
                                                -> assert K=prefix
  D  same request as C but with an EMPTY cache  -> assert C and D videos are byte-exact
                                                    (real-model final check of RNG
                                                    alignment + ring assembly)

Usage (remote):
  LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/ckpt CUDA_VISIBLE_DEVICES=2 \
  python examples/exact_prefix_reuse/e2e_telefuser_lingbot.py --frame-num 37 --prefix-chunks 2 \
      --out-dir /tmp/worldkv_e2e [--no-save-videos]

Output: manifest.json under --out-dir (per-request K / chunk count / per-frame sha256 /
assertion results). Exit 0 if all assertions pass, exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

# --- editable-finder shim: run before importing telefuser/cacheseek ---
# In a shared venv both are PEP660 editable installs whose meta_path finder takes
# priority over PYTHONPATH, so without stripping it we always import the old checkout.
# Set WORLDKV_REPO_ROOTS (colon-separated repo roots) to enable: strip the finder,
# prepend to sys.path, and self-check after import.
_WORLDKV_ROOTS = [r for r in os.environ.get("WORLDKV_REPO_ROOTS", "").split(":") if r]
if _WORLDKV_ROOTS:
    sys.meta_path = [
        f for f in sys.meta_path
        if not any(tag in getattr(f, "__module__", "") for tag in ("__editable___telefuser", "__editable___cacheseek"))
    ]
    for _r in reversed(_WORLDKV_ROOTS):
        if _r not in sys.path:
            sys.path.insert(0, _r)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # cacheseek repo root

import numpy as np
import torch
from PIL import Image

try:                                      # telefuser not in env -> fall back to $TELEFUSER checkout
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
    make_rolling_config,
)
from cacheseek.stores import InMemoryTierStore

if _WORLDKV_ROOTS:                       # self-check: imports must resolve to our clone
    import telefuser as _tf

    import cacheseek as _cs
    for _mod in (_tf, _cs):
        _src = inspect.getsourcefile(_mod) or ""
        assert any(_src.startswith(r) for r in _WORLDKV_ROOTS), f"{_mod.__name__} resolved to {_src} (editable finder not stripped?)"
        print(f"[worldkv-shim] {_mod.__name__} -> {_src}")

PROMPT = "a quiet stone courtyard, ancient walls, soft afternoon light"
SEED = 42
ORIG_H, ORIG_W = 480, 832


def load_image(image_path: str) -> Image.Image:
    """Prefer a real reference image (lingbot-world asset, same input as production);
    fall back to a deterministic synthetic image when none is given."""
    if image_path:
        return Image.open(image_path).convert("RGB")
    y, x = np.mgrid[0:ORIG_H, 0:ORIG_W]
    arr = np.stack(
        [(x / ORIG_W * 255).astype(np.uint8), (y / ORIG_H * 255).astype(np.uint8), ((x + y) % 256).astype(np.uint8)],
        axis=-1,
    )
    arr[180:300, 320:520] = (200, 60, 60)
    return Image.fromarray(arr)


def _apply_fork(poses: np.ndarray, fork_from: int) -> np.ndarray:
    """Apply a PURE rotation (yaw) from frame fork_from onward; earlier frames stay
    bit-identical to the main trajectory.

    Must be a pure rotation: the camera conditioning max-normalizes framewise translation
    over the whole trajectory (control.py compute_relative_poses: trans/norm(trans).max()).
    Changing tail translation shifts max_norm => the prefix control changes too => a correct
    cache miss; rotation leaves the translation norm untouched => the prefix is bit-identical
    => a hit.
    """
    forked = poses.copy()
    for i in range(fork_from, len(forked)):
        j = i - fork_from
        yaw = 0.03 * j
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=forked.dtype)
        forked[i, :3, :3] = rot @ forked[i, :3, :3]
    return forked


def make_trajectories(frame_num: int, fork_from: int, action_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (main-trajectory poses, forked poses, intrinsics). Prefer real
    poses.npy/intrinsics.npy."""
    if action_path:
        root = Path(action_path)
        poses = np.load(root / "poses.npy").astype(np.float32)[:frame_num]
        intrinsics = np.load(root / "intrinsics.npy").astype(np.float32)
        if intrinsics.ndim > 1:
            intrinsics = intrinsics[:frame_num]
        assert len(poses) >= frame_num, f"poses.npy has only {len(poses)} frames < frame_num={frame_num}"
    else:
        poses = np.tile(np.eye(4, dtype=np.float32), (frame_num, 1, 1))
        for i in range(frame_num):
            poses[i, 2, 3] = 0.05 * i                  # move forward
        intrinsics = np.tile(np.array([400.0, 400.0, ORIG_W / 2, ORIG_H / 2], dtype=np.float32), (frame_num, 1))
    return poses, _apply_fork(poses, fork_from), intrinsics


def run_request(pipeline, binding, frame_num: int, poses, intrinsics, *, image_path: str = "",
                save_dir: Path | None = None, tag: str = ""):
    session = LingBotWorldFastSessionConfig(
        prompt=PROMPT,
        image=load_image(image_path),
        control_mode="cam",
        frame_num=frame_num,
        seed=SEED,
        poses=poses,
        intrinsics=intrinsics,
        world_kv_binding=binding,
    )
    t0 = time.time()
    runtime = pipeline.create_runtime(session)
    frames: list[Image.Image] = []
    chunk_times = []
    while runtime.active:
        tc = time.time()
        frames.extend(pipeline.generate_next_chunk(runtime))
        chunk_times.append(round(time.time() - tc, 3))
    wall = round(time.time() - t0, 3)
    # Async writes: drain in-flight puts so the next request's lookup sees all of this
    # request's chunks. flush_s = write time not hidden by the chunk loop; smaller means
    # more of the async benefit was realized.
    tf = time.time()
    store = binding.mgr.store
    if hasattr(store, "flush"):
        store.flush()
    flush_s = round(time.time() - tf, 3)
    hashes = [hashlib.sha256(np.asarray(f).tobytes()).hexdigest() for f in frames]
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(save_dir / f"{tag}_{i:04d}.jpg", quality=92)
    return {
        "tag": tag,
        "fast_forward_k": binding.last_fast_forward,
        "n_chunks": runtime.current_chunk_index,
        "n_frames": len(frames),
        "wall_s": wall,
        "flush_s": flush_s,
        "chunk_times_s": chunk_times,
        "frame_sha256": hashes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-num", type=int, default=37)
    ap.add_argument("--prefix-chunks", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default="/tmp/worldkv_e2e")
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True,
                    help="save per-frame jpg to out-dir/frames/ (on by default; --no-save-videos to disable)")
    ap.add_argument("--image-path", type=str, default="", help="real reference image (lingbot-world asset); empty = synthetic fallback")
    ap.add_argument("--action-path", type=str, default="", help="directory containing poses.npy/intrinsics.npy; empty = synthetic trajectory")
    ap.add_argument("--store", type=str, default="inmem", choices=["inmem", "localdisk", "fluxon"], help="blob storage backend")
    ap.add_argument("--fluxon-config", type=str, default="", help="path to the Fluxon client external_config.yaml")
    ap.add_argument("--disk-root", type=str, default="/tmp/worldkv_diskstore", help="root dir for the localdisk backend (should be real local disk)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="number of A/B/C/D loop iterations (weights loaded once; for soak testing, each round uses an independent cache stack)")
    ap.add_argument("--aux-device", type=str, default="",
                    help="two-GPU usage: place the T5 text encoder (~11GB) + VAE on this device (e.g. cuda:1), keep the DiT on the main device. "
                         "Note: the LingBot pipeline's DiT itself has no tensor parallelism — two GPUs split memory placement, not compute")
    args = ap.parse_args()

    ckpt = os.environ.get("LINGBOT_WORLD_CHECKPOINT_DIR", "")
    assert ckpt, "Set LINGBOT_WORLD_CHECKPOINT_DIR"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames" if args.save_videos else None

    pipeline = LingBotWorldFastPipeline(device="cuda")
    cfg = LingBotWorldFastPipelineConfig(checkpoint_dir=ckpt)   # local_attn_size=7, sink_size=3 by default
    if args.aux_device:                                          # 2-gpu: T5+VAE -> aux device, DiT stays on main
        from telefuser.core.config import ModelRuntimeConfig
        _t, _i = (args.aux_device.split(":") + ["0"])[:2]
        aux = dict(device_type=_t, device_id=int(_i))
        cfg.text_encoding_config = ModelRuntimeConfig(**aux)
        cfg.vae_config = ModelRuntimeConfig(**aux)
        print(f"[2-gpu] text-encoder + VAE -> {args.aux_device}; DiT -> cuda:0", flush=True)
    pipeline.init(None, cfg)

    def make_store():
        if args.store == "fluxon":
            from cacheseek.stores import TensorStoreTierStore
            from cacheseek.stores.fluxon import FluxonKVStore
            assert args.fluxon_config, "--store fluxon requires --fluxon-config"
            # True async writes: put enqueues and returns, worker publishes ready after
            # draining (write latency moved out of the chunk loop).
            return TensorStoreTierStore(FluxonKVStore(config_path=args.fluxon_config), async_put=True)
        if args.store == "localdisk":
            from cacheseek.stores import TensorStoreTierStore
            from cacheseek.stores.tier import LocalDiskTensorStore
            # Same adapter and same async path as fluxon, so the only variable in the
            # baseline comparison is the backend.
            return TensorStoreTierStore(LocalDiskTensorStore(args.disk_root), async_put=True)
        return InMemoryTierStore()

    # The store shares one client (two same-named Fluxon members in one process would
    # collide); the empty-cache cold run is guaranteed by a fresh forest (trie index) --
    # old payloads in the store are unreachable without an index.
    shared_store = make_store()

    def fresh_stack():
        forest = NamespaceForest()
        mgr = WorldKVManager(
            forest, shared_store,
            make_rolling_config(local_attn_size=cfg.local_attn_size, sink_size=cfg.sink_size, chunk_size=3),
        )
        return forest, mgr

    for _rep in range(args.repeat):
        if args.repeat > 1:
            print(f"[repeat] ===== iteration {_rep + 1}/{args.repeat} =====", flush=True)
        forest, mgr = fresh_stack()
        fork_from_pixel = args.prefix_chunks * 3 * 4 + 2            # fork only after prefix latent frames (leave interpolation margin)
        poses_main, poses_fork, intr = make_trajectories(args.frame_num, fork_from_pixel, args.action_path)

        results = {}
        results["A"] = run_request(pipeline, LingBotWorldKVBinding(mgr, forest), args.frame_num, poses_main, intr,
                                   image_path=args.image_path, save_dir=frames_dir, tag="A")
        results["B"] = run_request(pipeline, LingBotWorldKVBinding(mgr, forest), args.frame_num, poses_main, intr,
                                   image_path=args.image_path, save_dir=frames_dir, tag="B")
        results["C"] = run_request(pipeline, LingBotWorldKVBinding(mgr, forest), args.frame_num, poses_fork, intr,
                                   image_path=args.image_path, save_dir=frames_dir, tag="C")
        forest_d, mgr_d = fresh_stack()                             # empty-cache cold-run reference
        results["D"] = run_request(pipeline, LingBotWorldKVBinding(mgr_d, forest_d), args.frame_num, poses_fork, intr,
                                   image_path=args.image_path, save_dir=frames_dir, tag="D")

        n_chunks = results["A"]["n_chunks"]
        checks = {
            "A_cold": results["A"]["fast_forward_k"] == 0,
            "B_full_hit": results["B"]["fast_forward_k"] == n_chunks,
            "B_frames_equal_A": results["B"]["frame_sha256"] == results["A"]["frame_sha256"],
            "C_prefix_hit": results["C"]["fast_forward_k"] == args.prefix_chunks,
            "D_cold": results["D"]["fast_forward_k"] == 0,
            "C_frames_equal_D": results["C"]["frame_sha256"] == results["D"]["frame_sha256"],
        }
        if not all(checks.values()):
            print(f"[repeat] iteration {_rep + 1} FAILED: {checks}", flush=True)
            break
    manifest = {
        "frame_num": args.frame_num, "prefix_chunks": args.prefix_chunks, "seed": SEED, "store": args.store,
        "pipeline": {"local_attn_size": cfg.local_attn_size, "sink_size": cfg.sink_size},
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "results": results, "checks": checks, "all_pass": all(checks.values()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"checks": checks, "all_pass": manifest["all_pass"],
                      "wall_s": {k: v["wall_s"] for k, v in results.items()},
                      "flush_s": {k: v["flush_s"] for k, v in results.items()},
                      "fast_forward_k": {k: v["fast_forward_k"] for k, v in results.items()}}, indent=2))
    return 0 if manifest["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
