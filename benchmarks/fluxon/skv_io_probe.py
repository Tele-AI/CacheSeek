# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Self-KV restore I/O decomposition probe (read-side, no model load).

The service-form E2E showed self-KV restore (~10s/chunk) and save (~8s/chunk)
both dominate over recompute (~3.6s/chunk) on the cold local-disk tier. This
probe attributes that cost without touching the DiT: it walks ONE saved chunk's
80 block files (40 layers x {k,v}) and times the three restore stages
separately, then micro-benchmarks the serialize layer (torch.save/load vs a raw
bf16 buffer view) to project the payoff of swapping it.

Run: PYTHONPATH=<phase-C cacheseek> CUDA_VISIBLE_DEVICES=2,3 python skv_io_probe.py \
        --cache-dir <run>/cache --device cuda:0
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch


def _t() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cache = Path(args.cache_dir).expanduser().resolve()
    index = json.loads((cache / "kv_index.json").read_text())

    # Group block keys by their self_kv chunk hash (the segment before ":block:").
    groups: dict[str, list[str]] = {}
    for key in index:
        if ":block:" not in key:
            continue
        gid, _, block = key.rpartition(":block:")
        if not block.startswith("layer"):  # skip meta.* blocks
            continue
        groups.setdefault(gid, []).append(key)

    # Pick a representative full chunk (80 layer k/v keys).
    full = {g: ks for g, ks in groups.items() if len(ks) == 80}
    if not full:
        raise SystemExit(f"no 80-key chunk group found; sizes={ {g: len(k) for g,k in groups.items()} }")
    gid, keys = next(iter(full.items()))
    keys.sort()
    print(f"[probe] cache={cache}")
    print(f"[probe] {len(full)} full chunk(s); probing group {gid[-16:]} with {len(keys)} keys", flush=True)

    paths = [cache / index[k] for k in keys]
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"[probe] chunk on-disk size = {total_bytes/1e9:.3f} GB", flush=True)

    # ---- Stage 1: pure disk read (bytes only) ----
    # Drop page cache effect note: first read is cold, re-run for warm number.
    t0 = time.perf_counter()
    blobs = [p.read_bytes() for p in paths]
    t_read = time.perf_counter() - t0

    # ---- Stage 2: pure CPU deserialize (torch.load on in-RAM bytes) ----
    t0 = time.perf_counter()
    tensors = [torch.load(io.BytesIO(b), weights_only=True, map_location="cpu") for b in blobs]
    t_load = time.perf_counter() - t0

    # ---- Stage 3: pure H2D (serial .to(device)) ----
    dev = torch.device(args.device)
    t0 = _t()
    _ = [t.to(dev) for t in tensors]
    t_h2d = _t() - t0

    print("\n[probe] === restore stage decomposition (one chunk, 80 keys) ===", flush=True)
    print(f"[probe]  1. disk read      : {t_read:6.2f}s  ({total_bytes/1e9/t_read:5.2f} GB/s)")
    print(f"[probe]  2. torch.load     : {t_load:6.2f}s  ({total_bytes/1e9/t_load:5.2f} GB/s)")
    print(f"[probe]  3. H2D .to(dev)   : {t_h2d:6.2f}s  ({total_bytes/1e9/t_h2d:5.2f} GB/s)")
    print(f"[probe]  sum               : {t_read+t_load+t_h2d:6.2f}s   (recompute ~3.6s/chunk)", flush=True)

    # ---- Micro: serialize layer comparison on one 47.9MB bf16 tensor ----
    sample = tensors[0].contiguous()
    print(f"\n[probe] === serialize micro-bench (one tensor {tuple(sample.shape)} {sample.dtype}) ===", flush=True)

    def torch_save(t: torch.Tensor) -> bytes:
        buf = io.BytesIO(); torch.save(t, buf); return buf.getvalue()

    def torch_load(b: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(b), weights_only=True, map_location="cpu")

    def raw_save(t: torch.Tensor) -> bytes:
        # bitcast bf16 -> uint16 (numpy has no bf16); store shape separately in real impl.
        return t.contiguous().view(torch.uint16).numpy().tobytes()

    def raw_load(b: bytes, shape, dtype) -> torch.Tensor:
        return torch.frombuffer(bytearray(b), dtype=torch.uint16).view(dtype).reshape(shape)

    N = 20
    for name, fn in (("torch.save", torch_save), ("raw bf16 buf", raw_save)):
        t0 = time.perf_counter()
        for _ in range(N):
            blob = fn(sample)
        dt = (time.perf_counter() - t0) / N
        print(f"[probe]  serialize  {name:14s}: {dt*1e3:7.2f} ms  ({len(blob)/1e6:.1f} MB)")

    blob_ts = torch_save(sample)
    blob_raw = raw_save(sample)
    for name, fn in (
        ("torch.load", lambda b: torch_load(b)),
        ("raw bf16 buf", lambda b: raw_load(b, sample.shape, sample.dtype)),
    ):
        b = blob_ts if name == "torch.load" else blob_raw
        t0 = time.perf_counter()
        for _ in range(N):
            _ = fn(b)
        dt = (time.perf_counter() - t0) / N
        print(f"[probe]  deserialize {name:14s}: {dt*1e3:7.2f} ms", flush=True)

    # correctness of raw round-trip
    rt = raw_load(raw_save(sample), sample.shape, sample.dtype)
    print(f"[probe]  raw round-trip bit-exact: {torch.equal(rt, sample)}", flush=True)

    # ---- Experiment A: one blob per chunk (stack 80 -> 1 tensor) ----
    import tempfile
    print("\n[probe] === experiment A: 1 blob/chunk (stack 80 layer-tensors) ===", flush=True)
    stacked = torch.stack([t.contiguous() for t in tensors], dim=0)  # (80,1,4680,40,128)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        t0 = time.perf_counter(); torch.save(stacked, f.name); t_save1 = time.perf_counter() - t0
        f.flush()
        t0 = time.perf_counter(); blob1 = Path(f.name).read_bytes(); t_read1 = time.perf_counter() - t0
        t0 = time.perf_counter(); big = torch.load(io.BytesIO(blob1), weights_only=True, map_location="cpu"); t_load1 = time.perf_counter() - t0
        t0 = _t(); _ = big.to(dev); t_h2d1 = _t() - t0
    gb = stacked.numel() * stacked.element_size() / 1e9
    print(f"[probe]  save 1 blob    : {t_save1:6.2f}s")
    print(f"[probe]  read 1 blob    : {t_read1:6.2f}s  ({gb/t_read1:5.2f} GB/s)")
    print(f"[probe]  load 1 blob    : {t_load1:6.2f}s  ({gb/t_load1:5.2f} GB/s)")
    print(f"[probe]  H2D  1 blob    : {t_h2d1:6.2f}s  ({gb/t_h2d1:5.2f} GB/s)")
    print(f"[probe]  read+load+h2d  : {t_read1+t_load1+t_h2d1:6.2f}s  vs 80-key {t_read+t_load+t_h2d:.2f}s", flush=True)

    # ---- Experiment B: thread the existing 80-key path ----
    from concurrent.futures import ThreadPoolExecutor
    print("\n[probe] === experiment B: ThreadPool(8) over 80 keys ===", flush=True)

    def one(p: Path):
        return torch.load(io.BytesIO(p.read_bytes()), weights_only=True, map_location="cpu").to(dev)

    t0 = _t()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, paths))
    t_thread = _t() - t0
    print(f"[probe]  threaded read+load+h2d: {t_thread:6.2f}s  vs serial {t_read+t_load+t_h2d:.2f}s", flush=True)


if __name__ == "__main__":
    main()
