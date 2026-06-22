"""Discriminators for the Fluxon self-KV bandwidth conclusion.

Settles whether raw's restore LOSS is fundamental or two removable confounds:
  D3: what does access() return for a plain BYTES field -- memoryview (zero-copy
      view into shm => bytes(v) is a removable artifact) or copied bytes?
  D1: raw GET via the adapter (does bytes(v)) vs raw-store get + frombuffer over
      the value directly (skips the copy). How close to dlpack's ~11 GB/s?
  D2: H2D per chunk (80 blocks=3.83GB) -- pageable .to(dev) vs pinned vs copy_
      into a pre-allocated GPU buffer (what the real restore actually does).

Run: CUDA_VISIBLE_DEVICES=2,3 <venv> skv_fluxon_disc.py --config <cfg> --device cuda:0
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

from cacheseek.stores import FluxonKVStore

warnings.filterwarnings("ignore", message=".*non-writable.*")

SHAPE = (1, 4680, 40, 128)
NB = 1 * 4680 * 40 * 128 * 2  # bytes/block
BLOCKS = 80
PHYS = NB * BLOCKS  # 3.83 GB


def _ok(res):
    return res.unwrap("ok") if (res is not None and res.is_ok()) else None


def _raw_bytes(t):
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def _from_raw(buf):
    return torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).reshape(SHAPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-tag", default="disc")
    args = ap.parse_args()

    store = FluxonKVStore(config_path=args.config)
    raw = store._store
    g = torch.Generator().manual_seed(0)
    originals = [
        torch.randn(*SHAPE, generator=g).to(torch.bfloat16).contiguous()
        for _ in range(BLOCKS)
    ]
    keys = [f"disc:{args.run_tag}:b{i}" for i in range(BLOCKS)]

    try:
        # put 80 blocks as raw bytes (via the adapter -> {"v": bytes})
        blobs = [_raw_bytes(t) for t in originals]
        for k, b in zip(keys, blobs):
            store.put(k, b)

        # ---- D3: access() return type for a BYTES field ----
        res = raw.get(keys[0])
        mh = _ok(_ok(res).wait())
        d = _ok(mh.access())
        v0 = d.get("v")
        print(f"[disc] D3 access() bytes-field type = {type(v0).__name__}", flush=True)

        # ---- D1: GET timing, adapter (bytes(v)) vs raw-store (no bytes copy) ----
        t0 = time.perf_counter()
        a = [_from_raw(store.get(k)) for k in keys]  # adapter: bytes(v) inside
        t_adapter = time.perf_counter() - t0
        del a

        def raw_get_tensor(k):
            mh = _ok(_ok(raw.get(k)).wait())
            v = _ok(mh.access()).get("v")
            # frombuffer directly over v (memoryview or bytes) -- no extra bytes() copy
            return (
                torch.frombuffer(v, dtype=torch.uint16)
                .view(torch.bfloat16)
                .reshape(SHAPE)
            )

        t0 = time.perf_counter()
        a = [raw_get_tensor(k) for k in keys]
        t_rawget = time.perf_counter() - t0
        ok_bits = torch.equal(a[0].clone(), originals[0])
        del a

        print(
            f"[disc] D1 GET adapter (bytes(v)+frombuffer): {t_adapter:5.2f}s "
            f"({PHYS / 1e9 / t_adapter:5.2f} GB/s)"
        )
        print(
            f"[disc] D1 GET raw-store (frombuffer over v): {t_rawget:5.2f}s "
            f"({PHYS / 1e9 / t_rawget:5.2f} GB/s)   bit-exact={ok_bits}"
        )
        print("[disc] D1 (dlpack reference from sweep ~11 GB/s)", flush=True)
    finally:
        for k in keys:
            try:
                store.remove(k)
            except Exception:  # noqa: BLE001
                pass

    # ---- D2: H2D per chunk (80 blocks = 3.83 GB) ----
    if not torch.cuda.is_available():
        print("[disc] D2 skipped (no CUDA)")
        return
    dev = torch.device(args.device)
    _ = originals[0].to(dev)  # warm ctx
    torch.cuda.synchronize()

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # (a) pageable .to(dev)
    t_page = timed(lambda: [t.to(dev) for t in originals])

    # (b) pinned .to(dev, non_blocking)
    pinned = [t.pin_memory() for t in originals]  # one-time alloc, not timed
    t_pin = timed(lambda: [p.to(dev, non_blocking=True) for p in pinned])

    # (c) copy_ from pinned into a pre-allocated GPU buffer (what restore does)
    gpu_bufs = [
        torch.empty(SHAPE, dtype=torch.bfloat16, device=dev) for _ in range(BLOCKS)
    ]
    t_buf = timed(
        lambda: [gb.copy_(p, non_blocking=True) for gb, p in zip(gpu_bufs, pinned)]
    )

    print("[disc] D2 H2D/chunk (3.83 GB):")
    print(
        f"[disc]   pageable .to(dev)            : {t_page:5.2f}s ({PHYS / 1e9 / t_page:5.2f} GB/s)"
    )
    print(
        f"[disc]   pinned .to(dev)              : {t_pin:5.2f}s ({PHYS / 1e9 / t_pin:5.2f} GB/s)"
    )
    print(
        f"[disc]   pinned -> existing buf copy_ : {t_buf:5.2f}s ({PHYS / 1e9 / t_buf:5.2f} GB/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
