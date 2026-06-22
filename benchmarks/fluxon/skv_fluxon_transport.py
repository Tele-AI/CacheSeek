# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Pure Fluxon transport timing (serialize/deserialize EXCLUDED).

Isolates the Fluxon put/get itself from torch ser/deser. Payloads are built
ONCE up front (outside timing); only the store call is timed.

  bytes regime (== torch_save & raw transport; they only differ in serialize):
      PUT = store.put(key, bytes)        GET = store.get(key)  (materialized bytes)
  dlpack regime:
      PUT = raw.put(key, {"v": tensor})
      GET split into:  access-only (transport, returns lazy view)
                       + clone (the actual pool->client materialization)

Run: CUDA_VISIBLE_DEVICES=2,3 <venv> skv_fluxon_transport.py --config <cfg> --sweep 1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

from cacheseek.stores import FluxonKVStore

warnings.filterwarnings("ignore", message=".*non-writable.*")

SHAPE = (1, 4680, 40, 128)
NB = 1 * 4680 * 40 * 128 * 2
BLOCKS = 80


def _ok(res):
    return res.unwrap("ok") if (res is not None and res.is_ok()) else None


def _raw_bytes(t):
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def _gbps(phys, t):
    return phys / 1e9 / t if t > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", default="1,2,3,4,5,6")
    ap.add_argument("--run-tag", default="tport")
    args = ap.parse_args()

    store = FluxonKVStore(config_path=args.config)
    raw = store._store
    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]

    g = torch.Generator().manual_seed(0)
    originals = [
        torch.randn(*SHAPE, generator=g).to(torch.bfloat16).contiguous()
        for _ in range(BLOCKS)
    ]
    blobs = [_raw_bytes(t) for t in originals]  # pre-built bytes (untimed)

    print(
        f"[tport] block {NB / 1e6:.1f} MB, chunk {NB * BLOCKS / 1e9:.2f} GB; "
        f"sweep={sweep}  (serialize EXCLUDED)",
        flush=True,
    )
    print(
        f"[tport] {'N':>2} | bytes PUT  bytes GET | dlpack PUT  dl GET-access  dl GET-clone",
        flush=True,
    )

    for N in sweep:
        nk = BLOCKS * N
        phys = NB * nk
        kb = [
            f"tport:{args.run_tag}:b:c{c}:l{i}" for c in range(N) for i in range(BLOCKS)
        ]
        kd = [
            f"tport:{args.run_tag}:d:c{c}:l{i}" for c in range(N) for i in range(BLOCKS)
        ]
        pay = [(originals[j % BLOCKS], blobs[j % BLOCKS]) for j in range(nk)]
        try:
            # ---- bytes regime ----
            t0 = time.perf_counter()
            for k, (_, b) in zip(kb, pay):
                store.put(k, b)
            t_bput = time.perf_counter() - t0

            t0 = time.perf_counter()
            for k in kb:
                _ = store.get(k)  # returns materialized bytes
            t_bget = time.perf_counter() - t0

            # ---- dlpack regime ----
            t0 = time.perf_counter()
            for k, (t, _) in zip(kd, pay):
                res = raw.put(k, {"v": t})
                fut = _ok(res)
                _ok(fut.wait())
            t_dput = time.perf_counter() - t0

            # GET access-only (transport, no clone)
            t0 = time.perf_counter()
            for k in kd:
                mh = _ok(_ok(raw.get(k)).wait())
                _ = _ok(mh.access()).get("v")  # lazy view, not materialized
            t_dacc = time.perf_counter() - t0

            # GET full (access + from_dlpack + clone)  -> clone time = full - access
            t0 = time.perf_counter()
            for k in kd:
                mh = _ok(_ok(raw.get(k)).wait())
                v = _ok(mh.access()).get("v")
                _ = torch.from_dlpack(v).clone()
            t_dfull = time.perf_counter() - t0
            t_dclone = t_dfull - t_dacc
        finally:
            for k in kb + kd:
                try:
                    store.remove(k)
                except Exception:  # noqa: BLE001
                    pass

        print(
            f"[tport] {N:>2} | "
            f"{_gbps(phys, t_bput):5.2f} ({t_bput:5.2f}s) {_gbps(phys, t_bget):5.2f} ({t_bget:5.2f}s) | "
            f"{_gbps(phys, t_dput):5.2f} ({t_dput:5.2f}s) "
            f"{_gbps(phys, t_dacc):6.2f} ({t_dacc:5.2f}s) "
            f"{_gbps(phys, t_dclone):6.2f} ({t_dclone:5.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
