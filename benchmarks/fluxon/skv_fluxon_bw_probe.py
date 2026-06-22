"""Fluxon self-KV store bandwidth probe (no model load).

Fakes lingbot-fast self-KV chunks and measures end-to-end **Fluxon** put/get
bandwidth across three serialization forms, sweeping chunk counts.

Sizing (one chunk):
    80 blocks = 40 layers x {k, v}; each tensor [batch, tokens, heads, head_dim]
    = [1, 4680, 40, 128] bf16 ~= 48 MB; one full chunk ~= 3.83 GB.

Serialization modes (--serialize):
  torch_save : PUT = torch.save(pickle) + store.put(bytes)   [current path]
               GET = store.get(bytes) + torch.load
  raw        : PUT = view(uint16).tobytes() + store.put(bytes)
               GET = store.get(bytes) + frombuffer
  dlpack     : PUT = raw store.put(key, {"v": tensor})       [Fluxon-native form]
               GET = raw store.get -> access() -> from_dlpack(view).clone()
               (bypasses our bytes adapter; build_flat_dict_ptrs hands the tensor
                pointer to Rust on put; access() returns a dlpack view on get)

Measurement is **end-to-end on each side** -- serialize folded into PUT,
deserialize/materialize folded into GET -- so the modes are directly comparable
as "save a chunk" / "restore a chunk to a usable CPU tensor". GB/s is on
PHYSICAL bytes (numel*2). The pickle tax shows up as torch_save being slower on
both sides; raw and dlpack should be close (dlpack only saves the .tobytes()
copy on PUT). (C) restore verdict = GET(N=1) [+ H2D] vs recompute 3.0-3.6 s.

Recovery: Fluxon list_keys() returns [] (no enumeration). --run-tag is
deterministic and --cleanup-only regenerates the exact keys to remove them.

Run on the Fluxon host (stack up, see scripts/fluxon/), GPU2/3:
    CUDA_VISIBLE_DEVICES=2,3 <venv> skv_fluxon_bw_probe.py \
        --config <external_config.yaml> --serialize dlpack --sweep 1,2,3,4,5,6 \
        --device cuda:0 --threads 8
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch

from cacheseek.stores import FluxonKVStore

# torch.frombuffer over read-only bytes warns per call; the view is what we want.
warnings.filterwarnings("ignore", message=".*non-writable.*")

RECOMPUTE_S_PER_CHUNK = (3.0, 3.6)  # LingBot one-chunk denoise (service E2E)
BSTAR_GBPS = 1.06  # per-restore breakeven = V_kv / t_recompute (scale-invariant)
SER_MODES = ("torch_save", "raw", "dlpack")


def _gbps(phys: int, t: float) -> float:
    return phys / 1e9 / t if t > 0 else float("nan")


# ---- serialization helpers --------------------------------------------------
def _torch_save(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _raw_bytes(t: torch.Tensor) -> bytes:
    # bf16 -> uint16 bitcast (numpy has no bf16); shape/dtype known from profile.
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def _from_raw(b: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.frombuffer(b, dtype=torch.uint16).view(torch.bfloat16).reshape(shape)


# ---- raw-store (dlpack) put/get, bypassing the bytes adapter ----------------
def _ok(res):
    if res is None or not res.is_ok():
        return None
    return res.unwrap("ok")


def _consume_err(res) -> str:
    if res is None:
        return "None result"
    if res.is_ok():
        _ = res.unwrap("ok")  # consume to satisfy strict Result
        return "(ok)"
    return f"{type(res.unwrap_error('err')).__name__}"


def _raw_put_tensor(raw, key: str, tensor: torch.Tensor) -> None:
    res = raw.put(key, {"v": tensor})
    fut = _ok(res)
    if fut is None:
        raise RuntimeError(f"dlpack put {key!r}: {_consume_err(res)}")
    if _ok(fut.wait()) is None:
        raise RuntimeError(f"dlpack put.wait {key!r} failed")


def _raw_get_tensor(raw, key: str) -> torch.Tensor:
    """GET via dlpack: access() returns a dlpack view; from_dlpack + clone
    materializes it into an owned CPU tensor (mirrors restore's copy-out)."""
    res = raw.get(key)
    fut = _ok(res)
    if fut is None:
        raise RuntimeError(f"dlpack get {key!r}: {_consume_err(res)}")
    mh = _ok(fut.wait())
    if mh is None:
        raise RuntimeError(f"dlpack get.wait {key!r} failed")
    d = _ok(mh.access())
    if d is None:
        raise RuntimeError(f"dlpack access {key!r} failed")
    v = d.get("v")
    if hasattr(v, "__dlpack__"):
        return torch.from_dlpack(v).clone()
    if isinstance(v, (bytes, bytearray, memoryview)):  # fallback if not wrapped
        return torch.frombuffer(bytes(v), dtype=torch.uint16).view(torch.bfloat16)
    raise TypeError(f"dlpack get {key!r}: unexpected 'v' type {type(v).__name__}")


# ---- payload build ----------------------------------------------------------
def _build_originals(layers, batch, tokens, heads, head_dim):
    """80 distinct random CPU contiguous bf16 tensors (one full chunk)."""
    g = torch.Generator().manual_seed(0)
    return [
        torch.randn(batch, tokens, heads, head_dim, generator=g)
        .to(torch.bfloat16)
        .contiguous()
        for _ in range(layers * 2)
    ]


def _suffixes(layers):
    return [f"layer{layer}.{kv}" for layer in range(layers) for kv in ("k", "v")]


def _keys(run_tag, mode, chunks, suffixes):
    return [f"skvbw:{run_tag}:{mode}:c{c}:{s}" for c in range(chunks) for s in suffixes]


def _cleanup(store, keys):
    leaked = 0
    for k in keys:
        try:
            store.remove(k)
        except Exception as exc:  # noqa: BLE001
            leaked += 1
            print(f"[probe]  remove failed {k!r}: {exc}")
    print(
        f"[probe]  cleanup: removed {len(keys) - leaked}/{len(keys)} keys"
        + (f"  ({leaked} LEAKED)" if leaked else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="fluxon_config_path")
    ap.add_argument("--serialize", choices=SER_MODES, default="dlpack")
    ap.add_argument("--sweep", default="", help="comma chunk counts, e.g. 1,2,3,4,5,6")
    ap.add_argument("--chunks", type=int, default=1, help="used when --sweep absent")
    ap.add_argument(
        "--threads", type=int, default=8, help="threaded-pass workers (1=skip)"
    )
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument(
        "--batch", type=int, default=1, help="2 if runtime keeps CFG cond/uncond"
    )
    ap.add_argument("--tokens-per-frame", type=int, default=1560, help="1560=480p")
    ap.add_argument("--latent-per-chunk", type=int, default=3)
    ap.add_argument("--device", default="", help="cuda:0 to also time H2D")
    ap.add_argument("--run-tag", default="probe")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    args = ap.parse_args()

    mode = args.serialize
    tokens = args.tokens_per_frame * args.latent_per_chunk
    shape = (args.batch, tokens, args.heads, args.head_dim)
    blocks_per_chunk = args.layers * 2
    suffixes = _suffixes(args.layers)
    sweep = [int(x) for x in args.sweep.split(",") if x.strip()] or [args.chunks]

    store = FluxonKVStore(config_path=args.config)
    raw = store._store  # raw FluxonKVCacheStore for the dlpack (tensor) path

    if args.cleanup_only:
        keys = _keys(args.run_tag, mode, max(sweep), suffixes)
        print(
            f"[probe] cleanup-only: {len(keys)} keys (mode={mode}, up to {max(sweep)} chunks)"
        )
        _cleanup(store, keys)
        return

    nbytes_block = args.batch * tokens * args.heads * args.head_dim * 2
    bytes_chunk = nbytes_block * blocks_per_chunk
    print(
        f"[probe] mode={mode}  block=[{args.batch},{tokens},{args.heads},{args.head_dim}] bf16"
        f"  ({nbytes_block / 1e6:.1f} MB/block, {bytes_chunk / 1e9:.2f} GB/chunk)"
    )
    print(
        f"[probe] sweep chunks={sweep}  threads={args.threads}  (PUT/GET are end-to-end)",
        flush=True,
    )

    originals = _build_originals(
        args.layers, args.batch, tokens, args.heads, args.head_dim
    )

    def save_one(kp):  # PUT: key <- tensor (serialize folded in)
        key, t = kp
        if mode == "torch_save":
            store.put(key, _torch_save(t))
        elif mode == "raw":
            store.put(key, _raw_bytes(t))
        else:
            _raw_put_tensor(raw, key, t)

    def restore_one(key) -> torch.Tensor:  # GET: key -> usable CPU tensor
        if mode == "torch_save":
            return torch.load(
                io.BytesIO(store.get(key)), weights_only=True, map_location="cpu"
            )
        if mode == "raw":
            return _from_raw(store.get(key), shape)
        return _raw_get_tensor(raw, key)

    rows = []
    bit_ok = True
    for N in sweep:
        keys = _keys(args.run_tag, mode, N, suffixes)
        items = [(k, originals[i % len(originals)]) for i, k in enumerate(keys)]
        phys = nbytes_block * len(keys)
        bad = []
        try:
            t0 = time.perf_counter()
            for kp in items:
                save_one(kp)
            t_put = time.perf_counter() - t0

            t0 = time.perf_counter()
            got = [restore_one(k) for k, _ in items]
            t_get = time.perf_counter() - t0

            sidx = range(0, len(items), max(1, len(items) // 8))
            bad = [
                items[i][0]
                for i in sidx
                if not torch.equal(got[i], originals[i % len(originals)])
            ]
            bit_ok = bit_ok and not bad
            del got

            t_put_thr = t_get_thr = float("nan")
            if args.threads > 1:
                with ThreadPoolExecutor(max_workers=args.threads) as ex:
                    t0 = time.perf_counter()
                    list(ex.map(save_one, items))
                    t_put_thr = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    got = list(ex.map(lambda kp: restore_one(kp[0]), items))
                    t_get_thr = time.perf_counter() - t0
                if not torch.equal(got[0], originals[0]):
                    bit_ok = False
                    bad.append("threaded:" + items[0][0])
                del got
        finally:
            if not args.keep:
                _cleanup(store, keys)

        rows.append((N, phys, t_put, t_get, t_put_thr, t_get_thr))
        print(
            f"[probe] N={N}  put {_gbps(phys, t_put):5.2f} GB/s ({t_put:6.2f}s)  "
            f"get {_gbps(phys, t_get):5.2f} GB/s ({t_get:6.2f}s)  "
            f"| thr put {_gbps(phys, t_put_thr):5.2f} get {_gbps(phys, t_get_thr):5.2f}  "
            f"| bit {'OK' if not bad else f'BAD {bad}'}",
            flush=True,
        )

    # ---- (C) restore verdict: GET(N=1) per chunk [+ H2D] vs recompute ----
    n1 = rows[0]
    t_get_chunk = n1[3] / n1[0]  # N=1 get time is already 1 chunk; /N keeps it general
    t_h2d_chunk = None
    if args.device and torch.cuda.is_available():
        dev = torch.device(args.device)
        s = originals[0].to(dev)  # warmup CUDA ctx
        del s
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in originals:
            _ = t.to(dev)
        torch.cuda.synchronize()
        t_h2d_chunk = time.perf_counter() - t0
    restore = t_get_chunk + (t_h2d_chunk or 0.0)
    lo, hi = RECOMPUTE_S_PER_CHUNK
    verdict = "WINS" if restore < lo else ("LOSES" if restore > hi else "MARGINAL")

    print(
        f"\n[probe] === mode={mode} summary (physical {bytes_chunk / 1e9:.2f} GB/chunk) ==="
    )
    print(
        f"[probe]  restore-side GET (incl deser): {t_get_chunk:5.2f}s/chunk "
        f"({_gbps(bytes_chunk, t_get_chunk):.2f} GB/s)"
    )
    if t_h2d_chunk is not None:
        print(
            f"[probe]  H2D .to({args.device:7s})          : {t_h2d_chunk:5.2f}s/chunk"
        )
    print(
        f"[probe]  ---> restore ~= {restore:5.2f}s/chunk  vs recompute {lo}-{hi}s  =>  {verdict}"
    )
    print(
        f"[probe]  (B*={BSTAR_GBPS} GB/s; pickle tax {'PRESENT' if mode == 'torch_save' else 'GONE'})",
        flush=True,
    )

    if not bit_ok:
        print("[probe]  FAIL: bit-exact mismatch", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
