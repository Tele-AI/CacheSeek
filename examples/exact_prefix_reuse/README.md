# Exact reuse — for AR-Diffusion / self-forcing models

CacheSeek's reuse path for **autoregressive video models** — AR-Diffusion and
other self-forcing architectures that generate a clip chunk by chunk, each chunk
conditioned on the frames generated before it (LingBot's world model is the
reference). Because generation is causal and deterministic given the action
prefix and seed, an identical action prefix reproduces identical chunks — so the
reuse is **exact and lossless**, the video analog of KV prefix caching in
autoregressive LLMs. When a new session's action prefix byte-for-byte matches one
computed before, CacheSeek replays the matched prefix's KV and recomputes only
the chunks that diverge.

The cache is a **forest of action-chain tries** — conceptually RadixAttention,
but with the prefix made of *actions* instead of tokens, and the cached unit a
*chunk-KV* instead of a token-KV. The governing invariant is
**`VALUE = pure_function(KEY)`**: the seed is derived from the node key and the
version hashes the real config blob, so an identical key always recomputes
identical KV — which is what makes the replay bit-exact.

The two examples go from a pure-Python toy to a real LingBot run, and **each runs
independently**.

| Example | What it proves | Needs | Runtime |
|---|---|---|---|
| `quickstart_trie.py` | Trie hit / fork / eviction-fallback over the real trie + manager code path (string stand-in payloads) | This repo only | ~5 s |
| `e2e_telefuser_lingbot.py` | Real LingBot world model: four requests, six assertions, **byte-identical video** for both exact replay and fork continuation | GPU + LingBot weights + TeleFuser checkout | cold ~17 s / hit ~4.5 s |

---

## Quickstart (no GPU, no weights)

Uses string stand-ins for the heavy KV payloads but runs them through the **real**
trie and manager code — the same paths the GPU run exercises — so the hit, fork,
and degradation behavior you see here matches the real thing.

```bash
python examples/exact_prefix_reuse/quickstart_trie.py
```

It demonstrates four behaviors in sequence:

1. **Cold session** — cache empty, generate 6 chunks and write the action chain.
2. **Identical trajectory** — full-chain hit: all 6 chunks fast-forwarded, zero recompute.
3. **Forked trajectory** — first 4 actions match, then diverge: prefix hit of 4, a new branch grows at the fork point.
4. **Eviction fallback** — drop a middle chunk's heavy KV: the hit gracefully shortens to the longest intact prefix, while the light skeleton latent stays readable.

It exits after printing that all assertions passed.

## End-to-end with real LingBot (byte-exact)

The canonical driver lives right here. It runs **four requests in a single
process** (no service; the trie forest and store are shared in-process) and
asserts byte-level video equivalence for both replay and fork continuation:

| Request | Trajectory | Expected fast-forward | Assertion |
|---|---|---|---|
| **A** | cold, full run | K = 0 | baseline |
| **B** | identical to A | K = all chunks (decode-only) | video **byte-identical to A** |
| **C** | shares first `--prefix-chunks` with A, then camera path forks | K = prefix | prefix replayed exactly |
| **D** | same request as C, but cold cache | K = 0 | **C and D byte-identical** |

The C == D check is the real test: it confirms the RNG draws of skipped chunks
are correctly *burned* and the rolling KV ring is reassembled to the exact
physical layout a cold run would have had — without it, replay silently diverges.

```bash
# Prerequisite: a TeleFuser checkout with the world-kv pipeline hooks
export TELEFUSER=/path/to/telefuser-internal        # auto-added to sys.path by the driver
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/lingbot-world-base-cam
export ASSETS=/path/to/lingbot-world/examples/00    # image.jpg + poses.npy + intrinsics.npy
ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 python examples/exact_prefix_reuse/e2e_telefuser_lingbot.py \
  --frame-num 37 --prefix-chunks 2 \
  --image-path $ASSETS/image.jpg --action-path $ASSETS/ \
  --out-dir /tmp/worldkv_e2e
```

**Reading the result.** Exit 0 means all six assertions passed; `--out-dir`
contains a `manifest.json` with each request's K, chunk count, per-frame sha256,
and assertion outcomes.

**Variants:**

- `--frame-num 81 --prefix-chunks 4` — longer run, exercises multi-round rolling eviction
- `--store localdisk --disk-root /path/on/local/nvme` — local-disk blob backend
- `--store fluxon --fluxon-config /path/to/external_config.yaml` — Fluxon backend (stack setup: [`../../scripts/fluxon/README.md`](../../scripts/fluxon/README.md))
- omit `--image-path` / `--action-path` to fall back to a synthetic image + trajectory

> If the `cacheseek` (or `telefuser`) in your venv is an editable install pointing
> at a different checkout, set `WORLDKV_REPO_ROOTS=$TELEFUSER:$PWD` so imports
> resolve to the intended repos. The driver already handles the other sharp edges
> for you: stripping the editable import finder, normalizing the forked camera
> trajectory to a pure rotation (LingBot's conditioning pipeline normalizes the
> whole trajectory), and quantizing the action key at 1e-6 before hashing.
