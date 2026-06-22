# Approximate reuse — for diffusion-architecture models

CacheSeek's reuse path for **diffusion models** — full-sequence, bidirectional
denoising that refines the whole clip over ~40 steps (Wan2.2 T2V is the
reference model). In this architecture the early denoise steps carry coarse,
prompt-level structure that *similar* requests share. So when a new request is
semantically close to one served before, CacheSeek loads the donor request's
early-denoise latents and skips the first K steps — the skipped compute is the
cache win. Because "similar" isn't "identical", the reuse is **approximate**.

A hit is decided in three stages:

1. **Encode** the prompt (and, on save, sampled video frames) with Qwen3-VL.
2. **Retrieve** nearest neighbors from the vector store (FAISS or Qdrant).
3. **Rerank** the top candidates; accept only above `rerank_score_threshold`.
   The rerank gate is what keeps "close in embedding space" from becoming a
   wrong reuse — see the note under *End-to-end with real models*.

The three examples go from a pure-Python toy to a real Wan2.2 service, and **each
runs independently** — pick the one that matches what you have available.

| Example | What it proves | Needs | Runtime |
|---|---|---|---|
| `quickstart_lifecycle.py` | The full `CacheService` lifecycle (build_query → lookup → apply_resume → on_response → save) against an in-memory fake engine | This repo only | ~5 s |
| `e2e_real_components.py` | Real Qwen encoder + reranker + FAISS (+ optional Fluxon): cold miss, exact-prompt hit with byte-identical latent round-trip, and the rerank gate rejecting an unrelated prompt | GPU + Qwen3-VL weights | ~3 min |
| `service/` + `e2e_service_smoke.py` | Real Wan2.2 generation behind HTTP: the same prompt sent twice, second request accelerated by a hit | Service deployment (GPU + weights + Fluxon/Qdrant) | varies |

---

## Quickstart (no GPU, no weights, no services)

The decoupled integration example. It drives the complete `CacheService`
lifecycle against a fake in-memory diffusion engine, so you can see exactly how
the five hooks a real `FrameworkAdapter` wires up fit together.

```bash
python examples/approximate_reuse/quickstart_lifecycle.py
```

It serves three requests and prints what the cache decides for each:

```
>>> request: 'a cinematic first-person walk through a city, blue van at the corner'
    cache miss — compute steps 0..20
    computed 20/20 steps (MISS)          # donor: cold, populates the cache

>>> request: 'a cinematic first person walk through a city, red van at the corner'
    apply_resume: SkipStep(k=10) — reuse donor '...blue van...' (sim=...)
    computed 10/20 steps (HIT)           # near-duplicate: skips the cached early steps

>>> request: 'underwater coral reef with a sea turtle gliding past'
    cache miss — compute steps 0..20
    computed 20/20 steps (MISS)          # unrelated: full compute
```

The near-duplicate reuses the donor's first 10 cached steps and computes only the
remaining 10 — that's the cross-request saving, end to end.

## End-to-end with real models (in-process, no video model needed)

Exercises the real production assembly chain — encoder factory → vector store →
KV store → async save → lookup — without a video generation model. `save` is fed
synthetic latents plus a real (or synthetic) frame sequence, so it validates the
plumbing, not Wan2.2 itself.

```bash
export QWEN_EMBED_PATH=/path/to/Qwen3-VL-Embedding-2B
export QWEN_RERANK_PATH=/path/to/Qwen3-VL-Reranker-2B
rm -rf /tmp/approx_e2e                      # start from a cold cache
CUDA_VISIBLE_DEVICES=0 python examples/approximate_reuse/e2e_real_components.py
```

The script asserts (exit 0 on success, with a JSON summary of the checks):

- cold lookup → **miss**
- same prompt → **hit**, `SkipStep(k=5)`, and the latent survives a **byte-identical
  KV round-trip**
- a rewritten prompt → reported as an approximate hit at lower similarity
- an **unrelated** prompt → **rejected by the 0.80 rerank threshold**

> **Why rerank is on by default.** With rerank *off*, the unrelated prompt
> sneaks through as a false hit at similarity ≈ 0.41 (above the 0.10 vector
> threshold). The reranker is the gate that catches it — this example is the A/B
> evidence for keeping `rerank_enabled=True`.

Config is supplied by [`e2e_ppl_config.py`](e2e_ppl_config.py), a `CACHE_CONFIG`
dict whose paths come from environment variables so it's portable across
machines. Override the work dir with `APPROX_E2E_DIR` (default `/tmp/approx_e2e`).
To use Fluxon as the KV backend instead of local disk:

```bash
APPROX_KV_STORE=fluxon FLUXON_CONFIG=/path/to/external_config.yaml \
  CUDA_VISIBLE_DEVICES=0 python examples/approximate_reuse/e2e_real_components.py
```

> If the `cacheseek` in your venv is an editable install pointing at a *different*
> checkout, set `WORLDKV_REPO_ROOTS=<this repo root>` to force imports to resolve
> here.

## In a live Wan2.2 service (real generation)

The full serving path: a real Wan2.2 service, with CacheSeek embedded in its
process, hit over HTTP.

**1. Start the service.** Presets (port, KV backend, cache mode) live at the top
of [`service/start_wan22_service.py`](service/start_wan22_service.py). Preview the
resolved config with `--dry-run` before a real launch:

```bash
bash examples/approximate_reuse/service/start_wan22_service.sh --preset s1_fresh --dry-run
bash examples/approximate_reuse/service/start_wan22_service.sh --preset s1_fresh
```

The `.sh` wrapper raises `ulimit -n 65536` (Wan2.2 worker spawn exhausts the
default 1024 FD limit) and finds TeleFuser's venv at a `TeleFuser/` sibling of
this repo.

**2. Send the same prompt twice.** The second request should hit and skip the
first K steps:

```bash
python examples/approximate_reuse/e2e_service_smoke.py \
    --base-url http://127.0.0.1:8007 \
    --cache-log /tmp/cacheseek_wan22/latent_cache_s1_fresh/logs/cache_service.log
```

**Reading the result.** The HTTP response doesn't expose a hit flag, so the smoke
test has two verdicts:

- **With `--cache-log`** (recommended): the service log's second-request
  `lookup HIT` line is the authoritative verdict; exit 0 iff the hit is confirmed
  (`cache_hit_confirmed`).
- **Without `--cache-log`**: falls back to a **timing heuristic** (`t2 < ratio·t1`,
  default `0.75`). Treat this as a rough signal only — the first request includes
  one-time `torch.compile` / autotune warmup that inflates the apparent speedup,
  while a real hit only skips `max_skip_step` steps (e.g. 3 of 40).

For deeper service experiments (forced-K sweeps, hit-rate measurement) use
[`../../scripts/run_experiment.py`](../../scripts/run_experiment.py). For bringing
up the Fluxon stack see [`../../scripts/fluxon/README.md`](../../scripts/fluxon/README.md).
