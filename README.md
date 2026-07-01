<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/cacheseek-logo.png">
    <img alt="CacheSeek" src="./docs/assets/cacheseek-logo.png" width="55%">
  </picture>
</p>
<h3 align="center">
Cross-request KV-cache middleware for world models — turn per-request cache into session-level continuation.
</h3>

<p align="center">
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-2196F3?labelColor=555555"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?labelColor=555555">
  <img alt="status: early release" src="https://img.shields.io/badge/status-early%20release-FF9800?labelColor=555555">
  <img alt="version: 0.1.0a1" src="https://img.shields.io/badge/pyproject-0.1.0a1-607D8B?labelColor=555555">
  <img alt="tests: smoke | ut | e2e" src="https://img.shields.io/badge/tests-smoke%20%7C%20ut%20%7C%20e2e-4CAF50?labelColor=555555">
</p>

CacheSeek is a cache **middle-layer** between the inference engine and distributed storage that lifts cross-request reuse out of the inference loop into one swappable layer. It ships **two reuse families, for two model architectures** — lossy *approximate* reuse for diffusion video, and lossless *exact* reuse for autoregressive-diffusion world models.

It is **open at both ends**: any **inference framework** above (via a swappable `FrameworkAdapter`) and any **distributed KV/latent store** below (Fluxon, Mooncake, Qdrant, local …), with **[TeleFuser](https://github.com/Tele-AI/TeleFuser)** and **[Fluxon](https://github.com/Tele-AI/Fluxon)** as the reference integrations, completing the **compute–cache–storage** stack.

## Two reuse families

### 1 · Exact reuse — AR-diffusion world models (LingBot-World-Fast)

Given a new session whose action prefix byte-for-byte matches one computed before, CacheSeek replays the matched prefix's KV (`FastForward`) and recomputes only the divergent tail. → `cacheseek/reuse/exact_prefix`

> A session generates chunks 0–4, **pauses and serializes its KV**, then a *fresh request* resumes from the cached KV for chunks 5–7 — bit-for-bit identical to never pausing. *(LingBot-World, first-person Great Wall.)*

<table>
<tr>
<td width="50%" align="center"><b>Session A</b> · chunks 0–4 &nbsp;<sub>(generated)</sub></td>
<td width="50%" align="center">⏸ save KV ▶ &nbsp; <b>Session B</b> · chunks 5–7 &nbsp;<sub>(resumed from cache)</sub></td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/91b23b48-42ad-40c6-bf40-7812d9649ce6" controls muted loop></video></td>
<td><video src="https://github.com/user-attachments/assets/f4cca44e-055d-4553-adf5-4b0822a77cf3" controls muted loop></video></td>
</tr>
</table>

<details>
<summary><b>▸ More session-continuation demos</b> — fantasy jungle · lake tree</summary>
<br>
<table>
<tr>
<td width="50%" align="center"><b>Session A</b> · chunks 0–4</td>
<td width="50%" align="center">⏸ save KV ▶ &nbsp; <b>Session B</b> · chunks 5–7 (from cache)</td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/fa3fd014-b6b3-4496-a539-6852103036d8" controls muted loop></video></td>
<td><video src="https://github.com/user-attachments/assets/1636140d-b636-445d-83e9-82b271b777b6" controls muted loop></video></td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/9d87a5a4-c464-4913-9266-5ccc71109c00" controls muted loop></video></td>
<td><video src="https://github.com/user-attachments/assets/e9303e88-f166-44df-a9c7-623a6c16451f" controls muted loop></video></td>
</tr>
</table>
</details>

This exact path powers three world-model serving patterns:

- **Session Continuation** — serialize the multi-layer KV, pointer index, action / camera trajectory, and cross-attention cache at request end; a new request with the same `session_id` restores the exact breakpoint.
- **Interactive Branch Reuse (prefix tree)** — for traffic that explores many action branches from a shared initial state, a matching prefix reuses the shared chunks' KV and recomputes only the divergent branch.
- **First-Chunk Warmup** — precompute and cache the first chunk for high-frequency initial conditions, flattening the cold-start latency spike.

### 2 · Approximate reuse — diffusion video (Wan2.2)

Given a new request *semantically close* to one served before, CacheSeek loads the donor's early-denoise latents and **skips the first K denoise steps** (`SkipStep`). The hit decision is prompt / video embedding → ANN retrieval → rerank gate. **26–33% end-to-end latency reduction** on Wan2.2-14B, and it stacks on top of in-request caches (cache-dit / DeepCache / TeaCache) and TeleFuser AdaTaylor. → `cacheseek/reuse/approximate`

| | | | |
|:---:|:---:|:---:|:---:|
| ![sidewalk donor](./docs/assets/examples/sidewalk-baseline.gif) | ![tray donor](./docs/assets/examples/tray-baseline.gif) | ![apple donor](./docs/assets/examples/apple-baseline.gif) | ![flowers donor](./docs/assets/examples/flowers-baseline.gif) |
| ![sidewalk cache](./docs/assets/examples/sidewalk-cache.gif) | ![tray cache](./docs/assets/examples/tray-cache.gif) | ![apple cache](./docs/assets/examples/apple-cache.gif) | ![flowers cache](./docs/assets/examples/flowers-cache.gif) |

<p align="center"><em>Wan2.2-14B T2V on 2×80GB Hopper. In each column the top clip is generated cold (full denoise); the bottom is a follow-up request served from cache — semantically close to the top, it reused the top's early-denoise latents as donor and skipped the first K denoise steps.</em></p>

---

## About this release

Reuse *inside* a single generation is already well-served — [`cache-dit`](https://github.com/vipshop/cache-dit)
alone bundles many such algorithms to skip redundant compute mid-request. CacheSeek is
complementary: it works a different axis — reuse *across* requests, carrying one request's
computed state forward so a later, related request can start from it instead of from scratch.

Its core move is to stop treating the KV cache as a per-request throwaway and manage it instead as a
**continuable state snapshot** — gathering the scattered hidden states and scheduling
indices behind one abstraction, exposed through standardized **serialization, migration,
and breakpoint-resume** interfaces. Those interfaces are the concrete seam between compute
and storage: retrieval, hit decisions, metadata, and policy for **cross-request** reuse
all live behind them, so neither the framework above nor the storage below has to know
how it is decided.

The diagram below places that seam in the full compute–cache–storage stack:

<p align="center">
  <img alt="CacheSeek architecture overview" src="./docs/assets/cacheseek-architecture.png" width="90%">
</p>

---

## Quickstart

### Install & verify (no GPU, no weights, no services)

Both demos pass on core deps using in-memory stubs:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python examples/exact_prefix_reuse/quickstart_trie.py        # exact-prefix trie: hit / fork / eviction (lossless)
python examples/approximate_reuse/quickstart_lifecycle.py    # approximate: a cache MISS, then a step-skipping HIT
```

> For wiring checks, `pip install -e ".[dev]"` then `pytest -m smoke`.

### Exact reuse — AR-diffusion world models (LingBot-World-Fast)

Run it end to end on real LingBot-World, byte-exact. It runs from **TeleFuser's venv** (the script imports the `telefuser` pipeline — same venv as the Wan2.2 path below):

```bash
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/lingbot-world-base-cam
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0 <telefuser>/.venv/bin/python \
  examples/exact_prefix_reuse/e2e_telefuser_lingbot.py \
  --frame-num 37 --prefix-chunks 2 --out-dir /tmp/worldkv_e2e
```

> Full ladder (trie → KV binding → e2e): [`examples/exact_prefix_reuse/`](./examples/exact_prefix_reuse/).

### Approximate reuse — diffusion video (Wan2.2)

This path goes through `CacheService.from_config(yaml)` — two calls bracket inference:

```python
from cacheseek import CacheService
cache = CacheService.from_config("config.yaml")

result = await cache.lookup(query)   # pre-inference: fetch reusable latents on a hit
await cache.save(query, outputs)     # post-inference: store reusable latents
```

The `config.yaml` maps 1:1 onto [`CacheConfig`](./cacheseek/service/config.py) (omitted keys use defaults) — every knob is documented under [Key configuration → Approximate reuse](#key-configuration) below, with [`quickstart.yaml`](./quickstart.yaml) as a ready-to-edit template.

Run it end to end with TeleFuser (needs GPU + Wan2.2-14B + Qwen3-VL weights):

```bash
<telefuser>/.venv/bin/pip install -e ".[all,dev]"    # 1. install INTO TeleFuser's venv, with all backends + test deps (see notes)
<telefuser>/.venv/bin/python -c "import cacheseek, torch; print(torch.__version__)"  # 1b. verify (expect 2.7.0+cu126)
pytest -m smoke                                      # 2. local check
$EDITOR examples/approximate_reuse/service/start_wan22_service.py    # 3. pick/add a PRESETS entry (telefuser_repo, port, KV backend)
python examples/approximate_reuse/service/start_wan22_service.py --dry-run   # 4a. preview resolved config, no GPU/infra
bash examples/approximate_reuse/service/start_wan22_service.sh --preset s1_rw_fluxon   # 4b. real launch (needs TeleFuser venv)
```

> **Extras.** `[all]` = `qdrant` + `faiss` + `encoder` (vector stores + Qwen3-VL embed/rerank deps); `[dev]` = test deps. Omit both for a bare lifecycle smoke; install `[all,dev]` for real hits / rerank. torch is pinned to `2.7.0+cu126` by TeleFuser — build its venv first.

---

## Key configuration

### Exact reuse — `WorldKVConfig`

Defined in [`cacheseek/reuse/exact_prefix/config.py`](./cacheseek/reuse/exact_prefix/config.py).

| Field | Effect |
|---|---|
| `window_chunks` | `W` — the local attention window, measured in chunks |
| `sink_chunks` | Pinned window head (the chunks at the head that are never evicted) |
| `break_even_k` | Smallest reused-prefix length that pays off; below it, skip the cache and just generate — harmless |
| `quant` | `none` (bf16, lossless default) / `int8` / `int4` — enable quantization only after a quality A/B passes |
| `commit_tier` | Where committed KV lives (e.g. Fluxon DRAM) |

### Approximate reuse — `CacheConfig`

All fields are defined in [`CacheConfig`](./cacheseek/service/config.py).

| Field | Default | Effect |
|---|---|---|
| `key_steps` | `[5, 10, 15, 20, 25]` | Which denoise steps to reuse |
| `max_skip_step` | `5` | Upper bound on how many steps a hit may skip |
| `rerank_score_threshold` | `0.80` | Reranker hit threshold |
| `rerank_top_k` | `5` | Number of rerank candidates |
| `kv_store_type` | `fluxon` | KV backend; choose from open-source distributed cache substrates Fluxon / Mooncake, or local disk (`local_file`) |
| `vector_store_type` | `faiss` | Vector backend; FAISS for local, Qdrant recommended for service deployments |

The full field list lives in the `CacheConfig` dataclass ([`./cacheseek/service/config.py`](./cacheseek/service/config.py)).

#### Staircase skip-step (rerank-score-tiered)

Opt-in (`staircase_skip_enabled=True`, default `False`; needs `rerank_enabled` and a rerank score). The skip depth is then tiered by the donor's rerank score instead of a flat `max_skip_step` — higher score, deeper reuse — always clamped to `max_skip_step` and snapped to a checkpointed step (`saved_steps`). No score → falls back to the flat "largest saved step ≤ `max_skip_step`".

> 📊 **Benchmark results to be added** — the fitted `skip_step_tau_table` thresholds and the per-tier speedup / drift on Wan2.2 will be published here.

---

## Integration

CacheSeek plugs into an engine through one integration shape per reuse family:

| Reuse family | Reference engine / model | Integration point | How it hooks in |
|---|---|---|---|
| **Exact prefix** (AR-diffusion world model) | TeleFuser × **LingBot-World** | `LingBotWorldKVBinding` (two runtime hooks) | `on_runtime_created` → action-prefix trie lookup + materialize cached KV; `on_chunk_finalized` → ingest the chunk's KV |
| **Approximate** (video DiT) | TeleFuser × **Wan2.2-14B** | `TeleFuserCacheAdapter` (a `FrameworkAdapter`) | `build_query` → `CacheService.lookup` → `apply_resume` (skip steps) → `on_response` → `CacheService.save` |

**Adding an AR-diffusion engine (exact):** wire the two binding hooks (lookup + materialize on session start; ingest per finalized chunk) into your runtime, keyed on the action prefix.

**Adding an inference framework (approximate):** implement the `FrameworkAdapter` Protocol (`build_query` / `apply_resume` / `on_response`) under `cacheseek/adapters/<framework>/` — caching strategy, KV, and vector backends stay untouched.

### Pluggable axes

| Axis | Current implementation | Planned / reserved | Abstraction |
|---|---|---|---|
| Caching strategy | **ExactPrefixCache** + **VideoBasedApproximateCache** | NIRVANA / ReDi / ReCon / Chorus, hybrid | `Strategy` |
| KV / tensor store | **Fluxon** + local disk (`local_file`) | Mooncake, others | `KVStore` |
| Vector store — approximate only | **FAISS** + `Qdrant` | Other vector retrieval backends | `VectorStore` |
| Encoder / reranker — approximate only | **Qwen3-VL** | — | `Encoder` / `Reranker` |
| Model profiles | **Wan2.2** (approximate) + **LingBot-World** (exact) | LTX-Video, OpenSora, HunyuanVideo, VLA | `ModelProfile` |

---

## Relation to in-request cache libraries

CacheSeek is **complementary** to [`cache-dit`](https://github.com/vipshop/cache-dit) /
DeepCache / TGate / TeaCache and stacks with them:

| Dimension | In-request cache library | **CacheSeek** |
|---|---|---|
| Scope | Inside a single denoise loop | Across independent requests, accumulated and persistent |
| Hit trigger | Inter-step feature delta, timestep rules | prompt / video embedding + rerank |
| Storage | In-process tensor cache, freed at end of inference | Persistent KV backend + vector index |
| Backend deps | Typically none | KVStore + VectorStore + MetadataStore + AuditLog |
| Optimization target | Cut duplicate compute inside one inference | Reuse recoverable intermediate state across similar requests |

---

## Repository layout

```
cacheseek/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── quickstart.yaml                ← Option A starter config
├── cacheseek/
│   ├── service/    ← CacheService orchestrator, CacheConfig, Protocol interfaces
│   ├── reuse/      ← reuse strategies: approximate (video DiT) / exact_prefix (trie)
│   ├── stores/     ← KV / tensor byte stores (memory / local_file / fluxon)
│   ├── backends/   ← vector (faiss/qdrant), encoder (Qwen3-VL), metadata, audit
│   ├── adapters/   ← framework adapters (telefuser)
│   └── engines/    ← engine-binding facade
├── examples/       ← runnable demos + Wan2.2 launcher
├── benchmarks/     ← Fluxon bandwidth / transport probes
├── scripts/        ← preflight / experiment utilities
├── docs/           ← README assets
└── tests/          ← smoke + unit + e2e
```

---

## Roadmap

### ✅ Shipped

- ✅ LingBot-World exact-prefix session continuation — lossless, bit-for-bit resume across requests (the `reuse/exact_prefix` path: trie + KV snapshot persistence)
- ✅ TeleFuser × Wan2.2-14B approximate reuse — first reference framework + approximate path, exercised by the live hit-pair e2e test on real Fluxon + Qdrant
- ✅ KV: Fluxon, local disk — Fluxon for distributed multi-instance pools, local disk (`local_file`) for single-machine smoke / dev
- ✅ Vector: FAISS, Qdrant — FAISS for embedded local search, Qdrant for service deployments with shared collections

### 🚧 Near term

- ⬜ Multi-strategy benchmark + demo docs — reproducible numbers + scripted demos for each Strategy impl so users can pick the right one for their workload
- ⬜ Active eviction (LRU / size-based) — wire the dormant `EvictionPolicy` Protocol into the save hot path so cache size stays under `max_cache_size_gb` automatically
- ⬜ Post-hit CLIP-quality evaluation doc — measure semantic / visual fidelity of skip-step outputs against the cold-run baseline so users can pick a `rerank_score_threshold` with quality numbers, not just hit rate
- ⬜ LTX-2.3 reuse strategy — add a `ModelProfile` + `Strategy` for LTX-Video 2.3, the second concrete model after Wan2.2 to validate the 5-axis seam end-to-end

### 🌱 Long term

- ⬜ vLLM-Omni / SGLang-Diffusion adapters — bring two more inference frameworks under the same `FrameworkAdapter` Protocol, broadening the A1 axis beyond TeleFuser
- ⬜ Exact-hash, hybrid, NIRVANA / ReDi / ReCon / Chorus strategies — swap-in alternative `Strategy` impls beyond video-approximate, covering exact-key reuse and concept-level retrieval
- ⬜ Profiles for world models, VLA, fuzzy LLM prefix reuse — extend `ModelProfile` to non-video DiT regimes, opening a third reuse pattern beyond per-step latent and exact KV

---


<p align="center">
  <strong><em>⭐ Star us if you love skipping the wait!</em></strong>
</p>

## Acknowledgments

### Infrastructure

- **[Fluxon](https://github.com/Tele-AI/Fluxon)** — currently the recommended distributed KV backend for cross-request latent / KV / state sharing.
- **[TeleFuser](https://github.com/Tele-AI/TeleFuser)** — the first integrated inference framework.

### Cross-request cache research

(Inspired this repo's strategy roadmap.)

| Paper | Venue | Contribution |
|---|---|---|
| [ReDi](https://arxiv.org/abs/2302.02285) | ICML 2023 | Cross-request trajectory retrieval; trajectory KB + Lipschitz bound enables diffusion step-skipping reuse |
| [NIRVANA](https://arxiv.org/abs/2312.04429) | NSDI 2024 | Approximate cache system design for diffusion serving |
| [ReCon](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7666_ECCV_2024_paper.php) | ECCV 2024 | Concept-level retrieval; concept KB + cosine-weighted noise aggregation |
| [FlexCache](https://arxiv.org/abs/2501.04012) | arXiv 2025 | Cross-request cache layout for video diffusion |
| [Chorus](https://arxiv.org/abs/2604.04451) | paper-only | Cross-request cache for video DiT |

### In-request DiT cache libraries (complementary to CacheSeek)

[`cache-dit`](https://github.com/vipshop/cache-dit) /
[DeepCache](https://github.com/horseee/DeepCache) /
[L2C](https://arxiv.org/abs/2406.01733) /
[TeaCache](https://arxiv.org/abs/2411.19108) /
[PAB](https://arxiv.org/abs/2408.12588) /
[AdaCache](https://arxiv.org/abs/2411.02397).

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
