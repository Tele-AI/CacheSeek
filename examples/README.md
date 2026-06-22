# Examples

Runnable examples for CacheSeek's two reuse families. **Which reuse strategy
applies depends on the model's architecture** — so the two folders are split
along that line, not by feature.

If you just want to see CacheSeek work, start with either quickstart — both run
on the core install alone (no GPU, no model weights, no external services):

```bash
pip install -e .
python examples/approximate_reuse/quickstart_lifecycle.py   # diffusion reuse, in 5s
python examples/exact_prefix_reuse/quickstart_trie.py        # AR / self-forcing reuse, in 5s
```

## Reuse strategy follows architecture

| Folder | Model architecture | Why this reuse fits | Hit means | Reuse type |
|---|---|---|---|---|
| [`approximate_reuse/`](approximate_reuse/) | **Diffusion** — full-sequence bidirectional denoising over ~40 steps (e.g. Wan2.2 T2V) | Early denoise steps carry coarse, prompt-level structure shared across *similar* requests | A *similar enough* past request exists (embedding + rerank) | **Lossy** — skip the first K denoise steps, seeded from a donor's cached latents (`SkipStep`) |
| [`exact_prefix_reuse/`](exact_prefix_reuse/) | **AR-Diffusion / self-forcing** — autoregressive, chunk-by-chunk generation conditioned on prior frames (e.g. LingBot world model) | Causal + deterministic: an identical action prefix reproduces identical chunks exactly, like KV prefix caching in autoregressive LLMs | An *identical* action prefix was computed before (hash trie) | **Lossless** — replay the matched prefix, recompute only the divergent tail (`FastForward`) |

Each folder's README explains its model class and walks every example with run
commands, prerequisites, and expected output.

## Verified in CI

`tests/test_examples_smoke.py` runs the zero-dependency quickstarts on every
test pass, so the entry-level examples can't silently rot. The GPU and service
paths are exercised by the opt-in `e2e` tests (see the repo root README).

## Related (not examples)

- [`../benchmarks/fluxon/`](../benchmarks/fluxon/) — Fluxon bandwidth / transport probes
- [`../scripts/fluxon/README.md`](../scripts/fluxon/README.md) — bringing up and debugging the Fluxon stack
- `../scripts/run_experiment.py` — service-level experiment driver (forced-K sweeps, hit-rate measurement)
