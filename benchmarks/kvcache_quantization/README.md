# LingBot Exact-Reuse KV Quantization Benchmark

`lingbot_exact_reuse.py` measures whether KIVI-compressed exact-prefix KV reuse keeps LingBot World output quality close to an unquantized reuse baseline while reducing forked-request latency versus a cold fork.

The script runs five requests:

- `main_cache_seed_quant`: cold main trajectory that writes the selected KIVI KV cache.
- `fork_cache_reuse_quant`: forked trajectory that reuses the quantized prefix.
- `main_cache_seed_none`: cold main trajectory for the unquantized baseline.
- `fork_cache_reuse_none`: forked trajectory that reuses unquantized prefix KV.
- `fork_cold_reference`: forked trajectory with an empty cache.

Quality metrics are aligned by absolute chunk index, so the fast-forwarded suffix is compared against the matching cold-reference chunks instead of frame zero.

## Requirements

- CUDA-capable environment with LingBot/Telefuser dependencies installed.
- `LINGBOT_WORLD_CHECKPOINT_DIR` set to the LingBot World checkpoint directory.
- If `telefuser` is not importable, set `TELEFUSER=/path/to/telefuser-internal`.
- If editable installs resolve to another checkout, set `WORLDKV_REPO_ROOTS=/path/to/telefuser:/path/to/CacheSeek`.

## Basic Usage

```bash
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/lingbot-world-base-cam
python benchmarks/kvcache_quantization/lingbot_exact_reuse.py \
  --quant kivi_int4 \
  --out-dir /tmp/worldkv_quant_benchmark_kivi_int4
```

`--quant` is required and accepts `kivi_int4` or `kivi_int8`. The unquantized `none` baseline is always run automatically.

The command prints a JSON summary and writes:

- `manifest.json`: run configuration, timings, speedups, quality metrics, cache stats, validity checks, and profiling metadata.
- Fork videos for quantized reuse, unquantized reuse, and the cold reference.
- Optional per-frame images when `--save-frames` is enabled.

The process exits with status `0` only when all prefix-hit, alignment, and timing-validity checks pass.

## Useful Options

- `--frame-num N`: total requested frames, default `37`.
- `--prefix-chunks N`: reusable prefix chunks before the fork, default `2`.
- `--prompt TEXT`, `--seed N`: generation inputs.
- `--store inmem|localdisk|fluxon`: cache backing store, default `inmem`.
- `--group-size N`: KIVI quantization group size, default `64`.
- `--video-format mp4|gif`: output video format, default `mp4`.
- `--aux-device cuda:1`: place text encoder and VAE on another CUDA device.

## Profiling

Benchmark timings are collected without profiler overhead. To diagnose a stage, run a separate profiling pass after the benchmark:

```bash
python benchmarks/kvcache_quantization/lingbot_exact_reuse.py \
  --quant kivi_int4 \
  --profile cprofile \
  --profile-target fork_cache_reuse_quant \
  --profile-scope create_runtime
```

Use `--torch-profile fork_cache_reuse_quant` for a separate `torch.profiler` trace. cProfile and torch profiling are intentionally mutually exclusive.

To choose where diagnostic artifacts are saved, pass `--profile-dir` for cProfile outputs or `--torch-profile-dir` for torch traces:

```bash
python benchmarks/kvcache_quantization/lingbot_exact_reuse.py \
  --quant kivi_int4 \
  --torch-profile fork_cache_reuse_quant \
  --profile-scope create_runtime \
  --torch-profile-dir /tmp/worldkv_quant_traces/kivi_int4
```

## Repeated Trials

This script currently runs one benchmark suite per invocation and records `"repeated_trials": false` in `manifest.json`. It does not yet have an in-script trial loop for repeated measurements, aggregation, or variance reporting.

For timing numbers intended for comparison, keep profiler flags disabled. Use a separate invocation only when collecting cProfile or torch traces, because profiler overhead is intentionally excluded from benchmark timings.

## A Thorough Example

```bash
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/lingbot-world-base-cam
export ASSETS=/path/to/lingbot-world-assets
source /path/to/telefuser-env/bin/activate
python benchmarks/kvcache_quant/lingbot_exact_prefix_quant.py \
  --image-path "$ASSETS/image.jpg" \
  --action-path "$ASSETS" \
  --quant kivi_int8 \
  --group-size 64 \
  --frame-num 45 \
  --prefix-chunks 3 \
  --store inmem \
  --out-dir /tmp/benchmark_int8 \
  --torch-profile-chrome-trace \
  --torch-profile fork_cache_reuse_quant \
  --profile-scope create_runtime \
  --no-torch-profile-detail \
```
