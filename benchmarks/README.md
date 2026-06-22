# benchmarks — 测量工具（非教学示例）

| 目录 | 测什么 |
|---|---|
| [`fluxon/`](fluxon/) | Fluxon KV 带宽（put/get、3 种序列化形态、chunk 数扫描）与传输/发现探针。实测参考（H100 同机 dlpack）：GET ≈12.2 GB/s，PUT ≈1.06 GB/s |

跑法见各脚本 docstring；需要 Fluxon 栈在跑（[`../scripts/fluxon/README.md`](../scripts/fluxon/README.md)）。
