#!/usr/bin/env bash
# Read-side I/O decomposition of self-KV restore. Points at an already-populated
# cache dir (from a prior service self-KV run) and times read/deserialize/H2D.
set -e
set -u

EXP_DIR="${EXP_DIR:?EXP_DIR must be set by caller}"
REPO_PATH="${REPO_PATH:?REPO_PATH must be set by caller}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR must point at a populated self-KV cache}"

ulimit -n 65536
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2,3

echo "=== skv_io_probe ==="
echo "  cache_dir : ${CACHE_DIR}"
echo "  device    : cuda:0 (physical 2)"
echo "===================="

cd "${REPO_PATH}"
exec ./.venv/bin/python "${EXP_DIR}/bin/skv_io_probe.py" \
    --cache-dir "${CACHE_DIR}" \
    --device "cuda:0"
