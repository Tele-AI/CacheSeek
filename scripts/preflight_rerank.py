#!/usr/bin/env python3
"""Preflight rerank scoring — confirm a (donor, request) pair will hit before generating.

Each Wan2.2-T2V video is ~5 min; rerank scoring is ~1s/pair, so gating a run
on the reranker (which decides cache hits) is ~300x cheaper than discovering
the hard way that everything missed.

Input — JSONL, one pair per line:
    {"name": "<id>", "donor": "<prompt>", "request": "<prompt>"}

Run (needs the reranker weights + cacheseek importable):
    export RERANKER_MODEL_PATH=/path/to/Qwen3-VL-Reranker-2B   # or pass --model
    python scripts/preflight_rerank.py --pairs pairs.jsonl --threshold 0.80

Inline single pair (no file):
    python scripts/preflight_rerank.py --donor "..." --request "..." --threshold 0.80

Reranker absolute scores depend on prompt style (length, detail), so they are
not a universal constant — scan candidates per new scenario to pick a threshold.

Exit codes:
    0  hit rate >= --min-hit (default 1.0 = all pairs must hit)
    1  hit rate too low
    2  bad input (file / JSON / missing field)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_pairs(args) -> list[dict]:
    if args.donor is not None and args.request is not None:
        return [{"name": args.name or "inline", "donor": args.donor, "request": args.request}]
    if args.pairs is None:
        print("[err] need --pairs FILE  or  --donor STR --request STR", file=sys.stderr)
        sys.exit(2)
    if not args.pairs.exists():
        print(f"[err] pairs file not found: {args.pairs}", file=sys.stderr)
        sys.exit(2)
    pairs = []
    with args.pairs.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[err] {args.pairs}:{ln}  JSON error: {e}", file=sys.stderr)
                sys.exit(2)
            for k in ("donor", "request"):
                if not isinstance(p.get(k), str) or not p[k].strip():
                    print(f"[err] {args.pairs}:{ln}  missing or empty {k!r}", file=sys.stderr)
                    sys.exit(2)
            p.setdefault("name", f"pair{ln}")
            pairs.append(p)
    if not pairs:
        print(f"[err] {args.pairs}: no pairs", file=sys.stderr)
        sys.exit(2)
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Preflight rerank scoring — confirm hit rate before expensive generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for usage and threshold guidance.",
    )
    ap.add_argument("--pairs", type=Path, help="JSONL of {name, donor, request} (one per line)")
    ap.add_argument("--donor", type=str, help="Inline donor prompt (with --request)")
    ap.add_argument("--request", type=str, help="Inline request prompt (with --donor)")
    ap.add_argument("--name", type=str, default=None, help="Name for the inline pair")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Hit iff score >= threshold (default 0.80, matches rerank_score_threshold)")
    ap.add_argument("--min-hit", type=float, default=1.0,
                    help="Exit 0 iff hit rate ≥ this fraction (default 1.0 = all must hit)")
    ap.add_argument("--model", default=os.environ.get("RERANKER_MODEL_PATH", ""),
                    help="Path to Qwen3-VL-Reranker-2B (or set $RERANKER_MODEL_PATH); "
                         "e.g. /path/to/Qwen3-VL-Reranker-2B")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSONL output of results")
    ap.add_argument("--quiet", action="store_true", help="Only print summary line")
    args = ap.parse_args()

    pairs = _load_pairs(args)
    if not args.model:
        print("[err] --model is required (or set RERANKER_MODEL_PATH env var); "
              "e.g. /path/to/Qwen3-VL-Reranker-2B", file=sys.stderr)
        sys.exit(2)
    if not args.quiet:
        print(f"[preflight] pairs={len(pairs)} threshold={args.threshold} "
              f"min_hit={args.min_hit*100:.0f}%", flush=True)

    # Heavy import (loads Qwen3-VL-Reranker-2B ~3GB on GPU); held until args validated.
    from cacheseek.backends.encoder.qwen3vl import Qwen3VLReranker

    rr = Qwen3VLReranker(model_path=args.model, device_id=args.device_id,
                         batch_size=args.batch_size)

    results = []
    hits = 0
    name_w = max(8, max(len(p["name"]) for p in pairs))
    if not args.quiet:
        print(f"\n{'name':<{name_w}}  {'score':>7}   {'gap':>7}   hit")
        print("-" * (name_w + 28))
    for p in pairs:
        score = float(rr.score_mm({"text": p["request"]}, [{"text": p["donor"]}])[0])
        hit = score >= args.threshold
        hits += int(hit)
        gap = score - args.threshold
        flag = "✓" if hit else "✗"
        if not args.quiet:
            print(f"{p['name']:<{name_w}}  {score:>7.4f}   {gap:>+7.4f}   {flag}")
        results.append({
            "name": p["name"], "score": score, "hit": hit,
            "threshold": args.threshold, "gap_to_threshold": gap,
        })

    rate = hits / len(pairs)
    if not args.quiet:
        print("-" * (name_w + 28))
    summary = f"hit rate: {hits}/{len(pairs)} = {rate*100:.1f}%  (need ≥ {args.min_hit*100:.0f}%)"
    print(summary)

    # Actionable hint if everything missed
    if rate < args.min_hit and not args.quiet:
        top = max(r["score"] for r in results)
        if top < args.threshold:
            print(f"\n[hint] no pair reached threshold {args.threshold}; max score = {top:.4f}.")
            if top < 0.7:
                print("       → either: lower --threshold to ~0.6 / rewrite requests to be longer / "
                      "less explicitly contradictory.")
            else:
                print(f"       → consider --threshold {top - 0.02:.2f} for this prompt style.")

    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"results → {args.out}")

    sys.exit(0 if rate >= args.min_hit else 1)


if __name__ == "__main__":
    main()
