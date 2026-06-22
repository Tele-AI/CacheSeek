"""approximate x TeleFuser service-level e2e smoke (pure HTTP, no heavy imports).

Prerequisite: the wan22 service is already running (start it with
service/start_wan22_service.sh in this directory, cache_mode=read_write). This script
submits the same prompt twice:

    1st time -> cache miss, full denoise (baseline time t1)
    2nd time -> should hit the approximate cache (SkipStep skips the first k steps)
                -> t2 markedly smaller than t1

Hit detection has two tiers (the service response does not expose a hit field, so HTTP
alone cannot confirm a hit):
  - Pass --cache-log pointing at the service cache_service.log (recommended): the second
    request's lookup HIT line in the log is AUTHORITATIVE; exit 0 iff a hit is confirmed.
  - Omit --cache-log: fall back to a TIMING heuristic (t2 < ratio*t1, default 0.75). Note
    the first request includes one-time warmup (torch.compile / autotune) that inflates
    speedup -- when a hit only skips max_skip_step steps (e.g. 3/40) the real speedup is
    far smaller, so timing is indicative only.

Usage:
    python examples/approximate_reuse/e2e_service_smoke.py \
        --base-url http://127.0.0.1:8007 \
        [--cache-log <latent_cache_dir>/logs/cache_service.log] \
        [--prompt "..."] [--ratio 0.75] [--timeout 1800]

Deep service experiments (K sweep / hit rate) -> scripts/run_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

DEFAULT_PROMPT = "a cinematic aerial shot of a coastal village at golden hour"


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def _hit_after(path: str, since: int) -> dict | None:
    """Scan cache_service.log lines [since:] for an approximate lookup HIT.

    Returns the first hit's {cache_id, step, sim}, else None. Authoritative:
    the service HTTP response does not expose the hit field.
    """
    for line in _read_lines(path)[since:]:
        if "lookup hit cache_id=" not in line:
            continue
        m = re.search(r"cache_id=(\w+)\s+step=(\d+)\s+sim=([0-9.]+)", line)
        return (
            {"cache_id": m.group(1), "step": int(m.group(2)), "sim": float(m.group(3))}
            if m
            else {"raw": line.strip()}
        )
    return None


def run_once(base: str, prompt: str, timeout: float, tag: str) -> float:
    t0 = time.time()
    resp = _post(f"{base}/v1/tasks/create", {"prompt": prompt, "seed": 42})
    task_id = resp.get("task_id") or resp.get("id")
    assert task_id, f"no task_id in create response: {resp}"
    print(f"[{tag}] task_id={task_id}", flush=True)
    while time.time() - t0 < timeout:
        st = _get(f"{base}/v1/tasks/{task_id}/status")
        status = str(st.get("status", "")).lower()
        if status in ("completed", "succeeded", "success", "finished"):
            dt = time.time() - t0
            print(f"[{tag}] done in {dt:.1f}s", flush=True)
            return dt
        if status in ("failed", "error", "cancelled"):
            sys.exit(f"[{tag}] task {status}: {st}")
        time.sleep(3)
    sys.exit(f"[{tag}] timeout after {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8007")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--ratio", type=float, default=0.75, help="timing-heuristic hit verdict: t2 < ratio*t1"
    )
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument(
        "--cache-log",
        default="",
        help="path to the service cache_service.log; if given, the log's lookup HIT is the authoritative verdict (timing is only a hint)",
    )
    args = ap.parse_args()

    _get(f"{args.base_url}/v1/service/status")  # health check (raises if unreachable)
    t1 = run_once(args.base_url, args.prompt, args.timeout, "miss-run")
    hit_since = len(_read_lines(args.cache_log)) if args.cache_log else 0
    t2 = run_once(args.base_url, args.prompt, args.timeout, "hit-run ")

    out = {
        "t1_miss_s": round(t1, 1),
        "t2_hit_s": round(t2, 1),
        "speedup": round(t1 / max(t2, 1e-6), 2),
        "hit_heuristic_pass": t2 < args.ratio * t1,
    }
    if args.cache_log:
        hit = _hit_after(args.cache_log, hit_since)
        out["cache_hit_confirmed"] = hit is not None
        out["cache_hit"] = hit
        out["note"] = "authoritative verdict = cache_hit_confirmed; timing is only a hint (first request includes warmup)"
        ok = hit is not None
    else:
        out["note"] = (
            "no --cache-log: falling back to the timing heuristic; for an authoritative hit check the cache log's lookup HIT line"
        )
        ok = out["hit_heuristic_pass"]
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
