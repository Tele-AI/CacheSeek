"""CacheSeek experiment driver — pure HTTP client.

POSTs each prompt of a JSONL dataset to an **already-running** TeleFuser
Wan2.2 service (started by ``examples/approximate_reuse/service/start_wan22_service.py``),
polls until completion, and appends a manifest. Zero TeleFuser / GPU /
cacheseek coupling — just ``httpx``.

The cache phase (``write_only`` / ``read_only`` / ``read_write``) is decided
by the **launcher's** ``CACHE_CONFIG``, not here. ``--phase`` is only a
manifest label (the ``experiment_id`` field) and is **optional** — it
defaults to ``rw`` for the single-pass read_write workflow; pass
``save``/``lookup`` only to label the two-phase write_only->read_only
workflow. Dataset rows tagged ``"group": "_skip"`` (the lookup warm-up
line) are skipped automatically.

Usage
-----
    # service already up via start_wan22_service.py with the desired cache_mode.
    # --phase is optional (defaults to "rw"); pass save/lookup only for the
    # two-phase write_only->read_only workflow.
    python scripts/run_experiment.py \\
        --dataset donors.jsonl \\
        --service-url http://127.0.0.1:8006 \\
        --output manifest.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ImportError:  # pragma: no cover — install-time guard
    print("[run_experiment] httpx is required: pip install httpx", file=sys.stderr)
    raise


DEFAULT_SEED = 42
DEFAULT_SERVICE_URL = "http://127.0.0.1:8006"
DEFAULT_TIMEOUT_S = 900.0  # 15 min per inference; ~360s typical for AdaTaylor ON


@dataclass
class TaskOutcome:
    """One row written to the manifest."""

    experiment_id: str
    prompt: str
    n: int
    seed: int = DEFAULT_SEED
    video_path: Optional[str] = None
    task_id: Optional[str] = None
    elapsed_s: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def load_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found at {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def append_manifest(manifest_path: Path, row: TaskOutcome) -> None:
    """Append one row (append-only keeps the writer single-process)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(row.to_jsonl() + "\n")


def _load_completed_n(manifest_path: Path) -> set[int]:
    """Return the set of ``n`` values already marked ``status=ok`` (for resume).
    Errors are NOT considered completed (so they retry on resume)."""
    if not manifest_path.exists():
        return set()
    done: set[int] = set()
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and isinstance(row.get("n"), int):
                done.add(row["n"])
    return done


def submit_task(
    client: httpx.Client,
    service_url: str,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    experiment_index: int,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    task: str = "t2v",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Submit one task to TeleFuser and block until completion."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "task": task,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "experiment_index": experiment_index,
    }

    base = service_url.rstrip("/")
    start = time.monotonic()
    resp = client.post(base + "/v1/tasks/create", json=payload, timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    task_id = body.get("task_id")
    if not task_id:
        raise RuntimeError(f"create returned no task_id: {body!r}")

    # Poll status — 30s interval to stay under TeleFuser rate limit (60 req/60s).
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        time.sleep(30.0)
        st_resp = client.get(base + f"/v1/tasks/{task_id}/status", timeout=10.0)
        st_resp.raise_for_status()
        st = st_resp.json()
        status = st.get("status") or st.get("task_status")
        if status == "completed":
            body.update(st)
            body["_driver_elapsed_s"] = time.monotonic() - start
            return body
        if status in ("failed", "error", "cancelled"):
            err = st.get("error") or st.get("message") or "Inference failed"
            raise RuntimeError(f"task {task_id} terminal status={status} error={err!r}")
    raise TimeoutError(f"task {task_id} did not complete within {timeout_s:.0f}s")


def run_phase(
    *,
    dataset: list[dict],
    service_url: str,
    manifest_path: Path,
    seed: int,
    negative_prompt: str,
    phase: str,
    resume: bool = False,
) -> None:
    """POST every non-``_skip`` row once; append outcome to the manifest.

    ``phase`` is only a manifest label — the cache_mode lives in the
    launcher's CACHE_CONFIG. Rows tagged ``"group": "_skip"`` (the lookup
    warm-up line) are skipped.
    """
    completed = _load_completed_n(manifest_path) if resume else set()
    if resume and completed:
        print(
            f"[run_experiment] resume: skipping {len(completed)} already-ok rows "
            f"(min={min(completed)} max={max(completed)})",
            file=sys.stderr,
        )
    n_post = sum(1 for r in dataset if r.get("group") != "_skip")
    print(f"[run_experiment] {phase} phase: {n_post} prompts → {manifest_path}", file=sys.stderr)

    with httpx.Client() as client:
        for offset, row in enumerate(dataset, start=1):
            if row.get("group") == "_skip":
                print(f"[run_experiment] n={offset}: skip (_skip marker)", file=sys.stderr)
                continue
            if offset in completed:
                continue
            prompt = row.get("prompt") or ""
            if not prompt:
                print(f"[run_experiment] skip n={offset}: empty prompt", file=sys.stderr)
                continue
            outcome = TaskOutcome(experiment_id=phase, prompt=prompt, n=offset, seed=seed)
            try:
                body = submit_task(
                    client,
                    service_url,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    experiment_index=offset,
                )
                outcome.status = "ok"
                outcome.task_id = body.get("task_id")
                outcome.video_path = body.get("output_path")
                outcome.elapsed_s = body.get("_driver_elapsed_s")
            except Exception as exc:
                outcome.status = "error"
                outcome.error = f"{type(exc).__name__}: {exc}"
                print(f"[run_experiment] {phase} n={offset} FAILED: {exc}", file=sys.stderr)
            append_manifest(manifest_path, outcome)
            print(
                f"[run_experiment] {phase} n={offset}/{len(dataset)} "
                f"status={outcome.status} elapsed={outcome.elapsed_s}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CacheSeek experiment driver — POST a JSONL dataset to a running TeleFuser service."
    )
    parser.add_argument(
        "--phase",
        choices=["save", "lookup", "rw"],
        default="rw",
        help="Manifest label only — written to experiment_id; never sent to the "
        "service, so it has zero effect on cacheseek (cache_mode lives in the "
        "launcher CACHE_CONFIG). Optional, defaults to 'rw' for the single-pass "
        "read_write workflow; pass save/lookup for the two-phase "
        "write_only->read_only workflow.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the prompt JSONL.")
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
        help=f"TeleFuser service base URL (default: {DEFAULT_SERVICE_URL}).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to append the manifest jsonl.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for inference (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Extra negative prompt (service ppl concatenates the canonical one).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already marked status=ok in the output manifest (errors retry).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if args.output.exists():
        print(
            f"[run_experiment] WARN: manifest {args.output} already exists; appending. "
            "Truncate manually (or use --resume) for a clean run.",
            file=sys.stderr,
        )

    run_phase(
        dataset=dataset,
        service_url=args.service_url,
        manifest_path=args.output,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        phase=args.phase,
        resume=args.resume,
    )

    print(f"[run_experiment] phase={args.phase} done → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
