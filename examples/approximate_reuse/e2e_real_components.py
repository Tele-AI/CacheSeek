"""approximate-reuse real-component e2e: real Qwen embedding(+reranker) + FAISS + optional Fluxon.

No dependency on a video-generation model -- save is fed a synthetic latent plus a frame
sequence from a real/synthetic image, exercising cacheseek's full production assembly chain
for semantic reuse (factory -> encoder -> vector store -> KV -> async save -> lookup hit):

    cold lookup     -> miss
    save(latent@step{5,10} + frames)
    same prompt lookup       -> hit + SkipStep.k=5 + latent bit-equal through the KV round trip
    paraphrased prompt       -> report hit/similarity (approximate semantics; hit expected,
                                similarity lower than exact)
    unrelated prompt         -> with rerank on (default) should be rejected by the 0.80 threshold

Usage (see the README in this directory for details):
    export QWEN_EMBED_PATH=/path/to/Qwen3-VL-Embedding-2B
    CUDA_VISIBLE_DEVICES=0 python examples/approximate/e2e_real_components.py \
        [--ppl examples/approximate/e2e_ppl_config.py] [--frames-image /path/to.jpg]

Exit 0 if all assertions pass; output JSON contains checks/info. Clear $APPROX_E2E_DIR
(default /tmp/approx_e2e) before rerunning. If the cacheseek in the running venv is an
editable install pointing at another checkout, set WORLDKV_REPO_ROOTS=<this repo root> to
enable the import-correction shim.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# --- editable-finder shim (optional): strip the venv cacheseek's PEP660 finder ---
_ROOTS = [r for r in os.environ.get("WORLDKV_REPO_ROOTS", "").split(":") if r]
if _ROOTS:
    sys.meta_path = [f for f in sys.meta_path if "__editable___cacheseek" not in getattr(f, "__module__", "")]
    for _r in reversed(_ROOTS):
        sys.path.insert(0, _r)

import torch
from PIL import Image

import cacheseek
import inspect
if _ROOTS:
    _src = inspect.getsourcefile(cacheseek) or ""
    assert any(_src.startswith(r) for r in _ROOTS), f"cacheseek resolved to {_src}"
    print(f"[shim] cacheseek -> {_src}", flush=True)

from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import SkipStep

P_MAIN = "an ancient stone courtyard with warm afternoon light"
P_PARA = "a quiet old courtyard of stone in soft afternoon sun"
P_FAR = "a futuristic city highway at night with neon cars"


def _frames(image_path: str) -> list[Image.Image]:
    if image_path:
        img = Image.open(image_path).convert("RGB").resize((416, 240))
    else:
        import numpy as np
        y, x = np.mgrid[0:240, 0:416]
        img = Image.fromarray(
            np.stack([(x % 256).astype("uint8"), (y % 256).astype("uint8"), ((x + y) % 256).astype("uint8")], axis=-1)
        )
    return [img] * 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl", default=str(Path(__file__).with_name("e2e_ppl_config.py")))
    ap.add_argument("--frames-image", default="", help="frame source image for save; empty = deterministic synthetic image")
    ap.add_argument("--visibility-timeout", type=float, default=120.0)
    args = ap.parse_args()

    pair = CacheServiceFactory.create_cache_service(args.ppl, True, "read_write")
    assert pair is not None, "factory assembly failed (see warning above)"
    service, _adapter = pair

    async def run():
        checks, info = {}, {}
        q = CacheQuery(prompt=P_MAIN, seed=42, task_type="t2v")
        checks["cold_miss"] = not (await service.lookup(q)).hit

        lat5 = torch.randn(1, 16, 8, 30, 52, dtype=torch.float16)
        outs = ModelOutputs(
            latent_states_dict={5: lat5, 10: torch.randn_like(lat5)},
            embedding_video_frames=_frames(args.frames_image),
            num_frames=8, final_step=25, saved_steps=[5, 10],
        )
        await service.save(q, outs)

        hit, t0 = None, time.time()                      # save enqueues async -> poll until visible
        while time.time() - t0 < args.visibility_timeout:
            r = await service.lookup(q)
            if r.hit:
                hit = r
                break
            await asyncio.sleep(2)
        checks["exact_hit"] = hit is not None
        if hit:
            info["exact_similarity"] = hit.matched_similarity
            checks["skip_step_hint"] = isinstance(hit.resume_hint, SkipStep) and hit.resume_hint.k == 5
            try:
                got = hit.payload.get_latent_at_step(5)
                checks["latent_roundtrip_equal"] = torch.equal(got.to(lat5.dtype), lat5)
            except Exception as e:
                info["latent_check_error"] = f"{type(e).__name__}: {e}"
                checks["latent_roundtrip_equal"] = False

        r_para = await service.lookup(CacheQuery(prompt=P_PARA, seed=42, task_type="t2v"))
        info["paraphrase"] = {"hit": r_para.hit, "similarity": r_para.matched_similarity}
        r_far = await service.lookup(CacheQuery(prompt=P_FAR, seed=42, task_type="t2v"))
        info["unrelated"] = {"hit": r_far.hit, "similarity": r_far.matched_similarity}
        return checks, info

    checks, info = asyncio.run(run())
    service.shutdown()
    out = {"checks": checks, "info": info, "all_pass": all(checks.values())}
    print(json.dumps(out, indent=2, default=str), flush=True)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
