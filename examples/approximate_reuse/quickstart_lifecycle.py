#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Minimal, host-agnostic CacheSeek example — no TeleFuser, no GPU, no qdrant.

This is the *decoupled* integration example: it shows the full CacheService
lifecycle against a fake in-memory diffusion engine, so you can see how
cross-request approximate reuse works without any model serving stack.

The five lifecycle steps a real FrameworkAdapter wires up
(see ``cacheseek.service.lifecycle.CacheService``):

    1. build_query   request            -> CacheQuery
    2. lookup        CacheService       -> LookupResult{hit, payload, resume_hint}
    3. apply_resume  resume_hint+payload -> seed the engine (skip early steps)
    4. on_response   raw engine output  -> ModelOutputs
    5. save          CacheService       -> persist for future hits

Run:
    python examples/minimal_cache_reuse.py
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root: run directly without pip install -e

from cacheseek import CacheQuery, CacheService
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.result import LookupResult, SkipStep

TOTAL_STEPS = 20          # a "full" denoise run is 20 steps
SIM_THRESHOLD = 0.5       # Jaccard word-overlap above which we reuse a donor


# --------------------------------------------------------------------------
# A toy "heavy payload": the cached early-denoise latent prefix. In real
# cacheseek this is VideoApproxPayload (latent tensors); here it's a list.
# --------------------------------------------------------------------------
@dataclass
class ToyLatentPayload:
    cached_prompt: str
    latent_prefix: list[int]   # latent snapshot up to `saved_through_step`
    saved_through_step: int


# --------------------------------------------------------------------------
# A minimal Strategy. Conforms structurally to cacheseek.service.interfaces.
# strategy.Strategy (just `async lookup` + `async save`) — no need to
# subclass BaseCacheStrategy or wire KV/vector backends for a demo.
# --------------------------------------------------------------------------
class InMemoryApproxCache:
    """Approximate reuse keyed by prompt word-overlap (Jaccard).

    Real strategies embed the prompt/video and search a vector store +
    rerank; here we just compare word sets to keep it dependency-free.
    """

    def __init__(self) -> None:
        self._store: dict[str, ToyLatentPayload] = {}

    @staticmethod
    def _words(prompt: str) -> set[str]:
        return set(prompt.lower().split())

    def _best_match(self, prompt: str) -> tuple[float, ToyLatentPayload | None]:
        q = self._words(prompt)
        best_sim, best = 0.0, None
        for payload in self._store.values():
            d = self._words(payload.cached_prompt)
            sim = len(q & d) / len(q | d) if (q | d) else 0.0
            if sim > best_sim:
                best_sim, best = sim, payload
        return best_sim, best

    async def lookup(self, query: CacheQuery, ctx=None) -> LookupResult:
        sim, payload = self._best_match(query.prompt)
        if payload is None or sim < SIM_THRESHOLD:
            return LookupResult.miss(reason="no_donor_above_threshold")
        # Hit: instruct the host to skip the first `saved_through_step` steps
        # (resume_hint = light instruction) and hand it the cached latent
        # prefix (payload = heavy data). This is the resume_hint / payload split.
        return LookupResult(
            hit=True,
            payload=payload,                       # type: ignore[arg-type]
            resume_hint=SkipStep(k=payload.saved_through_step),
            matched_similarity=sim,
            cached_prompt=payload.cached_prompt,
        )

    async def save(self, query: CacheQuery, outputs: ModelOutputs, ctx=None) -> None:
        through = outputs.final_step // 2          # cache the first half
        self._store[query.prompt] = ToyLatentPayload(
            cached_prompt=query.prompt,
            latent_prefix=outputs.extra["latent"][:through],
            saved_through_step=through,
        )


# --------------------------------------------------------------------------
# Fake "diffusion engine". A full run computes steps 0..TOTAL_STEPS. When the
# host applies a SkipStep(k) hint it seeds from the cached latent prefix and
# only computes steps k..TOTAL_STEPS — that skipped work is the cache win.
# --------------------------------------------------------------------------
def denoise(prompt: str, resume: LookupResult) -> tuple[list[int], int]:
    if resume.hit and isinstance(resume.resume_hint, SkipStep):
        k = resume.resume_hint.k
        latent = list[Any](resume.payload.latent_prefix)            # type: ignore[union-attr]
        start = k
        print(f"    apply_resume: SkipStep(k={k}) — reuse donor "
              f"'{resume.cached_prompt}' (sim={resume.matched_similarity:.2f}), "
              f"compute steps {k}..{TOTAL_STEPS}")
    else:
        latent, start = [], 0
        print(f"    cache miss — compute steps 0..{TOTAL_STEPS}")
    for step in range(start, TOTAL_STEPS):
        latent.append(step)                                    # toy "denoise"
    return latent, TOTAL_STEPS - start


async def serve(service: CacheService, prompt: str) -> None:
    print(f"\n>>> request: {prompt!r}")
    # 1. build_query
    query = CacheQuery(prompt=prompt, task_type="t2v")
    # 2. lookup
    result = await service.lookup(query)
    # 3. apply_resume + run engine
    latent, computed = denoise(prompt, result)
    print(f"    computed {computed}/{TOTAL_STEPS} steps "
          f"({'HIT' if result.hit else 'MISS'})")
    # 4. on_response
    outputs = ModelOutputs(final_step=TOTAL_STEPS, extra={"latent": latent})
    # 5. save
    await service.save(query, outputs)


async def main() -> None:
    # async_save=False → save runs inline, so the next lookup deterministically
    # sees what the previous request wrote (no save-worker race to wait on).
    service = CacheService(strategies=[InMemoryApproxCache()], async_save=False)
    try:
        # donor: cold, full compute, populates the cache
        await serve(service, "a cinematic first-person walk through a city, blue van at the corner")
        # near-duplicate: hits the donor, skips the cached early steps
        await serve(service, "a cinematic first person walk through a city, red van at the corner")
        # unrelated: misses, full compute
        await serve(service, "underwater coral reef with a sea turtle gliding past")
    finally:
        service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
