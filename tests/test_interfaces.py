"""Interface contract tests for cacheseek.service.interfaces + core data structures.

Covers:
- Core Protocol interfaces (``KVStore``, ``VectorStore``,
  ``MetadataStore``, ``AuditLog``, ``PromptEncoder``, ``VideoEncoder``,
  ``Reranker``, ``Strategy``, ``FrameworkAdapter``, ``EvictionPolicy``)
- ``Payload`` Protocol + ``PartialLoadSpec``
- ``ResumeHint`` sealed union (``SkipStep``, ``LoadStateSnapshot``,
  ``ReturnCachedOutput``, ``NoOp``)
- ``EvictionPolicy`` + ``LRUEviction`` default

These tests don't require GPU / fluxon / qdrant — they verify the
abstract shape + concrete impl conformance.
"""
from __future__ import annotations

import pytest


# ─── Core interface Protocols ───────────────────────────────────────────────


def test_core_interfaces_importable() -> None:
    from cacheseek.service.interfaces import (
        KVStore,
        VectorStore,
        MetadataStore,
        AuditLog,
        EvictionPolicy,
        PromptEncoder,
        VideoEncoder,
        Reranker,
        Strategy,
        FrameworkAdapter,
    )
    # All are runtime-checkable Protocols (have _is_runtime_protocol attr)
    for proto in (KVStore, VectorStore, MetadataStore, AuditLog, EvictionPolicy,
                  PromptEncoder, VideoEncoder, Reranker, Strategy, FrameworkAdapter):
        assert getattr(proto, "_is_runtime_protocol", False), \
            f"{proto.__name__} should be runtime-checkable Protocol"


def test_kv_store_protocol_satisfied_by_in_memory() -> None:
    """InMemoryKVStore satisfies the new core/interfaces/KVStore Protocol
    via structural typing (no inheritance change required)."""
    from cacheseek.stores import InMemoryKVStore
    from cacheseek.service.interfaces import KVStore

    kv = InMemoryKVStore()
    assert isinstance(kv, KVStore), "InMemoryKVStore should satisfy KVStore Protocol structurally"


def test_metadata_store_protocol_satisfied_by_local_manager(tmp_path) -> None:
    from cacheseek.backends.metadata import LocalCacheMetadataManager
    from cacheseek.service.interfaces import MetadataStore

    mgr = LocalCacheMetadataManager(metadata_cache_dir=tmp_path)
    assert isinstance(mgr, MetadataStore)


# ─── CacheQuery + LookupResult + ResumeHint sealed union ────────────────────


def test_cache_query_construct() -> None:
    from cacheseek import CacheQuery

    q = CacheQuery(prompt="hello", seed=42, task_type="t2v")
    assert q.prompt == "hello"
    assert q.seed == 42
    assert q.task_type == "t2v"
    assert q.model_profile is None
    assert q.hint_spec == {}


def test_lookup_result_miss() -> None:
    from cacheseek import LookupResult

    miss = LookupResult.miss()
    assert miss.hit is False
    assert miss.payload is None
    assert miss.resume_hint is None


def test_resume_hint_sealed_union() -> None:
    from cacheseek import (
        ResumeHint,
        SkipStep,
        LoadStateSnapshot,
        ReturnCachedOutput,
        NoOp,
    )

    hints = [
        SkipStep(k=5),
        LoadStateSnapshot(state_id="s0"),
        ReturnCachedOutput(output_uri="/tmp/x.mp4", media_type="video"),
        NoOp(reason="shadow_mode"),
    ]
    for h in hints:
        assert isinstance(h, ResumeHint), f"{type(h).__name__} not a ResumeHint subtype"
    # frozen dataclass — mutation should fail
    with pytest.raises((AttributeError, Exception)):
        hints[0].k = 10  # type: ignore[misc]


def test_unsupported_resume_hint_exception() -> None:
    from cacheseek import SkipStep, UnsupportedResumeHint

    with pytest.raises(UnsupportedResumeHint) as exc_info:
        raise UnsupportedResumeHint(SkipStep(k=5), "test_framework", [SkipStep])
    assert exc_info.value.framework == "test_framework"
    assert exc_info.value.hint.k == 5


# ─── Payload Protocol ──────────────────────────────────────────────────────


def test_payload_protocol_satisfied_by_video_approx() -> None:
    from cacheseek import Payload
    from cacheseek.reuse.approximate import VideoApproxPayload

    p = VideoApproxPayload(
        cache_id="abc123",
        step_to_latents={5: b"latent5", 10: b"latent10"},
    )
    assert isinstance(p, Payload), "VideoApproxPayload should satisfy Payload Protocol"
    assert p.cache_id == "abc123"
    assert p.schema_version == "video_approx_v1"
    assert p.estimated_size_bytes == len(b"latent5") + len(b"latent10")


def test_payload_to_kv_entries_naming() -> None:
    """KV key naming convention: ``{cache_id}_step{N}``."""
    from cacheseek.reuse.approximate import VideoApproxPayload

    p = VideoApproxPayload(
        cache_id="abc",
        step_to_latents={5: b"x", 10: b"y", 25: b"z"},
    )
    entries = list(p.to_kv_entries())
    keys = sorted(k for k, _ in entries)
    assert keys == ["abc_step10", "abc_step25", "abc_step5"]


def test_payload_partial_loading_spec() -> None:
    """from_kv_loader requires explicit partial_spec; loads only specified steps."""
    from cacheseek.reuse.approximate import (
        VideoApproxPayload,
        VideoApproxPartialSpec,
    )

    fake_kv = {"abc_step5": b"only5", "abc_step10": b"only10"}

    def loader(key: str):
        return fake_kv.get(key)

    # explicit "full" set of steps
    p_full = VideoApproxPayload.from_kv_loader(
        "abc", loader, partial_spec=VideoApproxPartialSpec(steps=(5, 10, 15, 20, 25))
    )
    assert 5 in p_full.step_to_latents
    assert 10 in p_full.step_to_latents
    # missing keys (15/20/25) silently absent — loader returned None
    assert 15 not in p_full.step_to_latents

    # partial load: only step 5 (single() helper)
    p_partial = VideoApproxPayload.from_kv_loader(
        "abc", loader, partial_spec=VideoApproxPartialSpec.single(5)
    )
    assert 5 in p_partial.step_to_latents
    assert 10 not in p_partial.step_to_latents


def test_payload_from_kv_loader_requires_partial_spec() -> None:
    """Default partial_spec was a 5×-read footgun; now mandatory."""
    from cacheseek.reuse.approximate import VideoApproxPayload

    with pytest.raises(TypeError):
        VideoApproxPayload.from_kv_loader("abc", lambda k: None)  # type: ignore[call-arg]


def test_payload_serialize_roundtrip() -> None:
    """torch.save + torch.load(weights_only=True) roundtrip."""
    import torch

    from cacheseek.reuse.approximate import VideoApproxPayload

    tensor = torch.randn(2, 3, 4, dtype=torch.float32)
    blob = VideoApproxPayload.serialize_tensor(tensor)
    restored = VideoApproxPayload.deserialize_tensor(blob)
    assert torch.allclose(tensor, restored)


# ─── EvictionPolicy ────────────────────────────────────────────────────────


def test_eviction_policy_protocol_satisfied_by_lru() -> None:
    from cacheseek.service.eviction import LRUEviction
    from cacheseek.service.interfaces import EvictionPolicy

    policy = LRUEviction()
    assert isinstance(policy, EvictionPolicy)


def test_lru_eviction_selects_oldest() -> None:
    from cacheseek.service.eviction import LRUEviction

    policy = LRUEviction()
    candidates = [
        ("id_old", {"size_mb": 50.0, "last_access_time": 1000.0}),
        ("id_mid", {"size_mb": 50.0, "last_access_time": 2000.0}),
        ("id_new", {"size_mb": 50.0, "last_access_time": 3000.0}),
    ]
    victims = policy.select_victims(candidates, size_to_free_bytes=60 * 1024 * 1024)
    assert victims[0] == "id_old"
    assert "id_new" not in victims  # newest should not be evicted first


def test_lru_eviction_handles_empty_candidates() -> None:
    from cacheseek.service.eviction import LRUEviction

    policy = LRUEviction()
    victims = policy.select_victims([], size_to_free_bytes=1000)
    assert victims == []


# ─── VideoApproxPayload + Strategy compat ──────────────────────────────────


def test_strategies_video_approximate_re_export() -> None:
    """Strategy class is accessible from the ``strategies`` namespace."""
    from cacheseek.reuse.approximate import VideoBasedApproximateCache

    # Class definition lives in core/strategies.py; new namespace is re-export.
    from cacheseek.reuse.approximate.strategy import (
        VideoBasedApproximateCache as core_class,
    )
    assert VideoBasedApproximateCache is core_class
