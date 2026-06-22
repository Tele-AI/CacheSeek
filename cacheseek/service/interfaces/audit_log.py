"""AuditLog Protocol — append-only event stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class AuditLog(Protocol):
    """Contract for append-only audit / event-stream logging.

    Default implementation writes JSONL files. Other backends (e.g.
    Kafka, ClickHouse, OTLP) can implement the same contract; the
    Protocol is intentionally minimal (one method) so it composes
    cleanly inside ``MetadataStore`` implementations or stands alone.

    Conformance:
    - Implementations must be thread-safe — events are recorded from
      many ``CacheService`` worker threads concurrently. JSONL append on
      POSIX with ``O_APPEND`` is atomic for small payloads; other
      backends must provide equivalent guarantees.
    - Implementations that buffer writes (open file handle, batch queue)
      should keep the buffer private and flush under an internal lock.
    """

    def record(self, event_type: str, payload: dict) -> None: ...


# Common event-type schemas (informational dataclasses).
# These describe the well-known `event_type` strings emitted by the existing
# metadata layer. They are NOT part of the Protocol surface — the contract is
# just `record(event_type, payload)` — but they document the payload shape an
# `AuditLog` implementation should preserve when round-tripping.


@dataclass
class HitPairEvent:
    """`event_type="hit_pair"` — one cache hit between request and stored cache.

    Mirrors the JSONL row written by the default file-based audit log.
    Field order matches `record_hit_pair` in `cacheseek.backends.metadata.local`.
    """

    timestamp: float
    request_prompt: str
    cache_id: str
    cached_prompt: str
    similarity: float
    task_type: str
    cache_type: str
    skip_step: int


@dataclass
class SimilarityScoreEvent:
    """`event_type="similarity_scores"` — candidate ranking snapshot per stage.

    Mirrors the JSONL row written by the default file-based audit log.
    Field order matches `record_similarity_scores` in `cacheseek.backends.metadata.local`.
    `stage` is typically `"vector_search"` or `"rerank"`.
    """

    timestamp: float
    request_prompt: str
    task_type: str
    cache_type: str
    stage: str
    candidates: list[dict]
