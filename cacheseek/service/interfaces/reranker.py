"""Reranker Protocol — second-stage scoring of vector-search candidates."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Contract for second-stage rerank scoring.

    The reranker is invoked after a coarse ``VectorStore.search`` returns
    a top-K shortlist; it produces a fine-grained relevance score for
    each candidate (typically by running a cross-encoder over query +
    candidate text). Higher score = more relevant.

    Conformance:
    - Model weights may be cached privately on the reranker, but
      ``score_mm(query, documents)`` must be deterministic for a fixed
      configuration.
    - Implementations must be thread-safe — rerank is called inline from
      the ``lookup`` hot path under concurrent traffic.
    - The returned list MUST have the same length as ``documents`` and
      be aligned by index — score ``i`` describes document ``i``.

    Multimodal-friendly signature:
    - ``query`` is a dict so callers can pass ``{"text": prompt, ...}``
      and future implementations can accept image / video keys without
      breaking the contract.
    - Each entry of ``documents`` is a dict carrying at minimum a
      ``"text"`` field; implementations may consume additional keys.
    """

    def score_mm(
        self,
        query: dict[str, object],
        documents: list[dict[str, object]],
    ) -> list[float]:
        """Return scores aligned with ``documents``, higher = more relevant."""
        ...
