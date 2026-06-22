# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""cacheseek core Protocol surface — contracts for backends and strategies.

This package collects the Protocols that define cacheseek's core boundary:

- ``KVStore``           — opaque byte-blob storage
- ``VectorStore``       — vector index + similarity search
- ``MetadataStore``     — cache index + access stats + eviction
- ``AuditLog``          — append-only event stream
- ``PromptEncoder`` /
  ``VideoEncoder``      — text / video → vector
- ``Reranker``          — second-stage scoring of vector-search candidates
- ``Strategy``          — ``lookup`` / ``save``
- ``FrameworkAdapter``  — bridge inference-framework hooks to cacheseek types

All Protocols are ``runtime_checkable``; ABC subclasses with matching
signatures satisfy ``isinstance(obj, Protocol)`` structurally.
"""

from __future__ import annotations

from cacheseek.service.eviction import EvictionPolicy
from cacheseek.service.interfaces.audit_log import AuditLog
from cacheseek.service.interfaces.encoder import PromptEncoder, VideoEncoder
from cacheseek.service.interfaces.framework_adapter import FrameworkAdapter
from cacheseek.service.interfaces.kv_store import KVStore, TensorKVStore
from cacheseek.service.interfaces.metadata_store import MetadataStore
from cacheseek.service.interfaces.reranker import Reranker
from cacheseek.service.interfaces.strategy import Strategy
from cacheseek.service.interfaces.vector_store import VectorStore

__all__ = [
    "KVStore",
    "TensorKVStore",
    "VectorStore",
    "MetadataStore",
    "AuditLog",
    "EvictionPolicy",
    "PromptEncoder",
    "VideoEncoder",
    "Reranker",
    "Strategy",
    "FrameworkAdapter",
]
