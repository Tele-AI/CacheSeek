# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""JSONL append-only audit log.

Standalone ``AuditLog`` implementation. The legacy
``LocalCacheMetadataManager`` still embeds ``record_hit_pair`` /
``record_similarity_scores`` writers; this class is the path forward for
strategies that want audit independent of metadata storage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JSONLAuditLog:
    """Append-only JSONL event log. Thread-safe via OS-level append atomicity for
    JSON lines under PIPE_BUF (4KB) on POSIX."""

    def __init__(self, log_path: Path | str):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one event to the log file."""
        line = json.dumps({
            "timestamp": float(time.time()),
            "event_type": event_type,
            **payload,
        }, ensure_ascii=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
