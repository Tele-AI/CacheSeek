# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Loguru sink management for the dedicated cache-service log (cache_service.log)."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_CACHE_MODULE_PREFIXES = (
    "cacheseek.service",
    "cacheseek.reuse",
    "cacheseek.stores",
    "cacheseek.adapters",
    "telefuser.service.cache.cache_service",
    "telefuser.service.cache.cache_factory",
)

_sink_id: int | None = None


def _cache_module_filter(record: dict) -> bool:
    name = record.get("name", "")
    return any(name.startswith(prefix) for prefix in _CACHE_MODULE_PREFIXES)


def setup_cache_log_sink(
    log_dir: str | Path,
    *,
    level: str = "DEBUG",
    rotation: str = "100 MB",
    retention: str = "7 days",
    fmt: str = ("[CACHE] {time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"),
) -> Path:
    """Install a loguru file sink scoped to cacheseek modules.

    Adds a file sink at ``<log_dir>/cache_service.log`` that only captures
    records whose logger name starts with one of the cacheseek/cache module
    prefixes, so cache logs are isolated from the host application's logs.
    Idempotent: a previously installed sink is removed first, so calling this
    again reconfigures rather than duplicating the sink. Writes are enqueued
    (async) to avoid blocking business threads. The created directory is made
    if missing.

    Args:
        log_dir: Directory to hold the log file; created if absent.
        level: Minimum log level captured by the sink.
        rotation: loguru rotation spec (e.g. size or time trigger).
        retention: loguru retention spec for pruning rotated files.
        fmt: loguru format string for each record.

    Returns:
        The path to the log file the sink writes to.
    """
    global _sink_id

    if _sink_id is not None:
        try:
            logger.remove(_sink_id)
        except ValueError as exc:
            logger.debug("Old cache log sink already removed (id={}): {}", _sink_id, exc)
        _sink_id = None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cache_service.log"

    _sink_id = logger.add(
        str(log_path),
        filter=_cache_module_filter,
        format=fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # async writes; do not block business threads
    )

    logger.info(
        "Cache log sink configured: path={} level={} rotation={} retention={}",
        log_path,
        level,
        rotation,
        retention,
    )
    return log_path


def remove_cache_log_sink() -> None:
    """Remove the cache log sink if one is installed (no-op otherwise)."""
    global _sink_id
    if _sink_id is not None:
        try:
            logger.remove(_sink_id)
        except ValueError as exc:
            logger.debug("Cache log sink already removed (id={}): {}", _sink_id, exc)
        _sink_id = None


def is_cache_log_sink_active() -> bool:
    """Return whether the cache log sink is currently installed."""
    return _sink_id is not None
