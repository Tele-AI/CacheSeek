"""CacheService — lifecycle orchestrator.

Lifecycle:

    [Framework] → FrameworkAdapter.build_query → CacheQuery
                ↓
                CacheService.lookup(query)
                  ├─ wait for in-flight vector_store upserts to settle
                  ├─ for strat in strategies: try lookup (first hit wins)
                  ├─ Strategy returns LookupResult
                ↓
                LookupResult{hit, payload, resume_hint}
                ↓
                FrameworkAdapter.apply_resume → engine
                ↓
                FrameworkAdapter.on_response → ModelOutputs
                ↓
                CacheService.save(query, outputs)  — async by default
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from cacheseek.service.eviction import EvictionPolicy, LRUEviction
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult

if TYPE_CHECKING:
    from cacheseek.service.interfaces.strategy import Strategy


class CacheService:
    """Lifecycle orchestrator wrapping ordered strategies + backends.

    Construction:
        service = CacheService(
            strategies=[VideoBasedApproximateCache(...)],
            eviction_policy=LRUEviction(),
            async_save=True,
            max_queue_size=8,
            on_full="drop",
            flush_on_shutdown=True,
            vector_wait_poll_s=0.05,
            vector_wait_warn_s=2.0,
            vector_wait_timeout_s=120.0,
        )

    Usage by FrameworkAdapter:
        result = await service.lookup(query)   # blocks if a save is mid-flight
        adapter.apply_resume(result, engine_ctx)
        outputs = adapter.on_response(req, raw_outputs)
        await service.save(query, outputs)     # returns immediately
    """

    def __init__(
        self,
        strategies: "list[Strategy]",
        eviction_policy: Optional[EvictionPolicy] = None,
        *,
        async_save: bool = True,
        max_queue_size: int = 8,
        on_full: str = "drop",
        flush_on_shutdown: bool = True,
        vector_wait_poll_s: float = 0.05,
        vector_wait_warn_s: float = 2.0,
        vector_wait_timeout_s: float = 120.0,
        cache_mode: Any = None,
    ) -> None:
        if not strategies:
            raise ValueError("CacheService requires at least one Strategy")
        self._strategies = list(strategies)
        self._eviction_policy = eviction_policy or LRUEviction()
        self._async_save = bool(async_save)
        self._on_full = self._normalize_on_full(on_full)
        self._flush_on_shutdown = bool(flush_on_shutdown)
        self._cache_mode = self._normalize_cache_mode(cache_mode)

        # vector_store-upsert barrier: lookup waits for in-flight save's
        # vector upsert before reading the index, so hot reads see what the
        # most recent save wrote.
        self._vector_wait_poll_s = max(0.001, float(vector_wait_poll_s))
        self._vector_wait_warn_s = max(0.0, float(vector_wait_warn_s))
        self._vector_wait_timeout_s = max(0.0, float(vector_wait_timeout_s))
        self._pending_vector_updates = 0  # counter: n saves in flight -- reserve +1 / release -1
        self._pending_lock = threading.Lock()  # guards the counter
        self._vector_update_idle = threading.Event()
        self._vector_update_idle.set()  # idle initially

        self._save_queue: Optional[Queue] = None
        self._save_worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        if async_save:
            self._save_queue = Queue(maxsize=max_queue_size)
            self._save_worker = threading.Thread(
                target=self._save_worker_loop,
                name="cacheseek-save-worker",
                daemon=True,
            )
            self._save_worker.start()

    @classmethod
    def from_config(cls, config_path: "str | Path") -> "CacheService":
        """Bootstrap a ``CacheService`` from a YAML config file.

        The YAML keys map 1:1 to fields on
        :class:`cacheseek.service.config.CacheConfig`. Any field omitted
        falls back to the dataclass default. The default strategy is
        :class:`VideoBasedApproximateCache`; KV / vector stores are
        instantiated lazily through ``ConnectionManager`` based on
        ``kv_store_type`` / ``vector_store_type``.

        Example::

            from cacheseek import CacheService
            cache = CacheService.from_config("quickstart.yaml")

        See ``quickstart.yaml`` at the repo root for an annotated
        starter template.
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover — install-time guard
            raise ImportError(
                "CacheService.from_config requires PyYAML. "
                "Install with: pip install pyyaml"
            ) from exc

        from cacheseek.backends.metadata.local import LocalCacheMetadataManager
        from cacheseek.service.config import CacheConfig, CacheMode
        from cacheseek.service.connection import ConnectionManager
        from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache

        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"CacheConfig yaml not found: {cfg_path}")

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"CacheConfig yaml must be a mapping, got "
                f"{type(raw).__name__} from {cfg_path}"
            )

        if "cache_mode" in raw and not isinstance(raw["cache_mode"], CacheMode):
            try:
                raw["cache_mode"] = CacheMode(str(raw["cache_mode"]).lower())
            except ValueError as exc:
                valid = [m.value for m in CacheMode]
                raise ValueError(
                    f"cache_mode={raw['cache_mode']!r} is not one of {valid}"
                ) from exc

        cache_config = CacheConfig(**raw)

        cache_root = Path(cache_config.latent_cache_dir)
        storage_dir = cache_root / "storage"
        metadata_dir = cache_root / "metadata"
        for d in (cache_root, storage_dir, metadata_dir):
            d.mkdir(parents=True, exist_ok=True)

        conn_mgr = ConnectionManager(cache_config, storage_dir=storage_dir)
        metadata_manager = LocalCacheMetadataManager(metadata_dir)
        strategy = VideoBasedApproximateCache(
            cache_config,
            conn_mgr.kv_store,
            conn_mgr.vector_store,
            metadata_manager,
            prompt_encoder=conn_mgr.prompt_encoder,
            video_encoder=conn_mgr.video_encoder,
            reranker=conn_mgr.reranker,
        )
        # Cascade-close hook: Strategy.shutdown closes the ConnectionManager.
        strategy._cacheseek_conn_mgr = conn_mgr

        return cls(
            strategies=[strategy],
            async_save=bool(cache_config.save_async_enabled),
            max_queue_size=int(cache_config.save_queue_size or 1),
            on_full=str(cache_config.save_on_full or "drop"),
            flush_on_shutdown=bool(cache_config.flush_on_shutdown),
            vector_wait_poll_s=float(cache_config.vector_wait_poll_s),
            vector_wait_warn_s=float(cache_config.vector_wait_warn_s),
            vector_wait_timeout_s=float(cache_config.vector_wait_timeout_s),
            cache_mode=cache_config.cache_mode,
        )

    async def lookup(self, query: CacheQuery) -> LookupResult:
        """Try each strategy in order; first hit wins.

        Blocks (with timeout) until any in-flight save's vector upsert
        is done — without this barrier the miss → save → hit chain races
        when the second lookup runs before the save worker has reached
        ``vector_store.upsert``.

        Returns ``LookupResult.miss()`` if all strategies miss.
        """
        if self._cache_mode == "write_only":
            logger.debug("CacheService.lookup skipped: cache_mode=write_only")
            return LookupResult.miss()

        await self._wait_vector_updates_done()

        for strat in self._strategies:
            try:
                result = await strat.lookup(query)
            except Exception as exc:
                logger.exception(
                    "cacheseek CacheService.lookup strategy={} failed err={}",
                    type(strat).__name__,
                    exc,
                )
                continue
            if result is not None and result.hit:
                return result
        return LookupResult.miss()

    async def save(self, query: CacheQuery, outputs: ModelOutputs) -> None:
        """
        ``async_save=True``: enqueue to worker (returns immediately).
        ``async_save=False``: run inline (mostly for tests).

        Reserves a vector-update slot before enqueue / inline-run so the
        next ``lookup`` call blocks until the upsert is durable.
        """
        if self._cache_mode == "read_only":
            logger.debug("CacheService.save skipped: cache_mode=read_only")
            return

        primary = self._strategies[0]
        if self._async_save and self._save_queue is not None:
            self._reserve_vector_update()
            try:
                if self._on_full == "block":
                    self._save_queue.put((primary, query, outputs))
                else:
                    self._save_queue.put_nowait((primary, query, outputs))
            except Full:
                if self._on_full == "sync":
                    logger.warning(
                        "cacheseek CacheService.save queue full, running save inline; qsize={}",
                        self._save_queue.qsize(),
                    )
                    self._run_save_inline(primary, query, outputs)
                    return
                self._release_vector_update()
                if self._on_full in {"drop", "downgrade"}:
                    logger.warning(
                        "cacheseek CacheService.save queue full,dropping; on_full={} qsize={}",
                        self._on_full,
                        self._save_queue.qsize(),
                    )
                    return
                raise
            except Exception:
                # enqueue itself failed — release immediately so we don't
                # leave the barrier waiting forever.
                self._release_vector_update()
                raise
        else:
            self._reserve_vector_update()
            try:
                await self._safe_strategy_save(primary, query, outputs)
            finally:
                self._release_vector_update()

    def shutdown(self) -> None:
        """Stop async save worker; optionally flush pending queue.

        Cascades shutdown to each strategy if it exposes ``shutdown()``;
        strategies own their KV / vector / metadata backend handles
        directly.
        """
        if self._save_worker is not None:
            # Stop worker BEFORE flushing — otherwise shutdown() and the
            # worker race on Queue.get(), causing a double-consumer drain.
            self._stop_event.set()
            self._save_worker.join(timeout=30.0)
            if self._save_worker.is_alive():
                logger.warning(
                    "CacheService.shutdown save worker did not exit within "
                    "timeout; skipping queue flush to avoid racing"
                )
            elif self._flush_on_shutdown and self._save_queue is not None:
                while True:
                    try:
                        item = self._save_queue.get_nowait()
                    except Empty:
                        break
                    self._run_save_inline(*item)

        for strat in self._strategies:
            shutdown = getattr(strat, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    logger.exception(
                        "cacheseek CacheService.shutdown strategy={} failed err={}",
                        type(strat).__name__,
                        exc,
                    )

    def _reserve_vector_update(self) -> None:
        with self._pending_lock:
            self._pending_vector_updates += 1
            self._vector_update_idle.clear()

    def _release_vector_update(self) -> None:
        with self._pending_lock:
            self._pending_vector_updates = max(0, self._pending_vector_updates - 1)
            if self._pending_vector_updates == 0:
                self._vector_update_idle.set()

    async def _wait_vector_updates_done(self) -> None:
        """Block lookup until the save worker has caught up.

        Polls ``vector_update_idle`` (threading.Event) at ``poll_s`` cadence
        from the asyncio loop;logs a warning at ``warn_s`` and gives up at
        ``timeout_s`` to avoid indefinite stalls under pathological save
        backlogs.
        """
        if self._vector_update_idle.is_set():
            return

        start = time.monotonic()
        warned = False
        with self._pending_lock:
            initial_pending = self._pending_vector_updates
        logger.info(
            "CacheService.lookup wait vector_update_idle start pending={}",
            initial_pending,
        )
        while not self._vector_update_idle.is_set():
            elapsed = time.monotonic() - start
            if (
                self._vector_wait_warn_s > 0
                and not warned
                and elapsed >= self._vector_wait_warn_s
            ):
                with self._pending_lock:
                    pending = self._pending_vector_updates
                logger.warning(
                    "CacheService.lookup wait vector_update_idle exceeded {:.2f}s pending={}",
                    self._vector_wait_warn_s,
                    pending,
                )
                warned = True
            if (
                self._vector_wait_timeout_s > 0
                and elapsed >= self._vector_wait_timeout_s
            ):
                with self._pending_lock:
                    pending = self._pending_vector_updates
                logger.warning(
                    "CacheService.lookup wait vector_update_idle timeout {:.2f}s "
                    "pending={}; continue with lookup (may read stale index)",
                    self._vector_wait_timeout_s,
                    pending,
                )
                return
            await asyncio.sleep(self._vector_wait_poll_s)
        logger.info(
            "CacheService.lookup wait vector_update_idle end elapsed={:.3f}s",
            time.monotonic() - start,
        )

    async def _safe_strategy_save(
        self, strat: "Strategy", query: CacheQuery, outputs: ModelOutputs
    ) -> None:
        """Wrap ``strat.save`` with exception logging — never crash the worker."""
        try:
            await strat.save(query, outputs)
        except Exception as exc:
            logger.exception(
                "cacheseek CacheService.save strategy={} failed err={}",
                type(strat).__name__,
                exc,
            )

    def _save_worker_loop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._save_queue.get(timeout=0.5)  # type: ignore[union-attr]
                except Empty:
                    continue
                try:
                    self._run_save_inline(*item, _loop=loop)
                except Exception as exc:
                    logger.exception("cacheseek save worker error: {}", exc)
        finally:
            loop.close()

    def _run_save_inline(
        self,
        strat: "Strategy",
        query: CacheQuery,
        outputs: ModelOutputs,
        _loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Run one save inline.

        Three call sites:
          1. ``_save_worker_loop`` — passes its own persistent loop via
             ``_loop``; uses ``run_until_complete`` on it.
          2. ``save()`` with ``on_full=sync`` after a Full enqueue — runs
             from inside the caller's asyncio loop.
          3. ``shutdown()`` flushing the queue — may run from sync code
             (process exit) or from inside an event loop (test fixture).

        For (2) and (3) we drive the coroutine manually via
        ``coro.send(None)``. The current ``Strategy.save`` impls have no
        ``await`` in their body — they're sync work wrapped in async
        sugar to satisfy the Protocol — so the coroutine completes in
        one ``send`` and never suspends. This avoids spinning up a
        throwaway event loop (and the
        ``Cannot run the event loop while another loop is running``
        error you'd hit if a fresh loop's ``run_until_complete`` is
        invoked while the caller's loop is active on this thread).

        If a future strategy adds a real ``await`` (e.g. async KV
        client), the coroutine will yield on first ``send`` and we
        fall back to running it on a one-shot worker thread with its
        own event loop, isolated from any caller-loop context.

        ``_release_vector_update`` runs in the ``finally`` so the
        barrier opens even on save exception. ``_safe_strategy_save``
        already swallows + logs strategy-side errors.
        """
        # Path 1: caller-supplied loop (persistent save worker).
        if _loop is not None:
            try:
                _loop.run_until_complete(
                    self._safe_strategy_save(strat, query, outputs)
                )
            finally:
                self._release_vector_update()
            return

        # Path 2/3: standalone call. Try sync drive first.
        coro = self._safe_strategy_save(strat, query, outputs)
        try:
            coro.send(None)
        except StopIteration:
            # Coroutine completed without ever awaiting — done.
            self._release_vector_update()
            return
        except BaseException:
            # _safe_strategy_save catches strategy errors itself; this
            # path means _safe_strategy_save's own scaffolding raised,
            # which would only happen on programmer error. Re-raise.
            self._release_vector_update()
            raise

        # The coroutine yielded — it has a real ``await`` somewhere.
        # Close the started coroutine and re-run on a worker thread with
        # its own event loop, isolated from any caller-loop context.
        coro.close()
        exc_holder: list[BaseException] = []

        def _drive_on_worker_thread() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self._safe_strategy_save(strat, query, outputs)
                )
            except BaseException as exc:  # noqa: BLE001 — re-surfaced below
                exc_holder.append(exc)
            finally:
                loop.close()

        thread = threading.Thread(
            target=_drive_on_worker_thread,
            name="cacheseek-save-inline-fallback",
            daemon=False,
        )
        thread.start()
        thread.join(timeout=30.0)
        self._release_vector_update()
        if thread.is_alive():
            logger.warning(
                "CacheService._run_save_inline fallback thread did not "
                "exit within 30s; barrier released but save may still be running"
            )
        if exc_holder:
            logger.exception(
                "CacheService._run_save_inline fallback thread raised: {}",
                exc_holder[0],
            )

    def _normalize_cache_mode(self, cache_mode: Any) -> str:
        value = getattr(cache_mode, "value", cache_mode)
        value = str(value or "read_write").strip().lower()
        if value not in {"read_write", "read_only", "write_only"}:
            logger.warning(
                "CacheService invalid cache_mode={!r}; falling back to read_write",
                value,
            )
            return "read_write"
        return value

    def _normalize_on_full(self, on_full: str) -> str:
        value = str(on_full or "drop").strip().lower()
        aliases = {
            "drop": "drop",
            "downgrade": "downgrade",
            "block": "block",
            "sync": "sync",
        }
        normalized = aliases.get(value)
        if normalized is None:
            logger.warning(
                "CacheService invalid save on_full={!r}; falling back to drop",
                value,
            )
            return "drop"
        return normalized


__all__ = ["CacheService"]
