# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""CacheService lifecycle tests — async save dispatch, vector-update
barrier, shutdown ordering, cache_mode short-circuits.

These cover the orchestrator concerns documented in
``cacheseek/core/lifecycle.py`` (review buckets 2 #15, #23). They do
NOT exercise strategy logic — strategies are stubbed via
``_AsyncRecorder`` which records calls and yields control as needed.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cacheseek.service.lifecycle import CacheService
from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult

pytestmark = pytest.mark.smoke


class _AsyncRecorder:
    """Async-strategy stub that records calls + lets tests delay save.

    - ``lookup_calls`` / ``save_calls`` collect each invocation.
    - ``save_block`` / ``save_release`` form a hand-built gate so the test
      can hold the worker thread inside ``save`` while it inspects the
      pending state.
    """

    def __init__(
        self, save_delay_s: float = 0.0, block_first_n: int | None = None
    ) -> None:
        self.lookup_calls: list[CacheQuery] = []
        self.save_calls: list[tuple[CacheQuery, ModelOutputs]] = []
        self.save_delay_s = save_delay_s
        # ``block_first_n=None`` means every save blocks while
        # save_delay_s > 0 (existing behavior). Setting it to e.g. 1
        # makes only the first save block — subsequent saves run through.
        self.block_first_n = block_first_n
        self.lookup_return: LookupResult = LookupResult.miss()
        self.save_release = threading.Event()
        self.save_release.set()  # default: no blocking
        self.save_should_raise: BaseException | None = None
        # Mark as "has config.cache_mode" so CacheService can derive default.
        self.config = type("_Cfg", (), {"cache_mode": "read_write"})()
        # Required by Strategy.shutdown cascade (no-op).
        self.shutdown_calls = 0

    async def lookup(self, query: CacheQuery) -> LookupResult:
        self.lookup_calls.append(query)
        return self.lookup_return

    async def save(self, query: CacheQuery, outputs: ModelOutputs, ctx=None) -> None:
        # Capture call BEFORE blocking so tests can observe the entry.
        call_index = len(self.save_calls)  # 0-based
        self.save_calls.append((query, outputs))
        should_block = self.save_delay_s > 0 and (
            self.block_first_n is None or call_index < self.block_first_n
        )
        if should_block:
            # Block the worker thread inside save until released.
            self.save_release.clear()
            self.save_release.wait(timeout=10.0)
        if self.save_should_raise is not None:
            raise self.save_should_raise

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _query() -> CacheQuery:
    return CacheQuery(prompt="lifecycle test", task_type="t2v")


def _outputs() -> ModelOutputs:
    return ModelOutputs(saved_steps=[5, 10])


def _wait(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate()`` until truthy or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ─── cache_mode short-circuits ────────────────────────────────────────────


async def test_lookup_skipped_when_write_only() -> None:
    strat = _AsyncRecorder()
    svc = CacheService([strat], async_save=False, cache_mode="write_only")
    try:
        result = await svc.lookup(_query())
        assert result.hit is False
        assert strat.lookup_calls == []  # short-circuited
    finally:
        svc.shutdown()


async def test_save_skipped_when_read_only() -> None:
    strat = _AsyncRecorder()
    svc = CacheService([strat], async_save=False, cache_mode="read_only")
    try:
        await svc.save(_query(), _outputs())
        assert strat.save_calls == []
    finally:
        svc.shutdown()


# ─── async save dispatch ──────────────────────────────────────────────────


async def test_async_save_dispatches_to_worker_thread() -> None:
    strat = _AsyncRecorder()
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=8,
        flush_on_shutdown=True,
        cache_mode="read_write",
    )
    try:
        await svc.save(_query(), _outputs())
        assert _wait(lambda: len(strat.save_calls) == 1), (
            f"save not dispatched; saw {len(strat.save_calls)} calls"
        )
    finally:
        svc.shutdown()


async def test_async_save_queue_full_drop_policy() -> None:
    """When queue fills and on_full=drop, extra saves are dropped without
    waiting on the worker (and without raising)."""
    strat = _AsyncRecorder(save_delay_s=10.0)
    strat.save_release.clear()  # block worker inside save
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=1,
        on_full="drop",
        flush_on_shutdown=False,
        cache_mode="read_write",
    )
    try:
        # First save → enters worker, blocks (because save_delay + cleared release)
        await svc.save(_query(), _outputs())
        # Wait until the worker actually picked it up
        assert _wait(lambda: len(strat.save_calls) >= 1, timeout=2.0)
        # Second save → enqueues into the now-empty queue
        await svc.save(_query(), _outputs())
        # Third save → queue full → dropped
        await svc.save(_query(), _outputs())
        # Release and let worker drain.
        strat.save_release.set()
    finally:
        strat.save_release.set()
        svc.shutdown()


# ─── vector update barrier ────────────────────────────────────────────────


async def test_lookup_waits_for_in_flight_save() -> None:
    """save() reserves a vector-update slot;the next lookup blocks until
    save completes (or hits timeout)."""
    strat = _AsyncRecorder(save_delay_s=0.5)
    strat.save_release.clear()
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=8,
        flush_on_shutdown=True,
        vector_wait_poll_s=0.01,
        vector_wait_warn_s=0,
        vector_wait_timeout_s=5.0,
        cache_mode="read_write",
    )
    try:
        await svc.save(_query(), _outputs())
        # Worker is now sitting inside save (release is cleared)
        assert _wait(lambda: len(strat.save_calls) >= 1, timeout=2.0)

        # Kick off lookup — it should block until release fires.
        lookup_started_at = time.monotonic()
        lookup_done_at: list[float] = []

        async def do_lookup():
            await svc.lookup(_query())
            lookup_done_at.append(time.monotonic())

        lookup_task = asyncio.create_task(do_lookup())
        # Give it a head start to enter the wait loop.
        await asyncio.sleep(0.1)
        assert not lookup_done_at, "lookup must not return while save is in flight"

        # Release worker → save finishes → barrier opens → lookup returns.
        strat.save_release.set()
        await lookup_task
        assert lookup_done_at, "lookup did not complete"
        elapsed = lookup_done_at[0] - lookup_started_at
        assert elapsed >= 0.05, "lookup must have actually waited at the barrier"
    finally:
        strat.save_release.set()
        svc.shutdown()


# ─── shutdown ordering (#15 race regression) ──────────────────────────────


async def test_shutdown_stops_worker_before_flushing() -> None:
    """The bucket-2 #15 fix: shutdown must `stop_event.set()` then
    `worker.join()` BEFORE flushing the queue, so the main thread and
    worker don't double-consume Queue.get().

    Also exercises the async-context safety of shutdown: calling it
    directly from inside a running event loop must not error out with
    ``Cannot run the event loop while another loop is running``.
    """
    strat = _AsyncRecorder()
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=4,
        flush_on_shutdown=True,
        cache_mode="read_write",
    )
    # Fill queue with three items (worker drains them one at a time)
    await svc.save(_query(), _outputs())
    await svc.save(_query(), _outputs())
    await svc.save(_query(), _outputs())

    # shutdown should not raise, should drain everything, and worker must be dead.
    # (Called directly from async context — internal fallback handles loop isolation.)
    svc.shutdown()

    # Strategy saw all 3 saves (some via worker, possibly some via flush).
    assert len(strat.save_calls) == 3
    assert svc._save_worker is not None and not svc._save_worker.is_alive()
    # Strategy.shutdown was cascaded.
    assert strat.shutdown_calls == 1


async def test_on_full_sync_runs_inline_from_async_context() -> None:
    """When the queue fills and ``on_full=sync``, the third save runs
    inline. Because ``save()`` is itself async, this inline run happens
    inside a running event loop — the implementation must dispatch to a
    worker thread to avoid 'Cannot run the event loop while another loop
    is running'.
    """
    # Worker holds save #1 (block_first_n=1) so the queue can fill;
    # saves #2 and #3 don't block.
    strat = _AsyncRecorder(save_delay_s=10.0, block_first_n=1)
    strat.save_release.clear()
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=1,
        on_full="sync",
        flush_on_shutdown=False,
        cache_mode="read_write",
    )
    try:
        # 1: enters worker, blocks
        await svc.save(_query(), _outputs())
        assert _wait(lambda: len(strat.save_calls) >= 1, timeout=2.0)
        # 2: queues
        await svc.save(_query(), _outputs())
        # 3: queue full → on_full=sync → runs inline via fallback driver
        # (without raising 'Cannot run the event loop while another loop
        # is running'). Worker is still parked on save #1; save #2 still
        # queued; only save #3 runs to completion here.
        await svc.save(_query(), _outputs())
        # save_calls now contains entries for #1 (entered, blocked) and
        # #3 (ran inline). #2 is still queued.
        assert len(strat.save_calls) == 2, (
            f"expected 2 save entries (#1 blocked + #3 inline), "
            f"saw {len(strat.save_calls)}"
        )
    finally:
        strat.save_release.set()
        svc.shutdown()


async def test_shutdown_with_flush_disabled_drops_pending() -> None:
    """``flush_on_shutdown=False`` discards items the worker hadn't yet
    consumed."""
    strat = _AsyncRecorder(save_delay_s=10.0)  # worker blocks inside first save
    strat.save_release.clear()
    svc = CacheService(
        [strat],
        async_save=True,
        max_queue_size=4,
        flush_on_shutdown=False,
        cache_mode="read_write",
    )
    await svc.save(_query(), _outputs())
    assert _wait(lambda: len(strat.save_calls) >= 1, timeout=2.0)
    # Now enqueue two more — they sit in the queue.
    await svc.save(_query(), _outputs())
    await svc.save(_query(), _outputs())

    # Release worker so it can finish the first one and exit on stop_event.
    strat.save_release.set()
    svc.shutdown()

    # Only the first save (the one that was already in flight) ran;
    # queued items were not flushed.
    assert len(strat.save_calls) == 1
