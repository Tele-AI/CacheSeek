from __future__ import annotations

import asyncio
from unittest.mock import patch

from cacheseek.service.outputs import ModelOutputs
from cacheseek.service.query import CacheQuery
from cacheseek.service.result import LookupResult


class _CountingStrategy:
    def __init__(self) -> None:
        self.lookup_calls = 0
        self.save_calls = 0

    async def lookup(self, query: CacheQuery, ctx=None) -> LookupResult:
        self.lookup_calls += 1
        return LookupResult.hit_skip_step(
            payload=object(),
            k=5,
            cache_id="cache-1",
            score=1.0,
            similarity=1.0,
        )

    async def save(self, query: CacheQuery, outputs: ModelOutputs, ctx=None) -> None:
        self.save_calls += 1


def test_cache_service_write_only_skips_lookup() -> None:
    from cacheseek.service.lifecycle import CacheService

    strategy = _CountingStrategy()
    service = CacheService([strategy], cache_mode="write_only", async_save=False)

    result = asyncio.run(service.lookup(CacheQuery(prompt="hello")))

    assert result.hit is False
    assert strategy.lookup_calls == 0


def test_cache_service_read_only_skips_save() -> None:
    from cacheseek.service.lifecycle import CacheService

    strategy = _CountingStrategy()
    service = CacheService([strategy], cache_mode="read_only", async_save=False)

    asyncio.run(service.save(CacheQuery(prompt="hello"), ModelOutputs(saved_steps=[5])))

    assert strategy.save_calls == 0


def test_qwen_encoder_reports_missing_transitive_dependency() -> None:
    from cacheseek.backends.encoder.qwen3vl import Qwen3VLEncoder

    def fake_import_module(module_name: str):
        if module_name == "cacheseek.reuse.approximate.models_src.models.qwen3_vl_embedding":
            raise ModuleNotFoundError("No module named 'torchvision'", name="torchvision")
        raise ModuleNotFoundError(f"No module named {module_name!r}", name=module_name)

    with patch("cacheseek.backends.encoder.qwen3vl.importlib.import_module", fake_import_module):
        try:
            Qwen3VLEncoder(model_path="/tmp/missing-model")
        except ImportError as exc:
            assert "torchvision" in str(exc)
        else:
            raise AssertionError("Qwen3VLEncoder should fail when torchvision is missing")
