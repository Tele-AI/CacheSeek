"""cacheseek 0.1 alpha smoke tests.

Validates:
1. Public API imports
2. CacheConfig instantiation + basic field access
3. ConnectionManager wiring with in-memory KV/Vector backends
4. Strategy module is loadable (heavy encoder deps are lazy and not exercised here)
5. TeleFuser adapter modules are importable

These tests run without GPU / models / fluxon / qdrant — they verify
the package itself is well-formed and import paths are sound.
End-to-end inference is covered separately.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest


pytestmark = pytest.mark.smoke


def test_top_level_import() -> None:
    import cacheseek

    assert hasattr(cacheseek, "__version__")
    assert cacheseek.__version__.startswith("0.1.")


def test_lazy_reexport_classes() -> None:
    import cacheseek

    # CacheConfig
    cfg_cls = cacheseek.CacheConfig
    assert cfg_cls is not None
    cfg = cfg_cls()
    assert hasattr(cfg, "kv_store_type")
    assert hasattr(cfg, "vector_store_type")
    assert hasattr(cfg, "key_steps")

    # Lifecycle types
    lookup_result_cls = cacheseek.LookupResult
    miss = lookup_result_cls.miss()
    assert miss.hit is False
    assert miss.payload is None
    assert miss.resume_hint is None

    skip_step_cls = cacheseek.SkipStep
    hint = skip_step_cls(k=5)
    assert hint.k == 5

    cache_query_cls = cacheseek.CacheQuery
    query = cache_query_cls(prompt="hello", task_type="t2v")
    assert query.prompt == "hello"
    assert query.task_type == "t2v"

    model_outputs_cls = cacheseek.ModelOutputs
    outputs = model_outputs_cls(saved_steps=[5, 10])
    assert outputs.saved_steps == [5, 10]

    # Orchestrator + adapter classes importable
    cache_service_cls = cacheseek.CacheService
    assert cache_service_cls is not None
    adapter_cls = cacheseek.TeleFuserCacheAdapter
    assert adapter_cls is not None
    # Adapter is stateless — should construct without args
    adapter = adapter_cls()
    assert adapter.FRAMEWORK_NAME == "telefuser"


def test_core_subpackages_import() -> None:
    """All core sub-packages should import cleanly without optional deps."""
    importlib.import_module("cacheseek.service.cache_types")
    importlib.import_module("cacheseek.service.config")
    importlib.import_module("cacheseek.service.connection")
    importlib.import_module("cacheseek.backends.metadata.local")
    importlib.import_module("cacheseek.service.lifecycle")
    importlib.import_module("cacheseek.service.log_monitor")


def test_storage_interface_and_memory_backend() -> None:
    from cacheseek.stores import KVStore, InMemoryKVStore

    assert isinstance(InMemoryKVStore(), KVStore)
    kv = InMemoryKVStore()
    kv.put("smoke:k1", b"hello")
    assert kv.get("smoke:k1") == b"hello"
    assert "smoke:k1" in kv.list_keys()
    kv.remove("smoke:k1")
    assert kv.get("smoke:k1") is None


def test_vector_store_interface_imports() -> None:
    from cacheseek.backends.vector import VectorStore  # noqa: F401


def test_metadata_store_interface_imports() -> None:
    from cacheseek.service.interfaces.metadata_store import MetadataStore  # noqa: F401


def test_metadata_local_manager_constructible(tmp_path) -> None:
    from cacheseek.backends.metadata import LocalCacheMetadataManager

    mgr = LocalCacheMetadataManager(metadata_cache_dir=tmp_path)
    assert mgr.lookup_prompt("nonexistent") is None
    assert mgr.get_cache_meta("nonexistent") is None


def test_connection_manager_with_local_only(tmp_path) -> None:
    """ConnectionManager builds with local_file KV + faiss vector (faiss optional)."""
    from cacheseek.service.config import CacheConfig
    from cacheseek.service.connection import ConnectionManager

    cfg = CacheConfig(
        enable_latent_cache=True,
        latent_cache_dir=str(tmp_path / "cache"),
        kv_store_type="local_file",
        vector_store_type="faiss",
        faiss_index_dir=str(tmp_path / "faiss"),
        vector_dim=8,
    )
    cm = ConnectionManager(cfg, storage_dir=tmp_path / "cache")
    kv = cm.kv_store
    assert kv is not None
    kv.put("smoke:cm", b"works")
    assert kv.get("smoke:cm") == b"works"
    cm.shutdown()


def test_telefuser_adapter_modules_importable() -> None:
    """The TeleFuser adapter imports `telefuser.utils.*` for profiling/utils.
    If telefuser is not installed,we still want top-level cacheseek to be usable.
    """
    try:
        importlib.import_module("cacheseek.adapters.telefuser.cache_factory")
    except ImportError as e:
        # Acceptable: TeleFuser not installed in this env. Skip but log.
        pytest.skip(f"TeleFuser not available; adapter import skipped: {e}")


def test_strategies_module_importable() -> None:
    """`strategies` re-exports the Strategy base + concrete VideoBasedApproximateCache.
    Heavy encoder deps are lazy at instantiation,not at import.
    """
    mod = importlib.import_module("cacheseek.reuse.approximate.strategy")
    assert hasattr(mod, "BaseCacheStrategy")
    assert hasattr(mod, "VideoBasedApproximateCache")
    assert hasattr(mod, "get_strategy_class")


def test_cache_service_factory_smoke_with_local_cache_config(tmp_path, monkeypatch) -> None:
    """CacheServiceFactory can bootstrap from a ppl-file CACHE_CONFIG locally.

    The real factory imports TeleFuser's `import_function_from_file`; this smoke
    test provides only that utility as a module stub, then configures CacheSeek
    to avoid model/vector optional dependencies.
    """

    def import_function_from_file(path: str, name: str):
        spec = importlib.util.spec_from_file_location("_cacheseek_smoke_ppl", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, name)

    telefuser_mod = types.ModuleType("telefuser")
    utils_pkg = types.ModuleType("telefuser.utils")
    utils_mod = types.ModuleType("telefuser.utils.utils")
    utils_mod.import_function_from_file = import_function_from_file
    utils_pkg.utils = utils_mod
    telefuser_mod.utils = utils_pkg
    monkeypatch.setitem(sys.modules, "telefuser", telefuser_mod)
    monkeypatch.setitem(sys.modules, "telefuser.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "telefuser.utils.utils", utils_mod)
    sys.modules.pop("cacheseek.adapters.telefuser.cache_factory", None)

    ppl_file = tmp_path / "wan22_smoke_ppl.py"
    ppl_file.write_text(
        "\n".join(
            [
                "CACHE_CONFIG = {",
                f"    'latent_cache_dir': {str(tmp_path / 'latent_cache')!r},",
                "    'cache_mode': 'write_only',",
                "    'kv_store_type': 'local_file',",
                "    'vector_store_type': '',",
                "    'video_embedding_enabled': False,",
                "    'cache_log_enabled': False,",
                "    'save_async_enabled': False,",
                "    'key_steps': [7, 9],",
                "}",
                "",
            ]
        )
    )

    factory_mod = importlib.import_module("cacheseek.adapters.telefuser.cache_factory")
    created = factory_mod.CacheServiceFactory.create_cache_service(
        ppl_file=str(ppl_file),
        enable_latent_cache=True,
    )

    assert created is not None
    service, adapter = created
    try:
        from cacheseek.service.result import LookupResult

        assert service._cache_mode == "write_only"
        assert adapter.apply_resume(LookupResult.miss(), engine_ctx=None)["saved_steps"] == [7, 9]
    finally:
        service.shutdown()
