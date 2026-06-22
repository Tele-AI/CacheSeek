"""cacheseek.adapters.telefuser — TeleFuser FrameworkAdapter + factory.

Eager exports:
- ``TeleFuserCacheAdapter``: ``FrameworkAdapter`` Protocol impl.

Lazy exports (require ``telefuser`` installed at runtime):
- ``CacheServiceFactory``: produces ``(CacheService, TeleFuserCacheAdapter)``
  pair. Imports ``telefuser.utils.*`` internally so it can only be loaded
  inside a TeleFuser venv.

For the orchestrator, import directly:

    from cacheseek.service.lifecycle import CacheService

The factory wires the strategy with its KV / vector / metadata backends
through a ``ConnectionManager`` handle attached on the strategy
(cascade-closed by ``Strategy.shutdown``).
"""
from cacheseek.adapters.telefuser.adapter import TeleFuserCacheAdapter

__all__ = ["TeleFuserCacheAdapter", "CacheServiceFactory"]


def __getattr__(name):
    """Lazy import of factory — requires `telefuser` installed."""
    if name == "CacheServiceFactory":
        from cacheseek.adapters.telefuser.cache_factory import (
            CacheServiceFactory as _CSF,
        )
        return _CSF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
