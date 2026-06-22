"""CacheServiceFactory — TeleFuser-specific factory for cacheseek lifecycle.

Returns a ``(CacheService, TeleFuserCacheAdapter)`` pair.

What the factory wires up:
- Build directory layout under ``app_cache_config.latent_cache_dir`` for
  ``storage`` / ``metadata`` / ``dit_cache`` subdirs.
- Construct a ``ConnectionManager`` (lazy-creates KV + vector store on
  first access).
- Construct ``LocalCacheMetadataManager`` for the metadata sidecar.
- Construct ``VideoBasedApproximateCache`` strategy with the three
  backend handles attached.
- Attach ``_cacheseek_conn_mgr`` to the strategy so
  ``Strategy.shutdown`` cascades close to the ``ConnectionManager``.
- Wrap the strategy in ``CacheService`` (ordered strategies, async save
  worker, eviction-policy slot) and pair with ``TeleFuserCacheAdapter``
  for framework-specific ``build_query`` / ``apply_resume`` /
  ``on_response``.

``TeleFuserCacheAdapter.apply_resume`` produces the ``latent_data``
dict via ``ResumeHint`` dispatch; no ppl-file hook is needed.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from cacheseek.service.ppl_loader import import_function_from_file

from cacheseek.adapters.telefuser.adapter import TeleFuserCacheAdapter
from cacheseek.backends.audit.jsonl import JSONLAuditLog
from cacheseek.backends.metadata.local import LocalCacheMetadataManager
from cacheseek.service.config import CacheConfig, CacheMode
from cacheseek.service.connection import ConnectionManager
from cacheseek.service.lifecycle import CacheService
from cacheseek.service.log_monitor import setup_cache_log_sink
from cacheseek.reuse.approximate.strategy import VideoBasedApproximateCache


_BANNER_SEP_WIDTH = 50  # mirrors telefuser.utils.logging._SEP_WIDTH


def _print_cache_service_banner(config: "CacheConfig", source: str) -> None:
    """Print Cache Service config on startup, mirroring TeleFuser Logging banner style."""
    # Gate ANSI escapes on TTY so piped / log-captured banners stay plain text
    # instead of leaking '\x1b[1;36m...' sequences.
    if sys.stdout.isatty():
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        DIM = "\033[2m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        CYAN = GREEN = YELLOW = BLUE = MAGENTA = DIM = BOLD = RESET = ""
    SEP = f"{DIM}─{'─' * _BANNER_SEP_WIDTH}─{RESET}"

    mode = getattr(config.cache_mode, "value", config.cache_mode)
    vector_store = (getattr(config, "vector_store_type", "") or "").lower() or "none"
    kv_store = (getattr(config, "kv_store_type", "") or "").lower() or "none"

    if vector_store == "qdrant":
        url = (getattr(config, "qdrant_url", "") or "").strip() or ":memory:"
        backend_desc = f"Qdrant {url}"
    elif vector_store == "faiss":
        idx_dir = getattr(config, "faiss_index_dir", None) or f"{config.latent_cache_dir}/faiss"
        backend_desc = f"FAISS {idx_dir}"
    else:
        backend_desc = vector_store

    lines = [
        SEP,
        f"{BOLD}{CYAN}Cache Service{RESET}  {DIM}initialized{RESET}  {DIM}(source: {source}, backend: cacheseek){RESET}",
        f"  {GREEN}●{RESET} mode={mode}  {BLUE}●{RESET} vector={vector_store}  {YELLOW}●{RESET} kv={kv_store}",
        f"  {DIM}Backend:{RESET} {backend_desc}",
        f"  {DIM}Cache dir:{RESET} {config.latent_cache_dir}"
        f"  {DIM}Collection:{RESET} {getattr(config, 'video_vector_collection', '-')}"
        f"  {DIM}Vector dim:{RESET} {getattr(config, 'vector_dim', '-')}",
        f"  {DIM}Strategy:{RESET} {getattr(config, 'cache_strategy_type', '-')}"
        f"  {DIM}Key steps:{RESET} {list(getattr(config, 'key_steps', []) or [])}",
    ]
    if getattr(config, "rerank_enabled", False):
        lines.append(
            f"  {MAGENTA}●{RESET} rerank=on"
            f"  {DIM}top_k:{RESET} {getattr(config, 'rerank_top_k', '-')}"
            f"  {DIM}threshold:{RESET} {getattr(config, 'rerank_score_threshold', '-')}"
        )
    else:
        lines.append(f"  {DIM}● rerank=off{RESET}")
    lines.append(
        f"  {DIM}Async save:{RESET} {getattr(config, 'save_async_enabled', '-')}"
        f"  {DIM}Queue:{RESET} {getattr(config, 'save_queue_size', '-')}"
        f"/{getattr(config, 'save_queue_warn_threshold', '-')}"
        f"  {DIM}On full:{RESET} {getattr(config, 'save_on_full', '-')}"
    )
    # Vector store (FAISS file) and KV store (Fluxon DRAM / local file) live
    # on independent durability tiers. A KV-backend restart can leave vector
    # entries pointing to evicted KV; Strategy.lookup detects this on the
    # first hit and lazy-evicts the stale vector + metadata entry, so no
    # manual reconciliation is needed.
    lines.append(
        f"  {YELLOW}⚠{RESET}  {DIM}vector / KV durability differ; stale vector entries"
        f" lazy-evicted on first KV miss (no manual reconcile needed){RESET}"
    )
    lines.append(SEP)

    # Use print so the banner survives independent of any loguru sink config.
    for line in lines:
        print(line, flush=True)


class CacheServiceFactory:
    """Build the lifecycle pair: ``(CacheService, TeleFuserCacheAdapter)``."""

    @staticmethod
    def create_cache_service(
        ppl_file: Optional[str],
        enable_latent_cache: Optional[bool],
        cache_mode: Optional[str] = None,
    ) -> Optional[Tuple[CacheService, TeleFuserCacheAdapter]]:
        """Build the lifecycle pair: ``(CacheService, TeleFuserCacheAdapter)``.

        Returns ``None`` (with a warning logged) if cache deps are missing,
        ``ppl_file`` is absent, or any wiring step fails — the caller
        treats ``None`` as "cache disabled, run uncached".

        Caller is responsible for keeping both the service and adapter
        alive for the request lifetime; ``service.shutdown()`` cascades
        to ``Strategy.shutdown()`` which closes the ``ConnectionManager``.
        """
        try:
            if ppl_file is None:
                raise ValueError(
                    "enable_latent_cache is enabled but no ppl_file provided. "
                    "Please provide a pipeline file containing the CACHE_CONFIG dict."
                )

            # 1. Resolve CacheConfig: ppl-file CACHE_CONFIG overrides defaults.
            ppl_cache_config = None
            ppl_cache_config_load_error = None
            try:
                ppl_cache_config = import_function_from_file(ppl_file, "CACHE_CONFIG")
                logger.info(f"Found CACHE_CONFIG in {ppl_file}")
            except AttributeError:
                ppl_cache_config = None
            except (ImportError, FileNotFoundError, KeyError, ValueError, OSError) as exc:
                ppl_cache_config_load_error = exc
                logger.warning(f"Failed to load CACHE_CONFIG from {ppl_file}: {exc}")
                ppl_cache_config = None

            cache_config_source = "CacheConfig"
            if isinstance(ppl_cache_config, CacheConfig):
                app_cache_config = ppl_cache_config
                cache_config_source = "ppl CACHE_CONFIG"
            elif isinstance(ppl_cache_config, dict):
                valid_keys = {field.name for field in fields(CacheConfig)}
                overrides = {k: v for k, v in ppl_cache_config.items() if k in valid_keys}
                unknown_keys = sorted(set(ppl_cache_config.keys()) - valid_keys)
                if unknown_keys:
                    logger.warning(f"Ignore unknown CACHE_CONFIG keys: {', '.join(unknown_keys)}")
                app_cache_config = CacheConfig(**overrides)
                cache_config_source = "ppl CACHE_CONFIG"
            else:
                app_cache_config = CacheConfig()

            if isinstance(app_cache_config.cache_mode, str):
                try:
                    app_cache_config.cache_mode = CacheMode(app_cache_config.cache_mode)
                except ValueError:
                    logger.warning(
                        f"Invalid cache_mode '{app_cache_config.cache_mode}' in CACHE_CONFIG, "
                        "fallback to default READ_WRITE"
                    )
                    app_cache_config.cache_mode = CacheConfig().cache_mode

            if enable_latent_cache is not None:
                app_cache_config.enable_latent_cache = enable_latent_cache
                cache_config_source = "command line"

            if cache_mode is not None:
                try:
                    app_cache_config.cache_mode = CacheMode(cache_mode)
                    cache_config_source = "command line"
                except ValueError:
                    logger.warning(f"Invalid cache_mode '{cache_mode}', using {app_cache_config.cache_mode}")

            # 2. Setup cache log sink early so subsequent failures land in cache_service.log.
            if getattr(app_cache_config, "cache_log_enabled", False) and setup_cache_log_sink:
                cache_log_dir = getattr(app_cache_config, "cache_log_dir", None)
                if not cache_log_dir:
                    cache_log_dir = str(Path(app_cache_config.latent_cache_dir) / "logs")
                setup_cache_log_sink(
                    log_dir=cache_log_dir,
                    level=getattr(app_cache_config, "cache_log_level", "DEBUG"),
                    rotation=getattr(app_cache_config, "cache_log_rotation", "100 MB"),
                    retention=getattr(app_cache_config, "cache_log_retention", "7 days"),
                )
                if ppl_cache_config_load_error is not None:
                    logger.warning(
                        "CACHE_CONFIG load failed during cache init, using defaults. Original error: {}",
                        ppl_cache_config_load_error,
                    )

            # 3. Build directory layout + backends + strategy directly.
            #    Strategy owns its KV / vector / metadata backend handles.
            cache_root = Path(app_cache_config.latent_cache_dir)
            storage_dir = cache_root / "storage"
            metadata_dir = cache_root / "metadata"
            dit_cache_dir = cache_root / "dit_cache"
            for d in (cache_root, storage_dir, metadata_dir, dit_cache_dir):
                d.mkdir(parents=True, exist_ok=True)

            conn_mgr = ConnectionManager(app_cache_config, storage_dir=storage_dir)
            metadata_manager = LocalCacheMetadataManager(metadata_dir)
            strategy = VideoBasedApproximateCache(
                app_cache_config,
                conn_mgr.kv_store,
                conn_mgr.vector_store,
                metadata_manager,
                prompt_encoder=conn_mgr.prompt_encoder,
                video_encoder=conn_mgr.video_encoder,
                reranker=conn_mgr.reranker,
            )
            # Cascade-close hook: Strategy.shutdown will close the ConnectionManager
            # (which owns KV + vector handles) when CacheService.shutdown fires.
            strategy._cacheseek_conn_mgr = conn_mgr
            # Attach JSONL audit log for observability (lookup_hit / save_stored / kv_missing_after_vector_hit events).
            audit_log_path = cache_root / "logs" / "audit.jsonl"
            strategy._cacheseek_audit_log = JSONLAuditLog(audit_log_path)

            # Pass CacheConfig.key_steps to adapter so the miss-path
            # latent_data dict carries the right saved_steps for the
            # pipeline to snapshot. 
            adapter_default_saved_steps = tuple(
                int(s) for s in (getattr(app_cache_config, "key_steps", None) or [])
            ) or None

            # 4. Build lifecycle: orchestrator + adapter.
            #    Pass vector_wait_* through so lookup blocks for in-flight
            #    save's vector upsert.
            service = CacheService(
                strategies=[strategy],
                async_save=bool(getattr(app_cache_config, "save_async_enabled", True)),
                max_queue_size=int(getattr(app_cache_config, "save_queue_size", 8) or 8),
                on_full=str(getattr(app_cache_config, "save_on_full", "drop") or "drop"),
                flush_on_shutdown=bool(getattr(app_cache_config, "flush_on_shutdown", True)),
                vector_wait_poll_s=float(getattr(app_cache_config, "vector_wait_poll_s", 0.05) or 0.05),
                vector_wait_warn_s=float(getattr(app_cache_config, "vector_wait_warn_s", 2.0) or 2.0),
                vector_wait_timeout_s=float(getattr(app_cache_config, "vector_wait_timeout_s", 120.0) or 120.0),
                cache_mode=app_cache_config.cache_mode,
            )
            adapter = TeleFuserCacheAdapter(
                default_saved_steps=adapter_default_saved_steps,
            )

            mode_value = getattr(app_cache_config.cache_mode, "value", app_cache_config.cache_mode)
            logger.info(
                "Cache service enabled (mode: {}, source: {}, backend: cacheseek)",
                mode_value,
                cache_config_source,
            )
            _print_cache_service_banner(app_cache_config, cache_config_source)
            return service, adapter
        except ValueError:
            raise
        except (ImportError, FileNotFoundError, KeyError, OSError) as e:
            # NameError / TypeError / AttributeError-on-our-code intentionally
            # propagate — those are bugs, not "cache backend unavailable".
            logger.warning(f"Failed to initialize cache service: {e}")
            return None
