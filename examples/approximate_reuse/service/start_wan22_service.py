#!/usr/bin/env python3
"""Start a TeleFuser Wan2.2 T2V service with a CacheSeek config in one file.

One launcher, several concrete configs selected with ``--preset`` (see ``PRESETS``
below). Run:

    bash examples/approximate_reuse/service/start_wan22_service.sh --preset s1_fresh

``--dry-run`` prints the effective service/cache config without starting the GPU
service. ``--preset`` defaults to ``DEFAULT_PRESET``.
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
import tempfile
from pathlib import Path
from typing import Any


CACHESEEK_REPO = Path(__file__).resolve().parents[2]
# Instance-specific roots; override via env on your host. Experiment writes go to
# local scratch /tmp, never a shared volume — see WORKDIR.
MODEL_ZOO_DIR = Path(
    os.environ.get("CACHESEEK_MODEL_ZOO", "/path/to/model_zoo")
).resolve()
DEFAULT_TELEFUSER_REPO = os.environ.get(
    "CACHESEEK_TELEFUSER_REPO", "/path/to/telefuser"
)
WORKDIR = Path(os.environ.get("CACHESEEK_WAN22_WORKDIR", "/tmp/cacheseek_wan22"))


# --- Shared config (everything that does NOT vary by preset) ------------------


def _base_service() -> dict[str, Any]:
    return {
        "host": "127.0.0.1",
        "task": "t2v",
        "parallelism": 2,
        "cuda_visible_devices": os.environ.get("CACHESEEK_CUDA_DEVICES", "0,1"),
        "telefuser_log_level": "DEBUG",
        "telefuser_max_queue_size": 600,
        # port + telefuser_cache_dir filled per preset
    }


_PPL_CONFIG_OVERRIDES: dict[str, Any] = {
    "model_root": str(MODEL_ZOO_DIR / "Wan2.2-T2V-A14B"),
}


def _base_cache() -> dict[str, Any]:
    return {
        # Basic cache.
        "enable_latent_cache": True,
        "cache_mode": "read_write",  # read_write | read_only | write_only
        "max_cache_size_gb": 10,
        "cache_log_enabled": True,
        "cache_log_dir": None,  # None => {latent_cache_dir}/logs
        "cache_log_level": "DEBUG",
        "cache_log_rotation": "100 MB",
        "cache_log_retention": "7 days",
        # KV store — preset overrides kv_store_type / fluxon_config_path.
        "kv_store_type": "local_file",  # local_file | fluxon
        "fluxon_config_path": "/path/to/fluxon-deploy/external_config.yaml",
        # Vector store.
        "vector_store_type": "qdrant",  # faiss | qdrant
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_api_key": None,
        "faiss_index_dir": None,  # None => {latent_cache_dir}/faiss
        "vector_dim": 2048,
        "cache_strategy_type": "video_approximate",
        # Lookup strategy.
        "key_steps": [3, 7, 11, 14],  # P0 evidence experiment grid (W04-0512)
        "max_skip_step": 3,  # match key_steps upper bound (lib default 5 too low)
        "lookup_mode": "video",
        # Text embedding.
        "text_embedding_model_path": str(MODEL_ZOO_DIR / "Qwen3-VL-Embedding-2B"),
        "text_embedding_instruction": "Represent the user's input",
        "text_embedding_device_id": 1,
        "text_embedding_torch_dtype": None,
        "text_embedding_attn_impl": None,
        # Video embedding.
        "video_embedding_enabled": True,
        "video_embedding_model_path": str(MODEL_ZOO_DIR / "Qwen3-VL-Embedding-2B"),
        "video_embedding_instruction": "Represent the user's input",
        "video_embedding_fps": 1.0,
        "video_embedding_max_frames": 16,
        "video_embedding_max_length": 8192,
        "video_embedding_min_pixels": 4096,
        "video_embedding_max_pixels": 1843200,
        "video_embedding_total_pixels": 7864320,
        "video_embedding_device_id": 1,
        "video_embedding_torch_dtype": None,
        "video_embedding_attn_impl": None,
        # Video vector search and rerank.
        "video_similarity_threshold": 0.10,
        # video_vector_collection filled per preset
        "rerank_enabled": True,
        "rerank_model_path": str(MODEL_ZOO_DIR / "Qwen3-VL-Reranker-2B"),
        "rerank_top_k": 5,
        "rerank_batch_size": 2,
        "rerank_device_id": 0,
        "rerank_torch_dtype": None,
        "rerank_score_threshold": 0.6,
        # Async save.
        "save_async_enabled": True,
        "save_queue_size": 2,
        "save_on_full": "drop",  # drop | sync | downgrade
        "save_queue_warn_threshold": 8,
        "vector_wait_warn_s": 2.0,
        "vector_wait_poll_s": 0.05,
        "vector_wait_timeout_s": 120.0,
        "flush_on_shutdown": True,
    }


# --- Presets: only the fields that differ between concrete deployments --------
#
# Each preset supplies its TeleFuser checkout + the handful of fields that vary;
# the large shared remainder comes from _base_service() / _base_cache(). Add a
# new deployment by adding an entry here — do not fork this file.
#
#   telefuser_cache_dir = <telefuser_repo>/<telefuser_cache_subdir>
#   latent_cache_dir    = <latent_cache_dir>            (absolute literal), or
#                         <telefuser_repo>/<latent_cache_subdir>
#   cache               = overrides merged onto _base_cache()

PRESETS: dict[str, dict[str, Any]] = {
    # Single-machine smoke: local_file KV + faiss vector — no daemons (no qdrant,
    # no fluxon). All writes land on local nvme /tmp (WORKDIR); PVC is read-only
    # for experiments (REPO-TOPOLOGY.md §9).
    "s1_fresh": {
        "telefuser_repo": DEFAULT_TELEFUSER_REPO,
        "port": 8007,
        "telefuser_cache_dir": str(WORKDIR / "telefuser_cache_s1_fresh"),
        "latent_cache_dir": str(WORKDIR / "latent_cache_s1_fresh"),
        "cache": {
            "kv_store_type": "local_file",
            "vector_store_type": "faiss",
            "video_vector_collection": "s1_fresh_repro",
        },
    },
    # Production-like read_write: fluxon DRAM pool KV + faiss vector. Needs a
    # running fluxon stack; set $FLUXON_CONFIG to its external_config.yaml.
    "s1_rw_fluxon": {
        "telefuser_repo": DEFAULT_TELEFUSER_REPO,
        "port": 18007,
        "telefuser_cache_dir": str(WORKDIR / "telefuser_cache_fluxon"),
        "latent_cache_dir": str(WORKDIR / "latent_cache_fluxon"),
        "cache": {
            "kv_store_type": "fluxon",
            "fluxon_config_path": os.environ.get("FLUXON_CONFIG", ""),
            "vector_store_type": "faiss",
            "video_vector_collection": "embodied_s1_rw",
        },
    },
}

DEFAULT_PRESET = "s1_fresh"


def _resolve_preset(
    name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(telefuser_repo, service_config, ppl_overrides, cache_config)``."""
    if name not in PRESETS:
        raise SystemExit(
            f"unknown --preset {name!r}; choices: {', '.join(sorted(PRESETS))}"
        )
    p = PRESETS[name]
    telefuser_repo = Path(p["telefuser_repo"]).resolve()

    service = _base_service()
    service["port"] = p["port"]
    service["telefuser_cache_dir"] = (
        str(p["telefuser_cache_dir"])
        if "telefuser_cache_dir" in p
        else str(telefuser_repo / p["telefuser_cache_subdir"])
    )

    cache = _base_cache()
    cache.update(p.get("cache", {}))
    if "latent_cache_dir" in p:
        cache["latent_cache_dir"] = p["latent_cache_dir"]
    else:
        cache["latent_cache_dir"] = str(telefuser_repo / p["latent_cache_subdir"])

    return telefuser_repo, service, dict(_PPL_CONFIG_OVERRIDES), cache


# Populated by main() once the preset is chosen.
TELEFUSER_REPO: Path = Path()
SERVICE_CONFIG: dict[str, Any] = {}
PPL_CONFIG_OVERRIDES: dict[str, Any] = {}
CACHE_CONFIG: dict[str, Any] = {}


# Launcher implementation below — normally no edits needed.


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _print_config() -> None:
    print("[start_wan22_service] service")
    print(f"  TELEFUSER_REPO={TELEFUSER_REPO}")
    for key, value in SERVICE_CONFIG.items():
        print(f"  {key}={_format_value(value)}")
    print("  server_config.enable_latent_cache=True")
    print()
    print("[start_wan22_service] PPL_CONFIG overrides")
    for key, value in PPL_CONFIG_OVERRIDES.items():
        print(f"  {key}={_format_value(value)}")
    print()
    print("[start_wan22_service] CACHE_CONFIG")
    for key, value in CACHE_CONFIG.items():
        print(f"  {key}={_format_value(value)}")
    print()
    print(
        "[start_wan22_service] run_server("
        f"task={SERVICE_CONFIG['task']}, "
        f"host={SERVICE_CONFIG['host']}, "
        f"port={SERVICE_CONFIG['port']}, "
        f"parallelism={SERVICE_CONFIG['parallelism']})"
    )


def _generated_ppl_source() -> str:
    cache_config = pprint.pformat(CACHE_CONFIG, width=100, sort_dicts=False)
    ppl_overrides = pprint.pformat(PPL_CONFIG_OVERRIDES, width=100, sort_dicts=False)
    return f'''"""Generated by cacheseek/examples/approximate_reuse/service/start_wan22_service.py."""

from examples.wan_video import wan22_14b_text_to_video_service as _wan22_service

PPL_CONFIG = dict(_wan22_service.PPL_CONFIG)
PPL_CONFIG.update({ppl_overrides})

CACHE_CONFIG = {cache_config}

_wan22_service.PPL_CONFIG = PPL_CONFIG
_wan22_service.CACHE_CONFIG = CACHE_CONFIG

get_pipeline = _wan22_service.get_pipeline
run_with_file = _wan22_service.run_with_file
'''


def _write_generated_ppl() -> Path:
    path = Path(tempfile.gettempdir()) / "cacheseek_wan22_t2v_service.py"
    path.write_text(_generated_ppl_source(), encoding="utf-8")
    return path


def _validate_for_runtime(preset: str) -> None:
    if str(TELEFUSER_REPO).startswith("/path/to/") or not TELEFUSER_REPO.exists():
        raise SystemExit(
            f"PRESETS['{preset}']['telefuser_repo'] does not exist on this host "
            f"(current: {TELEFUSER_REPO}). Fix it in "
            "examples/approximate_reuse/service/start_wan22_service.py before starting."
        )

    if str(MODEL_ZOO_DIR).startswith("/path/to/") or not MODEL_ZOO_DIR.is_dir():
        raise SystemExit(
            "Edit MODEL_ZOO_DIR in examples/approximate_reuse/service/start_wan22_service.py "
            f"before starting (current: {MODEL_ZOO_DIR})."
        )

    if CACHE_CONFIG["kv_store_type"] == "fluxon":
        fluxon_config = str(CACHE_CONFIG.get("fluxon_config_path") or "")
        if not fluxon_config or fluxon_config.startswith("/path/to/"):
            raise SystemExit(
                f"preset {preset!r} uses kv_store_type='fluxon' but fluxon_config_path is "
                "unset/placeholder. Set $FLUXON_CONFIG to the fluxon external_config.yaml "
                "(or use --preset s1_fresh for the daemon-free local_file+faiss smoke)."
            )

    if CACHE_CONFIG["vector_store_type"] == "qdrant" and not CACHE_CONFIG.get(
        "qdrant_url"
    ):
        raise SystemExit("Set qdrant_url when vector_store_type='qdrant'.")


def _start(preset: str) -> None:
    _validate_for_runtime(preset)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(SERVICE_CONFIG["cuda_visible_devices"])
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["TELEFUSER_LOG_LEVEL"] = str(SERVICE_CONFIG["telefuser_log_level"])
    # fluxon's Rust tracing default is debug — silence the keepalive spam.
    # (FLUXON_LOG=WARN on the command line is quietest; this is just a floor.)
    os.environ.setdefault("RUST_LOG", "info")

    sys.path.insert(0, str(TELEFUSER_REPO))
    os.chdir(TELEFUSER_REPO)
    generated_ppl = _write_generated_ppl()

    from telefuser.service.core.config import server_config
    from telefuser.service.main import run_server
    from telefuser.service_types import TaskType

    server_config.enable_latent_cache = bool(CACHE_CONFIG["enable_latent_cache"])
    server_config.cache_dir = str(SERVICE_CONFIG["telefuser_cache_dir"])
    server_config.log_level = str(SERVICE_CONFIG["telefuser_log_level"])
    server_config.max_queue_size = int(SERVICE_CONFIG["telefuser_max_queue_size"])

    run_server(
        str(generated_ppl),
        TaskType(str(SERVICE_CONFIG["task"]).lower()),
        int(SERVICE_CONFIG["port"]),
        str(SERVICE_CONFIG["host"]),
        str(SERVICE_CONFIG["telefuser_cache_dir"]),
        int(SERVICE_CONFIG["parallelism"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start TeleFuser Wan2.2 service with CacheSeek."
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS),
        help=f"Which concrete config to launch (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print effective config and exit."
    )
    parser.add_argument(
        "--print-venv",
        action="store_true",
        help="Print the preset's <telefuser_repo>/.venv/bin/python and exit (used by the .sh wrapper).",
    )
    args = parser.parse_args()

    global TELEFUSER_REPO, SERVICE_CONFIG, PPL_CONFIG_OVERRIDES, CACHE_CONFIG
    TELEFUSER_REPO, SERVICE_CONFIG, PPL_CONFIG_OVERRIDES, CACHE_CONFIG = (
        _resolve_preset(args.preset)
    )

    if args.print_venv:
        print(str(TELEFUSER_REPO / ".venv/bin/python"))
        return

    print(f"[start_wan22_service] preset={args.preset}")
    if args.dry_run:
        _print_config()
        return

    _print_config()
    _start(args.preset)


if __name__ == "__main__":
    main()
