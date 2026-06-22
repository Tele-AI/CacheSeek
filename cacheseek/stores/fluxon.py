# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Fluxon KV store adapter.

Fluxon is TeleAI's distributed KV backend, accessed via the ``fluxon_py`` Python
binding.

## External dependencies

- The ``fluxon_py`` package (not pip-installable; provided by the TeleAI Fluxon
  repo as pylib_src source or a prebuilt wheel).
- A Fluxon YAML config (``fluxon_config_path``) is required before startup,
  specifying instance_key, shared_memory_path, cluster_name, etc. See the
  deployment examples in the Fluxon repo.

## Import strategy (lazy)

1. Prefer local source under ``fluxon/pylib_src/`` at the repo root.
2. Fall back to the installed ``fluxon_py`` package.
3. If both fail, raise ``ImportError`` at the call site.

The module itself always imports as long as ``__init__`` is not actually invoked
(see ``__init__.py``).

## Fluxon protocol adaptation (Result/Future)

==============  =========================================================
Our interface   Native Fluxon API
==============  =========================================================
get(key)        store.get(key) -> Result[Future, Err]
                  -> Future.wait() -> Result[MemHolder, Err]
                    -> MemHolder.access() -> Result[dict, Err]
                      -> dict["v"] -> bytes
put(key, val)   store.put(key, {"v": val}) -> Result[Future, Err]
                  -> Future.wait() -> Result[bool, Err]
remove(key)     store.remove(key) -> Result[bool, Err]
==============  =========================================================

Every Result must be explicitly consumed (``unwrap`` / ``unwrap_error``);
otherwise a strict Result raises on destruction.

## Error policy

- ``get()``: KeyNotFound is treated as a cache miss (returns None); other backend
  errors raise RuntimeError.
- ``put()`` / ``remove()``: failures raise RuntimeError; the caller decides
  whether to roll back.
- ``list_keys()``: Fluxon offers no enumeration interface, so it returns an empty
  list.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class FluxonKVStore:
    """Fluxon KV store adapter. See the module docstring for the error policy."""

    _BYTES_FIELD_KEY = "v"  # Fluxon examples/tests use "v" as the bytes field key.

    def __init__(
        self,
        config_path: str | None = None,
        store: Any | None = None,
    ) -> None:
        """Construct the adapter from a Fluxon config or an injected raw store.

        Args:
            config_path: Path to a Fluxon YAML config used to build a new store.
                Required unless ``store`` is given.
            store: An already-constructed raw Fluxon store to wrap (singleton
                case); when provided, ``config_path`` is ignored and no import
                is performed.

        Raises:
            ValueError: Neither ``config_path`` nor ``store`` was supplied.
            ImportError: The Fluxon Python API could not be imported.
            RuntimeError: Store construction failed for the given config.
        """
        if store is not None:
            # Inject an already-constructed raw store (singleton case; skips a
            # redundant import).
            self._store = store
            return
        if not config_path:
            raise ValueError(
                "FluxonKVStore requires either config_path or an injected store."
            )

        fluxon = self._import_fluxon_layer()
        cfg = fluxon.FluxonKvClientConfig.from_file(config_path)
        res = fluxon.new_store(cfg)
        store_obj = self._unwrap_result_ok(res, op="new_store")
        if store_obj is None:
            err = self._unwrap_result_err(res, op="new_store")
            raise RuntimeError(
                f"FluxonKVStore init failed config_path={config_path!r} "
                f"err_type={type(err).__name__} err={err}"
            )
        self._store = store_obj

    def get(self, key: str) -> bytes | None:
        """Fetch the value bytes for ``key``, or None on a cache miss.

        KeyNotFound from the backend is mapped to a miss (None); any other
        backend failure is wrapped and re-raised as RuntimeError.
        """
        try:
            r = self._store.get(key)
            fut = self._unwrap_result_ok(r, op="get")
            if fut is None:
                err = self._unwrap_result_err(r, op="get")
                if self._is_key_not_found(err):
                    logger.debug("FluxonKV.get miss key={!r}", key)
                    return None
                raise RuntimeError(
                    f"FluxonKV.get backend error key={key!r} "
                    f"err_type={type(err).__name__} err={err}"
                )

            wait_res = fut.wait()
            value_obj = self._unwrap_result_ok(wait_res, op="get.wait")
            if value_obj is None:
                err = self._unwrap_result_err(wait_res, op="get.wait")
                if self._is_key_not_found(err):
                    return None
                raise RuntimeError(
                    f"FluxonKV.get.wait backend error key={key!r} "
                    f"err_type={type(err).__name__} err={err}"
                )

            return self._extract_bytes_from_value(value_obj, key=key)
        except Exception as exc:
            logger.exception("FluxonKV.get failed key={} err={}", key, exc)
            raise RuntimeError(f"FluxonKV.get failed key={key!r}: {exc}") from exc

    def put(self, key: str, value: bytes) -> None:
        """Store ``value`` bytes under ``key``, waiting for the write to complete.

        Any backend failure is wrapped and re-raised as RuntimeError.
        """
        try:
            # Fluxon put takes a flat dict; store the bytes under the fixed field key.
            res = self._store.put(key, {self._BYTES_FIELD_KEY: value})
            fut = self._unwrap_result_ok(res, op="put")
            if fut is None:
                err = self._unwrap_result_err(res, op="put")
                raise RuntimeError(err)

            wait_res = fut.wait()
            if self._unwrap_result_ok(wait_res, op="put.wait") is None:
                err = self._unwrap_result_err(wait_res, op="put.wait")
                raise RuntimeError(err)
        except Exception as exc:
            logger.exception("FluxonKV.put failed key={} err={}", key, exc)
            raise RuntimeError(f"FluxonKV.put failed key={key!r}: {exc}") from exc

    def remove(self, key: str) -> None:
        """Delete ``key`` from the backend; removing a missing key is a no-op.

        Any non-KeyNotFound backend failure is wrapped and re-raised as
        RuntimeError.
        """
        try:
            res = self._store.remove(key)
            ok = self._unwrap_result_ok(res, op="remove")
            if ok is None:
                err = self._unwrap_result_err(res, op="remove")
                # Removing a missing key is not an error.
                if self._is_key_not_found(err):
                    return
                raise RuntimeError(err)
        except Exception as exc:
            logger.exception("FluxonKV.remove failed key={} err={}", key, exc)
            raise RuntimeError(f"FluxonKV.remove failed key={key!r}: {exc}") from exc

    def list_keys(self) -> list[str]:
        """Fluxon offers no key-enumeration interface; returns an empty list."""
        logger.warning(
            "FluxonKVStore.list_keys() is not supported by the Fluxon backend; "
            "returning []."
        )
        return []

    # ------------------------------------------------------------------
    # TensorKVStore capability — DLPack zero-pickle tensor put/get.
    #
    # put_tensor hands the tensor's DLPack pointer to Fluxon (no Python bytes, no
    # pickle); on get_tensor, access() returns a DLPack view onto the mapped shared
    # memory, and from_dlpack + clone materializes an owned CPU tensor (decoupled
    # from the MemHolder lifetime). DLPack supports only CPU, C-contiguous tensors,
    # hence .detach().cpu().contiguous() before put.
    # ------------------------------------------------------------------
    def put_tensor(self, key: str, tensor: Any) -> None:
        """Store ``tensor`` under ``key`` via DLPack — no pickle, no Python bytes.

        The tensor is moved to CPU and made C-contiguous (DLPack constraints)
        before its DLPack pointer is handed to Fluxon. Any backend failure is
        wrapped and re-raised as RuntimeError.
        """
        import torch  # noqa: F401  (lazy — heavy dep stays out of import time)

        try:
            t = tensor.detach().to("cpu").contiguous()
            res = self._store.put(key, {self._BYTES_FIELD_KEY: t})
            fut = self._unwrap_result_ok(res, op="put_tensor")
            if fut is None:
                err = self._unwrap_result_err(res, op="put_tensor")
                raise RuntimeError(err)
            wait_res = fut.wait()
            if self._unwrap_result_ok(wait_res, op="put_tensor.wait") is None:
                err = self._unwrap_result_err(wait_res, op="put_tensor.wait")
                raise RuntimeError(err)
        except Exception as exc:
            logger.exception("FluxonKV.put_tensor failed key={} err={}", key, exc)
            raise RuntimeError(
                f"FluxonKV.put_tensor failed key={key!r}: {exc}"
            ) from exc

    def get_tensor(
        self,
        key: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any = None,
    ) -> Any | None:
        """Fetch a tensor for ``key``, or None on a cache miss.

        On a DLPack payload, materializes an owned CPU tensor via
        ``from_dlpack(...).clone()`` so it outlives the backend MemHolder. On a
        raw-bytes payload, reinterprets the bytes using ``dtype`` and ``shape``.
        ``device``, if given, moves the result onto that device.

        Args:
            key: Store key to read.
            shape: Target tensor shape, used to reshape a raw-bytes payload.
            dtype: torch dtype used to reinterpret a raw-bytes payload.
            device: Optional device to move the returned tensor onto.

        Raises:
            RuntimeError: Any backend failure other than a key-not-found miss
                (which returns None). Every error — including an unexpected
                payload type — is wrapped as RuntimeError, never propagated raw.
        """
        import torch

        try:
            r = self._store.get(key)
            fut = self._unwrap_result_ok(r, op="get_tensor")
            if fut is None:
                err = self._unwrap_result_err(r, op="get_tensor")
                if self._is_key_not_found(err):
                    return None
                raise RuntimeError(err)

            wait_res = fut.wait()
            mh = self._unwrap_result_ok(wait_res, op="get_tensor.wait")
            if mh is None:
                err = self._unwrap_result_err(wait_res, op="get_tensor.wait")
                if self._is_key_not_found(err):
                    return None
                raise RuntimeError(err)

            d_res = mh.access()
            if not d_res.is_ok():
                err = d_res.unwrap_error("get_tensor.access failed")
                raise RuntimeError(f"err_type={type(err).__name__} err={err}")
            d = d_res.unwrap("get_tensor.access ok")
            v = d.get(self._BYTES_FIELD_KEY)
            if hasattr(v, "__dlpack__"):
                # zero-copy view into the pool; clone to own it past the holder.
                t = torch.from_dlpack(v).clone()
            elif isinstance(v, (bytes, bytearray, memoryview)):
                t = (
                    torch.frombuffer(bytes(v), dtype=torch.uint8)
                    .view(dtype)
                    .reshape(shape)
                )
            else:
                raise TypeError(
                    f"Fluxon get_tensor payload field {self._BYTES_FIELD_KEY!r} is "
                    f"neither dlpack nor bytes (type={type(v).__name__})"
                )
            if device is not None:
                t = t.to(device)
            return t
        except Exception as exc:
            logger.exception("FluxonKV.get_tensor failed key={} err={}", key, exc)
            raise RuntimeError(
                f"FluxonKV.get_tensor failed key={key!r}: {exc}"
            ) from exc

    def _extract_bytes_from_value(
        self,
        value_obj: Any,
        *,
        key: str | None = None,
    ) -> bytes | None:
        """Coerce a Fluxon get() value into raw bytes (or None).

        Accepts a bytes-like value directly, or a MemHolder whose ``access()``
        returns a dict carrying the bytes under the fixed field key.

        Args:
            value_obj: The object returned by the resolved get() future.
            key: Optional key, included in error messages for context.

        Returns:
            The value bytes, or None if ``value_obj`` is None.

        Raises:
            TypeError: ``value_obj`` lacks a callable ``access()``, its payload
                is not a dict, or the payload's bytes field is not bytes-like.
            RuntimeError: ``access()`` raised or returned an error Result.
        """
        if value_obj is None:
            return None
        if isinstance(value_obj, (bytes, bytearray, memoryview)):
            return bytes(value_obj)

        context = f" key={key!r}" if key is not None else ""
        access = getattr(value_obj, "access", None)
        if access is None or not callable(access):
            raise TypeError(
                "FluxonKV value object is missing callable access() "
                f"type={type(value_obj).__name__}{context}"
            )
        try:
            d_res = access()
        except Exception as exc:
            logger.exception("Fluxon MemHolder.access failed{} err={}", context, exc)
            raise RuntimeError(
                f"Fluxon MemHolder.access failed{context}: {exc}"
            ) from exc

        if not d_res.is_ok():
            err = d_res.unwrap_error("mem.access failed")
            raise RuntimeError(
                "Fluxon MemHolder.access returned error "
                f"err_type={type(err).__name__} err={err}{context}"
            )

        d = d_res.unwrap("mem.access ok")
        if not isinstance(d, dict):
            raise TypeError(
                "Fluxon MemHolder.access returned non-dict payload "
                f"type={type(d).__name__}{context}"
            )

        v = d.get(self._BYTES_FIELD_KEY)
        if not isinstance(v, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Fluxon MemHolder.access payload does not contain bytes field "
                f"field={self._BYTES_FIELD_KEY!r} value_type={type(v).__name__}"
                f"{context}"
            )
        return bytes(v)

    def _unwrap_result_ok(self, res: Any, op: str) -> Any:
        """Consume the ok branch of a Fluxon Result.

        A Fluxon Result must be explicitly consumed via unwrap()/unwrap_error();
        otherwise a strict Result raises on destruction.
        """
        if res is None:
            return None
        if not res.is_ok():
            return None
        return res.unwrap(f"{op} failed")

    def _unwrap_result_err(self, res: Any, op: str) -> Any:
        """Consume the error branch of a Fluxon Result."""
        if res is None:
            return RuntimeError(f"{op} returned None")
        if res.is_ok():
            # Consume the ok value so the strict Result does not assert on an
            # unconsumed result at destruction.
            _ = res.unwrap(f"{op} ok")
            return None
        return res.unwrap_error(f"{op} failed")

    def _is_key_not_found(self, err: Any) -> bool:
        if err is None:
            return False
        name = type(err).__name__
        return name in {"KeyNotFoundError", "ChanKeyNotFoundError"}

    def _import_fluxon_layer(self):
        """Lazily import the Fluxon Python API.

        Prefer local source under ``fluxon/pylib_src/`` at the repo root (easier to
        debug), falling back to the installed ``fluxon_py`` package. Raise
        ImportError if both fail.
        """
        load_errors: list[str] = []

        # 1) Prefer local Fluxon source if present.
        try:
            import sys
            from pathlib import Path as _Path

            # __file__ -> stores/fluxon.py; parents[3] -> repo root
            repo_root = _Path(__file__).resolve().parents[3]
            fluxon_root = (
                repo_root / "fluxon" / "pylib_src"
            )  # the fluxon_py package lives under pylib_src
            if fluxon_root.is_dir():
                fluxon_path = str(fluxon_root)
                if fluxon_path not in sys.path:
                    sys.path.insert(0, fluxon_path)
                import fluxon_py as api  # type: ignore

                return api
        except Exception as exc:
            load_errors.append(
                f"local Fluxon source import failed type={type(exc).__name__} err={exc}"
            )

        # 2) Fall back to the installed fluxon_py package.
        try:
            import fluxon_py as api  # type: ignore

            return api
        except Exception as exc:
            load_errors.append(
                f"installed fluxon_py import failed type={type(exc).__name__} err={exc}"
            )

        detail = (
            "; ".join(load_errors) if load_errors else "no import attempts succeeded"
        )
        raise ImportError(
            f"Fluxon Python API not available (fluxon_py): {detail}. "
            "Install fluxon_py or place the source at <repo>/fluxon/pylib_src/."
        )
