# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Factory helpers for KV-cache quantization codecs."""

from __future__ import annotations

from typing import Any

from .types import KVLayerCodec, QuantScheme


def build_kv_codec(
    *,
    quant: str | QuantScheme = QuantScheme.NONE,
    group_size: int = 64,
    kv_layout: str = "H,T,D",
    key_group_axis: str | int = "T",
    value_group_axis: str | int = "D",
    scale_dtype: str = "float32",
    offset_dtype: str = "float32",
) -> KVLayerCodec | None:
    """Build an optional per-layer KV codec from explicit parameters."""

    if quant == QuantScheme.NONE.value:
        return None

    if quant == QuantScheme.KIVI_INT4.value:
        from .kivi import KIVICodec

        return KIVICodec(
            bits=4,
            group_size=group_size,
            layout=kv_layout,
            key_group_axis=key_group_axis,
            value_group_axis=value_group_axis,
            scale_dtype=scale_dtype,
            offset_dtype=offset_dtype,
        )

    if quant == QuantScheme.KIVI_INT8.value:
        from .kivi import KIVICodec

        return KIVICodec(
            bits=8,
            group_size=group_size,
            layout=kv_layout,
            key_group_axis=key_group_axis,
            value_group_axis=value_group_axis,
            scale_dtype=scale_dtype,
            offset_dtype=offset_dtype,
        )

    raise ValueError(f"unsupported KV quantization scheme: {quant!r}")


def build_kv_codec_from_config(cfg: Any) -> KVLayerCodec | None:
    """Build an optional per-layer KV codec from a config-like object.

    This helper expects attributes used by WorldKVConfig, but intentionally does
    not import exact_prefix.config so quant/ can stay reusable.
    """

    return build_kv_codec(
        quant=getattr(cfg, "quant", QuantScheme.NONE.value),
        group_size=getattr(cfg, "group_size", 64),
        kv_layout=getattr(cfg, "kv_layout", "H,T,D"),
        key_group_axis=getattr(cfg, "key_group_axis", "T"),
        value_group_axis=getattr(cfg, "value_group_axis", "D"),
        scale_dtype=getattr(cfg, "scale_dtype", "float32"),
        offset_dtype=getattr(cfg, "offset_dtype", "float32"),
    )


def quant_fingerprint_from_config(cfg: Any) -> dict[str, Any]:
    """Return config fields that must isolate cache namespaces."""

    return {
        "quant": getattr(cfg, "quant", QuantScheme.NONE.value),
        "group_size": getattr(cfg, "group_size", 64),
        "kv_layout": getattr(cfg, "kv_layout", "H,T,D"),
        "key_group_axis": getattr(cfg, "key_group_axis", "T"),
        "value_group_axis": getattr(cfg, "value_group_axis", "D"),
        "scale_dtype": getattr(cfg, "scale_dtype", "float32"),
        "offset_dtype": getattr(cfg, "offset_dtype", "float32"),
    }


__all__ = [
    "build_kv_codec",
    "build_kv_codec_from_config",
    "quant_fingerprint_from_config",
]
