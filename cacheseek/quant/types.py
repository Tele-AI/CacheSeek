# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Basic data types for KV-cache quantization.

This module is intentionally torch-free. Codec implementations may use torch,
but store/reuse layers should be able to inspect payload metadata without
importing or initializing tensor libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable


class TensorRole(str, Enum):
    """Which half of a per-layer attention KV pair a tensor represents."""

    KEY = "k"
    VALUE = "v"


class QuantScheme(str, Enum):
    """Supported KV-cache quantization schemes."""

    NONE = "none"
    KIVI_INT4 = "kivi_int4"
    KIVI_INT8 = "kivi_int8"


class QuantDType(str, Enum):
    """Storage dtype for quantized payload tensors."""

    UINT8 = "uint8"
    INT8 = "int8"
    INT32_PACKED = "int32_packed"


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Original tensor view needed to reconstruct a decoded KV tensor.

    Attributes:
        shape: Logical tensor shape before quantization.
        dtype: Original tensor dtype, stored as a portable string such as
            "bfloat16", "float16", or "float32".
        layout: Logical layout string, for example "H,T,D".
    """

    shape: tuple[int, ...]
    dtype: str
    layout: str  # e.g. "H,T,D" for (num_heads, num_tokens, head_dim)


@dataclass(frozen=True, slots=True)
class QuantTensorSpec:
    """Quantization metadata for one tensor component.

    group_axis identifies the original tensor axis split into fixed-size
    quantization groups. For layout "H,T,D", group_axis=1 means grouping
    along token axis T.
    """

    role: TensorRole
    scheme: QuantScheme
    bits: int
    storage_dtype: QuantDType
    group_size: int
    group_axis: int

    padded_shape: tuple[int, ...] | None = None
    pack_order: str = "low_to_high"

    scale_dtype: str = "float32"
    offset_dtype: str = "float32"
    offset_kind: Literal["minimum", "zero_point"] = "minimum"

    symmetric: bool = False


@dataclass(frozen=True, slots=True)
class QuantTensor:
    """Encoded representation of one key or value tensor.

    Tensor-bearing fields are typed as Any so this type can represent torch
    tensors, numpy arrays, or backend-native tensor handles.
    """

    tensor: TensorSpec
    quant: QuantTensorSpec
    qdata: Any
    scale: Any
    offset: Any | None = None

    @property
    def role(self) -> TensorRole:
        """Return the tensor role carried by the quantization spec."""

        return self.quant.role


@dataclass(frozen=True, slots=True)
class KVQuantizedLayer:
    """Quantized payload for one attention layer's KV pair."""

    key: QuantTensor
    value: QuantTensor

    def as_pair(self) -> tuple[QuantTensor, QuantTensor]:
        """Return (key, value) in the same order as runtime KV payloads."""

        return self.key, self.value


@runtime_checkable
class KVLayerCodec(Protocol):
    """Codec boundary used by reuse managers and KV stores."""

    @property
    def scheme(self) -> QuantScheme:
        """Quantization scheme implemented by this codec."""

        ...

    def encode_layer(self, key: Any, value: Any) -> KVQuantizedLayer:
        """Encode one attention layer's key/value tensors."""

        ...

    def decode_layer(self, payload: KVQuantizedLayer) -> tuple[Any, Any]:
        """Decode one attention layer payload back to ordinary tensors."""

        ...


__all__ = [
    "TensorRole",
    "QuantScheme",
    "QuantDType",
    "TensorSpec",
    "QuantTensorSpec",
    "QuantTensor",
    "KVQuantizedLayer",
    "KVLayerCodec",
]