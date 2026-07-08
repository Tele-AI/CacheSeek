# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Naive PyTorch implementation of KIVI-style KV quantization.

This codec is intentionally simple and kernel-free. It establishes the storage
format and correctness path first. The current materialization path decodes back
to ordinary tensors before seeding the runtime KV window.
"""

from __future__ import annotations

from typing import Any

from .types import (
    KVQuantizedLayer,
    QuantDType,
    QuantScheme,
    QuantTensor,
    QuantTensorSpec,
    TensorRole,
    TensorSpec,
)


class KIVICodec:
    """Naive KV codec following KIVI-style key/value grouping.

    For layout H,T,D:
        - key_group_axis="T" implements grouped per-channel quantization.
        - value_group_axis="D" implements grouped per-token quantization.

    This implementation is chunk/layer-level. It does not manage attention,
    rolling windows, ring buffers, or runtime cache eviction.
    """

    def __init__(
        self,
        *,
        bits: int = 4,
        group_size: int = 64,
        layout: str = "H,T,D",
        key_group_axis: str | int = "T",
        value_group_axis: str | int = "D",
        scale_dtype: str = "float32",
        offset_dtype: str = "float32",
    ) -> None:
        if bits not in (4, 8):
            raise ValueError(f"KIVICodec supports bits=4 or bits=8, got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        self.bits = int(bits)
        self.group_size = int(group_size)
        self.layout = layout
        self._layout_tokens = _parse_layout(layout)
        self.key_group_axis = key_group_axis
        self.value_group_axis = value_group_axis
        self.scale_dtype = scale_dtype
        self.offset_dtype = offset_dtype

    @property
    def scheme(self) -> QuantScheme:
        if self.bits == 4:
            return QuantScheme.KIVI_INT4
        return QuantScheme.KIVI_INT8

    def encode_layer(self, key: Any, value: Any) -> KVQuantizedLayer:
        return KVQuantizedLayer(
            key=self._encode_tensor(key, TensorRole.KEY),
            value=self._encode_tensor(value, TensorRole.VALUE),
        )

    def decode_layer(self, payload: KVQuantizedLayer) -> tuple[Any, Any]:
        if payload.key.quant.scheme != self.scheme or payload.value.quant.scheme != self.scheme:
            raise ValueError(
                "quantized layer scheme does not match codec: "
                f"key={payload.key.quant.scheme.value} "
                f"value={payload.value.quant.scheme.value} "
                f"codec={self.scheme.value}"
            )
        return self._decode_tensor(payload.key), self._decode_tensor(payload.value)

    def _encode_tensor(self, tensor: Any, role: TensorRole) -> QuantTensor:
        torch = _torch()

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"KIVICodec expects torch.Tensor, got {type(tensor).__name__}")

        if tensor.ndim != len(self._layout_tokens):
            raise ValueError(
                f"tensor rank {tensor.ndim} does not match layout {self.layout!r} "
                f"with {len(self._layout_tokens)} axes"
            )

        x = tensor.detach().to(device="cpu").contiguous()

        original = TensorSpec(
            shape=tuple(int(s) for s in x.shape),
            dtype=_dtype_name(x.dtype),
            layout=self.layout,
        )

        group_axis = self._group_axis(role, x.ndim)
        x_padded = _pad_to_group(
            x.to(torch.float32),
            axis=group_axis,
            group_size=self.group_size,
        )

        q, scale, offset = self._quantize_grouped(x_padded, axis=group_axis)

        if self.bits == 4:
            qdata = _pack_int4_to_int32(q)
            storage_dtype = QuantDType.INT32_PACKED
        else:
            qdata = q.to(torch.uint8).contiguous()
            storage_dtype = QuantDType.UINT8

        quant = QuantTensorSpec(
            role=role,
            scheme=self.scheme,
            bits=self.bits,
            storage_dtype=storage_dtype,
            group_size=self.group_size,
            group_axis=group_axis,
            padded_shape=tuple(int(s) for s in x_padded.shape),
            pack_order="low_to_high",
            scale_dtype=self.scale_dtype,
            offset_dtype=self.offset_dtype,
            offset_kind="minimum",
            symmetric=False,
        )

        return QuantTensor(
            tensor=original,
            quant=quant,
            qdata=qdata,
            scale=scale.to(dtype=_torch_dtype(self.scale_dtype)).contiguous(),
            offset=offset.to(dtype=_torch_dtype(self.offset_dtype)).contiguous(),
        )

    def _decode_tensor(self, payload: QuantTensor) -> Any:
        torch = _torch()
        quant = payload.quant

        if quant.offset_kind != "minimum":
            raise ValueError(f"KIVICodec only supports minimum offsets, got {quant.offset_kind!r}")
        if payload.offset is None:
            raise ValueError("quantized payload is missing offset tensor")
        if quant.bits not in (4, 8):
            raise ValueError(f"unsupported bit width: {quant.bits}")

        padded_shape = quant.padded_shape or payload.tensor.shape
        if len(padded_shape) != len(payload.tensor.shape):
            raise ValueError(
                "padded_shape rank does not match original tensor rank: "
                f"padded={padded_shape}, original={payload.tensor.shape}"
            )

        group_axis = _normalize_axis(quant.group_axis, len(padded_shape))

        for axis, (original_dim, padded_dim) in enumerate(zip(payload.tensor.shape, padded_shape)):
            original_dim = int(original_dim)
            padded_dim = int(padded_dim)
            if axis == group_axis:
                if padded_dim < original_dim:
                    raise ValueError(
                        "padded grouped axis is shorter than original axis: "
                        f"axis={axis}, padded={padded_dim}, original={original_dim}"
                    )
            elif padded_dim != original_dim:
                raise ValueError(
                    "only grouped axis may be padded: "
                    f"axis={axis}, padded={padded_dim}, original={original_dim}"
                )

        num_values = _numel(padded_shape)

        if quant.bits == 4:
            if quant.storage_dtype is not QuantDType.INT32_PACKED:
                raise ValueError(
                    f"int4 KIVI payload must use INT32_PACKED storage, got {quant.storage_dtype}"
                )
            if quant.pack_order != "low_to_high":
                raise ValueError(f"unsupported int4 pack_order={quant.pack_order!r}")
            q = _unpack_int4_from_int32(payload.qdata, num_values).reshape(padded_shape)
        else:
            if quant.storage_dtype is not QuantDType.UINT8:
                raise ValueError(
                    f"int8 KIVI payload must use UINT8 storage, got {quant.storage_dtype}"
                )
            q = payload.qdata.reshape(padded_shape)

        q_grouped = _reshape_grouped(
            q.to(torch.float32),
            axis=group_axis,
            group_size=quant.group_size,
        )

        # scale/offset were stored in grouped layout, e.g.:
        #   original [H,T,D], group_axis=T -> stats [H,D,num_groups]
        #   original [H,T,D], group_axis=D -> stats [H,T,num_groups]
        # Therefore they only need unsqueeze(-1), not movedim(group_axis, -1).
        scale = _reshape_stats_for_group(payload.scale)
        offset = _reshape_stats_for_group(payload.offset)

        decoded_grouped = q_grouped * scale.to(torch.float32) + offset.to(torch.float32)
        decoded_padded = _restore_grouped(decoded_grouped, axis=group_axis)
        decoded = _slice_to_shape(decoded_padded, payload.tensor.shape)

        return decoded.to(dtype=_torch_dtype(payload.tensor.dtype)).contiguous()

    def _quantize_grouped(self, tensor: Any, *, axis: int) -> tuple[Any, Any, Any]:
        torch = _torch()

        grouped = _reshape_grouped(tensor, axis=axis, group_size=self.group_size)
        offset = grouped.amin(dim=-1)
        maximum = grouped.amax(dim=-1)

        levels = float((1 << self.bits) - 1)
        scale = (maximum - offset) / levels

        # Avoid division by zero for constant groups. q becomes all zeros and
        # dequantizes back to offset, which equals the original constant value.
        scale_safe = torch.where(scale > 0, scale, torch.ones_like(scale))

        q = torch.round((grouped - offset.unsqueeze(-1)) / scale_safe.unsqueeze(-1))
        q = q.clamp_(0, int(levels)).to(torch.uint8)
        q = _restore_grouped(q, axis=axis)

        return q.contiguous(), scale.contiguous(), offset.contiguous()

    def _group_axis(self, role: TensorRole, ndim: int) -> int:
        axis = self.key_group_axis if role is TensorRole.KEY else self.value_group_axis

        if isinstance(axis, int):
            return _normalize_axis(axis, ndim)

        token = axis.strip().upper()
        try:
            return self._layout_tokens.index(token)
        except ValueError as exc:
            raise ValueError(f"axis {axis!r} not present in layout {self.layout!r}") from exc


def _parse_layout(layout: str) -> tuple[str, ...]:
    tokens = tuple(part.strip().upper() for part in layout.split(",") if part.strip())

    if not tokens:
        raise ValueError("layout must contain at least one axis name")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"layout axes must be unique, got {layout!r}")

    return tokens


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError(f"ndim must be positive, got {ndim}")

    axis = int(axis)
    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of bounds for ndim={ndim}")

    return axis


def _pad_to_group(tensor: Any, *, axis: int, group_size: int) -> Any:
    torch = _torch()

    axis = _normalize_axis(axis, tensor.ndim)
    size = int(tensor.shape[axis])

    if size <= 0:
        raise ValueError("cannot quantize an empty tensor axis")

    pad = (-size) % int(group_size)
    if pad == 0:
        return tensor

    last = tensor.select(axis, size - 1).unsqueeze(axis)
    shape = list(tensor.shape)
    shape[axis] = pad
    padding = last.expand(*shape)

    return torch.cat((tensor, padding), dim=axis)


def _reshape_grouped(tensor: Any, *, axis: int, group_size: int) -> Any:
    axis = _normalize_axis(axis, tensor.ndim)

    moved = tensor.movedim(axis, -1).contiguous()

    if moved.shape[-1] % group_size != 0:
        raise ValueError(
            f"grouped axis length {moved.shape[-1]} is not divisible by "
            f"group_size={group_size}"
        )

    groups = moved.shape[-1] // group_size
    return moved.reshape(*moved.shape[:-1], groups, group_size)


def _reshape_stats_for_group(stats: Any) -> Any:
    return stats.to(device="cpu").unsqueeze(-1).contiguous()


def _restore_grouped(grouped: Any, *, axis: int) -> Any:
    ndim = grouped.ndim - 1
    axis = _normalize_axis(axis, ndim)

    flat = grouped.reshape(*grouped.shape[:-2], grouped.shape[-2] * grouped.shape[-1])
    return flat.movedim(-1, axis).contiguous()


def _slice_to_shape(tensor: Any, shape: tuple[int, ...]) -> Any:
    slices = tuple(slice(0, int(size)) for size in shape)
    return tensor[slices]


def _pack_int4_to_int32(qdata: Any) -> Any:
    torch = _torch()

    flat = qdata.reshape(-1).to(torch.int64)

    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat((flat, torch.zeros(pad, dtype=flat.dtype)))

    nibbles = flat.reshape(-1, 8)
    packed = torch.zeros(nibbles.shape[0], dtype=torch.int64)

    for i in range(8):
        packed |= (nibbles[:, i] & 0xF) << (4 * i)

    return packed.to(torch.int32).contiguous()


def _unpack_int4_from_int32(packed: Any, num_values: int) -> Any:
    torch = _torch()

    words = packed.reshape(-1).to(torch.int64)
    parts = []

    for i in range(8):
        parts.append(((words >> (4 * i)) & 0xF).to(torch.uint8))

    return torch.stack(parts, dim=1).reshape(-1)[:num_values].contiguous()


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype(name: str) -> Any:
    torch = _torch()
    dtype_name = str(name).replace("torch.", "")

    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype name {name!r}") from exc


def _torch() -> Any:
    import torch

    return torch


__all__ = [
    "KIVICodec",
    "_pack_int4_to_int32",
    "_unpack_int4_from_int32",
]