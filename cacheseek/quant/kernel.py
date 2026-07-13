# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the CacheSeek project
"""Optional kernels for KIVI-style KV-cache quantization.

The public helpers preserve the storage contract established by
``cacheseek.quant.kivi``:

* grouped affine quantization uses minimum offsets and unsigned q values;
* int4 packing is flat, low nibble to high nibble, eight values per int32;
* CPU tensors, CUDA tensors without Triton, and environments without Triton all
  use PyTorch fallbacks.

When Triton is available and tensors are CUDA resident, the grouped quant/dequant
and int4 pack/unpack paths use fused Triton kernels. Otherwise the same helpers
run with ordinary torch ops on the tensor's current device, which gives a CUDA
PyTorch fallback without copying full KV tensors back to CPU.
"""

from __future__ import annotations

from typing import Any

try:  # Triton is an optional acceleration dependency.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional local install.
    triton = None
    tl = None


_TRITON_AVAILABLE = triton is not None and tl is not None


if _TRITON_AVAILABLE:

    @triton.jit
    def _round_half_to_even(x):
        floored = tl.floor(x)
        frac = x - floored
        floor_i = floored.to(tl.int32)
        is_half = frac == 0.5
        floor_is_odd = (floor_i & 1) == 1
        round_up = (frac > 0.5) | (is_half & floor_is_odd)
        return floored + round_up.to(tl.float32)

    @triton.jit
    def _quantize_grouped_last_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        offset_ptr,
        GROUP_SIZE: tl.constexpr,
        LEVELS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < GROUP_SIZE
        ptrs = x_ptr + group_id * GROUP_SIZE + offs

        x_min_load = tl.load(ptrs, mask=mask, other=float("inf")).to(tl.float32)
        x_max_load = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        mn = tl.min(x_min_load, axis=0)
        mx = tl.max(x_max_load, axis=0)
        scale = (mx - mn) / LEVELS
        scale_safe = tl.where(scale > 0.0, scale, 1.0)

        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        q = _round_half_to_even((x - mn) / scale_safe)
        q = tl.minimum(tl.maximum(q, 0.0), LEVELS).to(tl.uint8)

        tl.store(q_ptr + group_id * GROUP_SIZE + offs, q, mask=mask)
        tl.store(scale_ptr + group_id, scale)
        tl.store(offset_ptr + group_id, mn)

    @triton.jit
    def _dequantize_grouped_last_kernel(
        q_ptr,
        scale_ptr,
        offset_ptr,
        out_ptr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < GROUP_SIZE
        ptrs = group_id * GROUP_SIZE + offs

        q = tl.load(q_ptr + ptrs, mask=mask, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + group_id).to(tl.float32)
        offset = tl.load(offset_ptr + group_id).to(tl.float32)
        out = q * scale + offset
        tl.store(out_ptr + ptrs, out, mask=mask)

    @triton.jit
    def _pack_int4_flat_kernel(
        q_ptr,
        packed_ptr,
        NUM_VALUES: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ):
        word_offsets = tl.program_id(0) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        value_base = word_offsets * 8
        word_mask = value_base < NUM_VALUES
        packed = tl.zeros((BLOCK_WORDS,), dtype=tl.uint32)

        for i in range(8):
            value_offsets = value_base + i
            q = tl.load(q_ptr + value_offsets, mask=value_offsets < NUM_VALUES, other=0).to(
                tl.uint32
            )
            packed = packed | ((q & 0xF) << (4 * i))

        tl.store(packed_ptr + word_offsets, packed.to(tl.int32), mask=word_mask)

    @triton.jit
    def _unpack_int4_flat_kernel(
        packed_ptr,
        q_ptr,
        NUM_VALUES: tl.constexpr,
        BLOCK_VALUES: tl.constexpr,
    ):
        value_offsets = tl.program_id(0) * BLOCK_VALUES + tl.arange(0, BLOCK_VALUES)
        mask = value_offsets < NUM_VALUES
        word_offsets = value_offsets // 8
        shift = (value_offsets - word_offsets * 8) * 4
        packed = tl.load(packed_ptr + word_offsets, mask=mask, other=0).to(tl.uint32)
        q = ((packed >> shift) & 0xF).to(tl.uint8)
        tl.store(q_ptr + value_offsets, q, mask=mask)


def triton_available() -> bool:
    """Return whether optional Triton kernels were imported successfully."""

    return bool(_TRITON_AVAILABLE)


def can_use_triton(*tensors: Any) -> bool:
    """Return whether all provided tensors can run through the Triton path."""

    if not _TRITON_AVAILABLE or not tensors:
        return False
    devices = []
    for tensor in tensors:
        if tensor is None or not getattr(tensor, "is_cuda", False):
            return False
        devices.append(getattr(tensor, "device", None))
    return all(device == devices[0] for device in devices)


def quantize_grouped(
    tensor: Any,
    *,
    axis: int,
    group_size: int,
    bits: int,
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    """Quantize ``tensor`` in fixed-size groups along ``axis``.

    Returns ``(q, scale, offset, padded_shape)`` where ``q`` has the padded
    original layout and dtype ``uint8``. ``scale`` and ``offset`` are stored in
    grouped-stat layout: original axes with the grouped axis moved to the end and
    replaced by ``num_groups``.
    """

    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"quantize_grouped expects torch.Tensor, got {type(tensor).__name__}")
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    axis = _normalize_axis(axis, tensor.ndim)
    x = _pad_to_group(
        tensor.detach().to(torch.float32).contiguous(),
        axis=axis,
        group_size=group_size,
    )
    padded_shape = tuple(int(s) for s in x.shape)

    if not can_use_triton(x):
        q, scale, offset = _quantize_grouped_torch(x, axis=axis, group_size=group_size, bits=bits)
        return q, scale, offset, padded_shape

    moved = x.movedim(axis, -1).contiguous()
    groups = moved.shape[-1] // group_size
    grouped_shape = (*moved.shape[:-1], groups, group_size)
    grouped = moved.reshape(grouped_shape)
    rows = grouped.numel() // group_size

    q_grouped = torch.empty_like(grouped, dtype=torch.uint8)
    scale = torch.empty((*moved.shape[:-1], groups), dtype=torch.float32, device=x.device)
    offset = torch.empty_like(scale)

    block_size = _next_power_of_2(group_size)
    _quantize_grouped_last_kernel[(rows,)](
        grouped,
        q_grouped,
        scale,
        offset,
        GROUP_SIZE=group_size,
        LEVELS=(1 << bits) - 1,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )

    q = _restore_grouped(q_grouped, axis=axis)
    return q.contiguous(), scale.contiguous(), offset.contiguous(), padded_shape


def dequantize_grouped(
    qdata: Any,
    scale: Any,
    offset: Any,
    *,
    axis: int,
    group_size: int,
    original_shape: tuple[int, ...] | None = None,
    dtype: Any | None = None,
) -> Any:
    """Dequantize full ``uint8`` qdata from grouped KIVI metadata."""

    torch = _torch()
    if not isinstance(qdata, torch.Tensor):
        raise TypeError(f"dequantize_grouped expects torch.Tensor, got {type(qdata).__name__}")
    if offset is None:
        raise ValueError("offset tensor is required")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    axis = _normalize_axis(axis, qdata.ndim)
    if dtype is None:
        dtype = torch.float32

    if not can_use_triton(qdata, scale, offset):
        out = _dequantize_grouped_torch(
            qdata,
            scale,
            offset,
            axis=axis,
            group_size=group_size,
        )
        out = _slice_to_shape(out, original_shape) if original_shape is not None else out
        return out.to(dtype=dtype).contiguous()

    q = qdata.contiguous()
    moved = q.movedim(axis, -1).contiguous()
    if moved.shape[-1] % group_size != 0:
        raise ValueError(
            f"grouped axis length {moved.shape[-1]} is not divisible by group_size={group_size}"
        )

    groups = moved.shape[-1] // group_size
    expected_stats_shape = (*moved.shape[:-1], groups)
    if tuple(scale.shape) != expected_stats_shape or tuple(offset.shape) != expected_stats_shape:
        raise ValueError(
            "scale/offset shape mismatch: "
            f"expected {expected_stats_shape}, got scale={tuple(scale.shape)} "
            f"offset={tuple(offset.shape)}"
        )

    grouped = moved.reshape(*moved.shape[:-1], groups, group_size)
    out_grouped = torch.empty(grouped.shape, dtype=dtype, device=q.device)
    block_size = _next_power_of_2(group_size)
    rows = grouped.numel() // group_size

    _dequantize_grouped_last_kernel[(rows,)](
        grouped,
        scale.contiguous(),
        offset.contiguous(),
        out_grouped,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )

    out = _restore_grouped(out_grouped, axis=axis)
    out = _slice_to_shape(out, original_shape) if original_shape is not None else out
    return out.contiguous()


def pack_int4_to_int32(qdata: Any) -> Any:
    """Pack unsigned int4 values into int32 words, low nibble first."""

    torch = _torch()
    if not isinstance(qdata, torch.Tensor):
        raise TypeError(f"pack_int4_to_int32 expects torch.Tensor, got {type(qdata).__name__}")

    flat = qdata.reshape(-1).contiguous()
    num_values = int(flat.numel())
    num_words = (num_values + 7) // 8
    if num_words == 0:
        return torch.empty(0, dtype=torch.int32, device=qdata.device)

    if can_use_triton(flat):
        packed = torch.empty(num_words, dtype=torch.int32, device=flat.device)
        block_words = 256
        _pack_int4_flat_kernel[(triton.cdiv(num_words, block_words),)](
            flat,
            packed,
            NUM_VALUES=num_values,
            BLOCK_WORDS=block_words,
            num_warps=4,
        )
        return packed.contiguous()

    flat_i64 = flat.to(torch.int64)
    pad = (-num_values) % 8
    if pad:
        flat_i64 = torch.cat(
            (flat_i64, torch.zeros(pad, dtype=flat_i64.dtype, device=flat_i64.device))
        )

    nibbles = flat_i64.reshape(-1, 8)
    packed = torch.zeros(nibbles.shape[0], dtype=torch.int64, device=flat_i64.device)
    for i in range(8):
        packed |= (nibbles[:, i] & 0xF) << (4 * i)
    return packed.to(torch.int32).contiguous()


def unpack_int4_from_int32(packed: Any, num_values: int) -> Any:
    """Unpack flat int32-packed int4 values into a ``uint8`` tensor."""

    torch = _torch()
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"unpack_int4_from_int32 expects torch.Tensor, got {type(packed).__name__}")
    num_values = int(num_values)
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if num_values == 0:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)

    words = packed.reshape(-1).contiguous()
    if can_use_triton(words):
        out = torch.empty(num_values, dtype=torch.uint8, device=words.device)
        block_values = 256
        _unpack_int4_flat_kernel[(triton.cdiv(num_values, block_values),)](
            words,
            out,
            NUM_VALUES=num_values,
            BLOCK_VALUES=block_values,
            num_warps=4,
        )
        return out.contiguous()

    words_i64 = words.to(torch.int64)
    parts = []
    for i in range(8):
        parts.append(((words_i64 >> (4 * i)) & 0xF).to(torch.uint8))
    return torch.stack(parts, dim=1).reshape(-1)[:num_values].contiguous()


def _quantize_grouped_torch(
    tensor: Any,
    *,
    axis: int,
    group_size: int,
    bits: int,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    grouped = _reshape_grouped(tensor, axis=axis, group_size=group_size)
    offset = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    levels = float((1 << bits) - 1)
    scale = (maximum - offset) / levels
    scale_safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round((grouped - offset.unsqueeze(-1)) / scale_safe.unsqueeze(-1))
    q = q.clamp_(0, int(levels)).to(torch.uint8)
    return _restore_grouped(q, axis=axis).contiguous(), scale.contiguous(), offset.contiguous()


def _dequantize_grouped_torch(
    qdata: Any,
    scale: Any,
    offset: Any,
    *,
    axis: int,
    group_size: int,
) -> Any:
    torch = _torch()
    q_grouped = _reshape_grouped(qdata.to(torch.float32), axis=axis, group_size=group_size)
    decoded = q_grouped * scale.to(torch.float32).unsqueeze(-1) + offset.to(
        torch.float32
    ).unsqueeze(-1)
    return _restore_grouped(decoded, axis=axis)


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
    return torch.cat((tensor, last.expand(*shape)), dim=axis)


def _reshape_grouped(tensor: Any, *, axis: int, group_size: int) -> Any:
    axis = _normalize_axis(axis, tensor.ndim)
    moved = tensor.movedim(axis, -1).contiguous()
    if moved.shape[-1] % group_size != 0:
        raise ValueError(
            f"grouped axis length {moved.shape[-1]} is not divisible by group_size={group_size}"
        )
    groups = moved.shape[-1] // group_size
    return moved.reshape(*moved.shape[:-1], groups, group_size)


def _restore_grouped(grouped: Any, *, axis: int) -> Any:
    ndim = grouped.ndim - 1
    axis = _normalize_axis(axis, ndim)
    flat = grouped.reshape(*grouped.shape[:-2], grouped.shape[-2] * grouped.shape[-1])
    return flat.movedim(-1, axis).contiguous()


def _slice_to_shape(tensor: Any, shape: tuple[int, ...] | None) -> Any:
    if shape is None:
        return tensor
    return tensor[tuple(slice(0, int(size)) for size in shape)]


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError(f"ndim must be positive, got {ndim}")
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of bounds for ndim={ndim}")
    return axis


def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 2


def _torch() -> Any:
    import torch

    return torch


__all__ = [
    "can_use_triton",
    "dequantize_grouped",
    "pack_int4_to_int32",
    "quantize_grouped",
    "triton_available",
    "unpack_int4_from_int32",
]
