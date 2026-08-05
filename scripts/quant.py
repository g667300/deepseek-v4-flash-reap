#!/usr/bin/env python3
"""Dequantization schemes, one per checkpoint family.

Two families are supported today:

``nvfp4``
    nvidia/modelopt NVFP4: E2M1 nibbles,
    a per-16-column F8_E4M3 scale and an F32 global scale.  Implemented in
    :mod:`nvfp4`; re-exported here so callers only need one import.

``deepseek``
    The format DeepSeek ships in ``DeepSeek-V4-Flash-0731``.  Derived from the
    checkpoint's own ``inference/`` sources rather than guessed:

    * routed experts are FP4 -- ``convert.py`` stores ``[out, in/2]`` nibbles
      (advertised as ``I8`` in the safetensors header, the bytes are
      ``float4_e2m1fn_x2``) with ``FP4_TABLE`` = ``[0, .5, 1, 1.5, 2, 3, 4, 6]``
      then the same negated, i.e. bit 3 is the sign, and the **low nibble is the
      earlier element along K** (``convert.py:31-33``).  The scale is
      ``[out, in/32]`` F8_E8M0 -- one power-of-two per 32 elements along the
      reduction dim (``model.py`` ``Linear``: "1x32 quant on K").
    * everything else is FP8 E4M3 with a **128x128 block** scale, also E8M0
      (``config.json`` ``scale_fmt: ue8m0``).  ``convert.py:126`` spells the
      dequantization out: ``w.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
      .float() * scale[:, None, :, None]``.

    Note there is no second/global scale here -- unlike NVFP4 a single E8M0
    factor per block is the whole story.
"""

from __future__ import annotations

import torch

from nvfp4 import dequantize_nvfp4, unpack_fp4  # noqa: F401  (re-exported)

FP4_BLOCK = 32          # DeepSeek FP4: 32 elements along K share one scale
FP8_BLOCK = 128         # DeepSeek FP8: 128x128 weight blocks share one scale


def e8m0_to_float(scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode an E8M0 scale to a real number.

    E8M0 is a bare biased exponent: the stored byte ``b`` means ``2**(b - 127)``
    (``b == 255`` is NaN).  ``torch.float8_e8m0fnu`` knows how to cast itself, so
    prefer that when safetensors handed us the typed tensor; fall back to the
    bit trick when it arrives as raw ``uint8``.
    """
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.to(dtype) - 127.0)
    return scale.to(dtype)


def dequantize_deepseek_fp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a DeepSeek FP4 weight.

    :param weight: ``[out, in/2]`` uint8, two packed E2M1 values per byte,
        packed along the reduction (last) dim
    :param scale: ``[out, in/32]`` E8M0, one power-of-two per 32 columns
    """
    if weight.dtype != torch.uint8:
        weight = weight.view(torch.uint8)
    out_dim, packed = weight.shape
    in_dim = packed * 2
    n_groups = scale.shape[-1]
    if n_groups * FP4_BLOCK != in_dim:
        raise ValueError(
            f"scale groups {n_groups} * {FP4_BLOCK} != unpacked columns {in_dim} "
            f"(weight {tuple(weight.shape)}, scale {tuple(scale.shape)})"
        )
    if scale.shape[0] != out_dim:
        raise ValueError(f"scale rows {scale.shape[0]} != weight rows {out_dim}")

    unpacked = unpack_fp4(weight, torch.float32)  # [out, in]
    factor = e8m0_to_float(scale, torch.float32)  # [out, in/32]
    out = unpacked.unflatten(-1, (n_groups, FP4_BLOCK)) * factor.unsqueeze(-1)
    return out.flatten(-2).to(dtype)


def dequantize_deepseek_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a DeepSeek block-scaled FP8 weight.

    :param weight: ``[out, in]`` float8_e4m3fn
    :param scale: ``[ceil(out/128), ceil(in/128)]`` E8M0, one per weight block
    """
    out_dim, in_dim = weight.shape
    b_out, b_in = scale.shape
    if b_out * FP8_BLOCK < out_dim or b_in * FP8_BLOCK < in_dim:
        raise ValueError(
            f"scale {tuple(scale.shape)} too small for weight {tuple(weight.shape)} "
            f"at block size {FP8_BLOCK}"
        )
    factor = e8m0_to_float(scale, torch.float32)
    w = weight.to(torch.float32)
    if out_dim % FP8_BLOCK or in_dim % FP8_BLOCK:
        # Ragged tail: expand the scale per element and crop. DeepSeek's own
        # shapes are all multiples of 128, so this is defensive only.
        factor = factor.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
        return (w * factor[:out_dim, :in_dim]).to(dtype)
    w = w.unflatten(0, (b_out, FP8_BLOCK)).unflatten(-1, (b_in, FP8_BLOCK))
    w = w * factor[:, None, :, None]
    return w.reshape(out_dim, in_dim).to(dtype)


def dequantize_linear(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    scheme: str = "nvfp4",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one linear's weight from a dict of its raw tensors.

    Returns the plain ``weight`` untouched when no scale accompanies it, which
    is how both families mark "this tensor was left in BF16".

    :param scheme: ``nvfp4`` or ``deepseek``
    """
    w = tensors[f"{prefix}.weight"]
    if scheme == "nvfp4":
        scale = tensors.get(f"{prefix}.weight_scale")
        if scale is None:
            return w.to(dtype)
        return dequantize_nvfp4(w, scale, tensors[f"{prefix}.weight_scale_2"], dtype)
    if scheme == "deepseek":
        scale = tensors.get(f"{prefix}.scale")
        if scale is None:
            return w.to(dtype)
        # FP4 weights arrive packed two-per-byte, FP8 weights one-per-byte; the
        # element count is what tells them apart.
        if w.dtype in (torch.uint8, torch.int8):
            return dequantize_deepseek_fp4(w, scale, dtype)
        return dequantize_deepseek_fp8(w, scale, dtype)
    raise ValueError(f"unknown quantization scheme {scheme!r}")
