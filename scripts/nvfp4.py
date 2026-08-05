#!/usr/bin/env python3
"""NVFP4 dequantization for nvidia/modelopt-format checkpoints.

Derivation of the formula (all verified against compressed-tensors' own code
rather than guessed):

* ``ModelOptNvfp4Converter`` (compressed_tensors/entrypoints/convert/converters/
  modelopt_nvfp4.py) maps modelopt -> compressed-tensors as
  ``weight -> weight_packed``, ``weight_scale -> weight_scale`` (unchanged),
  ``weight_scale_2 -> weight_global_scale = 1 / weight_scale_2``.
* ``NVFP4PackedCompressor.decompress`` unpacks the nibbles and calls
  ``dequantize(x_q=unpacked, scale=weight_scale, global_scale=...)``.
* ``_dequantize`` (quantization/lifecycle/forward_helpers.py) computes
  ``scale = scale / global_scale`` and then ``x_q * scale``.

Substituting ``global_scale = 1 / weight_scale_2`` gives, straight from the
modelopt tensors::

    w = unpack_fp4(weight) * weight_scale.float() * weight_scale_2

with ``weight_scale`` (F8_E4M3) broadcast over each group of 16 columns and
``weight_scale_2`` an F32 scalar.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 16

# E2M1: 1 sign, 2 exponent, 1 mantissa bit -> 8 magnitudes, index 0..7
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_LUT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Signed 16-entry lookup table indexed directly by the raw nibble."""
    key = (device, dtype)
    if key not in _LUT_CACHE:
        pos = torch.tensor(_E2M1, dtype=dtype, device=device)
        _LUT_CACHE[key] = torch.cat([pos, -pos])  # nibble bit 3 == sign
    return _LUT_CACHE[key]


def unpack_fp4(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack a ``[m, n/2]`` uint8 tensor into ``[m, n]`` float values.

    The low nibble holds the first (even-column) value, the high nibble the
    second -- matching ``pack_fp4_to_uint8``'s ``low | (high << 4)``.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8, got {packed.dtype}")
    lut = _lut(packed.device, dtype)
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    # interleave so column order is preserved: [lo0, hi0, lo1, hi1, ...]
    out = torch.stack((lut[lo], lut[hi]), dim=-1)
    return out.flatten(-2)


def dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize modelopt NVFP4 tensors to a dense tensor.

    :param weight: ``[m, n/2]`` uint8, two packed E2M1 values per byte
    :param weight_scale: ``[m, n/16]`` F8_E4M3 per-group scale
    :param weight_scale_2: F32 scalar global scale
    :param dtype: output dtype
    """
    m, n_packed = weight.shape
    n = n_packed * 2
    n_groups = weight_scale.shape[-1]
    if n_groups * GROUP_SIZE != n:
        raise ValueError(
            f"scale groups {n_groups} * {GROUP_SIZE} != unpacked columns {n} "
            f"(weight {tuple(weight.shape)}, scale {tuple(weight_scale.shape)})"
        )
    if weight_scale.shape[0] != m:
        raise ValueError(
            f"scale rows {weight_scale.shape[0]} != weight rows {m}"
        )

    unpacked = unpack_fp4(weight, torch.float32)  # [m, n]
    scale = weight_scale.to(torch.float32) * weight_scale_2.to(torch.float32)
    out = unpacked.unflatten(-1, (n_groups, GROUP_SIZE)) * scale.unsqueeze(-1)
    return out.flatten(-2).to(dtype)


def dequantize_linear(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one linear's weight given a dict of its modelopt tensors.

    Falls back to returning the plain ``weight`` when the layer is not
    quantized (e.g. the BF16 shared experts / attention / MTP block).
    """
    w = tensors[f"{prefix}.weight"]
    scale = tensors.get(f"{prefix}.weight_scale")
    if scale is None:
        return w.to(dtype)
    return dequantize_nvfp4(w, scale, tensors[f"{prefix}.weight_scale_2"], dtype)
