#!/usr/bin/env python3
"""Verify the DeepSeek FP4/FP8 dequantization against independent oracles.

Nothing here trusts our own reading of the format:

1. every one of the 256 nibble-pair byte patterns is checked against torch's
   native ``float4_e2m1fn_x2`` cast (when this torch build has it);
2. E8M0 decoding is checked against torch's ``float8_e8m0fnu`` cast;
3. real expert weights are pushed through DeepSeek's *own*
   ``inference/convert.py:cast_e2m1fn_to_e4m3fn`` -- documented as a lossless
   FP4 -> FP8 re-encoding -- and dequantized down that second path, which must
   land on exactly the same numbers as dequantizing the FP4 directly;
4. real attention weights are compared against the block-FP8 formula spelled
   out in ``convert.py:126``.

Run: ``.venv/bin/python scripts/test_quant.py [--ckpt DIR]``  (CPU only)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import (  # noqa: E402
    FP4_BLOCK,
    FP8_BLOCK,
    dequantize_deepseek_fp4,
    dequantize_deepseek_fp8,
    e8m0_to_float,
    unpack_fp4,
)

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "models" / "DeepSeek-V4-Flash-0731"
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def load_convert_module(ckpt: Path):
    """Import the checkpoint's own inference/convert.py as a reference."""
    path = ckpt / "inference" / "convert.py"
    spec = importlib.util.spec_from_file_location("ds_convert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nibble_lut(convert) -> None:
    print("\n== FP4 nibble decoding ==")
    packed = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    ours = unpack_fp4(packed, torch.float32)

    # Oracle A: the checkpoint's own table, applied the way convert.py does.
    lo = (packed & 0x0F).long()
    hi = ((packed >> 4) & 0x0F).long()
    theirs = torch.stack([convert.FP4_TABLE[lo], convert.FP4_TABLE[hi]], dim=-1).flatten(1)
    check("all 256 byte patterns match convert.py's FP4_TABLE", torch.equal(ours, theirs))

    # Oracle B: torch's own packed-FP4 dtype, if this build has it.
    if hasattr(torch, "float4_e2m1fn_x2"):
        try:
            native = packed.view(torch.float4_e2m1fn_x2).to(torch.float32)
        except NotImplementedError as exc:  # no CPU cast kernel in this build
            print(f"  [SKIP] torch.float4_e2m1fn_x2 has no CPU cast ({exc.__class__.__name__})")
        else:
            check(
                "all 256 byte patterns match torch.float4_e2m1fn_x2",
                torch.equal(ours, native),
                f"torch {torch.__version__}",
            )
    else:
        print(f"  [SKIP] torch {torch.__version__} has no float4_e2m1fn_x2")


def test_e8m0() -> None:
    print("\n== E8M0 scale decoding ==")
    raw = torch.arange(255, dtype=torch.uint8)  # 255 == NaN, excluded
    ours = e8m0_to_float(raw)
    expected = torch.exp2(raw.to(torch.float64) - 127.0).to(torch.float32)
    check("2**(b-127) for b in 0..254", torch.equal(ours, expected))
    if hasattr(torch, "float8_e8m0fnu"):
        native = raw.view(torch.float8_e8m0fnu).to(torch.float32)
        check("matches torch.float8_e8m0fnu cast", torch.equal(ours, native))
        check("typed tensors decode identically", torch.equal(e8m0_to_float(raw.view(torch.float8_e8m0fnu)), ours))
    else:
        print(f"  [SKIP] torch {torch.__version__} has no float8_e8m0fnu")


def read(ckpt: Path, weight_map: dict, name: str) -> torch.Tensor:
    with safe_open(str(ckpt / weight_map[name]), framework="pt") as f:
        return f.get_tensor(name)


def test_real_fp4(ckpt: Path, weight_map: dict, convert) -> None:
    print("\n== FP4 experts (real weights) ==")
    prefix = "layers.5.ffn.experts.0.w1"
    w = read(ckpt, weight_map, f"{prefix}.weight")
    s = read(ckpt, weight_map, f"{prefix}.scale")
    print(f"  {prefix}: weight {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)} {s.dtype}")

    ours = dequantize_deepseek_fp4(w, s, torch.float32)
    check("shape is [out, in] with in = 2 * packed", ours.shape == (w.shape[0], w.shape[1] * 2))

    # Second path: convert.py re-encodes FP4 -> FP8 losslessly, carrying the
    # E8M0 block scale. Dequantizing *that* must reproduce the same tensor.
    w_i8 = w.view(torch.int8) if w.dtype != torch.int8 else w
    fp8, blk_scale = convert.cast_e2m1fn_to_e4m3fn(w_i8, e8m0_to_float(s))
    theirs = dequantize_deepseek_fp8(fp8, blk_scale, torch.float32)
    max_diff = (ours - theirs).abs().max().item()
    check(
        "matches convert.py's lossless FP4->FP8 re-encoding",
        torch.equal(ours, theirs),
        f"max|diff| = {max_diff:.3e}",
    )

    finite = torch.isfinite(ours).all().item()
    check("all values finite", bool(finite))
    print(f"  stats: std {ours.std():.5f}, max|w| {ours.abs().max():.5f}, "
          f"zeros {100 * (ours == 0).float().mean():.1f}%")

    # The scale must be applied along K in blocks of 32, not transposed: a
    # wrong orientation would still be finite, so check the block structure
    # explicitly against a hand-rolled loop on a small corner.
    corner_w, corner_s = w[:4, : 2 * FP4_BLOCK], s[:4, :4]
    manual = torch.empty(4, 4 * FP4_BLOCK, dtype=torch.float32)
    lut = torch.cat([convert.FP4_TABLE[:8], convert.FP4_TABLE[8:]])
    for r in range(4):
        for g in range(4):
            for e in range(FP4_BLOCK):
                col = g * FP4_BLOCK + e
                byte = int(corner_w[r, col // 2])
                nib = (byte & 0x0F) if col % 2 == 0 else ((byte >> 4) & 0x0F)
                manual[r, col] = lut[nib] * float(e8m0_to_float(corner_s)[r, g])
    check(
        "scale applies per 32 elements along K (hand-rolled corner)",
        torch.equal(manual, ours[:4, : 4 * FP4_BLOCK]),
    )


def test_real_fp8(ckpt: Path, weight_map: dict) -> None:
    print("\n== FP8 non-expert (real weights) ==")
    prefix = "layers.5.attn.wq_a"
    w = read(ckpt, weight_map, f"{prefix}.weight")
    s = read(ckpt, weight_map, f"{prefix}.scale")
    print(f"  {prefix}: weight {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)} {s.dtype}")

    ours = dequantize_deepseek_fp8(w, s, torch.float32)
    # convert.py:126 verbatim.
    scale = e8m0_to_float(s)
    theirs = (
        w.to(torch.float32)
        .unflatten(0, (-1, FP8_BLOCK))
        .unflatten(-1, (-1, FP8_BLOCK))
        .float()
        * scale[:, None, :, None].float()
    ).reshape(w.shape)
    check("matches convert.py's block-FP8 formula", torch.equal(ours, theirs))
    check("all values finite", bool(torch.isfinite(ours).all().item()))
    print(f"  stats: std {ours.std():.5f}, max|w| {ours.abs().max():.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    convert = load_convert_module(args.ckpt)
    test_nibble_lut(convert)
    test_e8m0()

    index = json.loads((args.ckpt / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    test_real_fp4(args.ckpt, weight_map, convert)
    test_real_fp8(args.ckpt, weight_map)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
