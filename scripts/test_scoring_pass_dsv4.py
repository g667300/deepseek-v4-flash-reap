#!/usr/bin/env python3
"""Verify the DeepSeek-V4 scoring-pass driver, on CPU.

Two things have to hold, and neither is obvious by inspection:

1. **Name coverage.** Every non-expert tensor in the real checkpoint maps to a
   parameter transformers actually has, and every parameter transformers wants
   is supplied. Checked against the real 167 GB checkpoint using only the
   safetensors index -- no tensor data is read.

2. **The three-phase split is exact.** The scoring pass does not run
   ``DeepseekV4DecoderLayer.forward``; it unrolls it so the experts can be
   streamed one at a time, folding the routed output back in as ``post ⊗ out``.
   That rests on the hyper-connection expand step being linear in the MoE
   output. Here a real (tiny) decoder layer is run both ways and the outputs
   compared.

Run: ``.venv/bin/python scripts/test_stage_a_dsv4.py``  (CPU only, no GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reap_saliency_dsv4 as drv  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "models" / "DeepSeek-V4-Flash-0731"
FAILED: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(label)


# --------------------------------------------------------------------------
# 1. name coverage against the real checkpoint
# --------------------------------------------------------------------------


def test_name_coverage(ckpt: Path) -> None:
    print("\n== native -> HF name coverage (real checkpoint) ==")
    from accelerate import init_empty_weights
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer

    config = drv.load_config(ckpt, "sdpa")
    weight_map = json.loads((ckpt / drv.PROFILE.index_name).read_text())["weight_map"]

    # one layer of each flavour: sliding+hash, compressed+hash, heavily+moe
    for layer_idx in (0, 2, 3, 42):
        prefix = f"layers.{layer_idx}."
        expert_prefix = f"{prefix}ffn.experts."
        native = [n for n in weight_map
                  if n.startswith(prefix) and not n.startswith(expert_prefix)]
        try:
            mapped = set()
            for name in native:
                suffix = name[len(prefix):]
                if suffix.endswith(".scale"):
                    continue
                mapped.add(drv.to_hf_name(suffix))
        except KeyError as exc:
            check(False, f"layer {layer_idx}: every tensor maps", str(exc))
            continue

        with init_empty_weights():
            layer = DeepseekV4DecoderLayer(config, layer_idx)
        del layer.mlp.experts
        wanted = {n for n, _ in layer.named_parameters()}
        wanted |= {n for n, _ in layer.named_buffers()}
        # rotary inverse frequencies are computed from the config, not stored
        wanted = {n for n in wanted if "inv_freq" not in n}

        missing = sorted(wanted - mapped)
        extra = sorted(mapped - wanted)
        kind = f"{config.layer_types[layer_idx]}/{config.mlp_layer_types[layer_idx]}"
        check(not missing, f"layer {layer_idx} [{kind}]: no parameter left unfilled",
              f"missing {missing[:4]}" if missing else f"{len(wanted)} params")
        check(not extra, f"layer {layer_idx} [{kind}]: no tensor maps to nothing",
              f"extra {extra[:4]}" if extra else "")


# --------------------------------------------------------------------------
# 2. the three-phase split reproduces the layer's own forward
# --------------------------------------------------------------------------


def tiny_config(ckpt: Path):
    """Shrink the real config rather than inventing one.

    Building a config from scratch is how a port silently diverges
    (head_dim, moe_layer_freq and friends quietly took defaults), so start from
    the shipped file and only narrow what makes the test cheap.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    native = json.loads((ckpt / "config.json").read_text())
    native.update(
        num_hidden_layers=4,
        num_attention_heads=4,
        index_n_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        vocab_size=256,
        num_hash_layers=1,
        compress_ratios=[0, 4, 128, 4],
        index_topk=8,
        sliding_window=8,
    )
    native.pop("quantization_config", None)
    config = DeepseekV4Config(**native)
    config._attn_implementation = "eager"
    return config


class FusedExpertSource:
    """Serves experts out of a live ``DeepseekV4Experts`` module.

    Stands in for the checkpoint reader so the split can be tested without
    inventing a fake FP4 checkpoint: same interface, weights sliced out of the
    fused ``gate_up_proj`` / ``down_proj`` the way the real loader slices them
    out of ``w1`` / ``w3`` / ``w2``.
    """

    def __init__(self, experts, inter: int):
        self.experts = experts
        self.inter = inter

    def load_expert(self, layer: int, expert: int):
        gate_up = self.experts.gate_up_proj[expert]
        return (gate_up[: self.inter], gate_up[self.inter :], self.experts.down_proj[expert])


def test_split_matches_forward(ckpt: Path, layer_idx: int, seq_len: int = 16) -> None:
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4DecoderLayer,
        DeepseekV4RotaryEmbedding,
    )

    torch.manual_seed(layer_idx)
    config = tiny_config(ckpt)
    dtype = torch.float32
    device = torch.device("cpu")
    hc, hidden_size = config.hc_mult, config.hidden_size
    is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
    kind = f"{config.layer_types[layer_idx]}/{config.mlp_layer_types[layer_idx]}"
    print(f"\n== three-phase split vs. layer.forward -- layer {layer_idx} [{kind}] ==")

    layer = DeepseekV4DecoderLayer(config, layer_idx).to(dtype).eval()
    for p in layer.parameters():
        nn.init.normal_(p, std=0.02)
    if is_hash:
        layer.mlp.gate.tid2eid = torch.randint(
            0, config.n_routed_experts, (config.vocab_size, config.num_experts_per_tok)
        )

    batch = 2
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    hidden = torch.randn(batch, seq_len, hc, hidden_size, dtype=dtype) * 0.1
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = DeepseekV4RotaryEmbedding(config=config)
    probe = torch.zeros(batch, seq_len, hidden_size, dtype=dtype)
    pos_emb = {
        "main": rotary(probe, position_ids=position_ids, layer_type="main"),
        "compress": rotary(probe, position_ids=position_ids, layer_type="compress"),
    }
    mask = create_sliding_window_causal_mask(
        config=config, inputs_embeds=probe, attention_mask=None,
        past_key_values=None, position_ids=position_ids,
    )

    with torch.no_grad():
        reference = layer(
            hidden,
            input_ids=input_ids,
            position_embeddings=pos_emb,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )

    # ---- the same thing, unrolled the way the scoring pass does it ----
    n_tok = batch * seq_len
    source = FusedExpertSource(layer.mlp.experts, config.moe_intermediate_size)
    act_fn, limit = layer.mlp.experts.act_fn, layer.mlp.experts.limit

    with torch.no_grad():
        post_a, comb_a, collapsed = layer.attn_hc(hidden)
        attn_out, _ = layer.self_attn(
            layer.input_layernorm(collapsed),
            position_embeddings=pos_emb,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )
        h = post_a.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb_a.to(dtype).transpose(-1, -2), hidden
        )
        post, comb, collapsed = layer.ffn_hc(h)
        xb = layer.post_attention_layernorm(collapsed)
        h = post.to(dtype).unsqueeze(-1) * layer.mlp.shared_experts(xb).unsqueeze(-2) \
            + torch.matmul(comb.to(dtype).transpose(-1, -2), h)

        x_cache = xb.reshape(n_tok, hidden_size)
        post_cache = post.reshape(n_tok, hc).float()
        out = h.reshape(n_tok, hc, hidden_size).clone()

        token_ids = input_ids.reshape(-1) if is_hash else None
        topk_i, topk_w = drv.route(layer.mlp.gate, x_cache, token_ids, device, 4096)
        drv.run_experts(source, layer_idx, act_fn, limit, config.n_routed_experts,
                        x_cache, topk_i, topk_w, out, post_cache, None, device, 4096)

    got = out.view(batch, seq_len, hc, hidden_size)
    diff = (got - reference).abs().max().item()
    scale = reference.abs().max().item()
    check(diff <= 1e-4 * max(scale, 1.0), "output matches layer.forward",
          f"max|diff| = {diff:.3e} (values up to {scale:.3f})")

    routed = int((topk_i >= 0).sum())
    check(routed == n_tok * config.num_experts_per_tok,
          f"every token routed to top-{config.num_experts_per_tok}")
    if is_hash:
        expected = layer.mlp.gate.tid2eid[input_ids.reshape(-1)]
        check(torch.equal(topk_i.long(), expected),
              "hash router picked exactly tid2eid[input_ids]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    test_name_coverage(args.ckpt)
    for layer_idx in (0, 1, 2, 3):
        test_split_matches_forward(args.ckpt, layer_idx)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
