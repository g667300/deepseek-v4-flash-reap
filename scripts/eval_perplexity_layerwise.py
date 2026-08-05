#!/usr/bin/env python3
"""Perplexity and next-token predictions, computed layer-sequentially from disk.

The point of this script is measuring what pruning cost. A pruned checkpoint
can be served normally and scored through an API, but the *unpruned* original
does not fit the serving host -- which is exactly the comparison that matters.
This reuses the scoring pass's machinery (stream one decoder layer at a time, dequantize
it, throw it away) so a 167 GB checkpoint can be scored on a 24 GB GPU, and
runs the same code path over both checkpoints so the numbers are comparable.

What it produces:

* **perplexity** over the held-out set -- directly comparable between runs
* **argmax predictions per position**, saved to the output JSON, so a separate
  comparison can report how often two checkpoints pick the same next token.
  Top-1 agreement says more than a perplexity delta about whether two models
  behave alike.

Usage::

    eval_perplexity_layerwise.py --ckpt models/DeepSeek-V4-Flash-0731 \\
        --tokens artifacts/ppl-holdout-dsv4.pt --out artifacts/ppl-original.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reap_saliency_dsv4 import (  # noqa: E402
    DeepseekSource,
    build_layer,
    load_config,
    route,
    run_experts,
    to_hf_name,
)
from saliency_common import alloc, human, _rss_gb  # noqa: E402

import model_profiles  # noqa: E402

PROFILE = model_profiles.DEEPSEEK_V4


def load_tokens(path: str, vocab_size: int, limit: int | None) -> list[torch.Tensor]:
    blob = torch.load(path, weights_only=True)
    samples = [torch.as_tensor(t, dtype=torch.long) for t in blob]
    if limit:
        samples = samples[:limit]
    hi = max(int(s.max()) for s in samples)
    if hi >= vocab_size:
        raise ValueError(f"token id {hi} exceeds vocab_size {vocab_size}")
    return samples


def build_head(config, source: DeepseekSource, device: torch.device, dtype: torch.dtype):
    """The final stage: collapse the hc streams, RMSNorm, then the vocab projection."""
    from accelerate import init_empty_weights
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperHead,
        DeepseekV4RMSNorm,
    )

    with init_empty_weights():
        hc_head = DeepseekV4HyperHead(config)
        norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # checkpoint names -> the two modules above
    hc_head.load_state_dict(
        {
            "hc_fn": source.get_global("hc_head_fn"),
            "hc_base": source.get_global("hc_head_base"),
            "hc_scale": source.get_global("hc_head_scale"),
        },
        assign=True,
    )
    norm.load_state_dict({"weight": source.get_global("norm.weight")}, assign=True)

    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.weight = nn.Parameter(source.get_global("head.weight"), requires_grad=False)

    # hc_head runs its Sinkhorn-free mixing in float32 in the reference
    # implementation; keep it there and only demote the projection.
    return (
        hc_head.to(device=device, dtype=torch.float32),
        norm.to(device=device, dtype=dtype),
        lm_head.to(device=device, dtype=dtype),
    )


@torch.no_grad()
def score_batch(hidden_b, ids_b, hc_head, norm, lm_head, chunk: int):
    """Return (sum of NLL, token count, argmax predictions) for one batch.

    Chunked over sequence positions: the vocab is 129,280 wide, so materializing
    logits for a whole 2048-token sample at once is gigabytes for no reason.
    """
    bsz, seq = ids_b.shape
    total_nll = 0.0
    total_n = 0
    preds = torch.empty(bsz, seq - 1, dtype=torch.int32)

    collapsed = hc_head(hidden_b.float()).to(norm.weight.dtype)
    collapsed = norm(collapsed)

    for s in range(0, seq - 1, chunk):
        e = min(s + chunk, seq - 1)
        logits = lm_head(collapsed[:, s:e]).float()
        targets = ids_b[:, s + 1 : e + 1]
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).to(logits.device),
            reduction="sum",
        )
        total_nll += float(nll)
        total_n += targets.numel()
        preds[:, s:e] = logits.argmax(-1).to(torch.int32).cpu()
        del logits

    return total_nll, total_n, preds


def run(args) -> int:
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    device = torch.device(args.device)
    dtype = torch.bfloat16
    ckpt = Path(args.ckpt)
    spill = Path(args.spill) if args.spill else None

    config = load_config(ckpt, args.attn)
    num_experts = config.n_routed_experts
    hidden_size = config.hidden_size
    hc = config.hc_mult
    n_layers = config.num_hidden_layers
    native = json.loads((ckpt / "config.json").read_text())
    hash_layers = set(PROFILE.hash_layers(native))
    print(f"checkpoint: {ckpt}")
    print(f"layers: {n_layers}, experts/layer: {num_experts}, "
          f"top_k: {config.num_experts_per_tok}, hc_mult: {hc}")

    source = DeepseekSource(ckpt, device, dtype)

    samples = load_tokens(args.tokens, config.vocab_size, args.max_samples)
    seq_len = min(len(s) for s in samples)
    samples = [s[:seq_len] for s in samples]
    n_samples = len(samples)
    n_tok = n_samples * seq_len
    print(f"scoring: {n_samples} samples x {seq_len} tokens = {n_tok:,} tokens")

    hidden = alloc((n_samples, seq_len, hc, hidden_size), dtype, spill, "hidden")
    x_cache = alloc((n_tok, hidden_size), dtype, spill, "moe_input")
    post_cache = alloc((n_tok, hc), torch.float32, spill, "hc_post")
    token_ids = torch.stack(samples).reshape(-1)
    print(f"host cache: {human(n_tok * hidden_size * (hc * 2 + 2) + n_tok * hc * 4)}")

    embed = nn.Embedding(config.vocab_size, hidden_size)
    embed.weight = nn.Parameter(source.get_global("embed.weight"), requires_grad=False)
    embed = embed.to(device=device, dtype=dtype)
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            ids = torch.stack(samples[i : i + args.batch_size]).to(device)
            e = embed(ids).unsqueeze(2).expand(-1, -1, hc, -1)
            hidden[i : i + args.batch_size] = e.cpu()
    del embed
    gc.collect()
    torch.cuda.empty_cache()

    rotary = DeepseekV4RotaryEmbedding(config=config).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    masks: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        sizes = {min(args.batch_size, n_samples - i) for i in range(0, n_samples, args.batch_size)}
        for bs in sorted(sizes):
            probe = torch.zeros(bs, seq_len, hidden_size, device=device, dtype=dtype)
            masks[bs] = create_sliding_window_causal_mask(
                config=config, inputs_embeds=probe, attention_mask=None,
                past_key_values=None, position_ids=position_ids,
            )
            if masks[bs] is None or masks[bs].dtype == torch.bool:
                raise RuntimeError(
                    f"attn implementation {config._attn_implementation!r} produced a "
                    f"{'None' if masks[bs] is None else 'bool'} mask; use --attn eager. "
                    "See the explanation in reap_saliency_dsv4.py -- a bool mask makes "
                    "DeepseekV4Attention invert the compressor's causality bias, which "
                    "measured perplexity 1.35 here instead of 8.88."
                )
            del probe
        probe = torch.zeros(1, seq_len, hidden_size, device=device, dtype=dtype)
        pos_emb = {
            "main": rotary(probe, position_ids=position_ids, layer_type="main"),
            "compress": rotary(probe, position_ids=position_ids, layer_type="compress"),
        }
        del probe

    hidden_flat = hidden.view(n_tok, hc, hidden_size)
    t_start = time.time()

    for L in range(n_layers):
        t0 = time.time()
        is_hash = L in hash_layers
        layer, act_fn, limit = build_layer(config, L, source, device, dtype)

        with torch.no_grad():
            for i in range(0, n_samples, args.batch_size):
                sl = slice(i, min(i + args.batch_size, n_samples))
                h = hidden[sl].to(device, non_blocking=True).to(dtype)

                post, comb, collapsed = layer.attn_hc(h)
                attn_out, _ = layer.self_attn(
                    layer.input_layernorm(collapsed),
                    position_embeddings=pos_emb,
                    position_ids=position_ids,
                    attention_mask=masks[sl.stop - sl.start],
                    past_key_values=None,
                )
                h = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
                    comb.to(dtype).transpose(-1, -2), h
                )

                post, comb, collapsed = layer.ffn_hc(h)
                xb = layer.post_attention_layernorm(collapsed)
                h = post.to(dtype).unsqueeze(-1) * layer.mlp.shared_experts(xb).unsqueeze(-2) \
                    + torch.matmul(comb.to(dtype).transpose(-1, -2), h)

                x_cache[sl.start * seq_len : sl.stop * seq_len] = xb.reshape(-1, hidden_size).cpu()
                post_cache[sl.start * seq_len : sl.stop * seq_len] = post.reshape(-1, hc).float().cpu()
                hidden[sl] = h.cpu()
                del h, post, comb, collapsed, attn_out, xb
        torch.cuda.empty_cache()

        ids = token_ids if is_hash else None
        topk_i, topk_w = route(layer.mlp.gate, x_cache, ids, device, args.route_chunk)
        run_experts(source, L, act_fn, limit, num_experts, x_cache, topk_i, topk_w,
                    hidden_flat, post_cache, None, device, args.expert_chunk)
        del topk_i, topk_w, layer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  layer {L:>2}/{n_layers - 1}  {time.time() - t0:6.1f}s"
              f"  | VRAM peak {torch.cuda.max_memory_allocated(device) / 1e9:5.1f} GB,"
              f" host RSS {_rss_gb():5.1f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    print(f"all layers done in {time.time() - t_start:.0f}s; scoring...")

    hc_head, norm, lm_head = build_head(config, source, device, dtype)
    total_nll = 0.0
    total_n = 0
    all_preds = torch.empty(n_samples, seq_len - 1, dtype=torch.int32)
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            sl = slice(i, min(i + args.batch_size, n_samples))
            h = hidden[sl].to(device, non_blocking=True)
            ids_b = torch.stack(samples[sl]).to(device)
            nll, n, preds = score_batch(h, ids_b, hc_head, norm, lm_head, args.logit_chunk)
            total_nll += nll
            total_n += n
            all_preds[sl] = preds
            del h, ids_b
            torch.cuda.empty_cache()

    ppl = float(torch.exp(torch.tensor(total_nll / total_n)))
    print(f"\nperplexity: {ppl:.4f}  ({total_n:,} scored tokens, {n_samples} sequences)")

    out = {
        "checkpoint": str(ckpt),
        "perplexity": ppl,
        "mean_nll": total_nll / total_n,
        "scored_tokens": total_n,
        "num_samples": n_samples,
        "seq_len": seq_len,
        "tokens_file": args.tokens,
        "predictions": all_preds.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")
    source.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True, help=".pt file of token-id sequences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="eager",
                    help="must be eager; see the mask guard in run()")
    ap.add_argument("--route-chunk", type=int, default=65536)
    ap.add_argument("--expert-chunk", type=int, default=65536)
    ap.add_argument("--logit-chunk", type=int, default=256,
                    help="sequence positions per logits chunk (vocab is 129k wide)")
    ap.add_argument("--spill", help="directory to back the host caches with files")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
