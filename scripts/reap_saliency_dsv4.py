#!/usr/bin/env python3
"""The scoring pass for DeepSeek-V4: layer-sequential REAP saliency, read straight from FP4.

Walks the model one decoder layer at a time, keeping the whole calibration
set's activations on the host and only ever putting one layer on the GPU.
Loading the checkpoint through ``transformers`` would dequantize everything to
BF16 at once; here each layer is built empty, filled from the shards, and
thrown away.

Two things make this more involved than a plain residual-stream transformer:

**Hyper-connections.** The residual is not one stream but ``hc_mult`` (4) of
them, carried as ``[B, S, 4, D]``, and each block mixes them in and out through
a learned, Sinkhorn-projected transform::

    post, comb, collapsed = attn_hc(hidden)
    hidden = post ⊗ attn(norm(collapsed)) + combᵀ @ hidden
    post, comb, collapsed = ffn_hc(hidden)
    hidden = post ⊗ mlp(norm(collapsed))  + combᵀ @ hidden

The three-phase split (attention for every batch, then the router, then one
pass over the experts) still works, because that last line is *linear* in the
MoE output: phase 1 computes ``combᵀ @ hidden`` plus the shared expert's
contribution and caches ``post``, and phase 3 adds ``post ⊗ expert_out`` per
token. The host cache is 4x a conventional model's -- 17.2 GB of hidden state
for 256x2048 -- so watch ``--max-samples``.

**Hash-routed layers.** The first ``num_hash_layers`` (3) layers pick experts by
a frozen ``tid2eid[input_ids]`` table rather than by score. Their experts still
have to run, or the hidden state handed to layer 3 would be wrong, but they are
not scored: pruning them would strand every token whose entry points at a
dropped expert. The surgery pass refuses to prune them for the same reason.

Usage::

    reap_saliency_dsv4.py --ckpt models/DeepSeek-V4-Flash-0731 \\
        --tokens artifacts/calib-dsv4-512x2048.pt --max-samples 256 \\
        --out artifacts/dsv4-saliency.json --state artifacts/dsv4-state.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_profiles  # noqa: E402
from quant import dequantize_linear  # noqa: E402
from saliency_common import SaliencyTracker, alloc, human, _rss_gb  # noqa: E402

PROFILE = model_profiles.DEEPSEEK_V4


# --------------------------------------------------------------------------
# native -> HF parameter names
# --------------------------------------------------------------------------

# Applied in order to the part of a tensor name after the block prefix. The
# full table was derived by dumping DeepseekV4DecoderLayer's parameters and
# matching them against the checkpoint; test_stage_a_dsv4.py asserts the two
# sides still line up exactly.
_RENAMES: tuple[tuple[str, str], ...] = (
    # indexer first: its own compressor is nested under the outer one in HF
    ("attn.indexer.compressor.ape", "self_attn.compressor.indexer.position_bias"),
    ("attn.indexer.compressor.wkv", "self_attn.compressor.indexer.kv_proj"),
    ("attn.indexer.compressor.wgate", "self_attn.compressor.indexer.gate_proj"),
    ("attn.indexer.compressor.norm", "self_attn.compressor.indexer.kv_norm"),
    ("attn.indexer.wq_b", "self_attn.compressor.indexer.q_b_proj"),
    ("attn.indexer.weights_proj", "self_attn.compressor.indexer.scorer.weights_proj"),
    ("attn.compressor.ape", "self_attn.compressor.position_bias"),
    ("attn.compressor.wkv", "self_attn.compressor.kv_proj"),
    ("attn.compressor.wgate", "self_attn.compressor.gate_proj"),
    ("attn.compressor.norm", "self_attn.compressor.kv_norm"),
    ("attn.attn_sink", "self_attn.sinks"),
    ("attn.wq_a", "self_attn.q_a_proj"),
    ("attn.wq_b", "self_attn.q_b_proj"),
    ("attn.q_norm", "self_attn.q_a_norm"),
    ("attn.wkv", "self_attn.kv_proj"),
    ("attn.kv_norm", "self_attn.kv_norm"),
    ("attn.wo_a", "self_attn.o_a_proj"),
    ("attn.wo_b", "self_attn.o_b_proj"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
    ("ffn.gate.bias", "mlp.gate.e_score_correction_bias"),
    ("ffn.gate.tid2eid", "mlp.gate.tid2eid"),
    ("ffn.gate.weight", "mlp.gate.weight"),
    ("ffn.shared_experts.w1", "mlp.shared_experts.gate_proj"),
    ("ffn.shared_experts.w3", "mlp.shared_experts.up_proj"),
    ("ffn.shared_experts.w2", "mlp.shared_experts.down_proj"),
)


def to_hf_name(suffix: str) -> str:
    """Map one in-layer tensor name from DeepSeek's naming to transformers'."""
    for native, hf in _RENAMES:
        if suffix == native or suffix.startswith(native + "."):
            return hf + suffix[len(native):]
    raise KeyError(f"no HF name known for {suffix!r}")


# --------------------------------------------------------------------------
# checkpoint access
# --------------------------------------------------------------------------


class DeepseekSource:
    """Serves dequantized, HF-named weights straight from the FP4/FP8 shards."""

    def __init__(self, ckpt: Path, device: torch.device, dtype: torch.dtype):
        self.ckpt = ckpt
        self.device = device
        self.dtype = dtype
        self.weight_map: dict[str, str] = json.loads(
            (ckpt / PROFILE.index_name).read_text()
        )["weight_map"]
        self._handles: dict[str, object] = {}

    def _get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        h = self._handles.get(shard)
        if h is None:
            path = self.ckpt / shard
            if not path.exists():
                raise FileNotFoundError(f"shard {shard} is missing (needed for {name})")
            h = safe_open(str(path), framework="pt")
            self._handles[shard] = h
        return h.get_tensor(name)

    def _dequant(self, prefix: str) -> torch.Tensor:
        """Dequantize the linear at ``prefix``, on the GPU to keep it quick."""
        raw = {f"{prefix}.weight": self._get(f"{prefix}.weight").to(self.device, non_blocking=True)}
        if f"{prefix}.scale" in self.weight_map:
            raw[f"{prefix}.scale"] = self._get(f"{prefix}.scale").to(self.device, non_blocking=True)
        return dequantize_linear(raw, prefix, scheme=PROFILE.quant_scheme, dtype=self.dtype)

    def load_expert(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (w1/gate, w3/up, w2/down) for one routed expert."""
        base = f"layers.{layer}.ffn.experts.{expert}"
        return tuple(self._dequant(f"{base}.{proj}") for proj in ("w1", "w3", "w2"))

    def layer_state_dict(self, layer: int) -> dict[str, torch.Tensor]:
        """Every tensor of the layer except the routed experts, HF-named."""
        prefix = f"layers.{layer}."
        expert_prefix = f"{prefix}ffn.experts."
        names = [
            n for n in self.weight_map
            if n.startswith(prefix) and not n.startswith(expert_prefix)
        ]
        out: dict[str, torch.Tensor] = {}
        for name in names:
            suffix = name[len(prefix):]
            if suffix.endswith(".scale"):
                continue  # consumed together with its .weight
            if suffix.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.weight_map:
                out[to_hf_name(suffix)] = self._dequant(name[: -len(".weight")]).cpu()
            else:
                out[to_hf_name(suffix)] = self._get(name)
        return out

    def get_global(self, name: str) -> torch.Tensor:
        return self._get(name)

    def close(self) -> None:
        self._handles.clear()


# --------------------------------------------------------------------------
# config / layer construction
# --------------------------------------------------------------------------


def load_config(ckpt: Path, attn_impl: str):
    """Build an HF config from DeepSeek's native config.json.

    ``DeepseekV4Config`` accepts the native keys directly: it folds the legacy
    ``compress_ratios`` / ``num_hash_layers`` / ``qk_rope_head_dim`` spellings
    into ``layer_types`` / ``mlp_layer_types`` / ``partial_rotary_factor``.
    Building a config by hand instead is exactly how this went wrong
    once, so don't.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    native = json.loads((ckpt / "config.json").read_text())
    config = DeepseekV4Config(**native)
    config._attn_implementation = attn_impl
    return config


def build_layer(config, layer_idx: int, source: DeepseekSource,
                device: torch.device, dtype: torch.dtype):
    """Materialize one decoder layer, without its routed experts.

    The fused ``DeepseekV4Experts`` parameters are 12.9 GB per layer in BF16, so
    the module is built on the meta device and the experts dropped -- this
    driver runs them itself, one at a time.
    """
    from accelerate import init_empty_weights
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer

    with init_empty_weights():
        layer = DeepseekV4DecoderLayer(config, layer_idx)

    limit = layer.mlp.experts.limit
    act_fn = layer.mlp.experts.act_fn
    del layer.mlp.experts

    sd = source.layer_state_dict(layer_idx)
    _missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"layer {layer_idx}: unexpected checkpoint tensors {unexpected[:5]}")

    layer = layer.to(device=device, dtype=dtype)
    # The router scores in float32 and the selection bias is float32 in the
    # checkpoint; demoting either would change which experts win the top-k.
    layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    if hasattr(layer.mlp.gate, "e_score_correction_bias"):
        layer.mlp.gate.e_score_correction_bias = layer.mlp.gate.e_score_correction_bias.float()
    if hasattr(layer.mlp.gate, "tid2eid"):
        layer.mlp.gate.tid2eid = layer.mlp.gate.tid2eid.long()

    stranded = [n for n, p in list(layer.named_parameters()) + list(layer.named_buffers())
                if p.device.type == "meta"]
    if stranded:
        raise RuntimeError(f"layer {layer_idx}: weights never loaded: {stranded[:8]}")
    return layer, act_fn, limit


# --------------------------------------------------------------------------
# the three phases
# --------------------------------------------------------------------------


@torch.no_grad()
def swiglu(xs: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
           w_down: torch.Tensor, act_fn, limit: float) -> torch.Tensor:
    """One expert's FFN, matching ``DeepseekV4Experts._apply_gate``.

    The clamp is not decoration: ``swiglu_limit`` is 10.0 for this model and
    both branches are clipped before the activation.
    """
    gate = nn.functional.linear(xs, w_gate)
    up = nn.functional.linear(xs, w_up)
    if limit and limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return nn.functional.linear(act_fn(gate) * up, w_down)


@torch.no_grad()
def route(gate, x_flat: torch.Tensor, token_ids: torch.Tensor | None,
          device: torch.device, chunk: int):
    """Run the router over the whole token cache, in chunks.

    Hash routers need the token ids that produced each row; score-based ones
    ignore them.
    """
    n = x_flat.shape[0]
    top_k = gate.top_k
    idx = torch.empty(n, top_k, dtype=torch.int32)
    wts = torch.empty(n, top_k, dtype=torch.float32)
    is_hash = token_ids is not None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # DeepSeek's own Gate scores in float32 (`linear(x.float(), w.float())`)
        # and build_layer keeps the router weights there; feeding bf16 in would
        # both fail the matmul and change which experts win the top-k.
        xs = x_flat[s:e].to(device, non_blocking=True).to(gate.weight.dtype)
        if is_hash:
            _, w, i = gate(xs, token_ids[s:e].to(device, non_blocking=True))
        else:
            _, w, i = gate(xs)
        idx[s:e] = i.to(torch.int32).cpu()
        wts[s:e] = w.float().cpu()
    return idx, wts


@torch.no_grad()
def run_experts(
    source: DeepseekSource, layer_idx: int, act_fn, limit: float, num_experts: int,
    x_flat: torch.Tensor, topk_i: torch.Tensor, topk_w: torch.Tensor,
    hidden_flat: torch.Tensor | None, post_flat: torch.Tensor | None,
    tracker: SaliencyTracker | None, device: torch.device, chunk: int,
) -> None:
    """One pass over the layer's experts, each dequantized exactly once.

    :param hidden_flat: ``[n_tok, hc_mult, hidden]`` -- expert output is folded
        in as ``post ⊗ out``, the hyper-connection's expand step
    :param tracker: if given, saliency is accumulated
    """
    for expert in range(num_experts):
        rows, slots = (topk_i == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w_gate, w_up, w_down = source.load_expert(layer_idx, expert)

        for s in range(0, rows.numel(), chunk):
            r = rows[s : s + chunk]
            xs = x_flat.index_select(0, r).to(device, non_blocking=True)
            out = swiglu(xs, w_gate, w_up, w_down, act_fn, limit)
            gates = topk_w[r, slots[s : s + chunk]].to(device, non_blocking=True)
            if tracker is not None:
                # REAP measures the output before the router coefficient
                tracker.add(expert, gates, torch.linalg.norm(out.float(), dim=-1))
            if hidden_flat is not None:
                weighted = out * gates[:, None]
                post = post_flat.index_select(0, r).to(device, non_blocking=True)
                hidden_flat.index_add_(
                    0, r, (post[:, :, None] * weighted[:, None, :]).to(hidden_flat.dtype).cpu()
                )
                del weighted, post
            del xs, out, gates

        del w_gate, w_up, w_down


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def load_tokens(args, vocab_size: int) -> list[torch.Tensor]:
    if args.random_tokens:
        n, seq = (int(x) for x in args.random_tokens.split(","))
        g = torch.Generator().manual_seed(args.seed)
        return [torch.randint(0, vocab_size, (seq,), generator=g) for _ in range(n)]
    blob = torch.load(args.tokens, weights_only=True)
    samples = [torch.as_tensor(t, dtype=torch.long) for t in blob]
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    hi = max(int(s.max()) for s in samples)
    if hi >= vocab_size:
        raise ValueError(f"token id {hi} exceeds vocab_size {vocab_size}")
    return samples


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
    hash_layers = set(PROFILE.hash_layers(json.loads((ckpt / "config.json").read_text())))
    scored_layers = [L for L in range(n_layers) if L not in hash_layers]
    print(f"layers: {n_layers} ({len(scored_layers)} scored, {sorted(hash_layers)} hash-routed), "
          f"experts/layer: {num_experts}, top_k: {config.num_experts_per_tok}, hc_mult: {hc}")

    source = DeepseekSource(ckpt, device, dtype)

    # ---- calibration data ----
    samples = load_tokens(args, config.vocab_size)
    seq_len = min(len(s) for s in samples)
    if any(len(s) != seq_len for s in samples):
        samples = [s[:seq_len] for s in samples]
        print(f"truncated all samples to the shortest length, {seq_len}")
    n_samples = len(samples)
    n_tok = n_samples * seq_len
    print(f"calibration: {n_samples} samples x {seq_len} tokens = {n_tok:,} tokens")

    hbytes = 4 if args.hidden_dtype == "fp32" else 2
    est = n_tok * hidden_size * hc * hbytes + n_tok * hidden_size * 2 + n_tok * hc * 4
    print(f"host cache: {human(est)} (hidden x{hc} streams + MoE input + hc post)"
          + (f", spilled to {spill}" if spill else ""))

    hdtype = torch.float32 if args.hidden_dtype == "fp32" else dtype
    hidden = alloc((n_samples, seq_len, hc, hidden_size), hdtype, spill, "hidden")
    x_cache = alloc((n_tok, hidden_size), dtype, spill, "moe_input")
    post_cache = alloc((n_tok, hc), torch.float32, spill, "hc_post")
    token_ids = torch.stack(samples).reshape(-1)

    if args.score_hash_layers:
        # Hash layers are normally left unscored because they cannot be pruned
        # on their own -- but the expert count is a single config value, so if
        # the rest of the model is pruned they have to come along. Scoring them
        # gives that unavoidable prune a principled survivor set instead of an
        # arbitrary one.
        scored_layers = list(range(n_layers))
        print("scoring hash-routed layers too (--score-hash-layers)")
    trackers = {L: SaliencyTracker(num_experts, device) for L in scored_layers}

    # ---- embeddings / resume ----
    state_path = Path(args.state) if args.state else None
    start_layer = 0
    if state_path and state_path.exists():
        st = torch.load(state_path, weights_only=True)
        start_layer = st["next_layer"]
        hidden.copy_(st["hidden"])
        for L, s in st["trackers"].items():
            trackers[L].load(s)
        print(f"resumed from {state_path}: continuing at layer {start_layer}")
    else:
        embed = nn.Embedding(config.vocab_size, hidden_size)
        embed.weight = nn.Parameter(source.get_global("embed.weight"), requires_grad=False)
        embed = embed.to(device=device, dtype=dtype)
        with torch.no_grad():
            for i in range(0, n_samples, args.batch_size):
                ids = torch.stack(samples[i : i + args.batch_size]).to(device)
                # every hyper-connection stream starts as a copy of the embedding
                e = embed(ids).unsqueeze(2).expand(-1, -1, hc, -1)
                hidden[i : i + args.batch_size] = e.to(hdtype).cpu()
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
                    f"{'None' if masks[bs] is None else 'bool'} mask; use --attn eager.\n"
                    "DeepseekV4Attention extends the mask over the compressor's extra KV "
                    "slots with `torch.cat([mask, block_bias.to(mask.dtype)])`. block_bias "
                    "is an additive float bias (0 = attend, -inf = block), so casting it to "
                    "bool inverts it: 0.0 becomes False (blocked) and -inf becomes True "
                    "(attended). Every query then sees compressed slots it must not, and "
                    "the future leaks in. Measured on this checkpoint: perplexity 1.35 "
                    "under sdpa against 8.88 under eager."
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

    last_layer = min(n_layers, args.max_layers) if args.max_layers else n_layers
    if last_layer < n_layers:
        print(f"stopping after layer {last_layer - 1} (--max-layers); "
              "the saliency written will be partial")

    for L in range(start_layer, last_layer):
        t0 = time.time()
        tracker = trackers.get(L)
        is_hash = L in hash_layers
        layer, act_fn, limit = build_layer(config, L, source, device, dtype)

        # ---- phase 1: hyper-connections + attention + shared expert ----
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
                # everything that does not depend on the routed experts; their
                # contribution is folded in during phase 3
                h = post.to(dtype).unsqueeze(-1) * layer.mlp.shared_experts(xb).unsqueeze(-2) \
                    + torch.matmul(comb.to(dtype).transpose(-1, -2), h)

                x_cache[sl.start * seq_len : sl.stop * seq_len] = xb.reshape(-1, hidden_size).cpu()
                post_cache[sl.start * seq_len : sl.stop * seq_len] = post.reshape(-1, hc).float().cpu()
                hidden[sl] = h.to(hdtype).cpu()
                del h, post, comb, collapsed, attn_out, xb
        torch.cuda.empty_cache()

        # ---- phases 2 and 3: router, then one pass over the experts ----
        ids = token_ids if is_hash else None
        topk_i, topk_w = route(layer.mlp.gate, x_cache, ids, device, args.route_chunk)
        run_experts(source, L, act_fn, limit, num_experts, x_cache, topk_i, topk_w,
                    hidden_flat, post_cache, tracker, device, args.expert_chunk)
        del topk_i, topk_w, layer
        gc.collect()
        torch.cuda.empty_cache()

        note = (f", {tracker.total / 1e6:.1f}M routings" if tracker
                else " (hash-routed, unscored)")
        if tracker and is_hash:
            note += " [hash]"
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  layer {L:>2}/{n_layers - 1}  {time.time() - t0:6.1f}s{note}"
              f"  | VRAM peak {peak:5.1f} GB, host RSS {_rss_gb():5.1f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)

        if state_path and ((L + 1) % args.state_every == 0 or L == last_layer - 1):
            tmp = state_path.with_suffix(".tmp")
            torch.save({"next_layer": L + 1, "hidden": hidden,
                        "trackers": {k: v.state() for k, v in trackers.items()}}, tmp)
            tmp.replace(state_path)

    print(f"total {time.time() - t_start:.0f}s")

    out = {
        "num_experts": num_experts,
        "layers": {str(L): trackers[L].mean.tolist()
                   for L in scored_layers if L < last_layer},
        "meta": {
            "source": str(ckpt),
            "profile": PROFILE.name,
            "num_samples": n_samples,
            "seq_len": seq_len,
            "num_tokens": n_tok,
            "attn_implementation": args.attn,
            "hash_layers_unscored": sorted(hash_layers),
            "layers_run": last_layer,
            "routed_tokens_per_layer": {str(L): trackers[L].total
                                        for L in scored_layers if L < last_layer},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")

    dead = {L: int((trackers[L].count == 0).sum())
            for L in scored_layers if L < last_layer}
    dead = {L: n for L, n in dead.items() if n}
    if dead:
        print(f"WARNING: experts never routed to (score 0, pruned first): {dead}")
    source.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="DeepSeek-V4 checkpoint directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokens", help=".pt file holding a list of token-id sequences")
    g.add_argument("--random-tokens", help="N,SEQ of random ids (smoke tests only)")
    ap.add_argument("--out", required=True, help="saliency JSON to write")
    ap.add_argument("--max-samples", type=int,
                    help="use only the first N calibration samples (host cache is "
                         "hc_mult x a conventional model's, so this is the main dial)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--attn",
        default="eager",
        help="attention implementation. Must be 'eager': transformers' mask "
        "helpers return None under sdpa/flash_attention_2, expecting the "
        "attention backend to apply causality itself via is_causal. That "
        "holds when the full model's forward runs, but this driver calls "
        "self_attn directly, so a None mask means every token attends to the "
        "future and saliency is measured on a bidirectional model",
    )
    ap.add_argument("--hidden-dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--route-chunk", type=int, default=65536)
    ap.add_argument("--expert-chunk", type=int, default=65536)
    ap.add_argument("--spill", help="directory to back the host caches with files")
    ap.add_argument("--state", help="resume/checkpoint file")
    ap.add_argument("--state-every", type=int, default=1)
    ap.add_argument("--score-hash-layers", action="store_true",
                    help="also accumulate saliency for the hash-routed layers")
    ap.add_argument("--max-layers", type=int,
                    help="stop after this many layers (smoke tests; saliency is partial)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
