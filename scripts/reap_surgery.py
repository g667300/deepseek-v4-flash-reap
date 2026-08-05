#!/usr/bin/env python3
"""The surgery pass: REAP expert pruning applied directly to a quantized checkpoint.

llm-compressor's ``prune_moe_layer`` leaves the surviving experts' weights
bit-for-bit unchanged -- it only (1) drops expert submodules, (2) slices the
router's ``weight`` / bias rows, and (3) updates the expert count in the
config. That makes the whole operation expressible as a file-level transform,
so we never dequantize: the output keeps the source's quantization format and
no GPU is involved.

Model-specific naming lives in :mod:`model_profiles`; the profile is detected
from ``config.json`` unless ``--profile`` says otherwise. For DeepSeek-V4 the
per-layer tensors are ``layers.{L}.ffn.experts.{E}.*`` plus a
``ffn.gate.{weight,bias}`` router.

Blocks that route by frozen token-id hash (DeepSeek-V4's first three layers,
whose ``ffn.gate.tid2eid`` names experts *by value*) are refused for pruning:
dropping an expert there would leave every token that maps to it without a
destination. They are copied through untouched.

Usage::

    reap_surgery.py --src models/DeepSeek-V4-Flash-0731 --dst artifacts/dsv4-reap50 \
        --saliency artifacts/dsv4-saliency.json --sparsity 0.5 --mtp drop
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_profiles  # noqa: E402


# --------------------------------------------------------------------------
# retained-expert selection
# --------------------------------------------------------------------------


def retained_from_saliency(saliency: dict[int, list[float]], sparsity: float) -> dict[int, list[int]]:
    """Keep the highest-saliency experts, dropping ``sparsity`` of them.

    Mirrors ``REAPSaliencyTracker.compute_retained_experts`` for the ungrouped
    case; both models here have n_group=1, so the group-limited branch
    degenerates to a plain global topk within the single group.
    """
    retained: dict[int, list[int]] = {}
    for layer, scores in saliency.items():
        n = len(scores)
        n_drop = int(n * sparsity)
        if n_drop == 0:
            raise ValueError(f"sparsity={sparsity} drops 0 of {n} experts")
        t = torch.tensor(scores, dtype=torch.float64)
        drop = set(torch.topk(t, n_drop, largest=False).indices.tolist())
        retained[layer] = [i for i in range(n) if i not in drop]
    return retained


def load_selection(args) -> tuple[dict[int, list[int]], int, int]:
    """Return (retained per layer index, original n_experts, new n_experts)."""
    if args.retained:
        blob = json.loads(Path(args.retained).read_text())
        retained = {int(k): list(v) for k, v in blob["layers"].items()}
        n_orig = int(blob["num_experts"])
    else:
        blob = json.loads(Path(args.saliency).read_text())
        saliency = {int(k): v for k, v in blob["layers"].items()}
        n_orig = int(blob["num_experts"])
        retained = retained_from_saliency(saliency, args.sparsity)

    counts = {len(v) for v in retained.values()}
    if len(counts) != 1:
        raise ValueError(f"all layers must retain the same expert count, got {sorted(counts)}")
    n_new = counts.pop()

    for layer, ids in retained.items():
        if sorted(ids) != ids:
            raise ValueError(f"layer {layer}: retained ids must be ascending")
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {layer}: duplicate retained ids")
        if ids and (ids[0] < 0 or ids[-1] >= n_orig):
            raise ValueError(f"layer {layer}: retained id out of range [0,{n_orig})")

    return retained, n_orig, n_new


# --------------------------------------------------------------------------
# per-shard rewrite
# --------------------------------------------------------------------------


def plan_shard(
    profile: model_profiles.ModelProfile,
    names: list[str],
    retained: dict[str, list[int]],
    drop_blocks: set[str],
) -> list[tuple[str, str, str | None]]:
    """Map source tensor names to (src_name, dst_name, router_block_or_None).

    Tensors that are pruned away are simply absent from the result.
    """
    out: list[tuple[str, str, str | None]] = []
    for name in names:
        m = profile.block_re.match(name)
        if m and m.group(1) in drop_blocks:
            continue

        m = profile.expert_re.match(name)
        if m:
            block, expert, rest = m.group(1), int(m.group(2)), m.group(3)
            keep = retained.get(block)
            if keep is None:  # block left unpruned (hash layer, --mtp keep)
                out.append((name, name, None))
                continue
            if expert not in keep:
                continue
            out.append((name, profile.expert_name(block, keep.index(expert), rest), None))
            continue

        m = profile.router_re.match(name)
        if m and m.group(1) in retained:
            out.append((name, name, m.group(1)))
            continue

        out.append((name, name, None))
    return out


def rewrite_shard(
    profile_name: str,
    src_path: str,
    dst_path: str,
    names: list[str],
    retained_json: str,
    drop_blocks_json: str,
    n_orig: int,
    value_maps_json: str = "{}",
) -> tuple[str, dict[str, int], int]:
    """Rewrite one shard. Returns (shard filename, {dst_name: nbytes}, total)."""
    profile = model_profiles.get(profile_name)
    retained = json.loads(retained_json)
    drop_blocks = set(json.loads(drop_blocks_json))
    value_maps = {k: torch.tensor(v, dtype=torch.long)
                  for k, v in json.loads(value_maps_json).items()}
    plan = plan_shard(profile, names, retained, drop_blocks)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src_path, framework="pt") as f:
        for src_name, dst_name, router_block in plan:
            t = f.get_tensor(src_name)
            if profile.expert_id_valued_re is not None and value_maps \
                    and profile.expert_id_valued_re.search(src_name):
                m = profile.block_re.match(src_name)
                vm = value_maps.get(m.group(1)) if m else None
                if vm is not None:
                    # values *are* expert ids: rewrite them, do not slice rows
                    t = vm[t.long()].to(t.dtype).contiguous()
            if router_block is not None:
                if t.shape[0] != n_orig:
                    raise ValueError(
                        f"{src_name}: expected router dim0={n_orig}, got {t.shape[0]}"
                    )
                idx = torch.tensor(retained[router_block], dtype=torch.long)
                t = t.index_select(0, idx).contiguous()
            tensors[dst_name] = t

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dst_path, metadata={"format": "pt"})

    sizes = {n: t.numel() * t.element_size() for n, t in tensors.items()}
    n_params = sum(t.numel() for t in tensors.values())
    return Path(dst_path).name, sizes, n_params


def build_value_map(
    gate_weight: torch.Tensor, keep: list[int], mode: str = "balanced"
) -> list[int]:
    """Map every original expert id to a surviving expert's new index.

    Survivors map to their own new position. A dropped expert's tokens have to
    go somewhere -- ``tid2eid`` must name exactly ``top_k`` experts for every
    token id -- so they are merged into a survivor. That is a merge, not a
    prune, and it is only defensible because it touches the few hash-routed
    layers, which have no score-based routing to renormalize.

    ``nearest`` sends each dropped expert to its most similar survivor,
    independently. That maximises similarity but concentrates load: measured on
    this checkpoint it left one expert of layer 2 holding 64,910 token ids
    against a 6,060 average, because popular survivors won many times over.

    ``balanced`` (the default) caps how many dropped experts any one survivor
    may absorb at ``ceil(n_dropped / n_kept)`` -- exactly one each at 50%
    sparsity -- and fills the assignment greedily from the most similar pair
    down. Hash routing is content-independent, so an even spread of token ids
    is what keeps every expert's share of the vocabulary comparable to what it
    had before pruning.
    """
    n_orig = gate_weight.shape[0]
    new_of = {old: i for i, old in enumerate(keep)}
    dropped = [e for e in range(n_orig) if e not in new_of]
    if not dropped:
        return [new_of[e] for e in range(n_orig)]

    w = torch.nn.functional.normalize(gate_weight.float(), dim=-1)
    keep_idx = torch.tensor(keep, dtype=torch.long)
    sim = w @ w.index_select(0, keep_idx).T          # [n_orig, n_keep]

    if mode == "nearest":
        nearest = sim.argmax(dim=1)                  # position within `keep`
        return [new_of.get(e, int(nearest[e])) for e in range(n_orig)]
    if mode != "balanced":
        raise ValueError(f"unknown remap mode {mode!r}")

    capacity = -(-len(dropped) // len(keep))         # ceil
    load = [0] * len(keep)
    assigned: dict[int, int] = {}
    pairs = sorted(
        ((float(sim[e, j]), e, j) for e in dropped for j in range(len(keep))),
        reverse=True,
    )
    for _score, e, j in pairs:
        if e in assigned or load[j] >= capacity:
            continue
        assigned[e] = j
        load[j] += 1
        if len(assigned) == len(dropped):
            break
    for e in dropped:  # only reachable if capacity*len(keep) < len(dropped)
        if e not in assigned:
            j = min(range(len(keep)), key=lambda k: load[k])
            assigned[e] = j
            load[j] += 1
    # not dict.get(e, assigned[e]) -- that evaluates assigned[e] for survivors too
    return [new_of[e] if e in new_of else assigned[e] for e in range(n_orig)]


def read_router_weight(src: Path, weight_map: dict, block: str) -> torch.Tensor:
    for suffix in (".ffn.gate.weight", ".mlp.gate.weight"):
        name = block + suffix
        if name in weight_map:
            with safe_open(str(src / weight_map[name]), framework="pt") as f:
                return f.get_tensor(name)
    raise KeyError(f"no router weight found for block {block}")


# --------------------------------------------------------------------------
# output checkpoint scaffolding
# --------------------------------------------------------------------------


def patch_configs(
    profile: model_profiles.ModelProfile,
    src: Path,
    dst: Path,
    n_new: int,
    dropped: set[str],
    meta: dict,
) -> None:
    cfg = json.loads((src / "config.json").read_text())
    profile.patch_config(cfg, n_new, dropped)
    cfg["reap_pruning"] = meta
    (dst / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")

    hq_path = src / "hf_quant_config.json"
    if hq_path.exists():
        hq = json.loads(hq_path.read_text())
        if dropped:
            ex = hq.get("quantization", {}).get("exclude_modules")
            if ex is not None:
                hq["quantization"]["exclude_modules"] = model_profiles._drop_ignore_patterns(
                    ex, dropped
                )
        (dst / "hf_quant_config.json").write_text(json.dumps(hq, indent=4) + "\n")


def copy_aux_files(
    profile: model_profiles.ModelProfile, src: Path, dst: Path, skip: set[str]
) -> list[str]:
    copied = []
    for p in sorted(src.iterdir()):
        if p.name in skip or p.suffix == ".safetensors":
            continue
        if p.is_dir():
            if p.name in profile.aux_dirs_to_copy:
                shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
                copied.append(p.name + "/")
            continue
        shutil.copy2(p, dst / p.name)
        copied.append(p.name)
    return copied


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source quantized checkpoint directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--profile", help="override profile detection (see model_profiles.PROFILES)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--saliency", help="saliency JSON from the scoring pass")
    g.add_argument("--retained", help="explicit retained-expert JSON")
    ap.add_argument("--sparsity", type=float, help="fraction of experts to drop (with --saliency)")
    ap.add_argument(
        "--mtp",
        choices=["drop", "prune-with-last", "keep"],
        default="drop",
        help="multi-token-prediction blocks: drop them, prune them using the last "
        "scored layer's retained set, or leave them untouched",
    )
    ap.add_argument(
        "--hash-layers",
        choices=["refuse", "prune-remap"],
        default="refuse",
        help="what to do with hash-routed blocks. 'refuse' leaves them at the "
        "original expert count -- correct in isolation, but both transformers "
        "and vLLM read one n_routed_experts for every layer, so the result will "
        "not load once other layers are pruned. 'prune-remap' prunes them too "
        "and rewrites tid2eid, sending each dropped expert's tokens to the "
        "surviving expert with the most similar router row",
    )
    ap.add_argument(
        "--hash-remap",
        choices=["balanced", "nearest"],
        default="balanced",
        help="with --hash-layers prune-remap, how dropped experts are merged "
        "into survivors: 'balanced' caps each survivor's intake so token ids "
        "stay evenly spread, 'nearest' maximises router-row similarity but "
        "lets popular survivors pile up (21x the average was measured)",
    )
    ap.add_argument("--workers", type=int, default=2, help="shards rewritten in parallel")
    ap.add_argument("--dry-run", action="store_true", help="report sizes without writing")
    args = ap.parse_args()

    if args.saliency and args.sparsity is None:
        ap.error("--sparsity is required with --saliency")

    src, dst = Path(args.src), Path(args.dst)
    config = json.loads((src / "config.json").read_text())
    profile = (
        model_profiles.get(args.profile) if args.profile else model_profiles.detect(src, config)
    )
    index = json.loads((src / profile.index_name).read_text())
    weight_map: dict[str, str] = index["weight_map"]

    retained_by_layer, n_orig, n_new = load_selection(args)
    retained = {profile.block_of_layer(L): ids for L, ids in retained_by_layer.items()}

    # ---- blocks that must not be pruned ----
    unprunable = set(profile.unprunable_blocks(config))
    remap_blocks = args.hash_layers == "prune-remap"
    refused = [] if remap_blocks else sorted(unprunable & set(retained))
    for block in refused:
        del retained[block]
    missing_scores = sorted(unprunable - set(retained)) if remap_blocks else []
    if missing_scores:
        raise ValueError(
            f"--hash-layers prune-remap needs saliency for {missing_scores}; "
            "rerun the scoring pass with --score-hash-layers and merge the results"
        )
    if not retained:
        print("nothing left to prune after excluding hash-routed blocks", file=sys.stderr)
        return 1

    # A tid2eid-style table names experts by value; slicing rows elsewhere in
    # that block would leave it pointing at experts that no longer exist.
    value_maps: dict[str, list[int]] = {}
    if profile.expert_id_valued_re is not None:
        for name in weight_map:
            if not profile.expert_id_valued_re.search(name):
                continue
            m = profile.block_re.match(name)
            if not (m and m.group(1) in retained):
                continue
            block = m.group(1)
            if not remap_blocks:
                raise ValueError(
                    f"{block} owns {name}, which stores expert ids as values; "
                    "pruning it would orphan the tokens that map to dropped experts"
                )
            value_maps[block] = build_value_map(
                read_router_weight(src, weight_map, block),
                retained[block],
                mode=args.hash_remap,
            )

    # ---- decide what happens to the MTP blocks ----
    all_blocks = profile.blocks_in(weight_map)
    mtp_blocks = profile.mtp_blocks(all_blocks, set(retained) | unprunable, weight_map)
    drop_blocks: set[str] = set()
    if args.mtp == "drop":
        drop_blocks = set(mtp_blocks)
    elif args.mtp == "prune-with-last":
        last = max(retained, key=lambda b: profile.layer_of_block(b) or -1)
        for block in mtp_blocks:
            retained[block] = list(retained[last])

    print(f"source        : {src}")
    print(f"profile       : {profile.name}  (quant: {profile.quant_scheme})")
    print(f"experts/layer : {n_orig} -> {n_new}  ({(1 - n_new / n_orig):.1%} pruned)")
    print(f"pruned blocks : {len(retained)}")
    if refused:
        print(f"left intact   : {len(refused)} hash-routed ({', '.join(refused)})")
    if remap_blocks:
        print(f"hash-routed   : pruned, tid2eid remapped ({len(value_maps)} blocks)")
    print(f"MTP {mtp_blocks or '-'} : {args.mtp}")

    # ---- group tensors by shard ----
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    if args.dry_run:
        total = estimate_output_size(profile, src, by_shard, retained, drop_blocks)
        print(f"\nestimated output size: {total / 1e9:.1f} GB")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    retained_json = json.dumps(retained)
    drop_json = json.dumps(sorted(drop_blocks))
    value_maps_json = json.dumps(value_maps)

    new_map: dict[str, str] = {}
    total_size = 0
    total_params = 0
    done = 0

    # spawn, not fork: computing the hash-layer value maps runs a torch matmul
    # in this process, which spins up torch's OpenMP pool. Forking after that
    # leaves the children deadlocked on a futex the moment they touch torch --
    # observed as three workers stuck at 1/48 shards with zero disk reads.
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                rewrite_shard,
                profile.name,
                str(src / shard),
                str(dst / shard),
                names,
                retained_json,
                drop_json,
                n_orig,
                value_maps_json,
            ): shard
            for shard, names in sorted(by_shard.items())
        }
        for fut in as_completed(futures):
            name, sizes, n_params = fut.result()
            for dst_name in sizes:
                new_map[dst_name] = name
            total_size += sum(sizes.values())
            total_params += n_params
            done += 1
            print(f"  [{done}/{len(futures)}] {name}  {sum(sizes.values()) / 1e9:.2f} GB", flush=True)

    (dst / profile.index_name).write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": total_params, "total_size": total_size},
                "weight_map": dict(sorted(new_map.items())),
            },
            indent=2,
        )
        + "\n"
    )

    meta = {
        "source": str(src),
        "profile": profile.name,
        "num_experts_original": n_orig,
        "num_experts_retained": n_new,
        "sparsity": round(1 - n_new / n_orig, 6),
        "mtp": args.mtp,
        "unpruned_blocks": refused,
        "hash_layers": args.hash_layers,
        "hash_remap": args.hash_remap,
        "remapped_blocks": sorted(value_maps),
        "retained_experts": {k: v for k, v in sorted(retained.items())},
    }
    patch_configs(
        profile, src, dst, n_new, drop_blocks,
        {k: v for k, v in meta.items() if k != "retained_experts"},
    )
    (dst / "reap_pruning.json").write_text(json.dumps(meta, indent=2) + "\n")
    copy_aux_files(
        profile, src, dst, skip={profile.index_name, "config.json", "hf_quant_config.json"}
    )

    print(f"\nwrote {len(new_map)} tensors, {total_size / 1e9:.1f} GB -> {dst}")
    return 0


def estimate_output_size(profile, src: Path, by_shard, retained, drop_blocks) -> int:
    """Sum output bytes without writing, by reading only safetensors headers."""
    total = 0
    for shard, names in by_shard.items():
        path = src / shard
        if not path.exists():
            continue
        kept = {s for s, _, _ in plan_shard(profile, names, retained, drop_blocks)}
        with safe_open(str(path), framework="pt") as f:
            for name in kept:
                sl = f.get_slice(name)
                nbytes = _dtype_size(sl.get_dtype())
                for d in sl.get_shape():
                    nbytes *= d
                total += nbytes
    return total


_DTYPE_BYTES = {
    "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "F64": 8, "I64": 8,
}


def _dtype_size(name: str) -> int:
    if name not in _DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {name}")
    return _DTYPE_BYTES[name]


if __name__ == "__main__":
    sys.exit(main())
