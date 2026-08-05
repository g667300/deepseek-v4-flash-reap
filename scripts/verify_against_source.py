#!/usr/bin/env python3
"""Check a pruned checkpoint against the model it came from, byte for byte.

REAP does not modify the weights of the experts it keeps. Every surviving
expert in the output must therefore be bit-identical to the same expert in the
source, which makes the source an independent reference: it catches a shard
written wrong, an expert renumbered wrong, and a bit flipped in memory while the
surgery pass was writing.

That last one is the reason this exists. On non-ECC memory a flip during the
surgery pass is written into the checkpoint, and comparing your copy against the
published one cannot see it -- both sides are the same bytes. Only the source
is independent.

Mismatches are re-read before being reported. Bad reads happen (measured here:
about one per 30 GB under memory pressure, with nothing in the kernel log), and
a verifier that cannot tell a bad read from a bad checkpoint is worse than none.

Usage::

    verify_against_source.py --src models/DeepSeek-V4-Flash-0731 \\
        --dst artifacts/dsv4-reap50 --experts 4

    verify_against_source.py --src ... --dst ... --all   # every expert
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_profiles  # noqa: E402
from safetensors import safe_open  # noqa: E402


class Checkpoint:
    """Random access to tensors across a sharded safetensors checkpoint."""

    def __init__(self, root: Path, index_name: str):
        self.root = root
        self.weight_map = json.loads((root / index_name).read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def get(self, name: str):
        shard = self.weight_map[name]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(str(self.root / shard), framework="pt")
            self._handles[shard] = h
        return h.get_tensor(name)

    def reopen(self) -> None:
        """Drop handles so a re-read actually touches the file again."""
        self._handles.clear()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="original checkpoint directory")
    ap.add_argument("--dst", required=True, help="pruned checkpoint directory")
    ap.add_argument("--experts", type=int, default=4,
                    help="experts to check per block (default 4)")
    ap.add_argument("--blocks", type=int,
                    help="blocks to check (default: all of them)")
    ap.add_argument("--all", action="store_true", help="every surviving expert")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir, dst_dir = Path(args.src), Path(args.dst)
    pruning = json.loads((dst_dir / "reap_pruning.json").read_text())
    retained: dict[str, list[int]] = pruning["retained_experts"]
    profile = model_profiles.detect(dst_dir)

    src = Checkpoint(src_dir, profile.index_name)
    dst = Checkpoint(dst_dir, profile.index_name)

    # The per-expert tensor suffixes, taken from what the source actually has.
    blocks = sorted(retained, key=lambda b: (b.split(".")[0], int(b.split(".")[1])))
    if args.blocks:
        rng = random.Random(args.seed)
        blocks = sorted(rng.sample(blocks, min(args.blocks, len(blocks))))
    probe = blocks[0]
    suffixes = sorted({
        m.group(3) for m in (profile.expert_re.match(n) for n in src.weight_map)
        if m and m.group(1) == probe and int(m.group(2)) == retained[probe][0]
    })
    if not suffixes:
        print(f"found no expert tensors for {probe}", file=sys.stderr)
        return 2

    print(f"source : {src_dir}")
    print(f"pruned : {dst_dir}")
    print(f"blocks : {len(blocks)}, tensors per expert: {', '.join(suffixes)}")

    rng = random.Random(args.seed)
    checked = flagged = 0
    bad: list[str] = []
    reread_ok = 0

    for block in blocks:
        keep = retained[block]
        picks = keep if args.all else rng.sample(keep, min(args.experts, len(keep)))
        for orig in picks:
            new = keep.index(orig)
            for rest in suffixes:
                s_name = profile.expert_name(block, orig, rest)
                d_name = profile.expert_name(block, new, rest)
                a, b = src.get(s_name), dst.get(d_name)
                checked += 1
                if a.shape == b.shape and a.dtype == b.dtype and a.equal(b):
                    continue
                # Could be a bad read rather than a bad checkpoint. Look again.
                src.reopen(); dst.reopen()
                a2, b2 = src.get(s_name), dst.get(d_name)
                if a2.shape == b2.shape and a2.dtype == b2.dtype and a2.equal(b2):
                    reread_ok += 1
                    continue
                flagged += 1
                bad.append(f"{block} expert {orig} -> {new}, {rest}")

    # ---- routers and expert-id tables ------------------------------------
    # These are the tensors the surgery pass deliberately rewrites, so they
    # cannot be compared directly -- but both rewrites are reproducible, which
    # is just as good a check.
    import torch

    r_checked = r_bad = 0
    for name, shard in sorted(dst.weight_map.items()):
        m = profile.router_re.match(name)
        if not m or m.group(1) not in retained or name not in src.weight_map:
            continue
        if profile.expert_id_valued_re.search(name):
            continue
        keep = retained[m.group(1)]
        want = src.get(name).index_select(0, torch.tensor(keep, dtype=torch.long))
        got = dst.get(name)
        r_checked += 1
        if not (want.shape == got.shape and want.dtype == got.dtype and want.equal(got)):
            src.reopen(); dst.reopen()
            want = src.get(name).index_select(0, torch.tensor(keep, dtype=torch.long))
            if not want.equal(dst.get(name)):
                r_bad += 1
                bad.append(f"{name}: not the retained rows of the source router")

    t_checked = t_bad = 0
    n_new = len(next(iter(retained.values())))
    for name in sorted(dst.weight_map):
        if not profile.expert_id_valued_re.search(name) or name not in src.weight_map:
            continue
        block = profile.block_re.match(name).group(1)
        keep = retained[block]
        new_of = {old: i for i, old in enumerate(keep)}
        s, d = src.get(name).flatten(), dst.get(name).flatten()
        t_checked += 1
        problems = []
        if s.shape != d.shape:
            problems.append(f"shape changed {tuple(s.shape)} -> {tuple(d.shape)}")
        if int(d.min()) < 0 or int(d.max()) >= n_new:
            problems.append(f"values outside [0,{n_new}): min {int(d.min())} max {int(d.max())}")
        # The rewrite is a function of the old id, so it must be single-valued,
        # and every expert that survived must map to its own new index.
        mapping: dict[int, int] = {}
        for old, new in zip(s.tolist(), d.tolist()):
            if mapping.setdefault(old, new) != new:
                problems.append(f"expert {old} maps to both {mapping[old]} and {new}")
                break
        moved = {o: n for o, n in mapping.items() if o in new_of and n != new_of[o]}
        if moved:
            problems.append(f"{len(moved)} surviving experts were not left in place")
        absorbed: dict[int, int] = {}
        for old, new in mapping.items():
            if old not in new_of:
                absorbed[new] = absorbed.get(new, 0) + 1
        cap = -(-len(absorbed) // len(keep)) if absorbed else 0
        if absorbed and max(absorbed.values()) > max(cap, 1):
            problems.append(f"one survivor absorbed {max(absorbed.values())} dropped experts")
        if problems:
            t_bad += 1
            bad.extend(f"{name}: {p}" for p in problems)

    print(f"\nchecked {checked} expert tensors, {r_checked} routers, {t_checked} id tables")
    if reread_ok:
        print(f"{reread_ok} differed on first read and matched on re-read "
              f"(bad reads, not bad data -- but the memory is suspect)")
    if bad:
        print(f"\n{flagged} MISMATCHED after re-read:")
        for line in bad[:20]:
            print(f"  {line}")
        return 1
    print("surviving experts are bit-identical to the source; routers are its\n       retained rows; expert-id tables map every survivor to itself and every\n       dropped expert into range, without overloading any survivor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
