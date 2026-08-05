#!/usr/bin/env python3
"""Compare two saliency runs by what they would actually prune.

Raw score differences are hard to read -- what matters is whether the two runs
would keep the same experts. Reports, per layer shared by both files:

* **Jaccard** over the retained sets at a given sparsity: 1.0 means the pruned
  checkpoints would be identical, so redoing the surgery would change nothing.
* **Spearman** over the full score vectors: catches ranking drift that a
  particular sparsity threshold happens to hide.

Usage::

    compare_saliency.py --sparsity 0.5 old.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def retained(scores: list[float], sparsity: float) -> set[int]:
    n = len(scores)
    n_drop = int(n * sparsity)
    t = torch.tensor(scores, dtype=torch.float64)
    drop = set(torch.topk(t, n_drop, largest=False).indices.tolist())
    return {i for i in range(n) if i not in drop}


def spearman(a: list[float], b: list[float]) -> float:
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    ra = ta.argsort().argsort().to(torch.float64)
    rb = tb.argsort().argsort().to(torch.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--sparsity", type=float, default=0.5)
    args = ap.parse_args()

    old = json.loads(Path(args.old).read_text())
    new = json.loads(Path(args.new).read_text())
    shared = sorted(set(old["layers"]) & set(new["layers"]), key=int)
    if not shared:
        print("the two files share no layers", file=sys.stderr)
        return 2

    n_experts = old["num_experts"]
    n_keep = n_experts - int(n_experts * args.sparsity)
    print(f"comparing {len(shared)} shared layers at sparsity {args.sparsity} "
          f"({n_keep}/{n_experts} kept)")
    print(f"  old: {args.old}")
    print(f"  new: {args.new}\n")
    print(f"{'layer':>6} {'Jaccard':>9} {'kept both':>10} {'differs':>8} {'Spearman':>10}")
    print("-" * 47)

    jac_all, sp_all = [], []
    for L in shared:
        so, sn = old["layers"][L], new["layers"][L]
        ro, rn = retained(so, args.sparsity), retained(sn, args.sparsity)
        inter, union = len(ro & rn), len(ro | rn)
        j = inter / union if union else 1.0
        sp = spearman(so, sn)
        jac_all.append(j)
        sp_all.append(sp)
        print(f"{L:>6} {j:>9.4f} {inter:>10} {len(ro - rn):>8} {sp:>10.4f}")

    print("-" * 47)
    print(f"{'mean':>6} {sum(jac_all)/len(jac_all):>9.4f} {'':>10} {'':>8} "
          f"{sum(sp_all)/len(sp_all):>10.4f}")
    print(f"\nJaccard  min {min(jac_all):.4f}  max {max(jac_all):.4f}")
    print(f"Spearman min {min(sp_all):.4f}  max {max(sp_all):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
