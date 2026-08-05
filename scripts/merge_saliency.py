#!/usr/bin/env python3
"""Merge saliency JSONs from separate scoring runs into one file.

The scoring pass can be run over a subset of layers (``--max-layers``, or with
``--score-hash-layers`` to cover blocks a normal run skips). The surgery wants a
single file, so combine them here rather than teaching it to read several.

Later files win on conflict, and the merged ``meta`` records where each layer
came from so a mixed-provenance file is not mistaken for one run.

Usage::

    merge_saliency.py --out artifacts/dsv4-saliency-full.json \\
        artifacts/dsv4-saliency.json artifacts/dsv4-saliency-hash.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    layers: dict[str, list[float]] = {}
    provenance: dict[str, str] = {}
    num_experts: int | None = None
    metas: list[dict] = []

    for path in args.inputs:
        blob = json.loads(path.read_text())
        n = int(blob["num_experts"])
        if num_experts is None:
            num_experts = n
        elif n != num_experts:
            raise ValueError(f"{path}: num_experts {n} != {num_experts}")
        for layer, scores in blob["layers"].items():
            if len(scores) != num_experts:
                raise ValueError(
                    f"{path}: layer {layer} has {len(scores)} scores, expected {num_experts}"
                )
            layers[layer] = scores
            provenance[layer] = path.name
        metas.append({"file": path.name, **blob.get("meta", {})})

    merged = {
        "num_experts": num_experts,
        "layers": dict(sorted(layers.items(), key=lambda kv: int(kv[0]))),
        "meta": {"merged_from": metas, "layer_source": provenance},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2) + "\n")

    ids = sorted(int(k) for k in layers)
    gaps = [i for i in range(ids[0], ids[-1] + 1) if i not in set(ids)]
    print(f"wrote {args.out}: {len(layers)} layers ({ids[0]}..{ids[-1]}), "
          f"{num_experts} experts each")
    if gaps:
        print(f"note: no scores for layers {gaps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
