#!/usr/bin/env python3
"""Held-out perplexity against a running vLLM (OpenAI-compatible) server.

Deliberately does *not* reimplement a local NVFP4 forward pass -- the surgery
output only ever gets evaluated through vLLM on the DGX Spark (see
PROGRESS.md), so this scores the exact same serving path instead of a
separate one that could disagree with it.

Scoring trick: POST to /v1/completions with the held-out token ids as
`prompt` (List[int], skips re-tokenization), `max_tokens: 0` (generate
nothing), `echo: True, logprobs: 1` -> vLLM returns each prompt token's
log-probability under the model, which is exactly next-token loss. The
first token has no preceding context so its logprob is null; drop it.

Usage::

    scripts/eval_perplexity.py --data artifacts/ppl-holdout.pt \\
        --base-url http://localhost:8000/v1 --out artifacts/ppl-result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import requests
import torch


def score_sequence(base_url: str, model: str, ids: list[int], timeout: float) -> list[float]:
    r = requests.post(
        f"{base_url}/completions",
        json={"model": model, "prompt": ids, "max_tokens": 0, "echo": True, "logprobs": 1},
        timeout=timeout,
    )
    r.raise_for_status()
    logprobs = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    return [lp for lp in logprobs if lp is not None]


def resolve_model(base_url: str, timeout: float) -> str:
    r = requests.get(f"{base_url}/models", timeout=timeout)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="artifacts/ppl-holdout.pt")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", help="model id as registered with the server "
                     "(default: first entry from /v1/models)")
    ap.add_argument("--limit", type=int, help="only score the first N sequences")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", help="write per-sequence + aggregate results as JSON")
    args = ap.parse_args()

    samples: list[list[int]] = torch.load(args.data, weights_only=True)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("no sequences to score", file=sys.stderr)
        return 1

    model = args.model or resolve_model(args.base_url, args.timeout)
    print(f"scoring {len(samples)} sequences against {args.base_url} (model={model})")

    all_logprobs: list[float] = []
    per_seq = []
    for i, ids in enumerate(samples):
        lps = score_sequence(args.base_url, model, ids, args.timeout)
        all_logprobs.extend(lps)
        seq_ppl = math.exp(-sum(lps) / len(lps))
        per_seq.append({"n_tokens": len(lps), "ppl": seq_ppl})
        print(f"  [{i + 1}/{len(samples)}] {len(lps)} tok  ppl={seq_ppl:.3f}", flush=True)

    mean_nll = -sum(all_logprobs) / len(all_logprobs)
    ppl = math.exp(mean_nll)
    print(f"\noverall perplexity: {ppl:.4f}  "
          f"({len(all_logprobs):,} scored tokens, {len(samples)} sequences)")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "data": args.data, "base_url": args.base_url, "model": model,
            "num_sequences": len(samples), "num_tokens": len(all_logprobs),
            "perplexity": ppl, "per_sequence": per_seq,
        }, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
