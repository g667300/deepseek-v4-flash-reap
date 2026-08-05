#!/usr/bin/env python3
"""Build a REAP calibration set from a weighted language mixture.

The mixture is what decides which experts survive: an expert that never sees
tokens it specializes in scores zero and gets pruned first. So the mix is a
config file, not a constant -- see ``calib/mix-dsv4.json``.

Three things this deliberately does:

* **Streams.** Every source is read with ``streaming=True`` and abandoned as
  soon as enough samples are collected, so a 100 Mbps line pulls tens of MB per
  source instead of downloading whole corpora.
* **Packs to a fixed length.** Documents are concatenated into exactly
  ``seq_len``-token samples. The scoring pass needs uniform lengths (it truncates to the
  shortest sample otherwise), and packing wastes no tokens on padding.
* **Runs each source in its own process.** Abandoning a ``datasets`` stream
  leaves prefetch threads and an ``httpx`` client behind; tearing those down in
  process breaks the *next* source ("Cannot send a request, as the client has
  been closed") and can abort the interpreter during finalization. A child
  process per source sidesteps all of it, and isolates a failing source.

Usage::

    build_calibration.py --mix calib/mix-dsv4.json --tokenizer models/DeepSeek-V4-Flash-0731 \
        --out artifacts/calib.pt
    build_calibration.py --mix calib/mix-dsv4.json --probe   # check sources reachable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def extract_text(row: dict, spec: dict, render) -> str | None:
    """Turn one dataset row into a training-shaped string."""
    kind = spec["format"]

    if kind == "text":
        text = row.get(spec["field"])
        return text if isinstance(text, str) else None

    if kind == "instruction":
        # single-turn instruction data -> render through the chat template so the
        # calibration distribution matches how the model is actually prompted
        instr = row.get(spec.get("instruction_field", "instruction")) or ""
        inp = row.get(spec.get("input_field", "input")) or ""
        resp = row.get(spec.get("output_field", "output")) or ""
        if not instr or not resp:
            return None
        user = f"{instr}\n\n{inp}" if inp else instr
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]
        return render(msgs)

    if kind == "messages":
        msgs = row.get(spec.get("field", "messages"))
        if not msgs:
            return None
        norm = []
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            if not role or not content:
                continue
            norm.append({"role": role, "content": content})
        if len(norm) < 2:
            return None
        return render(norm)

    raise ValueError(f"unknown format {kind!r}")


def iter_source(spec: dict, render):
    """Yield texts from one source, lazily. Never tears the stream down --
    the caller is expected to be a short-lived process."""
    from datasets import load_dataset

    kwargs = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["dataset"], **kwargs)
    if spec.get("shuffle_buffer"):
        ds = ds.shuffle(seed=spec.get("seed", 0), buffer_size=spec["shuffle_buffer"])
    for row in ds:
        text = extract_text(row, spec, render)
        if text:
            yield text


def collect_source(spec: dict, tokenizer, render, n_samples: int,
                   seq_len: int, pack: bool, min_tokens: int, eos_id: int | None):
    """Collect up to ``n_samples`` sequences of ``seq_len`` tokens."""
    samples: list[list[int]] = []
    buf: list[int] = []
    n_docs = 0
    n_skipped = 0

    for text in iter_source(spec, render):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_docs += 1
        if pack:
            buf.extend(ids)
            if eos_id is not None:
                buf.append(eos_id)
            while len(buf) >= seq_len and len(samples) < n_samples:
                samples.append(buf[:seq_len])
                buf = buf[seq_len:]
        else:
            if len(ids) < min_tokens:
                n_skipped += 1
                continue
            samples.append(ids[:seq_len])
        if len(samples) >= n_samples:
            break

    return samples, {"docs_read": n_docs, "docs_skipped": n_skipped}


def load_mix(args) -> tuple[dict, int, int, list[dict]]:
    mix = json.loads(Path(args.mix).read_text())
    seq_len = args.seq_len or mix["seq_len"]
    total = args.num_samples or mix["num_samples"]
    sources = mix["sources"]

    # Largest-remainder allocation so the per-source counts sum to `total`
    # exactly; plain rounding drifts by a sample or two per source.
    wsum = sum(s["weight"] for s in sources)
    exact = [total * s["weight"] / wsum for s in sources]
    for s, e in zip(sources, exact):
        s["_n"] = max(1, int(e))
    short = total - sum(s["_n"] for s in sources)
    if short > 0:
        order = sorted(range(len(sources)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
        for i in order[:short]:
            sources[i]["_n"] += 1
    return mix, seq_len, total, sources


def get_tokenizer(path: str, pack: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    eos_id = tokenizer.eos_token_id if pack else None
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    return tokenizer, eos_id


def make_renderer(tokenizer, path: str, chat_kwargs: dict):
    """Return ``f(messages) -> str`` for turning chat data into model-shaped text.

    Most models expose an HF ``chat_template``. DeepSeek-V4 instead ships its
    own encoder (``encoding/encoding_dsv4.py``), so the profile hands us that;
    ``chat_kwargs`` from the mix file is passed to whichever one applies, which
    is why the two families spell the same intent differently
    (``enable_thinking: false`` vs ``thinking_mode: "chat"``).
    """
    import model_profiles

    profile = model_profiles.detect_optional(Path(path))
    encoder = profile.chat_renderer(Path(path)) if profile else None
    if encoder is not None:
        return lambda msgs: encoder(msgs, **chat_kwargs)
    return lambda msgs: tokenizer.apply_chat_template(msgs, tokenize=False, **chat_kwargs)


# --------------------------------------------------------------------------
# child mode: one source, then exit
# --------------------------------------------------------------------------


def run_child(args) -> int:
    import torch

    mix, seq_len, _total, sources = load_mix(args)
    spec = next((s for s in sources if s["name"] == args.only_source), None)
    if spec is None:
        print(f"no source named {args.only_source!r}", file=sys.stderr)
        return 2

    pack = not args.no_pack
    tokenizer, eos_id = get_tokenizer(args.tokenizer, pack)
    chat_kwargs = mix.get("chat_template_kwargs", {})
    render = make_renderer(tokenizer, args.tokenizer, chat_kwargs)

    samples, stats = collect_source(spec, tokenizer, render, spec["_n"],
                                    seq_len, pack, args.min_tokens, eos_id)
    torch.save({"samples": samples, "stats": stats}, args.shard_out)
    return 0


# --------------------------------------------------------------------------


def probe(sources, tokenizer, render) -> int:
    print(f"probing {len(sources)} sources (2 rows each)\n", flush=True)
    bad = 0
    for s in sources:
        t0 = time.time()
        try:
            got = []
            for text in iter_source(s, render):
                got.append(text)
                if len(got) >= 2:
                    break
            if not got:
                raise RuntimeError("produced no usable rows")
            n_tok = len(tokenizer(got[0], add_special_tokens=False)["input_ids"])
            head = got[0][:110].replace("\n", "\\n")
            print(f"  OK   {s['name']:<18} {s['dataset']}  "
                  f"({time.time() - t0:.1f}s, first doc {n_tok} tok)")
            print(f"       {head}...", flush=True)
        except Exception as e:  # noqa: BLE001 - this is a reachability probe
            bad += 1
            print(f"  FAIL {s['name']:<18} {s['dataset']}\n       {type(e).__name__}: {e}",
                  flush=True)
    print(f"\n{len(sources) - bad}/{len(sources)} sources usable")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default="calib/mix-dsv4.json")
    ap.add_argument("--out", default="artifacts/calib.pt")
    ap.add_argument("--tokenizer", default="models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--num-samples", type=int, help="override the mix's num_samples")
    ap.add_argument("--seq-len", type=int, help="override the mix's seq_len")
    ap.add_argument("--no-pack", action="store_true",
                    help="one sample per document (truncated) instead of packing")
    ap.add_argument("--min-tokens", type=int, default=256,
                    help="with --no-pack, drop documents shorter than this")
    ap.add_argument("--probe", action="store_true",
                    help="pull 2 rows per source to check reachability and formats")
    ap.add_argument("--report", help="write a JSON report of the realized mixture")
    ap.add_argument("--only-source", help=argparse.SUPPRESS)  # child mode
    ap.add_argument("--shard-out", help=argparse.SUPPRESS)  # child mode
    args = ap.parse_args()

    if args.only_source:
        return run_child(args)

    import torch

    mix, seq_len, total, sources = load_mix(args)
    pack = not args.no_pack
    tokenizer, _eos = get_tokenizer(args.tokenizer, pack)
    chat_kwargs = mix.get("chat_template_kwargs", {})
    render = make_renderer(tokenizer, args.tokenizer, chat_kwargs)

    if args.probe:
        return probe(sources, tokenizer, render)

    if chat_kwargs:
        print(f"chat template kwargs: {chat_kwargs}")
    print(f"mix: {args.mix}   target: {total} samples x {seq_len} tokens "
          f"({'packed' if pack else 'per-document'})", flush=True)

    all_samples: list[list[int]] = []
    per_source: dict[str, dict] = {}
    lang_tokens: Counter[str] = Counter()
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="calib-") as td:
        for s in sources:
            t0 = time.time()
            shard = Path(td) / f"{s['name']}.pt"
            cmd = [sys.executable, __file__, "--mix", args.mix, "--tokenizer", args.tokenizer,
                   "--seq-len", str(seq_len), "--num-samples", str(total),
                   "--min-tokens", str(args.min_tokens),
                   "--only-source", s["name"], "--shard-out", str(shard)]
            if args.no_pack:
                cmd.append("--no-pack")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not shard.exists():
                failures.append(s["name"])
                tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                print(f"  {s['name']:<18} FAILED  {tail}", flush=True)
                continue

            blob = torch.load(shard, weights_only=True)
            got, stats = blob["samples"], blob["stats"]
            n_tok = sum(len(x) for x in got)
            all_samples.extend(got)
            per_source[s["name"]] = {
                "dataset": s["dataset"], "lang": s.get("lang", "?"),
                "requested": s["_n"], "collected": len(got), "tokens": n_tok, **stats,
            }
            lang_tokens[s.get("lang", "?")] += n_tok
            short = "" if len(got) == s["_n"] else f"  (SHORT of {s['_n']})"
            print(f"  {s['name']:<18} {len(got):>5} samples  {n_tok:>10,} tok  "
                  f"{time.time() - t0:6.1f}s{short}", flush=True)

    if not all_samples:
        print("no samples collected", file=sys.stderr)
        return 1

    g = torch.Generator().manual_seed(mix.get("seed", 0))
    order = torch.randperm(len(all_samples), generator=g).tolist()
    all_samples = [all_samples[i] for i in order]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, args.out)

    tok_total = sum(lang_tokens.values())
    print(f"\nlanguage mix by tokens ({tok_total:,} total):")
    for lang, n in lang_tokens.most_common():
        print(f"  {lang:<6} {n / tok_total:6.1%}  {n:>11,}")
    lengths = {len(x) for x in all_samples}
    print(f"samples: {len(all_samples)}, lengths: "
          f"{'uniform ' + str(lengths.pop()) if len(lengths) == 1 else sorted(lengths)[:5]}")
    print(f"wrote {args.out}")
    if failures:
        print(f"WARNING: sources that produced nothing: {failures}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "mix": args.mix, "seq_len": seq_len, "num_samples": len(all_samples),
            "packed": pack, "chat_template_kwargs": chat_kwargs,
            "per_source": per_source, "language_tokens": dict(lang_tokens),
            "failed_sources": failures,
        }, indent=2) + "\n")
        print(f"wrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    # A `datasets` stream that was abandoned mid-iteration keeps prefetch threads
    # alive; they can abort the interpreter during finalization. Everything is
    # already flushed to disk by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
