#!/usr/bin/env python3
"""Surgery-pass checks for the DeepSeek-V4 profile, on a synthetic tiny checkpoint.

Covers what is specific to this checkpoint: the ``layers.{L}.ffn.*`` naming, the
``ffn.gate.bias`` router bias, MTP blocks living in their own ``mtp.{N}``
namespace, and -- the one that would quietly corrupt a model -- hash-routed
layers whose ``tid2eid`` table names experts by value and so must be left
alone.

Run: ``.venv/bin/python scripts/test_reap_surgery_dsv4.py``  (CPU only)
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SURGERY = HERE / "reap_surgery.py"

N_EXPERTS = 8
HIDDEN = 16
INTER = 8
VOCAB = 32
N_LAYERS = 6          # 0-2 hash-routed, 3-5 top-k routed
N_HASH = 3
MTP_BLOCKS = 2
KEEP = [0, 2, 5, 7]   # retained expert ids, same for every pruned layer

FAILED: list[str] = []
ORIGINAL_TID2EID: dict[str, torch.Tensor] = {}


def check(ok: bool, label: str) -> None:
    print(f"  {'ok' if ok else 'FAIL'}: {label}")
    if not ok:
        FAILED.append(label)


def expert_tensors(block: str, expert: int, tag: float) -> dict[str, torch.Tensor]:
    """One expert's FP4 payload, tagged so we can trace it after renumbering."""
    out = {}
    for proj, (o, i) in (("w1", (INTER, HIDDEN)), ("w3", (INTER, HIDDEN)), ("w2", (HIDDEN, INTER))):
        w = torch.full((o, i // 2), int(tag) % 128, dtype=torch.int8)
        w[0, 0] = int(tag) % 128
        out[f"{block}.ffn.experts.{expert}.{proj}.weight"] = w
        out[f"{block}.ffn.experts.{expert}.{proj}.scale"] = torch.full(
            (o, max(1, i // 32)), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    return out


def build_source(src: Path) -> None:
    src.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),
        "head.weight": torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),
        "norm.weight": torch.randn(HIDDEN, dtype=torch.bfloat16),
    }
    blocks = [f"layers.{L}" for L in range(N_LAYERS)] + [f"mtp.{i}" for i in range(MTP_BLOCKS)]
    for b, block in enumerate(blocks):
        hashed = block.startswith("layers.") and int(block.split(".")[1]) < N_HASH
        tensors[f"{block}.attn.wq_a.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{block}.attn_norm.weight"] = torch.randn(HIDDEN, dtype=torch.bfloat16)
        tensors[f"{block}.ffn.shared_experts.w1.weight"] = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16)
        # router: dim 0 is indexed by expert and must be sliced
        tensors[f"{block}.ffn.gate.weight"] = (
            torch.arange(N_EXPERTS, dtype=torch.float32).unsqueeze(1).expand(N_EXPERTS, HIDDEN).clone().bfloat16()
        )
        if hashed:
            table = torch.randint(0, N_EXPERTS, (VOCAB, 2), dtype=torch.int64)
            tensors[f"{block}.ffn.gate.tid2eid"] = table
            ORIGINAL_TID2EID[block] = table.clone()
        else:
            tensors[f"{block}.ffn.gate.bias"] = torch.arange(N_EXPERTS, dtype=torch.float32)
        for e in range(N_EXPERTS):
            tensors.update(expert_tensors(block, e, tag=b * 100 + e))

    # two shards, split arbitrarily -- the rewrite is per-shard
    names = sorted(tensors)
    half = len(names) // 2
    weight_map = {}
    for shard, chunk in (("model-00001-of-00002.safetensors", names[:half]),
                         ("model-00002-of-00002.safetensors", names[half:])):
        save_file({n: tensors[n] for n in chunk}, str(src / shard), metadata={"format": "pt"})
        weight_map.update({n: shard for n in chunk})

    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0}, "weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "n_routed_experts": N_EXPERTS,
        "num_hash_layers": N_HASH,
        "num_hidden_layers": N_LAYERS,
        "num_nextn_predict_layers": 1,
        "hidden_size": HIDDEN,
    }, indent=2))
    (src / "tokenizer_config.json").write_text("{}")
    (src / "inference").mkdir()
    (src / "inference" / "model.py").write_text("# reference implementation\n")


def run_surgery(src: Path, dst: Path, retained: Path, mtp: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SURGERY), "--src", str(src), "--dst", str(dst),
         "--retained", str(retained), "--mtp", mtp, "--workers", "1"],
        capture_output=True, text=True,
    )


def load_all(dst: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(dst.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


def run_case(src: Path, dst: Path, retained: Path, mtp: str) -> None:
    print(f"\n=== --mtp {mtp} ===")
    proc = run_surgery(src, dst, retained, mtp)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        check(False, f"surgery exited {proc.returncode}")
        return
    t = load_all(dst)
    cfg = json.loads((dst / "config.json").read_text())

    check(cfg["n_routed_experts"] == len(KEEP), f"n_routed_experts -> {len(KEEP)}")

    # -- pruned layers: renumbered 0..3, payload bit-identical --
    for L in range(N_HASH, N_LAYERS):
        block = f"layers.{L}"
        ids = sorted({int(k.split(".experts.")[1].split(".")[0])
                      for k in t if k.startswith(f"{block}.ffn.experts.")})
        check(ids == list(range(len(KEEP))), f"{block}: experts renumbered 0..{len(KEEP) - 1}")
        tags = [t[f"{block}.ffn.experts.{i}.w1.weight"][0, 0].item() for i in range(len(KEEP))]
        check(tags == [(L * 100 + e) % 128 for e in KEEP],
              f"{block}: payloads bit-identical, old {KEEP} -> new 0..{len(KEEP) - 1}")
        rows = t[f"{block}.ffn.gate.weight"][:, 0].float().tolist()
        check(rows == [float(e) for e in KEEP], f"{block}: gate.weight rows = {KEEP}")
        bias = t[f"{block}.ffn.gate.bias"].tolist()
        check(bias == [float(e) for e in KEEP], f"{block}: gate.bias rows = {KEEP}")

    # -- hash layers: untouched, tid2eid still valid --
    for L in range(N_HASH):
        block = f"layers.{L}"
        ids = sorted({int(k.split(".experts.")[1].split(".")[0])
                      for k in t if k.startswith(f"{block}.ffn.experts.")})
        check(ids == list(range(N_EXPERTS)), f"{block}: hash-routed, all {N_EXPERTS} experts kept")
        check(t[f"{block}.ffn.gate.weight"].shape[0] == N_EXPERTS,
              f"{block}: gate.weight not sliced")
        check(f"{block}.ffn.gate.tid2eid" in t, f"{block}: tid2eid preserved")
        check(int(t[f"{block}.ffn.gate.tid2eid"].max()) < N_EXPERTS,
              f"{block}: tid2eid still points at existing experts")

    # -- MTP blocks --
    mtp_keys = [k for k in t if k.startswith("mtp.")]
    if mtp == "drop":
        check(not mtp_keys, "MTP blocks removed entirely")
        check(cfg["num_nextn_predict_layers"] == 0, "num_nextn_predict_layers -> 0")
    elif mtp == "prune-with-last":
        for i in range(MTP_BLOCKS):
            ids = sorted({int(k.split(".experts.")[1].split(".")[0])
                          for k in mtp_keys if k.startswith(f"mtp.{i}.ffn.experts.")})
            check(ids == list(range(len(KEEP))), f"mtp.{i}: pruned with the last layer's set")
        check(cfg["num_nextn_predict_layers"] == 1, "num_nextn_predict_layers stays 1")
    else:
        ids = sorted({int(k.split(".experts.")[1].split(".")[0])
                      for k in mtp_keys if k.startswith("mtp.0.ffn.experts.")})
        check(ids == list(range(N_EXPERTS)), f"mtp.0: untouched, all {N_EXPERTS} experts kept")

    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    check(set(idx["weight_map"]) == set(t), "index weight_map matches stored tensors")
    check((dst / "inference" / "model.py").exists(), "inference/ directory copied")
    check((dst / "reap_pruning.json").exists(), "reap_pruning.json written")


def test_hash_remap(src: Path, work: Path) -> None:
    """--hash-layers prune-remap must leave tid2eid pointing at live experts."""
    print("\n=== --hash-layers prune-remap ===")
    retained = work / "retained-all.json"
    retained.write_text(json.dumps({
        "num_experts": N_EXPERTS,
        "layers": {str(L): KEEP for L in range(N_LAYERS)},
    }))
    dst = work / "dst-remap"
    proc = subprocess.run(
        [sys.executable, str(SURGERY), "--src", str(src), "--dst", str(dst),
         "--retained", str(retained), "--mtp", "drop", "--workers", "1",
         "--hash-layers", "prune-remap"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        check(False, f"surgery exited {proc.returncode}")
        return
    t = load_all(dst)

    for L in range(N_HASH):
        block = f"layers.{L}"
        ids = sorted({int(k.split(".experts.")[1].split(".")[0])
                      for k in t if k.startswith(f"{block}.ffn.experts.")})
        check(ids == list(range(len(KEEP))), f"{block}: pruned to {len(KEEP)} experts")
        table = t[f"{block}.ffn.gate.tid2eid"]
        check(int(table.max()) < len(KEEP) and int(table.min()) >= 0,
              f"{block}: tid2eid only names surviving experts "
              f"(range [{int(table.min())}, {int(table.max())}])")
        # a token that pointed at a survivor must now point at its new index
        original = ORIGINAL_TID2EID[block]
        survivors = {old: new for new, old in enumerate(KEEP)}
        kept_positions = [(r, c) for r in range(original.shape[0])
                          for c in range(original.shape[1])
                          if int(original[r, c]) in survivors]
        ok = all(int(table[r, c]) == survivors[int(original[r, c])]
                 for r, c in kept_positions[:200])
        check(ok, f"{block}: surviving experts keep their tokens (renumbered)")

        # balanced mode: every survivor absorbs the same number of dropped
        # experts, so no expert ends up holding a disproportionate share of
        # the vocabulary
        n_drop = N_EXPERTS - len(KEEP)
        absorbed = collections.Counter()
        for e in range(N_EXPERTS):
            if e in survivors:
                continue
            # find where this dropped expert's tokens went
            for r in range(original.shape[0]):
                for c in range(original.shape[1]):
                    if int(original[r, c]) == e:
                        absorbed[int(table[r, c])] += 1
                        break
                else:
                    continue
                break
        cap = -(-n_drop // len(KEEP))
        counts = collections.Counter()
        for e in range(N_EXPERTS):
            if e not in survivors:
                pos = [(r, c) for r in range(original.shape[0])
                       for c in range(original.shape[1]) if int(original[r, c]) == e]
                if pos:
                    counts[int(table[pos[0]])] += 1
        check(not counts or max(counts.values()) <= cap,
              f"{block}: no survivor absorbs more than {cap} dropped expert(s) "
              f"(max {max(counts.values()) if counts else 0})")

    cfg = json.loads((dst / "config.json").read_text())
    check(cfg["n_routed_experts"] == len(KEEP), "n_routed_experts uniform across layers")
    rp = json.loads((dst / "reap_pruning.json").read_text())
    check(sorted(rp["remapped_blocks"]) == [f"layers.{L}" for L in range(N_HASH)],
          "reap_pruning.json records which blocks were remapped")


def test_guard_without_hash_config(src: Path, work: Path) -> None:
    """Without num_hash_layers, the tid2eid guard must still refuse."""
    print("\n=== guard: hash layer requested for pruning ===")
    bad_src = work / "src-nohash"
    bad_src.mkdir()
    for p in src.iterdir():
        if p.is_dir():
            continue
        (bad_src / p.name).write_bytes(p.read_bytes())
    cfg = json.loads((src / "config.json").read_text())
    del cfg["num_hash_layers"]
    (bad_src / "config.json").write_text(json.dumps(cfg))

    retained = work / "retained-all.json"
    retained.write_text(json.dumps({
        "num_experts": N_EXPERTS,
        "layers": {str(L): KEEP for L in range(N_LAYERS)},
    }))
    proc = run_surgery(bad_src, work / "dst-guard", retained, "drop")
    check(proc.returncode != 0, "refuses to prune a block that owns tid2eid")
    check("expert ids as values" in (proc.stderr + proc.stdout),
          "error explains why (tid2eid stores expert ids)")


def main() -> int:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        build_source(src)
        print(f"built synthetic DeepSeek-V4 checkpoint: {N_EXPERTS} experts, "
              f"{N_LAYERS} layers ({N_HASH} hash-routed), {MTP_BLOCKS} MTP blocks")

        retained = td / "retained.json"
        retained.write_text(json.dumps({
            "num_experts": N_EXPERTS,
            "layers": {str(L): KEEP for L in range(N_HASH, N_LAYERS)},
        }))
        for mtp in ("drop", "prune-with-last", "keep"):
            run_case(src, td / f"dst-{mtp}", retained, mtp)
        test_hash_remap(src, td)
        test_guard_without_hash_config(src, td)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
