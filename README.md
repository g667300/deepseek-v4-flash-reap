# deepseek-v4-flash-reap

Pruning pipeline for
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
using REAP (Router-weighted Expert Activation Pruning): 256 experts per layer
down to 128 (50% sparsity), 166.9 GB down to **82.4 GB**.

Two goals. The first is to **run on a DGX Spark**: 82.4 GB fits its 128 GB of
unified memory with room left to serve, and the 166.9 GB original does not. A
128 GB unified-memory PC may work as a target too, though none was tested here.

The second is that **the calibration set is yours to build**. REAP keeps the
experts your own data actually uses, so what survives is decided by the mix you
feed it — `calib/mix-dsv4.json` here is weighted toward Japanese, English and
code, and swapping it is the supported way to aim the result at something else.

## How this differs from the usual REAP run

The reference implementation ([llm-compressor's REAP
modifier](https://arxiv.org/abs/2510.13999)) runs as one pass over a model
loaded through transformers: it records routing while calibration data flows
through, and prunes each MoE block in place as the pass reaches it.

That path cannot read this checkpoint, which is **already quantized** in
DeepSeek's own conventions:

* Tensors carry DeepSeek's native names (`layers.{L}.ffn.experts.{E}.w1.weight`),
  not the `model.layers.*.mlp.*` that transformers expects — this repo carries an
  explicit rename map to build a layer at all.
* The quantization is FP4 (E2M1, 32-element blocks, E8M0 scales) for routed
  experts and FP8 (E4M3, 128x128 blocks) elsewhere. compressed-tensors does not
  read that, so loading means dequantizing first, and saving back through the
  normal path would re-encode weights that REAP never touched.

So the run is split in two:

* **The scoring pass** (`reap_saliency_dsv4.py`, GPU) walks the model one layer
  at a time, dequantizing each layer just long enough to push calibration data
  through it, and writes out a saliency score per expert. Nothing is modified.
* **The surgery pass** (`reap_surgery.py`, CPU) reads those scores and rewrites
  the safetensors shards, keeping the winning experts and dropping the rest.

The split works because of a property of REAP itself: **it never modifies the
weights of the experts that survive.** Pruning is a slice of the expert list and
of the matching router rows, nothing more. The surgery is therefore a file-level
transform that preserves the original FP4/FP8 encoding exactly — no
dequantize-requantize round trip and no GPU.

Japanese: [README.jp.md](README.jp.md)

## Requirements

What it takes to actually run this end to end:

| | |
|---|---|
| RAM | **64 GB minimum**, 128 GB recommended |
| VRAM | **11 GB** (CUDA) |
| Free disk | **~290 GB** |
| Time | ~40 min of compute at 128 GB of RAM, 70+ min at 64 GB, plus a 166.9 GB download |

Measured on a desktop (RTX 4090). **A DGX Spark should be able to run the
pipeline itself as well, but that was not tried here** — it was only ever used
to serve the result.

Serving is a separate matter: that host has to hold the 82.4 GB checkpoint in
memory. Where these numbers come from, and what each stage needs
on its own, is in [Reference environment](#reference-environment).

## Results

Measured on the pruned checkpoint served through vLLM. **There is no
unpruned control**: the 166.9 GB original does not fit the evaluation host, so
these are absolute health checks rather than a measured degradation.

| Benchmark | Result | Baseline |
|---|---|---|
| Japanese JCommonsenseQA (full 1,119, 3-shot) | 0.9088 ± 0.0086 | 0.20 random |
| English MMLU (570-item diagnostic) | 0.6526 ± 0.0194 | 0.25 random |
| Chinese global_mmlu_zh (full 400) | 0.4975 ± 0.0250 | 0.25 random |
| Perplexity (262,016 held-out tokens) | 6.5891 | — |
| RULER 4K / 16K / 32K / 64K (12-task mean) | 98.68 / 96.10 / 96.93 / 94.51 | 85.6 effective-length threshold |

Retrieval is untouched by the prune — the eight needle-in-a-haystack tasks score
1.000 at every length up to 32K. What erodes is reading comprehension over the
retrieved span: `ruler_qa_squad` is the worst task at all four lengths, falling
from 0.842 at 4K to 0.542 at 64K.

The checkpoint is published as
[noooop/DeepSeek-V4-Flash-REAP-noMTP](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP-noMTP).

What these numbers do and do not establish, the limits of the method as applied
here, and what is still untested are in
[`docs/TECHNICAL_NOTES.jp.md`](docs/TECHNICAL_NOTES.jp.md) (Japanese).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

Verified with torch 2.11.0+cu128 / transformers 5.14.1 / lm_eval 0.4.12.
`CUDA_VISIBLE_DEVICES` may be needed on hosts where an unsupported GPU is
present alongside the one you want to use.

## Reference environment

What the timings on this page were measured on, and what each stage actually
needs.

| Stage | Needs | Measured here |
|---|---|---|
| Calibration | CPU only, a tokenizer | minutes |
| Scoring pass | CUDA GPU, under 11 GB VRAM. **Host RAM is the real constraint** (21.5 GB cache at 256 x 2048) | RTX 4090 + 64 GB host: 51 s/layer, ~37 min for 43 layers without `--spill`; ~78 s/layer (~55 min) with it |
| Surgery pass | No GPU, no dequantization. Disk for source **and** output (166.9 + 82.4 GB); two shards in flight per worker | minutes, disk-bound |
| Serving | 121 GiB unified memory (DGX Spark), 82.4 GB checkpoint resident | see [Serving](#serving-vllm) |

The scoring pass is bound by host-side work rather than GPU FLOPs. Per layer it gathers
tens of GB of activations out of the host cache and moves roughly 34 GB across
PCIe, while the expert math itself runs at a few percent of the card's peak
(VRAM never exceeds 11 GB of a 24 GB card). **A faster GPU therefore buys very
little; host memory and its bandwidth buy more.**

Running the scoring pass on the unified-memory serving host instead is untested. Expect
roughly the same range, or a little slower.

## Fetching the model

```bash
./scripts/dl.sh deepseek-ai/DeepSeek-V4-Flash-0731 models/DeepSeek-V4-Flash-0731 start
./scripts/dl.sh deepseek-ai/DeepSeek-V4-Flash-0731 models/DeepSeek-V4-Flash-0731 status
```

`dl.sh` is resumable. `dl_ratelimited.sh` paces the download in bytes/sec for
links that must be shared; `dl_one_file.sh` grabs a single remaining file.

## Pipeline

```bash
# Calibration set (Japanese/English/code-weighted, rendered with DeepSeek's own
# prompt format rather than an HF chat template -- the model ships none)
.venv/bin/python scripts/build_calibration.py --mix calib/mix-dsv4.json \
    --tokenizer models/DeepSeek-V4-Flash-0731 --out artifacts/calib.pt

# Scoring pass: layer-sequential saliency (GPU, ~35 min on a host that fits the cache)
.venv/bin/python scripts/reap_saliency_dsv4.py \
    --ckpt models/DeepSeek-V4-Flash-0731 --tokens artifacts/calib.pt \
    --max-samples 256 --out artifacts/saliency.json

# Surgery pass: file-level rewrite of the safetensors (no GPU, minutes)
.venv/bin/python scripts/reap_surgery.py \
    --src models/DeepSeek-V4-Flash-0731 --dst artifacts/dsv4-reap50 \
    --saliency artifacts/saliency.json --sparsity 0.5 \
    --mtp drop --hash-layers prune-remap
```

### Host memory for the scoring pass

The scoring pass is GPU-light (VRAM peaks around 10 GB) but host-heavy: it keeps the
whole calibration activation cache in system memory, **21.5 GB for 256 x 2048
tokens**. Most of that is the hidden state, which hyper-connections make 4x a
conventional model's (`hc_mult=4`).

128 GB is comfortable and runs the command above as written. **64 GB is
marginal.** The cache plus the process resident set leaves little headroom, and
running it there alongside other work froze the host outright — the kernel
thrashed instead of OOM-killing anything, so nothing was logged. On a 64 GB
host, run it like this instead:

```bash
systemd-run --user --scope -p MemoryMax=42G -p MemorySwapMax=0 \
    .venv/bin/python scripts/reap_saliency_dsv4.py \
    --ckpt models/DeepSeek-V4-Flash-0731 --tokens artifacts/calib.pt \
    --max-samples 256 --out artifacts/saliency.json \
    --spill /var/tmp/dsv4-spill --state artifacts/saliency-state.pt --state-every 3
```

* `--spill DIR` backs the caches with a file-backed shared mmap, so the pages
  are reclaimable page cache rather than anonymous memory. It costs **at least
  45% wall time and grows from there**: 77 s/layer at the start against 53 s
  without spill, drifting to 120 s by layer 25. The hidden state is rewritten
  every layer, so the run writes **35-40 GB per layer** counting `--state`, and
  a consumer SSD falls off its SLC cache well before the run ends — at that
  point the process stalls on writeback (measured: 0.7-1.1 GB/s at 90% device
  utilization, 60-100 ms write latency). **The limit is the drive's sustained
  write rate, not the RAM.** Budget 70+ min rather than 55.
* `--state FILE --state-every N` checkpoints the hidden state and the trackers
  every N layers, so an interrupted run resumes rather than restarting.
* The scope caps the process, so a runaway is killed instead of taking the
  machine down. The limit is reached and held by reclaim, which is the intended
  behaviour: anonymous memory stays around 1.6 GB and no OOM kill occurs.

`--max-samples 512` doubles the cache to 43 GB, which no longer fits page cache
on a 64 GB host — expect real disk I/O of roughly 86 GB per layer there.

### Hash-routed layers

DeepSeek-V4's first three layers route by a frozen `tid2eid` table (token id →
expert id) rather than by a learned score, and `n_routed_experts` is a single
value for the whole model. Pruning any layer therefore forces those three to
the same expert count.

`--hash-layers prune-remap` prunes them and rewrites `tid2eid`, sending each
dropped expert's token ids to the surviving expert whose router row points in
the most similar direction, with a balanced assignment so no survivor absorbs a
disproportionate share. This is a merge, not a pure prune, and it applies to 3
of 43 layers. `refuse` (the default) rejects the operation instead, which is
correct in isolation but produces a checkpoint that will not load.

## Tests

No GPU required:

```bash
.venv/bin/python scripts/test_quant.py             # FP4/FP8 dequant vs DeepSeek's own convert.py
.venv/bin/python scripts/test_reap_surgery_dsv4.py # surgery pass on a synthetic checkpoint
.venv/bin/python scripts/test_scoring_pass_dsv4.py      # three-phase split vs layer.forward
```

## Serving (vLLM)

To move the result to another host, `scripts/push_model.sh <src> <host:/dest>`
wraps rsync in a retry loop — an 82 GB transfer over ssh dropped every 35-45 GB
here, and resuming is the only practical answer. Pass `--checksum` when the
destination already holds an earlier build: the surgery pass rewrites every shard, so
mtimes all change while most bytes do not.

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v artifacts/dsv4-reap50:/model:ro \
    --entrypoint vllm <vllm_image> serve /model \
    --served-model-name dsv4-reap50 --gpu-memory-utilization 0.75 \
    --max-model-len 65536 --max-num-seqs 16 --kv-cache-dtype fp8
```

**`--gpu-memory-utilization` is not the same dial on a unified-memory host.**
There it is taken out of system RAM, so vLLM's usual 0.9 reserves ~109 GiB of a
121 GiB machine and leaves the OS about 8 GiB. Long context then collapses:
measured at 32K with 16 concurrent requests, generation fell to 0.1-2.2 tokens/s
while the engine reported only 4-6% of its KV cache in use — the pool was never
the constraint. Pointing a second client at it in that state hung the machine
hard enough to need a power cycle. 0.75 leaves ~28 GiB free and holds up.

`--kv-cache-dtype fp8` is mandatory: DeepSeek-V4's sparse-MLA kernel rejects
anything else. vLLM 0.25.1's stock image pins FlashInfer 0.6.13 while its own
code passes arguments that only exist in 0.6.14, so it crashes on load; a
derived image with `flashinfer-python==0.6.14` and
`FLASHINFER_DISABLE_VERSION_CHECK=1` (flashinfer-cubin never shipped 0.6.14)
works. Later vLLM releases may not need this.

## Evaluation

JCommonsenseQA reads upstream's raw JSON directly (its HF dataset ships a
loading script that `datasets>=4` will not run), so fetch that once first:

```bash
./scripts/fetch_eval_data.sh
```

Then, from the repository root:

```bash
MODEL=dsv4-reap50 TOKENIZER=models/DeepSeek-V4-Flash-0731 TAG=dsv4-50 \
    scripts/run_eval_suite.sh
```

The scripts default to `http://localhost:8000`. Set `BASE` when the server runs
elsewhere, e.g. `BASE=http://10.0.0.5:8000/v1/completions`.

Runs Japanese commonsense (JCommonsenseQA), English knowledge (MMLU), 15
languages (global_mmlu), perplexity and long context (RULER 4k-64k), cheapest
first so an interrupted run still fills the table. `SKIP="ruler"` and similar
skip stages by name (`jcqa mmlu multilingual ppl ruler`).

## License

Code: MIT ([LICENSE](LICENSE)). The base model
(DeepSeek-V4-Flash-0731) carries its own license; see
[its model page](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
