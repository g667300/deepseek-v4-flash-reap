#!/usr/bin/env bash
# Assemble a Hugging Face upload directory from a pruned checkpoint.
#
#   make_hf_upload.sh <src-ckpt-dir> <dest-dir> [results.md]
#
# e.g. make_hf_upload.sh artifacts/dsv4-reap50 artifacts/hf/DeepSeek-V4-Flash-REAP-50
#
# The weights are hardlinked, not copied, so a second staging directory costs
# no disk as long as it lands on the same filesystem (it falls back to copying
# if it does not). Editing a file in the upload directory would therefore edit
# the checkpoint too -- everything this script rewrites (the model card) is
# created fresh rather than modified in place.
#
# The optional third argument is a markdown fragment dropped into the card
# under "Evaluation"; without it the card carries a placeholder, because
# shipping a model card with unfilled numbers is worse than shipping none.
#
# What ends up in the directory:
#   * safetensors shards + index      hardlinked
#   * config / tokenizer / generation / LICENSE / encoding / inference  copied
#   * reap_pruning.json               copied -- which experts survived, per layer
#   * README.md                       generated here, NOT the base model card
#   * .gitattributes                  LFS patterns, so shards upload as LFS
set -euo pipefail

SRC="${1:?usage: $0 <src-ckpt-dir> <dest-dir> [results.md]}"
DST="${2:?usage: $0 <src-ckpt-dir> <dest-dir> [results.md]}"
RESULTS="${3:-}"

BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
LICENSE_ID="${LICENSE_ID:-mit}"
PIPELINE="${PIPELINE:-text-generation}"
# Where the card points readers for the pipeline. Set it to empty to drop the
# section entirely, which is the right thing for a fork with nowhere to point.
REPO_URL="${REPO_URL-https://github.com/g667300/deepseek-v4-flash-reap}"
# The card title. Defaults to the directory name, but the two are not the same
# thing: the upload target is a repo id given to `hf upload` at push time, and
# the staging directory is only where the files sit. Set MODEL_NAME to the
# repo's model name so the card reads correctly without renaming anything.
MODEL_NAME="${MODEL_NAME:-$(basename "${DST%/}")}"

[[ -d "$SRC" ]] || { echo "not a directory: $SRC" >&2; exit 2; }
[[ -f "$SRC/reap_pruning.json" ]] || { echo "no reap_pruning.json in $SRC -- is this a surgery-pass output?" >&2; exit 2; }

mkdir -p "$DST"

echo "== linking weights"
for f in "$SRC"/*.safetensors "$SRC"/model.safetensors.index.json; do
  [[ -e "$f" ]] || continue
  ln -f "$f" "$DST/$(basename "$f")" 2>/dev/null || cp -f "$f" "$DST/$(basename "$f")"
done

echo "== copying config and tokenizer"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         LICENSE reap_pruning.json; do
  [[ -e "$SRC/$f" ]] && cp -f "$SRC/$f" "$DST/$f"
done
for d in encoding inference; do
  [[ -d "$SRC/$d" ]] && cp -rf "$SRC/$d" "$DST/"
done

cat > "$DST/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
EOF

# --- facts for the card, read from the checkpoint rather than retyped --------
read -r N_ORIG N_KEEP SPARSITY MTP HASH TOTAL_GB N_LAYERS N_HASH <<<"$(
python3 - "$SRC" <<'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
p = json.loads((src / "reap_pruning.json").read_text())
cfg = json.loads((src / "config.json").read_text())
idx = json.loads((src / "model.safetensors.index.json").read_text())
total = idx.get("metadata", {}).get("total_size", 0) / 1e9
# The count of hash-routed layers is a property of the architecture, so read it
# rather than assume: remapped_blocks is what the surgery pass actually rewrote.
n_hash = cfg.get("num_hash_layers")
if n_hash is None:
    n_hash = len(p.get("remapped_blocks", []))
print(p["num_experts_original"], p["num_experts_retained"],
      f'{p["sparsity"]*100:g}', p["mtp"], p["hash_layers"],
      f"{total:.1f}", cfg.get("num_hidden_layers", "?"), n_hash)
PY
)"

echo "== writing model card ($N_ORIG -> $N_KEEP experts, ${TOTAL_GB} GB)"
{
  cat <<EOF
---
license: ${LICENSE_ID}
base_model: ${BASE_MODEL}
library_name: transformers
pipeline_tag: ${PIPELINE}
tags:
  - moe
  - pruning
  - reap
---

# ${MODEL_NAME}

[${BASE_MODEL}](https://huggingface.co/${BASE_MODEL}) with its routed experts
pruned from **${N_ORIG} to ${N_KEEP} per layer** (${SPARSITY}% sparsity) by REAP
(Router-weighted Expert Activation Pruning), leaving **${TOTAL_GB} GB** across
${N_LAYERS} layers.

The point of the exercise is to fit a single 128 GB unified-memory host with
room left over to serve, which the original does not.

## What was changed

REAP scores each expert by how much the router actually leans on it over a
calibration set, then drops the lowest scorers. **It does not modify the weights
of the experts that survive**, so this checkpoint keeps the original
quantization exactly: FP4 for routed experts, FP8 elsewhere. Nothing was
requantized, retrained, or distilled.

Three things are worth knowing before you use it:

* **The first ${N_HASH} layers route by a frozen token-id table, not by a
  learned score.** \`n_routed_experts\` is one value for the whole model, so
  pruning any layer forces those to the same count. Their table was rewritten to
  send each dropped expert's token ids to the surviving expert whose router row
  points in the most similar direction, balanced so no survivor absorbs a
  disproportionate share. That is a merge, not a pure prune.
* **The MTP (multi-token prediction) blocks were ${MTP}ped.** Speculative
  decoding through them is not available on this checkpoint.
* **Hash-layer handling: \`${HASH}\`.** See \`reap_pruning.json\` for the exact
  surviving expert ids per layer.

## Evaluation
EOF

  if [[ -n "$RESULTS" && -f "$RESULTS" ]]; then
    echo
    cat "$RESULTS"
  else
    cat <<'EOF'

<!-- TODO: paste the measured table here before publishing. -->

Not filled in yet.
EOF
  fi

  cat <<EOF

**There is no unpruned control.** The original does not fit the evaluation host,
so these are absolute health checks, not a measured degradation against the
model this came from.

## Calibration

Which experts survive is decided entirely by the calibration mix. This one was
weighted toward Japanese, English and code; a checkpoint aimed at other
languages or domains needs its own scoring pass rather than this file.
EOF

  if [[ -n "$REPO_URL" ]]; then
    cat <<EOF

## Reproducing

Pipeline, scripts and the reasoning behind each choice:
<${REPO_URL}>
EOF
  fi

  # Append the upstream card rather than dropping it. This model ships no HF
  # chat template, so its "Chat Template" section is the only documentation of
  # the prompt format -- losing it would leave users guessing. Its own YAML
  # frontmatter is stripped: a second --- block mid-file would render as a
  # horizontal rule and a stray table.
  if [[ -f "$SRC/README.md" ]]; then
    echo
    echo "---"
    echo
    echo "# Original model card: ${BASE_MODEL}"
    echo
    awk 'BEGIN{fm=0} NR==1 && $0=="---" {fm=1; next} fm==1 && $0=="---" {fm=0; next} fm==0 {print}' \
      "$SRC/README.md"
  fi
} > "$DST/README.md"

echo "== done: $DST"
du -sh "$DST"
echo "   (weights are hardlinks -- 'du' counts them, the disk does not)"
