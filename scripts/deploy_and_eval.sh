#!/usr/bin/env bash
# The whole path from a finished surgery pass to a filled-in results table:
# push the checkpoint, bring vLLM up on the remote host, then run the suite.
#
#   TOKENIZER=models/DeepSeek-V4-Flash-0731 \
#   PPL_DATA=artifacts/ppl-holdout-dsv4.pt \
#   TAG=dsv4-50 \
#     scripts/deploy_and_eval.sh artifacts/dsv4-reap50 spark:/srv/models/dsv4-reap50
#
# Required environment:
#   TOKENIZER  local tokenizer path (lm_eval reads the model name as an HF
#              repo id without it)
#   PPL_DATA   held-out token ids built with *this model's* tokenizer
#   TAG        goes into the output filenames
#
# Everything in serve_remote.sh (NAME, IMAGE, EXTRA, ...) and
# run_eval_suite.sh (SKIP, CONCURRENT, ...) is honoured and passed through.
# SKIP_PUSH=1 skips the transfer, SKIP_SERVE=1 evaluates whatever is already
# serving.
#
# The tokenizer check below is not ceremony. The suite defaults PPL_DATA to
# the file built for a different model, and the only symptom is a bare
# "400 Bad Request" from vLLM several stages into a multi-hour run.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

SRC="${1:?usage: $0 <local-ckpt-dir> <[user@]host:/remote/dir>}"
DST="${2:?usage: $0 <local-ckpt-dir> <[user@]host:/remote/dir>}"

: "${TOKENIZER:?set TOKENIZER to a local tokenizer path}"
# No apostrophes in the :? messages -- bash parses that text for quotes, so a
# lone ' there swallows everything up to the next one (it ate the heredoc below).
: "${PPL_DATA:?set PPL_DATA to held-out ids built with the tokenizer of this model}"
: "${TAG:?set TAG, used in output filenames}"

host="${DST%%:*}"
remote_dir="${DST#*:}"
NAME="${NAME:-$(basename "${remote_dir%/}")}"
PORT="${PORT:-8000}"

# --- the check that would have saved this run --------------------------------
"$ROOT/.venv/bin/python" - "$SRC" "$PPL_DATA" <<'PY'
import json, sys, torch
from pathlib import Path

ckpt, data = Path(sys.argv[1]), Path(sys.argv[2])
vocab = json.loads((ckpt / "config.json").read_text())["vocab_size"]
seqs = torch.load(data, weights_only=True)
hi = max(int(max(s)) for s in seqs)
if hi >= vocab:
    sys.exit(
        f"{data} holds token id {hi}, but {ckpt.name} has vocab_size {vocab}.\n"
        "That file was tokenized for a different model -- rebuild it with "
        "build_calibration.py against this checkpoint's tokenizer."
    )
print(f"ppl data ok: {len(seqs)} sequences, max token id {hi} < vocab {vocab}")
PY

if [[ "${SKIP_SERVE:-0}" != "1" ]]; then
  "$HERE/serve_remote.sh" "$SRC" "$DST"
fi

export MODEL="$NAME" TOKENIZER TAG PPL_DATA
export BASE="${BASE:-http://${host#*@}:${PORT}/v1/completions}"

echo "== evaluating ${MODEL} at ${BASE}"
exec bash "$HERE/run_eval_suite.sh"
