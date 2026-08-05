#!/usr/bin/env bash
# Global-MMLU across every language it ships, in one lm_eval invocation.
#
# The point is not the absolute scores but the spread: calibration was
# ja 35% / en 35% / code 26% / zh 4% and nothing else, so the eleven languages
# never seen by the scoring pass show what REAP actually removed. A flat profile across
# all fifteen would mean the pruning was not selective enough to be worth the
# targeting -- deeper sparsity would then be on the table.
#
# All languages go in one invocation so they share the harness startup and the
# HTTP connection pool; lm_eval reports each as its own row.
set -u

VENV=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin
BASE="${BASE:-http://localhost:8000/v1/completions}"
MODEL="${MODEL:-dsv4-reap50}"
TOKENIZER="${TOKENIZER:-models/DeepSeek-V4-Flash-0731}"
CONCURRENT="${CONCURRENT:-16}"
OUT="${OUT:-artifacts/global-mmlu-dsv4}"

LANGS="ar bn de en es fr hi id it ja ko pt sw yo zh"
TASKS=$(for l in $LANGS; do printf "global_mmlu_%s," "$l"; done | sed 's/,$//')

echo "############ global_mmlu: $LANGS"
echo "############ concurrency ${CONCURRENT}, started $(date +%H:%M:%S)"
"$VENV/lm_eval" run --model local-completions \
  --model_args "model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${CONCURRENT},max_retries=3,max_length=65536" \
  --tasks "$TASKS" \
  --output_path "$OUT"
echo "############ finished rc=$? at $(date +%H:%M:%S)"
