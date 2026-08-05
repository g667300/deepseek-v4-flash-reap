#!/usr/bin/env bash
# The whole evaluation battery against one served model, in the order that
# yields information fastest: the cheap language checks first, the multi-hour
# RULER sweep last, so a run cut short still leaves the comparison table filled.
#
# Usage:
#   MODEL=dsv4-reap50 TOKENIZER=models/DeepSeek-V4-Flash-0731 TAG=dsv4-50 \
#     scripts/run_eval_suite.sh
#
# Skip stages with SKIP="ruler ppl" (names: jcqa mmlu multilingual ppl ruler).
# JCommonsenseQA needs --include_path eval_tasks; global_mmlu, mmlu and RULER
# are stock lm_eval tasks.
set -u

VENV=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin
BASE="${BASE:-http://localhost:8000/v1/completions}"
MODEL="${MODEL:?set MODEL to the served model name}"
TOKENIZER="${TOKENIZER:?set TOKENIZER to a local tokenizer path}"
TAG="${TAG:?set TAG, used in output filenames}"
CONCURRENT="${CONCURRENT:-16}"
TIMEOUT="${TIMEOUT:-300}"
SKIP="${SKIP:-}"
MAXLEN="${MAXLEN:-65536}"

# PPL_DATA has no default on purpose. It used to fall back to a file built with
# a different model's tokenizer, and the only symptom was a bare 400 Bad
# Request from vLLM several stages into a multi-hour run.
PPL_DATA="${PPL_DATA:?set PPL_DATA to held-out ids built with the tokenizer of this model}"

# Long context is a different regime, not just a bigger number. lm_eval's
# timeout is a *total* per request, queue wait included, and at 32K this engine
# runs only 6-7 requests at once (bandwidth, not KV cache -- usage sits at 25%)
# while generating 2-4 tokens/s in aggregate. Sixteen in flight therefore leaves
# ten queued past the 300 s default, and the client kills a run the server is
# still working through. Fewer in flight, and far longer to wait.
RULER_CONCURRENT="${RULER_CONCURRENT:-8}"
RULER_TIMEOUT="${RULER_TIMEOUT:-1800}"

ARGS="model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${CONCURRENT},max_retries=3,max_length=${MAXLEN},timeout=${TIMEOUT}"
RULER_ARGS="model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${RULER_CONCURRENT},max_retries=3,max_length=${MAXLEN},timeout=${RULER_TIMEOUT}"

run_stage() {
  local name="$1"; shift
  case " $SKIP " in *" $name "*) echo "#### skip ${name}"; return;; esac
  echo "#### ${name} started $(date +%H:%M:%S)"
  "$@"
  echo "#### ${name} finished rc=$? at $(date +%H:%M:%S)"
}

run_stage jcqa "$VENV/lm_eval" run --model local-completions \
  --model_args "$ARGS" --tasks jcommonsenseqa_local \
  --include_path eval_tasks --output_path "artifacts/eval-${TAG}"

run_stage mmlu "$VENV/lm_eval" run --model local-completions \
  --model_args "$ARGS" --tasks mmlu --limit 10 \
  --output_path "artifacts/eval-${TAG}"

LANGS="ar bn de en es fr hi id it ja ko pt sw yo zh"
ML=$(for l in $LANGS; do printf "global_mmlu_%s," "$l"; done | sed 's/,$//')
run_stage multilingual "$VENV/lm_eval" run --model local-completions \
  --model_args "$ARGS" --tasks "$ML" --output_path "artifacts/eval-${TAG}"

run_stage ppl "$VENV/python" -u scripts/eval_perplexity.py \
  --data "${PPL_DATA}" \
  --base-url "${BASE%/completions}" --model "$MODEL" \
  --out "artifacts/ppl-${TAG}-result.json"

# One length per invocation, always: passing several to max_seq_lengths together
# with --limit takes the first N of the concatenated set, so only the shortest
# length actually gets evaluated. RULER_LENGTHS narrows the sweep when some
# lengths are already measured and only the rest need redoing.
for LEN in ${RULER_LENGTHS:-4096 16384 32768 65536}; do
  run_stage ruler "$VENV/lm_eval" run --model local-completions \
    --model_args "$RULER_ARGS" \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad \
    --limit 10 --metadata "{\"max_seq_lengths\":[${LEN}]}" \
    --output_path "artifacts/ruler-${TAG}"
done

echo "#### suite complete at $(date +%H:%M:%S)"
