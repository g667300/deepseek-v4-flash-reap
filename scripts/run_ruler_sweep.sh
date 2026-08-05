#!/usr/bin/env bash
# Run RULER once per context length, sequentially, and check that each length
# actually produced numbers.
#
#   run_ruler_sweep.sh 4096 16384 32768 65536
#
# Environment: BASE, MODEL, TOKENIZER, OUT, LIMIT, CONCURRENT, TIMEOUT, MAXLEN.
#
# Why not one invocation with several lengths: `--metadata max_seq_lengths`
# builds samples for every length and concatenates them into one dataset, and
# `--limit N` then takes the first N documents -- which are all the shortest
# length. A four-length run with --limit 10 silently evaluates 4096 only.
#
# `ruler_qa_hotpot` is left out on purpose: lm-eval hardcodes
# http://curtis.ml.cmu.edu/datasets/hotpot/... which no longer resolves, and
# its failure aborts the whole run.
#
# Two guards, both paid for in wasted hours:
#
# * **The served context must cover the length under test.** A length larger
#   than the served window drops every document and reports -1. Equal is fine:
#   65536 measures correctly against --max-model-len 65536, because RULER sizes
#   its contexts to fit the budget rather than overrun it.
#   Beware the sibling trap when reading results by eye: every successful run
#   also carries a `4096,none` entry of -1 for the task default it did not
#   evaluate. That is normal. Only the key for the length under test means
#   anything, which is what the check below reads.
# * **Long context is a different regime.** lm_eval's timeout is a total per
#   request including queue wait. At 32K this engine runs 6-7 requests at once
#   (bandwidth-bound; KV cache sits at 25%) and generates a few tokens/s, so
#   sixteen in flight leaves ten queued past the 300 s default and the client
#   kills a run the server is still working through.
set -uo pipefail

VENV=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin
BASE="${BASE:-http://localhost:8000/v1/completions}"
MODEL="${MODEL:-dsv4-reap50}"
TOKENIZER="${TOKENIZER:-models/DeepSeek-V4-Flash-0731}"
OUT="${OUT:-artifacts/ruler-${MODEL}}"
LIMIT="${LIMIT:-10}"
CONCURRENT="${CONCURRENT:-8}"
TIMEOUT="${TIMEOUT:-1800}"
HEARTBEAT="${HEARTBEAT:-120}"   # seconds between progress lines in the main log
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad
N_TASKS=12

[[ $# -ge 1 ]] || { echo "usage: $0 <length> [<length>...]" >&2; exit 2; }

api="${BASE%/completions}"
served_len=$(curl -sf -m 15 "${api}/models" \
  | "$VENV/python" -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len", 0))' 2>/dev/null || echo 0)
if [[ "${served_len:-0}" -gt 0 ]]; then
  echo "server reports max_model_len=${served_len}"
else
  echo "warning: could not read max_model_len from ${api}/models; skipping the fit check" >&2
fi

rc_all=0
for LEN in "$@"; do
  if [[ "${served_len:-0}" -gt 0 && "${served_len}" -lt "${LEN}" ]]; then
    echo "SKIP ${LEN}: the model is served with a ${served_len}-token window." >&2
    echo "     Restart it with --max-model-len >= ${LEN}; every task would report -1." >&2
    rc_all=1
    continue
  fi
  # Give lm_eval the served window, not the length under test: it is the budget
  # the server will actually honour.
  maxlen="${MAXLEN:-${served_len:-65536}}"
  [[ "${maxlen}" -ge "${LEN}" ]] || maxlen="${LEN}"

  echo "############ RULER @ ${LEN} (limit ${LIMIT}, ${CONCURRENT} concurrent, ${TIMEOUT}s) started $(date +%H:%M:%S)"

  # Each length gets its own raw log, and this loop prints a heartbeat into the
  # main one. lm_eval draws tqdm bars with carriage returns on stderr, so a
  # `tail` of the raw log looks frozen even when the run is healthy, and the
  # GPU idles for ten minutes while samples are generated locally -- neither
  # silence nor an idle GPU means the job died. The heartbeat says so directly.
  mkdir -p "$OUT"
  logf="${OUT}/ruler-${LEN}.log"
  "$VENV/lm_eval" run --model local-completions \
    --model_args "model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${CONCURRENT},max_retries=3,max_length=${maxlen},timeout=${TIMEOUT}" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --metadata "{\"max_seq_lengths\":[${LEN}]}" \
    --output_path "$OUT" > "$logf" 2>&1 &
  pid=$!
  start=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT"
    kill -0 "$pid" 2>/dev/null || break
    last=$(tr '\r' '\n' < "$logf" 2>/dev/null | grep -vE '^[[:space:]]*$' | tail -1)
    printf '  [%s] %s running, %ds elapsed | %.110s\n' \
      "$(date +%H:%M:%S)" "$LEN" "$((SECONDS - start))" "${last:-(no output yet)}"
  done
  wait "$pid"
  rc=$?
  echo "############ RULER @ ${LEN} finished rc=${rc} at $(date +%H:%M:%S) (log: ${logf})"
  [[ $rc -eq 0 ]] || { rc_all=1; continue; }

  # An exit code of 0 is not evidence of a measurement. Read the results back.
  latest=$(find "$OUT" -name 'results_*.json' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [[ -z "$latest" ]]; then
    echo "  no results file under ${OUT}" >&2; rc_all=1; continue
  fi
  "$VENV/python" - "$latest" "$LEN" "$N_TASKS" <<'PY' || rc_all=1
import json, sys
path, length, n_expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
res = json.load(open(path))["results"]
key = f"{length},none"
vals, missing, bad = {}, [], []
for name, r in res.items():
    if name.startswith(("niah_", "ruler_")):
        if key not in r:
            missing.append(name)
        elif r[key] < 0:
            bad.append(name)
        else:
            vals[name] = r[key]
if missing:
    sys.exit(f"  FAILED: {len(missing)} tasks have no '{key}' metric "
             f"(the run did not evaluate at this length): {', '.join(sorted(missing)[:4])} ...")
if bad:
    sys.exit(f"  FAILED: {len(bad)} tasks reported a negative score, which means every "
             f"document was dropped: {', '.join(sorted(bad)[:4])} ...")
if len(vals) < n_expected:
    sys.exit(f"  FAILED: only {len(vals)} of {n_expected} tasks reported")
worst = min(vals, key=vals.get)
print(f"  {length}: {len(vals)}-task mean {sum(vals.values())/len(vals)*100:.2f}"
      f"   perfect {sum(1 for v in vals.values() if v == 1.0)}/{len(vals)}"
      f"   worst {worst} {vals[worst]:.3f}")
PY
done

echo "############ sweep complete at $(date +%H:%M:%S) (rc=${rc_all})"
exit "$rc_all"
