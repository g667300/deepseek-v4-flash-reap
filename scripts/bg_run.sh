#!/usr/bin/env bash
# Two modes: launch a long job in the background, or watch its log.
#
# Launch: scripts/bg_run.sh <log-name> -- <command...>
#   Runs the command under nohup+disown, writing to artifacts/<log-name>.log,
#   prints the PID and returns immediately.
#   e.g. scripts/bg_run.sh eval-foo -- .venv/bin/lm_eval run --model local-completions ...
#
# Watch: scripts/bg_run.sh watch <logfile> [interval_sec=180] [success_regex]
#   Polls the log at the given interval and prints one progress line per poll
#   (handling tqdm's \r-separated output). Exits 0 when success_regex matches,
#   exits 1 on the usual failure signatures (Traceback / Error / CUDA out of
#   memory / Killed / Segmentation fault). One stdout line per event, so it can
#   drive an external notifier.
#   e.g. scripts/bg_run.sh watch artifacts/eval-foo.log 180 'jcommonsenseqa_local.*none'
set -euo pipefail

mode="${1:-}"

if [[ "$mode" == "watch" ]]; then
  LOG="${2:?usage: $0 watch <logfile> [interval_sec] [success_regex]}"
  INTERVAL="${3:-180}"
  SUCCESS_RE="${4:-}"
  ERROR_RE='Traceback \(most recent|^Error:|Exception:|CUDA out of memory|Killed|Segmentation fault'

  while true; do
    if [[ -f "$LOG" ]]; then
      if [[ -n "$SUCCESS_RE" ]] && grep -qE "$SUCCESS_RE" "$LOG"; then
        echo "=== success ==="
        tail -c 3000 "$LOG" | tr '\r' '\n' | tail -20
        exit 0
      fi
      if grep -qE "$ERROR_RE" "$LOG"; then
        echo "=== failure ==="
        tail -c 3000 "$LOG" | tr '\r' '\n' | tail -30
        exit 1
      fi
      snapshot=$(tr '\r' '\n' < "$LOG" | grep -v '^[[:space:]]*$' | tail -1)
      echo "progress: ${snapshot:-(no output yet)}"
    else
      echo "log not created yet: $LOG"
    fi
    sleep "$INTERVAL"
  done
fi

NAME="${1:?usage: $0 <log-name> -- <command...>  |  $0 watch <logfile> [interval] [success_regex]}"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
[[ $# -ge 1 ]] || { echo "usage: $0 <log-name> -- <command...>" >&2; exit 1; }

mkdir -p artifacts
LOG="artifacts/${NAME}.log"
rm -f "$LOG"

nohup "$@" > "$LOG" 2>&1 &
pid=$!
disown

echo "started PID ${pid} -> ${LOG}"
