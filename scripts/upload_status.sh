#!/usr/bin/env bash
# Progress, rate and ETA for a running `hf upload`.
#
#   upload_status.sh [logfile] [staging-dir]
#
# Defaults to artifacts/hf-upload.log and the directory the log says it is
# uploading. Add a number to sample twice and report the current rate as well
# as the average:
#
#   upload_status.sh                 # one shot
#   upload_status.sh '' '' 30        # also measure the rate over 30 seconds
#
# `hf upload` prints lines like
#
#   Uploading... 77/77 files checked, 9/48 uploaded (798MB transferred), 0 committed in 0 commit(s)
#
# which say nothing about how far along that is, and carry no timestamps. This
# fills both gaps: the total comes from the staging directory, the elapsed time
# from the process itself.
set -uo pipefail

LOG="${1:-artifacts/hf-upload.log}"
DIR="${2:-}"
SAMPLE="${3:-0}"

[[ -f "$LOG" ]] || { echo "no such log: $LOG" >&2; exit 2; }

# The upload process, so elapsed time is measured rather than guessed.
pid=$(pgrep -f "hf upload" | head -1 || true)
elapsed=0
[[ -n "$pid" ]] && elapsed=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')

# Total to send: the staging directory, taken from the command line if not given.
if [[ -z "$DIR" && -n "$pid" ]]; then
  DIR=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i ~ /\//  && $i !~ /bin|^-/) print $i}' | tail -2 | head -1)
fi
total=0
[[ -n "$DIR" && -d "$DIR" ]] && total=$(du -sb "$DIR" 2>/dev/null | cut -f1)

read_bytes() {
  # Last "(NNN<unit> transferred)" in the log, as bytes.
  tr '\r' '\n' < "$LOG" | grep -oE '\([0-9.]+[KMGT]?B transferred\)' | tail -1 \
    | grep -oE '[0-9.]+[KMGT]?B' \
    | awk '{ n=$0; sub(/[KMGT]?B$/,"",n); u=$0; sub(/^[0-9.]+/,"",u);
             m = (u=="KB")?1e3:(u=="MB")?1e6:(u=="GB")?1e9:(u=="TB")?1e12:1;
             printf "%.0f", n*m }'
}

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1}B"; }
hms() { printf '%dh%02dm' $(( ${1:-0} / 3600 )) $(( (${1:-0} % 3600) / 60 )); }

sent=$(read_bytes); sent=${sent:-0}
files=$(tr '\r' '\n' < "$LOG" | grep -oE '[0-9]+/[0-9]+ uploaded' | tail -1)
commits=$(tr '\r' '\n' < "$LOG" | grep -oE 'in [0-9]+ commit\(s\)' | tail -1 | grep -oE '[0-9]+')

echo "files    : ${files:-unknown}"
echo "sent     : $(human "$sent")$( [[ "$total" -gt 0 ]] && printf ' of %s (%.1f%%)' "$(human "$total")" "$(awk -v s="$sent" -v t="$total" 'BEGIN{print s*100/t}')" )"
[[ "$elapsed" -gt 0 ]] && echo "elapsed  : $(hms "$elapsed")"

if [[ "$elapsed" -gt 0 && "$sent" -gt 0 ]]; then
  avg=$(awk -v s="$sent" -v e="$elapsed" 'BEGIN{printf "%.0f", s/e}')
  echo "average  : $(human "$avg")/s"
  if [[ "$total" -gt 0 ]]; then
    left=$(( total - sent ))
    [[ "$left" -lt 0 ]] && left=0
    echo "eta (avg): $(hms "$(awk -v l="$left" -v r="$avg" 'BEGIN{printf "%.0f", (r>0)? l/r : 0}')")"
  fi
fi

if [[ "${SAMPLE:-0}" -gt 0 ]]; then
  sleep "$SAMPLE"
  now=$(read_bytes); now=${now:-0}
  rate=$(awk -v a="$sent" -v b="$now" -v t="$SAMPLE" 'BEGIN{printf "%.0f", (b-a)/t}')
  echo "current  : $(human "$rate")/s (over ${SAMPLE}s)"
  if [[ "$total" -gt 0 && "$rate" -gt 0 ]]; then
    echo "eta (now): $(hms "$(awk -v l="$(( total - now ))" -v r="$rate" 'BEGIN{printf "%.0f", l/r}')")"
  fi
fi

# Success is not visible in the progress lines: they end on "0 committed in 0
# commit(s)" and the result is a separate "Uploaded" line carrying the commit
# URL. Checking only the counter reports a finished upload as a failed one.
done_url=$(tr '\r' '\n' < "$LOG" | grep -oE 'https://huggingface\.co/[^ ]+/commit/[0-9a-f]+' | tail -1)

if [[ -n "$done_url" ]]; then
  echo "state    : finished"
  echo "commit   : ${done_url}"
elif [[ "${commits:-0}" -gt 0 ]]; then
  echo "state    : committed"
elif [[ -z "$pid" ]]; then
  echo "state    : no hf upload process running and no commit in the log -- it stopped early"
else
  echo "state    : uploading (pid ${pid})"
fi
