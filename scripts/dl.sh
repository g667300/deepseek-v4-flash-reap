#!/usr/bin/env bash
# Unthrottled HF Hub download control (via the hf CLI). start/stop are safe to
# repeat -- progress resumes from .incomplete files. REPO and DEST are required
# positional arguments:
#
#   ./scripts/dl.sh <repo> <dest> start
#
#   ./scripts/dl.sh org/model /path/to/dest start           # start in background (log: download_<dest>.log)
#   ./scripts/dl.sh org/model /path/to/dest stop            # interrupt now (progress is kept)
#   ./scripts/dl.sh org/model /path/to/dest finish-and-stop # SIGINT, then wait for the hf CLI to clean up
#                                                             # (no file-boundary guarantee, just gentler than stop)
#   ./scripts/dl.sh org/model /path/to/dest status          # progress, rate, ETA
#   ./scripts/dl.sh org/model /path/to/dest watch           # progress every 10s (Ctrl-C exits, download continues)
#
# Paths resolve relative to the script's own location, so it behaves the same
# whether invoked as ./scripts/dl.sh or /path/to/scripts/dl.sh.
#
# The manifest (used for the size total) and the log are keyed on DEST's
# directory name, so concurrent downloads of different models do not collide.
# dl_ratelimited.sh and dl_one_file.sh share the same naming convention.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: $0 <repo> <dest> {start|stop|finish-and-stop|status|watch}" >&2
    echo "  e.g. $0 org/other-model /path/to/dest start" >&2
    exit 1
}

[ $# -ge 3 ] || usage
REPO="$1"; DEST="$2"; ACTION="$3"; shift 3
SLUG="$(basename "$DEST")"
LOG="$HERE/download_${SLUG}.log"
MANIFEST="$HERE/.dl_manifest_${SLUG}.tsv"
VENV_PY="$HERE/.venv/bin/python"

# hf CLI: prefer the project's .venv, fall back to whatever is on PATH
if [ -x "$HERE/.venv/bin/hf" ]; then
    HF_BIN="$HERE/.venv/bin/hf"
else
    HF_BIN="$(command -v hf)" || { echo "hf command not found (neither .venv/bin/hf nor on PATH)" >&2; exit 1; }
fi

ensure_manifest() {
    if [ ! -s "$MANIFEST" ]; then
        echo "fetching manifest..." >&2
        "$VENV_PY" -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('$REPO', files_metadata=True)
for f in info.siblings:
    print(f'{f.rfilename}\t{f.size or 0}')
" > "$MANIFEST"
    fi
}

total_bytes() { awk -F'\t' '{s+=$2} END{print s+0}' "$MANIFEST"; }

pid_of() { pgrep -f "bin/hf download $REPO" | head -1; }

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }

got_bytes() { du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0; }

case "$ACTION" in

start)
    if [ -n "$(pid_of)" ]; then echo "already running (PID $(pid_of))"; exit 0; fi
    HF_XET_HIGH_PERFORMANCE=1 nohup "$HF_BIN" download "$REPO" \
        --local-dir "$DEST" --max-workers 16 >"$LOG" 2>&1 &
    sleep 2
    echo "started (PID $(pid_of))  repo: $REPO  dest: $DEST  log: $LOG"
    ;;

stop)
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "no running process"; exit 0; fi
    kill "$p"; sleep 3
    [ -n "$(pid_of)" ] && { kill -9 "$(pid_of)"; sleep 2; }
    echo "stopped. fetched so far: $(human "$(got_bytes)")  (start resumes)"
    ;;

finish-and-stop)
    # The hf CLI is an external tool, so unlike dl_ratelimited.sh there is no
    # signal handler to make it "finish the current file only". With
    # --max-workers 16 several files are in flight anyway, so a clean "next
    # file boundary" does not really exist. All this can do is send SIGINT
    # (which most Python CLIs handle more gracefully than SIGTERM) and wait for
    # the hf CLI's own cleanup. No guarantee about file boundaries -- it is
    # simply gentler than an immediate kill.
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "no running process"; exit 0; fi
    echo "sent SIGINT, waiting for the hf CLI to clean up (up to 2 min)..."
    kill -INT "$p" 2>/dev/null
    for _ in $(seq 1 24); do
        kill -0 "$p" 2>/dev/null || break
        sleep 5
    done
    if kill -0 "$p" 2>/dev/null; then
        echo "still alive after 2 min, forcing"
        kill -9 "$p" 2>/dev/null
        sleep 2
    fi
    echo "stopped. fetched so far: $(human "$(got_bytes)")  (start resumes)"
    ;;

status)
    ensure_manifest
    t=$(total_bytes)
    g=$(got_bytes)
    pct=$(awk -v g="$g" -v t="$t" 'BEGIN{printf "%.1f", g*100/t}')
    echo "fetched  : $(human "$g") / $(human "$t")  (${pct}%)"
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "state    : stopped"; exit 0; fi
    echo "state    : running (PID $p)"
    sleep 20
    g2=$(got_bytes)
    rate=$(( (g2 - g) / 20 ))
    if [ "$rate" -le 0 ]; then
        echo "rate     : unmeasurable (0 B/s -- verifying chunks, or stalled)"
    else
        eta=$(( (t - g2) / rate ))
        printf "rate     : %s/s\nETA      : %dh%dm  (done around %s)\n" \
            "$(human "$rate")" $((eta/3600)) $(((eta%3600)/60)) \
            "$(date -d "+${eta} seconds" '+%m/%d %H:%M')"
    fi
    ;;

watch)
    # Tailing the hf CLI's own log looks fine but its output is block-buffered
    # once redirected to a file, so updates appear to freeze when they have
    # not. Diffing du's actual size is less misleading, so compute it here
    # every 10 seconds instead.
    ensure_manifest
    t=$(total_bytes)
    prev_g=$(got_bytes)
    prev_t=$(date +%s)
    while true; do
        sleep 10
        g=$(got_bytes)
        now_t=$(date +%s)
        pct=$(awk -v g="$g" -v tot="$t" 'BEGIN{printf "%.1f", g*100/tot}')
        dt=$((now_t - prev_t)); db=$((g - prev_g))
        if [ "$dt" -gt 0 ] && [ "$db" -gt 0 ]; then
            rate=$((db / dt))
            eta=$(( (t - g) / rate ))
            extra="$(human "$rate")/s  ETA $((eta/3600))h$(((eta%3600)/60))m"
        else
            extra="measuring (may be waiting on reconstruction)"
        fi
        printf "\r%s  %s / %s (%s%%)  %s  %s          \n" "$(date '+%H:%M:%S')" \
            "$(human "$g")" "$(human "$t")" "$pct" \
            "$([ -n "$(pid_of)" ] && echo running || echo STOPPED)" "$extra"
        prev_g=$g; prev_t=$now_t
    done
    ;;

*)
    usage
    ;;
esac
