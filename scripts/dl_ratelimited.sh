#!/usr/bin/env bash
# Fetch an HF Hub repo with a bandwidth cap, bypassing the hf CLI (hf_xet).
# An independent download path from dl.sh. REPO and DEST are required
# positional arguments:
#
#   ./scripts/dl_ratelimited.sh org/other-model /path/to/dest start 4000000
#
# Why a separate path is needed:
#   The hf CLI uses hf_xet (a Rust high-performance transfer layer) internally.
#   Whether userspace shapers like trickle can reliably throttle that is
#   untested. Kernel-level tc/ifb shaping is not an option here -- it hard-hung
#   the machine this was developed on.
#
# Why hand-rolled Python pacing instead of curl --limit-rate:
#   This repo's safetensors shards all 302-redirect to the Xet CDN bridge
#   (us.aws.cdn.hf.co/xet-bridge-us/...). curl's --limit-rate stalls the
#   transfer to 0 B/s against that CDN regardless of the value -- even an
#   effectively unlimited 50MB/s. Without a rate limit it streams fine, and
#   rate-limiting the repo's non-Xet metadata files is fine too. The symptom
#   points at a keep-alive interaction between curl and that CDN; the root
#   cause was never pinned down. scripts/dl_ratelimited_core.py works around it
#   by streaming with requests and sleeping between chunks (token bucket).
#
#   Serial by design: on a link whose upstream is the bottleneck, parallelism
#   does not raise total throughput, and serial means the requested rate is the
#   total rate.
#
# Units: the rate is always bytes per second (no K/M/G suffixes as in curl, no
#   mbit as in tc). For 32 Mbit/s use 32_000_000 / 8 = 4000000.
#
# Bytes already staged by `hf download` (dl.sh) under
# .cache/huggingface/download/*.incomplete are in Xet's own chunk format and
# cannot be reused here. Switching tools restarts any partial file; files
# already complete are skipped on a size match.
#
#   ./scripts/dl_ratelimited.sh <repo> <dest> start 4000000   # start/resume at 4,000,000 B/s (32 Mbit/s)
#   ./scripts/dl_ratelimited.sh <repo> <dest> stop            # interrupt now (this script resumes later)
#   ./scripts/dl_ratelimited.sh <repo> <dest> finish-and-stop # finish the in-flight file, then stop
#                                                               # (use this when handing off to dl.sh -- stopping
#                                                               #  on a file boundary avoids double fetches)
#   ./scripts/dl_ratelimited.sh <repo> <dest> cancel-finish   # cancel a pending finish-and-stop and keep going
#   ./scripts/dl_ratelimited.sh <repo> <dest> status          # progress (shows a pending stop, if any)
#   ./scripts/dl_ratelimited.sh <repo> <dest> watch           # progress every 10s (Ctrl-C exits)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: $0 <repo> <dest> {start <rate_bytes_per_sec>|stop|finish-and-stop|cancel-finish|status|watch}" >&2
    echo "  e.g. $0 org/other-model /path/to/dest start 4000000   (4,000,000 B/s ~= 32 Mbit/s)" >&2
    exit 1
}

[ $# -ge 3 ] || usage
REPO="$1"; DEST="$2"; ACTION="$3"; shift 3
SLUG="$(basename "$DEST")"
MANIFEST="$HERE/.dl_manifest_${SLUG}.tsv"
LOG="$HERE/download_ratelimited_${SLUG}.log"
PIDFILE="$HERE/.dl_ratelimited_${SLUG}.pid"
FINISH_MARKER="$HERE/.dl_ratelimited_${SLUG}.finishing"
VENV_PY="$HERE/.venv/bin/python"
CORE="$HERE/scripts/dl_ratelimited_core.py"

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

pid_of() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }

total_bytes() { awk -F'\t' '{s+=$2} END{print s+0}' "$MANIFEST"; }

got_bytes() {
    local s=0 f sz local_sz
    while IFS=$'\t' read -r f sz; do
        local_sz=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
        [ "$local_sz" -gt "$sz" ] && local_sz=$sz
        s=$((s + local_sz))
    done < "$MANIFEST"
    echo "$s"
}

current_file() {
    grep -v "attempt .*failed at offset" "$LOG" 2>/dev/null \
        | grep "^=== " | tail -1 | sed -E 's/^=== (.+) \([^)]*\) ===$/\1/'
}

current_file_bytes() {
    local f="$1" sz
    sz=$(awk -F'\t' -v f="$f" '$1==f{print $2; exit}' "$MANIFEST")
    echo "${sz:-0}"
}

case "$ACTION" in

start)
    RATE="${1:?specify a rate in bytes per second (e.g. 4000000 = 32 Mbit/s)}"
    if [ -n "$(pid_of)" ]; then echo "already running (PID $(pid_of))"; exit 0; fi
    ensure_manifest
    nohup "$VENV_PY" "$CORE" --repo "$REPO" --dest "$DEST" --rate "$RATE" >"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    echo "started (PID $(cat "$PIDFILE"))  rate: ${RATE} B/s  log: $LOG"
    ;;

stop)
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "no running process"; rm -f "$PIDFILE"; exit 0; fi
    kill "$p" 2>/dev/null
    sleep 1
    kill -9 "$p" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped. fetched so far: $(human "$(got_bytes)")  (start resumes)"
    ;;

finish-and-stop)
    # Safe stop before handing off to another tool (e.g. dl.sh). Finishes the
    # in-flight file first, so nothing straddles a file boundary and gets
    # fetched twice. This command blocks until done; the pending state is
    # visible from status, and cancel-finish (from another terminal) aborts it.
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "no running process"; rm -f "$PIDFILE"; exit 0; fi
    echo "waiting for the current file to finish (kill -USR1 $p)..."
    kill -USR1 "$p" 2>/dev/null
    while kill -0 "$p" 2>/dev/null; do
        sleep 3
    done
    rm -f "$PIDFILE"
    echo "stopped safely. fetched so far: $(human "$(got_bytes)")  (no partial files)"
    ;;

cancel-finish)
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "no running process"; exit 0; fi
    if [ ! -f "$FINISH_MARKER" ]; then echo "no pending stop"; exit 0; fi
    kill -USR2 "$p" 2>/dev/null
    sleep 1
    if [ -f "$FINISH_MARKER" ]; then
        echo "cancel may have failed (marker still present)"
    else
        echo "pending stop cancelled, continuing with the next file"
    fi
    ;;

status)
    ensure_manifest
    g=$(got_bytes); t=$(total_bytes)
    pct=$(awk -v g="$g" -v t="$t" 'BEGIN{printf "%.1f", g*100/t}')
    echo "fetched  : $(human "$g") / $(human "$t")  (${pct}%)"
    p="$(pid_of)"
    if [ -z "$p" ]; then echo "state    : stopped"; exit 0; fi
    echo "state    : running (PID $p)"
    if [ -f "$FINISH_MARKER" ]; then
        echo "pending  : will stop after this file (cancel with cancel-finish)"
    fi
    cf="$(current_file)"
    [ -n "$cf" ] && echo "fetching : $cf"
    sleep 20
    g2=$(got_bytes)
    rate=$(( (g2 - g) / 20 ))
    if [ "$rate" -le 0 ]; then
        echo "rate     : unmeasurable (0 B/s -- burst boundary, or stalled)"
    else
        eta=$(( (t - g2) / rate ))
        printf "rate     : %s/s\nETA (all)      : %dh%dm  (done around %s)\n" \
            "$(human "$rate")" $((eta/3600)) $(((eta%3600)/60)) \
            "$(date -d "+${eta} seconds" '+%m/%d %H:%M')"
        if [ -n "$cf" ]; then
            cf_sz=$(current_file_bytes "$cf")
            cf_local=$(stat -c%s "$DEST/$cf" 2>/dev/null || echo 0)
            if [ "$cf_sz" -gt 0 ] && [ "$cf_local" -lt "$cf_sz" ]; then
                cf_eta=$(( (cf_sz - cf_local) / rate ))
                cf_pct=$(awk -v g="$cf_local" -v t="$cf_sz" 'BEGIN{printf "%.1f", g*100/t}')
                printf "ETA (this file): %dm  (%s%% done, around %s)\n" \
                    $((cf_eta/60)) "$cf_pct" "$(date -d "+${cf_eta} seconds" '+%H:%M')"
            fi
        fi
    fi
    ;;

watch)
    while true; do
        bash "$0" status
        echo "---"
    done
    ;;

*)
    usage
    ;;
esac
