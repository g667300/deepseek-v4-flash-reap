#!/usr/bin/env bash
# Fetch just the next not-yet-downloaded file from an HF Hub repo at full speed
# (hf CLI), then exit. REPO and DEST are required positional arguments:
#
#   ./scripts/dl_one_file.sh org/other-model /path/to/dest
#
# Why this exists: dl.sh (a full hf download) runs until the whole repo is
# done, which is awkward when you only have a few spare minutes of link time.
# This grabs one file (~9 GB on average, ~14 min at full speed) and stops, so
# it can be run opportunistically whenever the link is free.
#
# Shares the manifest and destination with dl_ratelimited.sh, and the two
# recognize each other's completed files: the hf CLI verifies content SHA256
# and skips files that are already whole, so nothing is fetched twice.
#
# Use ./scripts/dl_ratelimited.sh <repo> <dest> status to check progress (the
# manifest is shared).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 2 ]; then
    echo "usage: $0 <repo> <dest>" >&2
    echo "  e.g. $0 org/other-model /path/to/dest" >&2
    exit 1
fi
REPO="$1"; DEST="$2"
SLUG="$(basename "$DEST")"
HF_BIN="$HERE/.venv/bin/hf"
MANIFEST="$HERE/.dl_manifest_${SLUG}.tsv"
VENV_PY="$HERE/.venv/bin/python"
RATELIMITED_PIDFILE="$HERE/.dl_ratelimited_${SLUG}.pid"

# Both tools share a manifest and destination, so running them concurrently
# risks two writers on the same file.
if [ -f "$RATELIMITED_PIDFILE" ] && kill -0 "$(cat "$RATELIMITED_PIDFILE")" 2>/dev/null; then
    echo "dl_ratelimited.sh is running. Stop it safely first with" \
         "'./scripts/dl_ratelimited.sh <repo> <dest> finish-and-stop'." >&2
    exit 1
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

next_incomplete_file() {
    # Prefer files not started at all (0 bytes). The hf CLI cannot resume a
    # partial file written by another tool -- it re-fetches the whole thing --
    # so those are a last resort.
    local f sz local_sz first_partial=""
    while IFS=$'\t' read -r f sz; do
        local_sz=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
        [ "$sz" -eq 0 ] && continue
        [ "$local_sz" -eq "$sz" ] && continue
        if [ "$local_sz" -eq 0 ]; then
            echo "$f"
            return 0
        fi
        [ -z "$first_partial" ] && first_partial="$f"
    done < "$MANIFEST"
    if [ -n "$first_partial" ]; then
        echo "$first_partial"
        return 0
    fi
    return 1
}

ensure_manifest
f="$(next_incomplete_file)"
if [ -z "$f" ]; then
    echo "all files complete"
    exit 0
fi

echo "fetching: $f"
HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$REPO" --include "$f" --local-dir "$DEST"
