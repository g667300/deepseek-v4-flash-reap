#!/usr/bin/env bash
# Copy a checkpoint directory to a remote host, resuming after dropped links.
#
#   push_model.sh <src-dir> <[user@]host:/dest/dir> [--checksum] [--dry-run]
#
# e.g. push_model.sh artifacts/dsv4-reap50 spark:/srv/models/dsv4-reap50
#
# An 82 GB transfer over ssh does not reliably survive in one piece: a broken
# pipe every 35-45 GB was the norm here, on a link that otherwise sustained
# 540 MB/s. rsync resumes cleanly given --partial --inplace, so the fix is to
# keep retrying until it exits 0 -- each pass skips whatever already matches,
# so the retries converge instead of restarting the transfer.
#
# Retries are limited to the exit codes that mean "the link died" (socket,
# protocol, timeout, ssh). Anything else -- a full disk, a bad path, an
# interrupt -- aborts immediately rather than hammering a broken setup.
#
# --checksum compares by content instead of size+mtime. Use it when the
# destination already holds an earlier build of the same checkpoint: the surgery
# rewrites every shard, so every mtime changes while most bytes do not, and
# without it you re-send the whole checkpoint to deliver a few MB of real
# difference. It costs a full read of both sides, so it is not the default.
set -euo pipefail

SRC="${1:?usage: $0 <src-dir> <[user@]host:/dest/dir> [--checksum] [--dry-run]}"
DST="${2:?usage: $0 <src-dir> <[user@]host:/dest/dir> [--checksum] [--dry-run]}"
shift 2

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
BACKOFF="${BACKOFF:-10}"

RSYNC_OPTS=(-a --partial --inplace --info=progress2,stats1)
for arg in "$@"; do
  case "$arg" in
    --checksum) RSYNC_OPTS+=(--checksum) ;;
    --dry-run)  RSYNC_OPTS+=(--dry-run) ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

[[ -d "$SRC" ]] || { echo "not a directory: $SRC" >&2; exit 2; }
[[ "$DST" == *:* ]] || { echo "destination must be [user@]host:/path, got: $DST" >&2; exit 2; }

# Trailing slashes matter: copy the *contents* of src into dst.
SRC="${SRC%/}/"
DST="${DST%/}/"

host="${DST%%:*}"
remote_dir="${DST#*:}"
ssh "$host" "mkdir -p '${remote_dir}'"

src_bytes=$(du -sb "$SRC" | cut -f1)
echo "pushing $(numfmt --to=iec "$src_bytes" 2>/dev/null || echo "$src_bytes bytes") to ${DST}"

attempt=0
while :; do
  attempt=$((attempt + 1))
  set +e
  rsync "${RSYNC_OPTS[@]}" "$SRC" "$DST"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    break
  fi

  case $rc in
    # socket / protocol / timeout / ssh -- the link died, so resume.
    10|12|23|24|30|35|255) ;;
    *)
      echo "rsync failed with exit $rc; not a dropped link, aborting" >&2
      exit "$rc"
      ;;
  esac

  if [[ $attempt -ge $MAX_ATTEMPTS ]]; then
    echo "still failing after ${attempt} attempts (last exit ${rc}), giving up" >&2
    exit "$rc"
  fi

  dst_bytes=$(ssh "$host" "du -sb '${remote_dir}' 2>/dev/null | cut -f1" || echo 0)
  echo "=== attempt ${attempt} dropped (exit ${rc}); ${dst_bytes:-0} of ${src_bytes} bytes landed; retrying in ${BACKOFF}s ==="
  sleep "$BACKOFF"
done

echo "=== transfer complete after ${attempt} attempt(s) ==="
