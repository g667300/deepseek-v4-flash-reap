#!/usr/bin/env bash
# Put a pruned checkpoint on a remote host and bring vLLM up on it, blocking
# until the server actually answers.
#
#   serve_remote.sh <local-ckpt-dir> <[user@]host:/remote/dir>
#
# e.g. serve_remote.sh artifacts/dsv4-reap50 spark:/srv/models/dsv4-reap50
#
# Environment (all optional):
#   NAME           container name          (default: basename of the remote dir)
#   IMAGE          vLLM image              (default: vllm-dsv4:fi0614)
#   PORT           host port               (default: 8000)
#   MAXLEN         --max-model-len         (default: 65536)
#   MAXSEQS        --max-num-seqs          (default: 16)
#   GPUUTIL        --gpu-memory-utilization(default: 0.75, see below)
#   MEMORY         docker --memory         (default: 115g)
#   EXTRA          extra vllm serve args   (default: --kv-cache-dtype fp8)
#   READY_TIMEOUT  seconds to wait to load (default: 1800)
#   SKIP_PUSH=1    the checkpoint is already there, only (re)start the server
#
# Notes earned the hard way, encoded here so they are not rediscovered:
#
# * --entrypoint must be given explicitly. An image committed from a container
#   started with `--entrypoint bash` defaults to bash, and `docker run IMAGE
#   serve /model` then silently runs nothing.
# * --kv-cache-dtype fp8 is mandatory for DeepSeek-V4 (its sparse-MLA kernel
#   rejects anything else) and not universal. It lives in EXTRA for that reason.
# * FLASHINFER_DISABLE_VERSION_CHECK=1 pairs with an image carrying
#   flashinfer-python 0.6.14; flashinfer-cubin never shipped that version.
# * The load takes minutes (measured: 291 s for 82.4 GB), and a checkpoint that
#   overruns the host's memory hangs the whole machine rather than failing, so
#   this waits on /health and gives up rather than polling forever.
# * GPUUTIL defaults to 0.75, not vLLM's 0.9, because on a unified-memory host
#   that fraction is taken out of system RAM. At 0.9 on a 121 GiB machine vLLM
#   reserves ~109 GiB for weights and KV, leaving the OS about 8 GiB. Long
#   context then collapses: measured at 32K with 16 concurrent requests,
#   generation fell to 0.1-2.2 tokens/s with the engine reporting only 4-6% KV
#   cache in use -- the pool was never the constraint, the host running out of
#   everything else was. Adding a second client at that point hung the machine
#   hard enough to need a power cycle. Reserve less; the KV pool is not what
#   long context is short of.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SRC="${1:?usage: $0 <local-ckpt-dir> <[user@]host:/remote/dir>}"
DST="${2:?usage: $0 <local-ckpt-dir> <[user@]host:/remote/dir>}"

[[ "$DST" == *:* ]] || { echo "destination must be [user@]host:/path, got: $DST" >&2; exit 2; }
host="${DST%%:*}"
remote_dir="${DST#*:}"
remote_dir="${remote_dir%/}"

NAME="${NAME:-$(basename "$remote_dir")}"
IMAGE="${IMAGE:-vllm-dsv4:fi0614}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-65536}"
MAXSEQS="${MAXSEQS:-16}"
GPUUTIL="${GPUUTIL:-0.75}"
MEMORY="${MEMORY:-115g}"
EXTRA="${EXTRA---kv-cache-dtype fp8}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

if [[ "${SKIP_PUSH:-0}" != "1" ]]; then
  "$HERE/push_model.sh" "$SRC" "$DST"
fi

# Free the name and the port. Anything already serving has to go: the host
# holds one model at a time.
echo "== (re)starting ${NAME} on ${host} from ${IMAGE}"
ssh "$host" "docker rm -f '${NAME}' >/dev/null 2>&1 || true"

# shellcheck disable=SC2086  # EXTRA is deliberately word-split
ssh "$host" "docker run -d --name '${NAME}' \
  --gpus all --ipc=host --memory=${MEMORY} --memory-swap=${MEMORY} \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -p ${PORT}:8000 \
  -v '${remote_dir}':/model:ro \
  --entrypoint vllm '${IMAGE}' serve /model \
  --served-model-name '${NAME}' --gpu-memory-utilization ${GPUUTIL} \
  --max-model-len ${MAXLEN} --max-num-seqs ${MAXSEQS} ${EXTRA}" >/dev/null

url="http://${host#*@}:${PORT}"
echo "== waiting for ${url}/health (up to ${READY_TIMEOUT}s)"
start=$SECONDS
while ! curl -sf "${url}/health" >/dev/null 2>&1; do
  # Anchored: `-f name=X` is a substring match, so an unrelated container whose
  # name merely contains NAME would make a dead container look alive and this
  # loop would wait out the full timeout instead of failing with the logs.
  if ! ssh "$host" "docker ps -q -f name='^${NAME}\$'" | grep -q .; then
    echo "container exited during load; last 40 lines:" >&2
    ssh "$host" "docker logs --tail 40 '${NAME}'" >&2 || true
    exit 1
  fi
  if (( SECONDS - start >= READY_TIMEOUT )); then
    echo "not ready after ${READY_TIMEOUT}s; last 40 lines:" >&2
    ssh "$host" "docker logs --tail 40 '${NAME}'" >&2 || true
    exit 1
  fi
  sleep 15
done

echo "== ready after $((SECONDS - start))s: ${url}, served as '${NAME}'"
