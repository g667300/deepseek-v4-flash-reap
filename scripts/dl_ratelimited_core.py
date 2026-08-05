#!/usr/bin/env python3
"""Rate-limited sequential downloader for an HF Hub repo.

Exists because naive rate limiting hangs against this repo's file host:
every safetensors shard 302-redirects to the Xet CDN bridge
(``us.aws.cdn.hf.co/xet-bridge-us/...``). Both curl's ``--limit-rate`` and a
first version of this script that paced a single streaming connection with
``time.sleep()`` between chunks degraded to a crawl (or a full stall) after
a while: pausing mid-read on an open connection leaves data sitting unread
in the kernel socket buffer, which starves the TCP receive window, and this
particular CDN seems to punish that pattern hard and doesn't recover.

Fix: never pause *inside* an open connection. Instead, fetch each file in
large ``Range`` bursts at full, unthrottled speed (plain ``requests.get``,
no manual chunk pacing), close that request, and only sleep *between*
bursts -- exactly long enough that burst_size / (transfer_time + sleep)
works out to the target rate. Confirmed empirically: repeated full-speed
32MB range bursts sustain ~6MB/s indefinitely with no degradation, whereas
pacing reads within one connection decayed from ~4MB/s to ~25KB/s within a
minute against the same host.

Each burst also opens a *fresh* connection (a plain ``requests.get`` call,
no ``requests.Session`` reuse) -- a pooled/kept-alive connection reused for
a second Range request against this CDN hung indefinitely (no data, no
error, no timeout) on the second request every time this was tried with a
shared Session. A closed-then-reopened connection per burst did not.

Resuming is just starting the burst loop from the current on-disk size --
there is no separate single big streaming request to special-case.

After a file finishes downloading, its SHA256 is verified against the LFS
hash HF Hub reports for that path (non-LFS files, e.g. README.md, have no
such hash and are size-checked only). This only runs once, right after a
fresh completion -- files already complete on disk when the run starts are
size-matched and skipped without re-hashing every time.

SIGUSR1 requests a clean handoff to a different downloader (e.g. switching
back to `dl.sh`/hf CLI for an unthrottled overnight run): finish the file
currently in flight, then exit before starting the next one, so no
partially-downloaded file is left behind for the other tool to choke on.
SIGUSR2 cancels a pending SIGUSR1 request -- the run continues past the
current file instead of stopping. A marker file next to --dest records
whether a finish is pending, so the wrapper script's `status` can report it
without needing to talk to this process directly.
"""
from __future__ import annotations

import argparse
import hashlib
import signal
import sys
import time
from pathlib import Path

import requests

BURST_BYTES = 32_000_000
PROGRESS_INTERVAL = 30  # seconds; one progress line per interval (per-burst would be far too noisy)

# SIGUSR1 requests a clean handoff: finish the file currently in flight
# (never interrupted mid-file -- that's what leaves a partial file another
# tool, e.g. hf CLI, can't resume and would re-fetch from scratch), then
# exit before starting the next one. Checked between files, not between
# bursts, so there is never a partially-downloaded file left on disk.
_finish_requested = False
_finish_marker: Path | None = None  # set in main() once --dest is known


def _handle_finish_signal(signum, frame) -> None:
    global _finish_requested
    _finish_requested = True
    if _finish_marker is not None:
        _finish_marker.touch()
    print("\ngot SIGUSR1: will finish the current file, then exit", file=sys.stderr, flush=True)


def _handle_cancel_signal(signum, frame) -> None:
    global _finish_requested
    _finish_requested = False
    if _finish_marker is not None and _finish_marker.exists():
        _finish_marker.unlink()
    print("\ngot SIGUSR2: pending stop cancelled, continuing with the next file",
          file=sys.stderr, flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def verify_sha256(dest: Path, expected_sha256: str) -> None:
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256.lower():
        raise RuntimeError(f"SHA256 mismatch: got {actual}, expected {expected_sha256}")


def download_file(url: str, dest: Path, expected_size: int,
                   rate: float, timeout: float, retries: int,
                   expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    offset = dest.stat().st_size if dest.exists() else 0
    if expected_size and offset == expected_size:
        return  # already complete
    if expected_size and offset > expected_size:
        offset = 0  # stale/corrupt leftover -- restart this file clean

    mode = "r+b" if offset else "wb"
    progress_t0 = time.time()
    progress_offset0 = offset
    with open(dest, mode) as f:
        if offset:
            f.seek(offset)
        while expected_size == 0 or offset < expected_size:
            end = offset + BURST_BYTES - 1
            if expected_size:
                end = min(end, expected_size - 1)

            for attempt in range(1, retries + 1):
                try:
                    t0 = time.time()
                    r = requests.get(url, headers={"Range": f"bytes={offset}-{end}"},
                                     timeout=timeout)
                    if r.status_code not in (200, 206):
                        r.raise_for_status()
                    data = r.content
                    break
                except Exception as e:  # noqa: BLE001 - retry-all is the point here
                    print(f"  attempt {attempt}/{retries} failed at offset {offset}: "
                          f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    if attempt == retries:
                        raise
                    time.sleep(min(30, 2 ** attempt))

            if not data:
                break  # server says nothing more to give
            f.write(data)
            f.flush()
            offset += len(data)

            if rate > 0:
                elapsed = time.time() - t0
                should_take = len(data) / rate
                if should_take > elapsed:
                    time.sleep(should_take - elapsed)

            if r.status_code == 200:
                break  # server ignored Range and sent the whole thing -- done either way

            now = time.time()
            if now - progress_t0 >= PROGRESS_INTERVAL:
                rate_actual = (offset - progress_offset0) / (now - progress_t0)
                line = f"  {human(offset)}"
                if expected_size:
                    line += f"/{human(expected_size)} ({offset * 100 / expected_size:.1f}%)"
                line += f"  {human(rate_actual)}/s"
                if expected_size and rate_actual > 0:
                    eta = (expected_size - offset) / rate_actual
                    line += f"  ETA {int(eta // 60)}m{int(eta % 60)}s"
                print(line, flush=True)
                progress_t0 = now
                progress_offset0 = offset

    final = dest.stat().st_size
    if expected_size and final != expected_size:
        raise RuntimeError(f"size mismatch after download: got {final}, expected {expected_size}")

    if expected_sha256:
        print(f"  verifying SHA256... ({human(final)})", flush=True)
        verify_sha256(dest, expected_sha256)
        print("  SHA256 ok", flush=True)


def fetch_manifest(repo: str) -> list[tuple[str, int, str | None]]:
    """Return a list of (rfilename, size, sha256). sha256 exists only for
    LFS-tracked files; non-LFS files (README.md and friends) get None."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    return [(f.rfilename, f.size or 0, f.lfs.sha256 if f.lfs else None)
            for f in info.siblings]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--include", default=None,
                     help="restrict to this single rfilename (exact match)")
    ap.add_argument("--rate", type=float, required=True, help="bytes/sec, 0 = unlimited")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    global _finish_marker
    dest_root = Path(args.dest)
    _finish_marker = dest_root.parent / ".dl_ratelimited.finishing"
    if _finish_marker.exists():
        _finish_marker.unlink()  # stale marker from a previous run

    signal.signal(signal.SIGUSR1, _handle_finish_signal)
    signal.signal(signal.SIGUSR2, _handle_cancel_signal)

    manifest = fetch_manifest(args.repo)
    if args.include:
        manifest = [(n, s, h) for n, s, h in manifest if n == args.include]
        if not manifest:
            print(f"file not found in repo: {args.include}", file=sys.stderr)
            return 1
    total = sum(sz for _, sz, _ in manifest)
    print(f"{len(manifest)} files, {human(total)} total, rate={human(args.rate)}/s", flush=True)

    for name, size, sha256 in manifest:
        dest = dest_root / name
        if size and dest.exists() and dest.stat().st_size == size:
            continue

        url = f"https://huggingface.co/{args.repo}/resolve/main/{name}"
        print(f"=== {name} ({human(size)}) ===", flush=True)
        try:
            download_file(url, dest, size, args.rate, args.timeout, args.retries,
                          expected_sha256=sha256)
        except Exception as e:  # noqa: BLE001
            print(f"  giving up on {name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return 1

        if _finish_requested:
            if _finish_marker.exists():
                _finish_marker.unlink()
            print(f"finished {name}; exiting without starting the next file (safe handoff point)",
                  flush=True)
            return 0

    print("all files complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
