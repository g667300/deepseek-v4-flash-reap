#!/usr/bin/env bash
# Fetch the raw benchmark data that eval_tasks/ reads directly.
#
# Only JCommonsenseQA needs this: its HF dataset ships as a loading script,
# which datasets>=4 refuses to execute, so the task points at upstream's raw
# JSON instead. Everything else in the suite (MMLU, global_mmlu, RULER) comes
# from stock lm_eval tasks and needs nothing here.
#
#   ./scripts/fetch_eval_data.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/eval_data/jcommonsenseqa"
BASE="https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jcommonsenseqa-v1.2"

mkdir -p "$DEST"
for f in train-v1.2.json valid-v1.2.json; do
    if [ -s "$DEST/$f" ]; then
        echo "already present: eval_data/jcommonsenseqa/$f"
        continue
    fi
    echo "fetching $f ..."
    curl -fsSL "$BASE/$f" -o "$DEST/$f"
done

echo "done. JCommonsenseQA data is in eval_data/jcommonsenseqa/"
