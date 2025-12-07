#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="../../analysis/data/raw/Files for Aous"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

# Optional: start fresh summary
rm -f summaries/pupil_hht_summary.csv

for f in "$DATA_DIR"/*.csv; do
  base="$(basename "$f" .csv)"
  echo ">>> Processing $base"

  python pupil_hht_sim.py \
    --data-file "$f" \
    --save-plots \
    --no-show \
    | tee "$LOG_DIR/${base}.log"
done
