#!/bin/bash
# After all four runs are trained and evaluated on the held-out sets (scaling/RUNBOOK.md section 6):
#   1. a fixed sample of each run's OWN training clips (up to 2000, 10 s segments) evaluated at every checkpoint
#      -> compute to target on training motions, the paper's Fig. 3(b) protocol;
#   2. every run's world model scored on the same pooled held-out windows, at every checkpoint step;
#   3. tables and figures.
# Usage (one GPU pod, from the repository root):  DATA=/workspace/data bash scaling/post_runs.sh
set -euo pipefail
: "${DATA:?set DATA to the shared volume}"
RUNS="${RUNS:-D1 D7 D40 Dall}"
N="${N_TRAIN_CLIPS:-2000}"

for S in $RUNS; do
  lib=$DATA/lib/train_eval_${S}_seg10.pt
  if [ ! -f "$lib" ]; then
    # reproducible sample: shuf seeded by a fixed byte stream
    shuf -n "$N" --random-source=<(yes 0) "$DATA/subsets/$S.txt" > "$DATA/subsets/train_eval_$S.txt"
    uv run python scripts/data/merge_library_parts.py --parts "$DATA/parts" --clip-list "$DATA/subsets/train_eval_$S.txt" \
        --out "$DATA/lib/train_eval_$S.pt"
    uv run python scaling/segment_library.py --library "$DATA/lib/train_eval_$S.pt" --out "$lib"
  fi
  for c in "$DATA"/logs/"$S"/ckpt_00*.pt; do
    out=$DATA/results/$S/$(basename "$c" .pt)
    if grep -q '"train"' "$out/summary.json" 2>/dev/null; then continue; fi
    uv run python scaling/eval_heldout.py --ckpt "$c" --out-dir "$out" --library train="$lib" 2>&1 \
        | grep -E "^\[train\] mean" | sed "s|^|$S $(basename "$c" .pt) |"
  done
done

for c in "$DATA"/logs/Dall/ckpt_00*.pt "$DATA"/logs/D40/ckpt_00*.pt; do basename "$c" .pt; done | sort -u | while read -r step; do
  out=$DATA/results/wm_error_$step.json
  [ -f "$out" ] && continue
  ckpts=$(for S in $RUNS; do if [ -f "$DATA/logs/$S/$step.pt" ]; then echo "$DATA/logs/$S/$step.pt"; fi; done)
  # held-out windows only (motion, performer, lafan1), not the training samples
  uv run python scaling/wm_error.py --ckpt $ckpts --windows "$DATA/results/*/$step/[mpl]*_windows.pt" --out "$out" 2>&1 | tail -4
done

uv run --with matplotlib python scaling/analyze.py --data "$DATA" --out-dir "$DATA/analysis"
