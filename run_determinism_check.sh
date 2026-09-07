#!/usr/bin/env bash
# run_determinism_check.sh
# ---------------------------------------------------------------------------
# Runs the two bf16 determinism-check training arms sequentially (seed=42,
# seed=43). Aborts before run B if run A fails, so a broken run A never wastes
# GPU time on a run B that could not possibly be compared to it.
#
# Usage:
#   bash run_determinism_check.sh
#   nohup bash run_determinism_check.sh > logs/driver.log 2>&1 &   # survives
#                                                                    terminal close
# ---------------------------------------------------------------------------
set -e   # abort immediately if either training run exits non-zero

mkdir -p logs trained_models/determinism_check

CKPT_A=./trained_models/determinism_check/segformer_b2_bf16_seedA.pth
CKPT_B=./trained_models/determinism_check/segformer_b2_bf16_seedB.pth

# Refuse to run if either output path already exists, so a re-run of this
# script can never silently overwrite a completed result.
for f in "$CKPT_A" "$CKPT_B"; do
  if [ -f "$f" ]; then
    echo "!! $f already exists. Move or delete it before re-running, or edit"
    echo "   this script's CKPT_A/CKPT_B paths."
    exit 1
  fi
done

echo "=== Run A: seed=42 ==="
date
DETAIL=0 USE_FOAM=0 AUG2B=0 SWAD=0 WARM_START="" SEED=42 \
CKPT="$CKPT_A" \
python train_segformer_dual_branch.py 2>&1 | tee logs/bf16_seedA.log

echo ""
echo "=== Run A complete. Starting Run B: seed=43 ==="
date
DETAIL=0 USE_FOAM=0 AUG2B=0 SWAD=0 WARM_START="" SEED=43 \
CKPT="$CKPT_B" \
python train_segformer_dual_branch.py 2>&1 | tee logs/bf16_seedB.log

echo ""
echo "=== Both runs complete ==="
date
ls -la "$CKPT_A" "$CKPT_B"
