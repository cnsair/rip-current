#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Foam-gap lambda sweep launcher (RTX 4090, bf16).
#
# Trains an epoch-matched set of arms — baseline (lambda=0) plus a lambda sweep —
# each from scratch under the fixed seed (set_seed(42) in the script), each to a
# DISTINCT checkpoint, then evaluates every checkpoint on the 4,349-image test set.
#
# All knobs are passed via environment variables read by train_segformer.py:
#   USE_FOAM (0/1), FOAM_LAMBDA, FOAM_MARGIN, CKPT, EPOCHS, MONITOR
#
# Usage:
#   chmod +x run_foam_sweep.sh
#   ./run_foam_sweep.sh
#
# Adjust EPOCHS / MONITOR / the LAMBDAS array to taste. MONITOR=recall or f2 is
# worth trying alongside the default miou (see the manuscript discussion).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EPOCHS=${EPOCHS:-50}        # same budget for every arm (epoch-matched comparison)
BATCH=${BATCH:-2}           # MUST match across arms; 2 matches the Table III baseline
MONITOR=${MONITOR:-miou}    # checkpoint-selection metric: miou | recall | f2
MARGIN=${MARGIN:-0.07}      # foam-gap target contrast (gentler than the old 0.10)
CKPT_DIR=${CKPT_DIR:-./trained_models}
TEST_IMAGES=${TEST_IMAGES:-data_local/test_local/rip_vis_val_images/images}
SEG_PATH=${SEG_PATH:-./segformer-b2-local}

LAMBDAS=(0.05 0.10 0.15)    # lambda values to sweep (treatment arms)

mkdir -p "$CKPT_DIR" results

run_train () {  # $1=use_foam $2=lambda_tag $3=lambda_val $4=ckpt
  echo; echo "==================================================================="
  echo "TRAIN  arm=$2  use_foam=$1  lambda=$3  margin=$MARGIN  monitor=$MONITOR  batch=$BATCH  epochs=$EPOCHS"
  echo "==================================================================="
  USE_FOAM="$1" FOAM_LAMBDA="$3" FOAM_MARGIN="$MARGIN" \
  EPOCHS="$EPOCHS" BATCH="$BATCH" MONITOR="$MONITOR" CKPT="$4" \
    python train_segformer.py
}

run_eval () {   # $1=label $2=ckpt
  echo; echo "---- EVAL  $1  ($2) ----"
  python evaluate_test_set.py \
    --family segformer \
    --checkpoint "$2" \
    --segformer-path "$SEG_PATH" \
    --label "$1" \
    || echo "eval failed for $1"
}

# 1) Baseline arm (lambda = 0)
BASE_CKPT="$CKPT_DIR/segformer_b2_baseline.pth"
run_train 0 baseline 0.0 "$BASE_CKPT"
run_eval segformer_b2_baseline "$BASE_CKPT"

# 2) Treatment arms (lambda sweep)
for L in "${LAMBDAS[@]}"; do
  TAG="l$(echo "$L" | tr -d '.')"          # 0.10 -> l010
  CKPT="$CKPT_DIR/segformer_b2_foam_${TAG}.pth"
  run_train 1 "foam_$TAG" "$L" "$CKPT"
  run_eval "segformer_b2_foam_${TAG}" "$CKPT"
done

echo; echo "Sweep complete. Per-arm aggregate CSVs are in results/."
echo "Compare test mIoU AND recall across arms, then run significance_test.py"
echo "on the best treatment arm vs the baseline (paired Wilcoxon, per-image mIoU)."
