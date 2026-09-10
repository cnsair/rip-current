#!/usr/bin/env bash
# run_sawi_repetitions.sh
# ---------------------------------------------------------------------------
# Runs the full per-seed SAWI pipeline for the remaining repetitions required
# by reviewer comment 1 (five independent runs, before and after SAWI).
#
# Per seed, in order:
#   1. Train Arm 1a  (warm-start from theta0, bf16, SWAD collection on)
#   2. Evaluate the fine-tuned endpoint        -> "before SAWI"
#   3. Evaluate that run's SWAD average
#   4. Interpolate theta0 <-> that run's SWAD at alpha = 0.10 and 0.70
#   5. Re-estimate BatchNorm on training images
#   6. Evaluate both interpolated models       -> "after SAWI"
#
# alpha is FIXED at the two values the selection rules already returned. The
# rules are not re-derived per seed, so no validation-partition evaluation is
# needed here — that is deliberate and saves ~35 min per seed.
#
# Usage:
#   DRY_RUN=1 bash run_sawi_repetitions.sh      # print commands, run nothing
#   bash run_sawi_repetitions.sh                # seeds 44 45 46
#   bash run_sawi_repetitions.sh 45 46          # explicit subset
#
# GPU is assumed EXCLUSIVE. Do not start while seed43 or any evaluation is
# still running.
# ---------------------------------------------------------------------------
set -u

ANCHOR="./trained_models/segformer_b2_local.pth"
SEGPATH="./segformer-b2-local"
CKDIR="./trained_models/determinism_check"
TEST_IMAGES="data_local/test_local/rip_vis_val_images/images"
TEST_MASKS="data_local/test_local/rip_vis_val_images/masks"
TRAIN_IMAGES="data_local/train_local/images"
RESULTS="results_determinism_check"

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(44 45 46)

run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '    %q ' "$@"; echo
  else
    "$@" || { echo "!! FAILED: $*"; exit 1; }
  fi
}

evaluate() {   # checkpoint  label
  local ckpt="$1" label="$2"
  # Under DRY_RUN nothing has been trained, so the file legitimately does not
  # exist yet; only enforce the guard on a real run.
  if [ "${DRY_RUN:-0}" != "1" ] && [ ! -f "$ckpt" ]; then
    echo "!! MISSING checkpoint, aborting: $ckpt"; exit 1
  fi
  run env "EVAL_IMAGES=$TEST_IMAGES" "EVAL_MASKS=$TEST_MASKS" \
      "EVAL_RESULTS=$RESULTS" \
      python evaluate_test_set.py --family segformer \
        --checkpoint "$ckpt" --segformer-path "$SEGPATH" --label "$label"
}

# ── Pre-flight ─────────────────────────────────────────────────────────────
echo "run_sawi_repetitions.sh  |  seeds: ${SEEDS[*]}"
echo ""

if [ ! -f "$ANCHOR" ]; then
  echo "!! Anchor not found: $ANCHOR"; exit 1
fi
for f in train_segformer_dual_branch.py evaluate_test_set.py \
         wise_ft_interpolate.py bn_recalibrate.py; do
  [ -f "$f" ] || { echo "!! Script not found in cwd: $f"; exit 1; }
done

# Refuse to clobber a completed run.
for S in "${SEEDS[@]}"; do
  if [ -f "$CKDIR/segformer_b2_ft_bf16_seed${S}.pth" ]; then
    echo "!! $CKDIR/segformer_b2_ft_bf16_seed${S}.pth already exists."
    echo "   Move it aside, or pass only the seeds that still need running."
    exit 1
  fi
done

mkdir -p "$CKDIR" "$RESULTS" logs

# ── Per-seed pipeline ──────────────────────────────────────────────────────
for S in "${SEEDS[@]}"; do
  CKPT="$CKDIR/segformer_b2_ft_bf16_seed${S}.pth"
  SWAD="$CKDIR/segformer_b2_ft_bf16_seed${S}_swad.pth"
  WDIR="./trained_models/wise_seed${S}"

  echo ""
  echo "============================================================"
  echo "  SEED $S  —  step 1/6: training"
  echo "  started $(date)"
  echo "============================================================"

  # tee is used rather than run() so the training log is captured verbatim.
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    DETAIL=0 USE_FOAM=0 AUG2B=0 SWAD=1 EPOCHS=30 \\"
    echo "    WARM_START=$ANCHOR SEED=$S CKPT=$CKPT \\"
    echo "    python train_segformer_dual_branch.py 2>&1 | tee logs/ft_bf16_seed${S}.log"
  else
    DETAIL=0 USE_FOAM=0 AUG2B=0 SWAD=1 EPOCHS=30 \
    WARM_START="$ANCHOR" SEED="$S" CKPT="$CKPT" \
    python train_segformer_dual_branch.py 2>&1 | tee "logs/ft_bf16_seed${S}.log"
    # PIPESTATUS[0] is python's exit code; $? would be tee's.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "!! Training failed for seed $S — stopping."; exit 1
    fi
  fi

  echo ""
  echo "  SEED $S  —  step 2/6: evaluate fine-tuned endpoint (before SAWI)"
  evaluate "$CKPT" "ft_bf16_seed${S}"

  echo ""
  echo "  SEED $S  —  step 3/6: evaluate this run's SWAD average"
  evaluate "$SWAD" "swad_bf16_seed${S}"

  echo ""
  echo "  SEED $S  —  step 4/6: interpolate theta0 <-> SWAD at alpha 0.10, 0.70"
  run python wise_ft_interpolate.py \
      --anchor "$ANCHOR" --finetuned "$SWAD" \
      --outdir "$WDIR" --alphas 0.1 0.7

  echo ""
  echo "  SEED $S  —  step 5/6: BatchNorm re-estimation on training images"
  run python bn_recalibrate.py --wise-dir "$WDIR" \
      --train-images "$TRAIN_IMAGES" --batches 400 --batch-size 8 --seed 0

  echo ""
  echo "  SEED $S  —  step 6/6: evaluate interpolated models (after SAWI)"
  evaluate "$WDIR/segformer_b2_wise_a010.pth" "sawi_a010_seed${S}"
  evaluate "$WDIR/segformer_b2_wise_a070.pth" "sawi_a070_seed${S}"

  echo ""
  echo "  SEED $S complete — $(date)"
done

echo ""
echo "============================================================"
echo "  All seeds complete."
echo "  Aggregates in $RESULTS/:"
for S in "${SEEDS[@]}"; do
  echo "    ft_bf16_seed${S}   swad_bf16_seed${S}   sawi_a010_seed${S}   sawi_a070_seed${S}"
done
echo "============================================================"
