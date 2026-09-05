#!/usr/bin/env bash
# run_clean_val_eval.sh
# ---------------------------------------------------------------------------
# Re-runs validation-dependent selection on the deduplicated subsets, to answer
# reviewer comment 4. Inference only — no retraining, no checkpoint regeneration.
#
#   Part A: alpha sweep on BOTH subsets (t6, n=976 and t14, n=614).
#           Establishes that the selected interpolation coefficient does not
#           depend on leaked validation images.
#   Part B: architecture ranking on t6 only.
#
# Usage:
#   bash run_clean_val_eval.sh            # run everything
#   bash run_clean_val_eval.sh alpha      # Part A only
#   bash run_clean_val_eval.sh arch       # Part B only
#   DRY_RUN=1 bash run_clean_val_eval.sh  # print commands without executing
# ---------------------------------------------------------------------------
set -u

SEGFORMER_PATH="./segformer-b2-local"

# Directory holding the SAWI interpolation checkpoints (segformer_b2_wise_aXXX.pth)
SAWI_DIR="./trained_models/wise_swad"

# The early-stopped baseline theta_0. EDIT THIS to the real path.
THETA0_CKPT="./trained_models/segformer_b2_best.pth"

# Architecture checkpoints for Part B: "family|checkpoint|label"
# EDIT: fill in the real paths and confirm each --family string matches what
# evaluate_test_set.py expects.
ARCH_MODELS=(
  "segformer|./trained_models/segformer_b2_best.pth|segformer_b2"
  "unet|./trained_models/unet_best.pth|unet"
  "deeplabv3plus|./trained_models/deeplabv3plus_best.pth|deeplabv3plus"
  "attention_unet|./trained_models/attention_unet_scse_best.pth|attention_unet_scse"
  "manet|./trained_models/manet_best.pth|manet"
  "diffusion|./trained_models/diffusion_best.pth|diffusion"
)

SUBSETS=("t6:data_local/val_clean_t6" "t14:data_local/val_clean_t14")

run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  $*"; else eval "$@"; fi
}

evaluate() {  # subset_dir results_dir family checkpoint label
  local sdir="$1" rdir="$2" fam="$3" ckpt="$4" lab="$5"
  if [ ! -f "$ckpt" ]; then
    echo "  !! MISSING, skipped: $ckpt"
    return
  fi
  local extra=""
  [ "$fam" = "segformer" ] && extra="--segformer-path $SEGFORMER_PATH"
  run "EVAL_IMAGES=$sdir/images EVAL_MASKS=$sdir/masks EVAL_RESULTS=$rdir \
       python evaluate_test_set.py --family $fam --checkpoint $ckpt $extra \
       --label $lab"
}

# --------------------------------------------------------------- Part A -----
part_alpha() {
  echo "=== Part A: alpha sweep on both clean subsets ==="
  mapfile -t CKPTS < <(ls "$SAWI_DIR"/*.pth 2>/dev/null | sort)
  if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "!! No checkpoints in $SAWI_DIR — check SAWI_DIR."; return
  fi
  echo "Found ${#CKPTS[@]} interpolation checkpoints in $SAWI_DIR"

  for entry in "${SUBSETS[@]}"; do
    tag="${entry%%:*}"; sdir="${entry#*:}"
    rdir="results_val_clean_${tag}"
    n=$(ls "$sdir/images" 2>/dev/null | wc -l)
    echo ""
    echo "--- subset $tag ($n images) -> $rdir ---"

    # theta_0 anchor: the alpha=0 reference point of the sweep
    evaluate "$sdir" "$rdir" segformer "$THETA0_CKPT" "theta0_${tag}"

    for ckpt in "${CKPTS[@]}"; do
      base=$(basename "$ckpt" .pth)
      evaluate "$sdir" "$rdir" segformer "$ckpt" "${base}_${tag}"
    done
  done
}

# --------------------------------------------------------------- Part B -----
part_arch() {
  echo ""
  echo "=== Part B: architecture ranking on t6 only ==="
  sdir="data_local/val_clean_t6"; rdir="results_val_clean_t6_arch"
  for m in "${ARCH_MODELS[@]}"; do
    IFS='|' read -r fam ckpt lab <<< "$m"
    evaluate "$sdir" "$rdir" "$fam" "$ckpt" "${lab}_t6"
  done
}

case "${1:-all}" in
  alpha) part_alpha ;;
  arch)  part_arch ;;
  all)   part_alpha; part_arch ;;
  *)     echo "usage: bash run_clean_val_eval.sh [alpha|arch|all]"; exit 1 ;;
esac

echo ""
echo "Done. Results in results_val_clean_t6/, results_val_clean_t14/,"
echo "results_val_clean_t6_arch/."
echo "Next: compare the argmax alpha across t6 and t14 against the full-partition"
echo "selection. Invariance is the result comment 4 requires."
