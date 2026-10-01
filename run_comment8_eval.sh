#!/usr/bin/env bash
# run_comment8_eval.sh
# ---------------------------------------------------------------------------
# Evaluates the weight-averaging baselines of reviewer comment 8 on the RipVIS
# transfer partition: 4 baselines x 5 seeds + the uniform soup = 21 runs.
#
# Resumable: a model whose aggregate CSV already exists is skipped, so an
# interrupted run can be restarted without repeating completed work.
#
# Usage:
#   DRY_RUN=1 bash run_comment8_eval.sh     # print commands, run nothing
#   bash run_comment8_eval.sh               # everything not yet done
#   FORCE=1 bash run_comment8_eval.sh       # re-evaluate even if CSV exists
#   bash run_comment8_eval.sh 42 43         # restrict to these seeds
# ---------------------------------------------------------------------------
set -u

SEGPATH="./segformer-b2-local"
TEST_IMAGES="data_local/test_local/rip_vis_val_images/images"
TEST_MASKS="data_local/test_local/rip_vis_val_images/masks"
RESULTS="results_comment8"

# baseline file stem -> results label prefix
declare -A BASELINES=(
  ["baseline_swa"]="swa"
  ["baseline_lastk5"]="lastk5"
  ["baseline_ema"]="ema"
  ["baseline_swadwin"]="swadwin"
)

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 43 44 45 46)

mkdir -p "$RESULTS"
done_n=0; skip_n=0; miss_n=0

evaluate() {   # checkpoint  label
  local ckpt="$1" label="$2"

  if [ ! -f "$ckpt" ]; then
    echo "  !! MISSING, skipped: $ckpt"
    miss_n=$((miss_n+1)); return
  fi
  if [ "${FORCE:-0}" != "1" ] && [ -f "$RESULTS/${label}_aggregate.csv" ]; then
    echo "  -- already done, skipped: $label"
    skip_n=$((skip_n+1)); return
  fi

  local -a cmd=(env "EVAL_IMAGES=$TEST_IMAGES" "EVAL_MASKS=$TEST_MASKS"
                "EVAL_RESULTS=$RESULTS"
                python evaluate_test_set.py --family segformer
                --checkpoint "$ckpt" --segformer-path "$SEGPATH" --label "$label")

  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '  %q ' "${cmd[@]}"; echo
  else
    echo "  >> $label"
    "${cmd[@]}" || { echo "!! FAILED on $label"; exit 1; }
  fi
  done_n=$((done_n+1))
}

echo "run_comment8_eval.sh  |  seeds: ${SEEDS[*]}  |  results -> $RESULTS"
date

for S in "${SEEDS[@]}"; do
  echo ""
  echo "--- seed $S ---"
  for stem in "${!BASELINES[@]}"; do
    evaluate "./trained_models/baselines_seed${S}/${stem}.pth" \
             "${BASELINES[$stem]}_seed${S}"
  done
done

echo ""
echo "--- soups ---"
for f in ./trained_models/soups/soup_*.pth; do
  [ -e "$f" ] || continue
  base=$(basename "$f" .pth)
  evaluate "$f" "$base"
done

echo ""
echo "============================================================"
echo "  evaluated: $done_n   already done: $skip_n   missing: $miss_n"
date
echo "  aggregates in $RESULTS/"
echo "============================================================"
