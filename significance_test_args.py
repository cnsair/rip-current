"""
Runs Wilcoxon signed-rank tests between a chosen model and baselines.

Two modes:
  1. SWEEP (original behaviour, default): best model vs ALL other models
         python significance_test.py
         python significance_test.py --best segformer_b2_dual
  2. PAIR: one specific comparison, with a selectable alternative hypothesis
         python significance_test.py --best segformer_b2_dual \
             --baseline segformer_b2 --alternative two-sided
         python significance_test.py --best segformer_b2_dual \
             --baseline segformer_b2 --alternative greater

The Wilcoxon signed-rank test is used (not a paired t-test) because:
  - Segmentation metrics are not normally distributed
  - The test operates on paired per-image differences
  - It is the standard test for this type of comparison in CV papers

CHANGE vs previous version:
  - argparse replaces the hardcoded BEST_MODEL / one-sided-only test.
  - `--alternative {greater,two-sided,less}` because a one-sided "greater"
    test can NEVER detect that the new model is worse or merely equivalent;
    for the dual-vs-baseline comparison a two-sided test is the honest
    primary analysis, with the pre-registered one-sided test reported
    alongside it.
  - Zero-difference pairs are dropped explicitly (wilcoxon's default) and
    the count is reported: on 4,349 images many pairs may tie exactly
    (e.g. images where both models predict all-background), and the
    effective n matters for interpreting the p-value.
  - Median difference and rank-biserial effect size are reported: with
    n≈4,000, even a trivial mean difference can reach p<0.05, so the
    effect size is what the manuscript should quote next to the p-value.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "results"
ALPHA       = 0.05


def load_all(results_dir):
    files = sorted(Path(results_dir).glob("*_per_image.csv"))
    data = {}
    for f in files:
        label = f.stem.replace("_per_image", "")
        data[label] = pd.read_csv(f).set_index("image")
    return data


def compare(data, best, baseline, metric, alternative):
    """One paired Wilcoxon comparison. Returns a result-row dict."""
    shared = data[best].index.intersection(data[baseline].index)
    best_s = data[best].loc[shared, metric].values
    base_s = data[baseline].loc[shared, metric].values

    diff    = best_s - base_s
    n_total = len(diff)
    n_zero  = int(np.sum(diff == 0))
    n_eff   = n_total - n_zero          # pairs actually used by the test

    stat, p = wilcoxon(diff, alternative=alternative)

    # Rank-biserial effect size r = 1 - 2*W_minus / (n(n+1)/2), computed on
    # the nonzero pairs — standard effect size companion to Wilcoxon.
    nz = diff[diff != 0]
    if len(nz) > 0:
        ranks   = pd.Series(np.abs(nz)).rank().values
        w_plus  = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        r_rb    = (w_plus - w_minus) / (w_plus + w_minus)
    else:
        r_rb = 0.0

    return {
        "model_a":        best,
        "model_b":        baseline,
        "metric":         metric,
        "alternative":    alternative,
        "mean_a":         f"{best_s.mean():.4f}",
        "mean_b":         f"{base_s.mean():.4f}",
        "mean_diff":      f"{diff.mean():+.4f}",
        "median_diff":    f"{np.median(diff):+.4f}",
        "wilcoxon_stat":  f"{stat:.2f}",
        "p_value":        f"{p:.4g}",
        "significant":    "YES" if p < ALPHA else "NO",
        "effect_r":       f"{r_rb:+.3f}",
        "n_images":       n_total,
        "n_ties_dropped": n_zero,
    }


def main():
    ap = argparse.ArgumentParser(description="Wilcoxon signed-rank tests on per-image metrics")
    ap.add_argument("--best", default="segformer_b2",
                    help="Label of the model of interest (default: segformer_b2)")
    ap.add_argument("--baseline", default=None,
                    help="PAIR mode: compare --best against this one label only. "
                         "Omit for the original all-vs-best sweep.")
    ap.add_argument("--metric", default="miou",
                    help="Column from the per-image CSVs (default: miou)")
    ap.add_argument("--alternative", default="greater",
                    choices=["greater", "two-sided", "less"],
                    help="H1 for the test. 'greater' = best > baseline "
                         "(original behaviour); use 'two-sided' as the honest "
                         "primary test when the direction is not certain.")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    data = load_all(args.results_dir)
    if args.best not in data:
        raise ValueError(f"'{args.best}' not found. Available: {list(data.keys())}")

    if args.baseline:                                   # ── PAIR mode ──
        if args.baseline not in data:
            raise ValueError(f"'{args.baseline}' not found. Available: {list(data.keys())}")
        targets = [args.baseline]
        suffix  = f"{args.best}_vs_{args.baseline}_{args.metric}_{args.alternative}"
    else:                                               # ── SWEEP mode ──
        targets = [l for l in data if l != args.best]
        suffix  = f"{args.best}_vs_all_{args.metric}_{args.alternative}"

    rows = []
    for label in targets:
        row = compare(data, args.best, label, args.metric, args.alternative)
        rows.append(row)
        print(f"  {args.best} vs {label:<28} "
              f"Δ{args.metric}={row['mean_diff']}  median Δ={row['median_diff']}  "
              f"p={row['p_value']} ({args.alternative})  r={row['effect_r']}  "
              f"{'*' if row['significant'] == 'YES' else ''}")

    out_path = f"{args.results_dir}/significance_tests_{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
