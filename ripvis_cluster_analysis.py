#!/usr/bin/env python3
"""
Cluster-robust re-analysis of RipVIS cross-dataset results, correcting the
pseudoreplication in the previous frame-level statistics (supervisor
comments 6 and 7).

The 4,349 RipVIS test frames come from 36 videos; adjacent frames are
near-duplicates, so frames are NOT independent samples. This script
aggregates per-frame scores to per-video means and performs all inference
over videos. Six "-NR-" videos contain no rip annotations and are excluded
from Recall/F2 analysis (they belong to a separate false-positive report).

Effective n for every safety-metric claim = 30 rip-bearing videos.

INPUTS
------
1. frame_to_video.csv   (produced alongside this script)
      columns: file_name, image_id, video_id, is_NR, n_rip_ann, has_rip

2. One per-image score CSV per model, each with columns:
      file_name, recall, f2, miou        (add/rename as your files require)
   Provide the baseline file and one or more comparison files.

USAGE
-----
  python ripvis_cluster_analysis.py \
      --lookup   frame_to_video.csv \
      --baseline scores_theta0.csv \
      --compare  scores_sawi_a010.csv scores_sawi_a070.csv \
      --metrics  recall f2 miou \
      --out      ripvis_cluster_results.csv

Depends on: numpy, pandas, scipy.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG = np.random.default_rng(42)
N_BOOT = 10000


def load_scores(path, metrics):
    df = pd.read_csv(path)
    key = 'file_name' if 'file_name' in df.columns else df.columns[0]
    df = df.rename(columns={key: 'file_name'})
    return df[['file_name'] + metrics]


def to_video(df, lookup, metrics):
    """Join to lookup, drop NR videos, aggregate to per-video means."""
    m = df.merge(lookup[['file_name', 'video_id', 'is_NR']], on='file_name', how='inner')
    m = m[m['is_NR'] == 0]                       # rip-bearing videos only
    per_video = m.groupby('video_id')[metrics].mean()
    return per_video                             # index = video_id, one row per video


def bca_ci(deltas_per_video, n_boot=N_BOOT, alpha=0.05):
    """Bias-corrected and accelerated bootstrap CI for the mean paired delta,
    resampling whole videos with replacement."""
    x = np.asarray(deltas_per_video, dtype=float)
    n = len(x)
    theta_hat = x.mean()

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)              # resample videos
        boot[b] = x[idx].mean()

    # bias-correction z0
    prop = np.mean(boot < theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    from scipy.stats import norm
    z0 = norm.ppf(prop)

    # acceleration via jackknife over videos
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5) + 1e-12
    a = num / den

    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.quantile(boot, adj(zl))
    hi = np.quantile(boot, adj(zu))
    return theta_hat, lo, hi


def analyse(baseline_pv, compare_pv, metrics, label):
    """All statistics over the shared set of videos."""
    vids = baseline_pv.index.intersection(compare_pv.index)
    out = []
    for met in metrics:
        b = baseline_pv.loc[vids, met].values
        c = compare_pv.loc[vids, met].values
        delta = c - b                             # per-video paired delta

        # two-sided paired Wilcoxon over videos
        try:
            stat, p = wilcoxon(c, b, alternative='two-sided', zero_method='wilcox')
        except ValueError:
            stat, p = np.nan, np.nan              # all-zero differences

        # rank-biserial effect size for paired Wilcoxon
        diffs = c - b
        nz = diffs[diffs != 0]
        if len(nz):
            ranks = pd.Series(np.abs(nz)).rank().values
            r_pos = ranks[nz > 0].sum()
            r_neg = ranks[nz < 0].sum()
            rb = (r_pos - r_neg) / ranks.sum()
        else:
            rb = 0.0

        mean_delta, lo, hi = bca_ci(delta)
        frac_won = float(np.mean(delta > 0))

        out.append({
            'comparison': label, 'metric': met, 'n_videos': len(vids),
            'mean_delta_pp': round(mean_delta * 100, 3),
            'ci95_lo_pp': round(lo * 100, 3), 'ci95_hi_pp': round(hi * 100, 3),
            'wilcoxon_p_two_sided': None if np.isnan(p) else round(float(p), 5),
            'rank_biserial_r': round(rb, 3),
            'frac_videos_won': round(frac_won, 3),
            'ci_excludes_zero': bool(lo > 0 or hi < 0),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookup', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--compare', nargs='+', required=True)
    ap.add_argument('--metrics', nargs='+', default=['recall', 'f2', 'miou'])
    ap.add_argument('--out', default='ripvis_cluster_results.csv')
    args = ap.parse_args()

    lookup = pd.read_csv(args.lookup)
    n_vid = lookup.loc[lookup.is_NR == 0, 'video_id'].nunique()
    print(f'Rip-bearing videos (analysis n): {n_vid}')
    print(f'NR videos (excluded from Recall/F2): '
          f'{lookup.loc[lookup.is_NR==1,"video_id"].nunique()}')

    base_pv = to_video(load_scores(args.baseline, args.metrics), lookup, args.metrics)

    all_rows = []
    for cpath in args.compare:
        cmp_pv = to_video(load_scores(cpath, args.metrics), lookup, args.metrics)
        label = cpath.split('/')[-1].replace('.csv', '') + '_vs_baseline'
        all_rows += analyse(base_pv, cmp_pv, args.metrics, label)

    res = pd.DataFrame(all_rows)
    res.to_csv(args.out, index=False)
    pd.set_option('display.width', 160, 'display.max_columns', 20)
    print('\n' + res.to_string(index=False))
    print(f'\nWrote {args.out}')
    print('\nReport the mean_delta_pp with its 95% CI as the headline number. '
          'A CI that excludes zero is the cluster-robust positive result; '
          'the two-sided p is secondary and will be far weaker than the old '
          'frame-level p — that is the correct measurement, not a regression.')


if __name__ == '__main__':
    main()
