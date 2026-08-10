#!/usr/bin/env python3
"""
Test whether the cross-dataset Recall gain of SAWI concentrates on narrow rip
necks, which would convert the manuscript's rip-specificity argument from a
mechanism claim into a measured result.

Rationale
---------
Section V-G argues that predicted rip extent is the quantity least constrained
by the training objective, because rip necks are narrow, boundaries are
diffuse, and mIoU is dominated by the background class. If that argument is
correct, an intervention that restores extent should help most where extent
matters most: on the narrowest necks, where a contraction of two or three
pixels per side removes the largest proportion of the annotated foreground.

This script tests that prediction directly. It requires no retraining and no
new inference beyond the per-image score files already produced.

Two statistics are reported:

  Binned deltas    Images are grouped into width bins (terciles by default),
                   and the paired delta (method minus baseline) is reported
                   per bin. If the argument holds, the delta is largest in the
                   narrowest bin.

  Trend test       Spearman rank correlation between per-image rip width and
                   per-image paired delta, over all rip-bearing images. A
                   negative coefficient means the gain grows as necks narrow.
                   This is the single number to quote.

Clustering
----------
If --lookup is supplied (frame_to_video.csv from the RipVIS analysis), all
statistics are additionally computed with the video as the unit of analysis,
since frames within a sequence are not independent. Quote the video-level
figures.

Usage
  python stratify_gain_by_geometry.py \
      --geometry geometry_test/per_image.csv \
      --baseline scores_theta0.csv \
      --method   scores_sawi_a070.csv \
      --metric   recall \
      --lookup   frame_to_video.csv \
      --out      stratified_gain

Score CSVs need a file_name column and the metric column.

Depends on: numpy, pandas, scipy.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RNG = np.random.default_rng(42)


def load_scores(path, metric):
    df = pd.read_csv(path)
    key = 'file_name' if 'file_name' in df.columns else df.columns[0]
    df = df.rename(columns={key: 'file_name'})
    if metric not in df.columns:
        sys.exit(f'{path} has no column "{metric}". Columns: {list(df.columns)}')
    return df[['file_name', metric]].rename(columns={metric: 'score'})


def cluster_bootstrap_mean(x, groups, n_boot=10000):
    """Bootstrap the mean of x by resampling whole groups."""
    x = np.asarray(x, float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[p] for p in pick])
        means[b] = x[sel].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', required=True, help='per_image.csv from measure_rip_geometry.py')
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--metric', default='recall')
    ap.add_argument('--width-col', default='width_mean_median',
                    help='geometry column to stratify on. Options include '
                         'width_mean_median (default), width_mean_min '
                         '(narrowest neck in the image), area_frac_total.')
    ap.add_argument('--bins', type=int, default=3, help='number of quantile bins')
    ap.add_argument('--lookup', default=None, help='frame_to_video.csv for clustering')
    ap.add_argument('--out', default='stratified_gain')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    geo = pd.read_csv(args.geometry)
    b = load_scores(args.baseline, args.metric).rename(columns={'score': 'base'})
    m = load_scores(args.method, args.metric).rename(columns={'score': 'meth'})

    df = geo.merge(b, on='file_name').merge(m, on='file_name')
    # Recall is undefined without positives: keep rip-bearing images only
    if 'n_regions' in df.columns:
        df = df[df['n_regions'] > 0]
    df = df.dropna(subset=[args.width_col])
    df['delta'] = df['meth'] - df['base']

    if len(df) < 30:
        print(f'WARNING: only {len(df)} images after merging. Check that '
              f'file_name strings match between the geometry and score files.')

    lines = []
    lines.append(f'Metric              : {args.metric}')
    lines.append(f'Stratifying variable: {args.width_col}')
    lines.append(f'Rip-bearing images  : {len(df)}')
    lines.append('')

    # ---------------- image-level ----------------
    rho, p = spearmanr(df[args.width_col], df['delta'])
    lines.append('TREND TEST (image level)')
    lines.append(f'  Spearman rho(width, delta) = {rho:+.4f}   p = {p:.3g}')
    lines.append(f'  Interpretation: rho < 0 means the gain increases as necks narrow.')
    lines.append('')

    df['bin'] = pd.qcut(df[args.width_col], args.bins,
                        labels=[f'Q{i+1}' for i in range(args.bins)],
                        duplicates='drop')
    lines.append('BINNED PAIRED DELTA (image level)')
    lines.append(f'  {"bin":>4s} {"n":>6s} {"width range":>18s} '
                 f'{"mean delta (pp)":>16s} {"median (pp)":>12s}')
    for bn, grp in df.groupby('bin', observed=True):
        w = grp[args.width_col]
        lines.append(f'  {str(bn):>4s} {len(grp):>6d} '
                     f'{w.min():>8.2f}-{w.max():<9.2f} '
                     f'{grp["delta"].mean()*100:>16.3f} '
                     f'{grp["delta"].median()*100:>12.3f}')
    lines.append('')

    # ---------------- video level ----------------
    if args.lookup and os.path.exists(args.lookup):
        lk = pd.read_csv(args.lookup)
        cols = ['file_name', 'video_id'] + (['is_NR'] if 'is_NR' in lk.columns else [])
        dv = df.merge(lk[cols], on='file_name', how='inner')
        if 'is_NR' in dv.columns:
            dv = dv[dv['is_NR'] == 0]
        if len(dv):
            pv = dv.groupby('video_id').agg(
                width=(args.width_col, 'mean'),
                delta=('delta', 'mean'),
                n_frames=('delta', 'size')).reset_index()
            lines.append(f'VIDEO LEVEL  (n = {len(pv)} sequences)  <-- quote these')
            rho_v, p_v = spearmanr(pv['width'], pv['delta'])
            lines.append(f'  Spearman rho(width, delta) = {rho_v:+.4f}   p = {p_v:.3g}')
            lo, hi = cluster_bootstrap_mean(dv['delta'].values, dv['video_id'].values)
            lines.append(f'  overall mean delta = {dv["delta"].mean()*100:+.3f} pp   '
                         f'95% CI [{lo*100:+.3f}, {hi*100:+.3f}] (cluster bootstrap)')
            lines.append('')
            nb = min(args.bins, max(2, len(pv) // 5))
            pv['bin'] = pd.qcut(pv['width'], nb,
                                labels=[f'Q{i+1}' for i in range(nb)],
                                duplicates='drop')
            lines.append(f'  BINNED PAIRED DELTA (video level, {nb} bins)')
            lines.append(f'    {"bin":>4s} {"n_vid":>6s} {"width range":>18s} '
                         f'{"mean delta (pp)":>16s}')
            for bn, grp in pv.groupby('bin', observed=True):
                lines.append(f'    {str(bn):>4s} {len(grp):>6d} '
                             f'{grp["width"].min():>8.2f}-{grp["width"].max():<9.2f} '
                             f'{grp["delta"].mean()*100:>16.3f}')
            lines.append('')
            try:
                st, pw = wilcoxon(pv['delta'], alternative='two-sided')
                lines.append(f'  two-sided paired Wilcoxon over videos: p = {pw:.4g}')
            except ValueError:
                lines.append('  Wilcoxon not computable (all deltas zero)')
            lines.append('')
            pv.to_csv(os.path.join(args.out, 'per_video.csv'), index=False)
        else:
            lines.append('No rows survived the lookup merge; check file_name formats.')
            lines.append('')

    lines.append('HOW TO READ THIS')
    lines.append('  A negative Spearman rho, with the largest mean delta in the')
    lines.append('  narrowest bin, supports the argument of Section V-G: the')
    lines.append('  intervention recovers most where predicted extent is least')
    lines.append('  constrained. That is a rip-specific result rather than a')
    lines.append('  generic one, and it can be stated as such.')
    lines.append('  A rho near zero means the gain is uniform across rip widths.')
    lines.append('  In that case do NOT claim narrow-neck specificity; report the')
    lines.append('  null result, which is still informative.')

    df.to_csv(os.path.join(args.out, 'per_image_merged.csv'), index=False)
    txt = '\n'.join(lines)
    open(os.path.join(args.out, 'summary.txt'), 'w').write(txt + '\n')
    print(txt)
    print(f'\nWrote {args.out}/summary.txt')


if __name__ == '__main__':
    main()
