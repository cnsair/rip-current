#!/usr/bin/env python3
"""
Test whether the SAWI cross-dataset gain differs between static-installation
and mobile acquisition, using the video sequence as the unit of analysis.

This answers a different question from ripvis_cluster_analysis.py. That script
asks whether SAWI helps overall; this one asks where it helps. Section V-G
predicts that fixed shore-based cameras should be the most exposed to the
run-to-run instability, because their far-field rip necks subtend the fewest
pixels and therefore lie closest to the regime in which predicted extent is
least constrained. This script tests that prediction.

No inference is required. The per-image score files already exist; the only new
input is a modality label per video, produced with
make_modality_contact_sheet.py.

Statistics
----------
Frames within a sequence are not independent, so every figure is computed with
the video as the unit: per-image scores are averaged within each video, and the
paired delta is taken per video. Confidence intervals come from a bootstrap
that resamples whole videos within each modality group. With roughly 15 to 20
videos per group the intervals will be wide; report them as they are.

Usage
  python stratify_by_modality.py \
      --lookup   frame_to_video.csv \
      --modality video_modality.csv \
      --baseline scores_theta0.csv \
      --method   scores_sawi_a070.csv \
      --metric   recall \
      --out      modality_results

Depends on: numpy, pandas, scipy.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

RNG = np.random.default_rng(42)
NBOOT = 10000


def load_scores(path, metric, name):
    df = pd.read_csv(path)
    key = "file_name" if "file_name" in df.columns else df.columns[0]
    df = df.rename(columns={key: "file_name"})
    if metric not in df.columns:
        sys.exit(f"{path} has no column '{metric}'. Columns: {list(df.columns)}")
    return df[["file_name", metric]].rename(columns={metric: name})


def boot_ci(x, n=NBOOT, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    means = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--metric", default="recall")
    ap.add_argument("--label", default="SAWI", help="name for the method column")
    ap.add_argument("--out", default="modality_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lk = pd.read_csv(args.lookup)
    md = pd.read_csv(args.modality)
    md["modality"] = md["modality"].astype(str).str.strip().str.lower()
    bad = sorted(set(md.modality) - {"static", "mobile"})
    if bad:
        sys.exit(f"Unrecognised modality values: {bad}. Use 'static' or 'mobile'.")

    b = load_scores(args.baseline, args.metric, "base")
    m = load_scores(args.method, args.metric, "meth")

    df = (b.merge(m, on="file_name")
            .merge(lk[["file_name", "video_id"] + (["is_NR"] if "is_NR" in lk.columns else [])],
                   on="file_name")
            .merge(md[["video_id", "modality"]], on="video_id"))
    if "is_NR" in df.columns:
        df = df[df.is_NR == 0]
    if not len(df):
        sys.exit("No rows survived the merge. Check that file_name strings match.")
    df["delta"] = df.meth - df.base

    # video is the unit of analysis
    pv = (df.groupby(["modality", "video_id"])
            .agg(delta=("delta", "mean"), base=("base", "mean"),
                 meth=("meth", "mean"), n_frames=("delta", "size"))
            .reset_index())
    pv.to_csv(os.path.join(args.out, "per_video.csv"), index=False)

    L = []
    L.append(f"Metric              : {args.metric}")
    L.append(f"Method              : {args.label}")
    L.append(f"Frames after merge  : {len(df)}")
    L.append(f"Videos              : {pv.video_id.nunique()}")
    L.append("")
    L.append(f"{'modality':>9} {'n_vid':>6} {'frames':>7} {'baseline':>9} "
             f"{'method':>8} {'mean delta (pp)':>16} {'95% CI (pp)':>20} "
             f"{'won':>6} {'p':>8}")
    L.append("-" * 96)

    groups = {}
    for mod, g in pv.groupby("modality"):
        d = g.delta.values
        lo, hi = boot_ci(d)
        try:
            _, p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            p = np.nan
        groups[mod] = d
        L.append(f"{mod:>9} {len(g):>6d} {int(g.n_frames.sum()):>7d} "
                 f"{g.base.mean():>9.4f} {g.meth.mean():>8.4f} "
                 f"{d.mean()*100:>16.3f} "
                 f"[{lo*100:>+7.3f}, {hi*100:>+7.3f}] "
                 f"{100*np.mean(d>0):>5.0f}% "
                 f"{p:>8.4f}")

    # pooled
    d_all = pv.delta.values
    lo, hi = boot_ci(d_all)
    try:
        _, p_all = wilcoxon(d_all, alternative="two-sided")
    except ValueError:
        p_all = np.nan
    L.append("-" * 96)
    L.append(f"{'all':>9} {len(pv):>6d} {int(pv.n_frames.sum()):>7d} "
             f"{pv.base.mean():>9.4f} {pv.meth.mean():>8.4f} "
             f"{d_all.mean()*100:>16.3f} "
             f"[{lo*100:>+7.3f}, {hi*100:>+7.3f}] "
             f"{100*np.mean(d_all>0):>5.0f}% {p_all:>8.4f}")
    L.append("")

    # between-group comparison
    if len(groups) == 2:
        a, bb = groups.get("static"), groups.get("mobile")
        if a is not None and bb is not None and len(a) > 1 and len(bb) > 1:
            try:
                _, pb = mannwhitneyu(a, bb, alternative="two-sided")
            except ValueError:
                pb = np.nan
            diff = a.mean() - bb.mean()
            bootd = np.array([RNG.choice(a, len(a), True).mean()
                              - RNG.choice(bb, len(bb), True).mean()
                              for _ in range(NBOOT)])
            dlo, dhi = np.quantile(bootd, [.025, .975])
            L.append("BETWEEN-GROUP COMPARISON  (static minus mobile)")
            L.append(f"  difference in mean delta : {diff*100:+.3f} pp")
            L.append(f"  95% CI                   : [{dlo*100:+.3f}, {dhi*100:+.3f}] pp")
            L.append(f"  Mann-Whitney U, two-sided: p = {pb:.4f}")
            L.append("")
            if dlo > 0:
                L.append("  The gain is larger on static installations, and the interval")
                L.append("  excludes zero. This supports the prediction of Section V-G.")
            elif dhi < 0:
                L.append("  The gain is larger on mobile acquisition, and the interval")
                L.append("  excludes zero. This runs against the prediction of Section V-G,")
                L.append("  and should be reported as such.")
            else:
                L.append("  The interval spans zero: these data do not distinguish the two")
                L.append("  modalities. Report the null result rather than the prediction.")
    L.append("")
    L.append("Frames within a sequence are not independent; every figure above uses")
    L.append("the video as the unit of analysis. Quote the per-video counts, not the")
    L.append("frame counts, as n.")

    txt = "\n".join(L)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\nWrote {args.out}/summary.txt and per_video.csv")


if __name__ == "__main__":
    main()
