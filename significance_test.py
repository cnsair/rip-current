"""
Runs Wilcoxon signed-rank tests between your best model and every baseline.
Saves results/significance_tests.csv.

The Wilcoxon signed-rank test is used (not a paired t-test) because:
  - Segmentation metrics are not normally distributed
  - The test operates on paired per-image differences
  - It is the standard test for this type of comparison in CV papers
"""

import csv
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR  = "results"
BEST_MODEL   = "manet_swin_tiny"   # change to your actual best model label
METRIC       = "miou"              # primary comparison metric
ALPHA        = 0.05                # significance threshold

# Load all per-image result files
files = sorted(Path(RESULTS_DIR).glob("*_per_image.csv"))
data  = {}
for f in files:
    label = f.stem.replace("_per_image", "")
    df    = pd.read_csv(f).set_index("image")
    data[label] = df

if BEST_MODEL not in data:
    raise ValueError(f"Best model '{BEST_MODEL}' not found. "
                     f"Available: {list(data.keys())}")

best_scores = data[BEST_MODEL][METRIC].values
rows = []

for label, df in data.items():
    if label == BEST_MODEL:
        continue
    baseline_scores = df[METRIC].values

    # Align on shared images
    shared = data[BEST_MODEL].index.intersection(df.index)
    best_s = data[BEST_MODEL].loc[shared, METRIC].values
    base_s = df.loc[shared, METRIC].values

    diff   = best_s - base_s
    stat, p = wilcoxon(diff, alternative="greater")   # one-sided: best > baseline

    mean_diff = diff.mean()
    significant = "YES" if p < ALPHA else "NO"

    rows.append({
        "baseline":       label,
        "best_model":     BEST_MODEL,
        "metric":         METRIC,
        "mean_best":      f"{best_s.mean():.4f}",
        "mean_baseline":  f"{base_s.mean():.4f}",
        "mean_diff":      f"{mean_diff:+.4f}",
        "wilcoxon_stat":  f"{stat:.2f}",
        "p_value":        f"{p:.4f}",
        "significant":    significant,
        "n_images":       len(shared),
    })
    print(f"  {BEST_MODEL} vs {label:<30}  "
          f"Δ{METRIC}={mean_diff:+.4f}  p={p:.4f}  {'*' if p<ALPHA else ''}")

out_path = f"{RESULTS_DIR}/significance_tests_{BEST_MODEL}_vs_all.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  Significance tests saved: {out_path}")