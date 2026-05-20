"""
Runs Wilcoxon signed-rank tests between your best model and every baseline.
Saves results/significance_tests.csv.

The Wilcoxon signed-rank test is used (not a paired t-test) because:
  - Segmentation metrics are not normally distributed
  - The test operates on paired per-image differences
  - It is the standard test for this type of comparison in CV papers

Run after all models are evaluated:
    python significance_test.py
"""

import csv
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR  = "results"
BEST_MODEL   = "segformer_b2"   # change to your actual best model label
METRIC       = "miou"            # primary comparison metric
ALPHA        = 0.05              # significance threshold

# Display names for the graphic table
# Keys must match the file label prefixes in results/*_per_image.csv
DISPLAY_NAMES = {
    "unet_resnet50":                 "UNet / ResNet-50",
    "unet_resnet34":                 "UNet / ResNet-34",
    "unet_mobilenet_v2":             "UNet / MobileNet-V2",
    "unet_efficientnet-b2":          "UNet / EfficientNet-B2",
    "unet_swin_tiny":                "UNet / Swin-Tiny",
    "unet_maxvit_tiny":              "UNet / MaxViT-Tiny",
    "unet_mambaout_tiny":            "UNet / MambaOut-Tiny",
    "unet_convnext_tiny":            "UNet / ConvNeXt-Tiny",
    "deeplabv3plus_resnet50":        "DeepLabV3+ / ResNet-50",
    "deeplabv3plus_efficientnet-b2": "DeepLabV3+ / EfficientNet-B2",
    "attention_unet_resnet50":       "Attn-UNet / ResNet-50",
    "attention_unet_swin_tiny":      "Attn-UNet / Swin-Tiny",
    "manet_resnet50":                "MANet / ResNet-50",
    "manet_resnet34":                "MANet / ResNet-34",
    "manet_mobilenet_v2":            "MANet / MobileNet-V2",
    "manet_efficientnet-b2":         "MANet / EfficientNet-B2",
    "manet_swin_tiny":               "MANet / Swin-Tiny",
    "manet_maxvit_tiny":             "MANet / MaxViT-Tiny",
    "manet_mambaout_tiny":           "MANet / MambaOut-Tiny",
    "manet_convnext_tiny":           "MANet / ConvNeXt-Tiny",
    "diffusion_resnet50":            "Diffusion / ResNet-50 \u2020",
}

def get_family_color(label):
    """Return colour based on architectural family."""
    if label.startswith("unet"):         return "#2E7D32"   # green
    if label.startswith("deeplabv3"):    return "#1565C0"   # blue
    if label.startswith("attention"):    return "#E65100"   # orange
    if label.startswith("manet"):        return "#6A1B9A"   # purple
    if label.startswith("diffusion"):    return "#B71C1C"   # red
    return "#555555"

    # "CNN Encoder-Decoder":    "#E8F4E8",   # light green
    # "DeepLabV3+":             "#E8EEF4",   # light blue-grey
    # "Attention UNet (scSE)":  "#F4EDE8",   # light orange
    # "MANet (Attn-Residual)":  "#F4E8F4",   # light purple
    # "SegFormer":              "#FFFACD",   # light yellow
    # "Diffusion":              "#FFE8E8",   # light red

# ── Original working logic — unchanged ───────────────────────────────────────

# Load all per-image result files
files = sorted(Path(RESULTS_DIR).glob("*_per_image.csv"))
data  = {}
for f in files:
    label     = f.stem.replace("_per_image", "")
    df        = pd.read_csv(f).set_index("image")
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

    mean_diff   = diff.mean()
    significant = "YES" if p < ALPHA else "NO"

    rows.append({
        "baseline":      label,
        "best_model":    BEST_MODEL,
        "metric":        METRIC,
        "mean_best":     f"{best_s.mean():.4f}",
        "mean_baseline": f"{base_s.mean():.4f}",
        "mean_diff":     f"{mean_diff:+.4f}",
        "wilcoxon_stat": f"{stat:.2f}",
        "p_value":       f"{p:.4f}",
        "significant":   significant,
        "n_images":      len(shared),
    })
    print(f"  {BEST_MODEL} vs {label:<30}  "
          f"\u0394{METRIC}={mean_diff:+.4f}  p={p:.4f}  {'*' if p<ALPHA else ''}")

out_path = f"{RESULTS_DIR}/significance_tests_{BEST_MODEL}_vs_all.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  Significance tests saved: {out_path}")

# ── Graphic table — added only here, original logic above is untouched ────────
#
# One row per baseline model. Columns: Baseline, Mean (Baseline),
# Mean (Best Model), Delta, p-value, Significant.
#
# Row background:
#   Light green  = p < alpha and best > baseline  (significant win)
#   Light grey   = p >= alpha                     (not significant)
#   Light red    = p < alpha and best < baseline  (should not occur with
#                  one-sided alternative="greater", included for safety)
#
# Baseline name is colour-coded by architectural family.
# Rows sorted: significant wins first (by delta desc), then not significant.

rows_sig   = sorted([r for r in rows if r["significant"] == "YES"],
                    key=lambda r: float(r["mean_diff"]), reverse=True)
rows_insig = sorted([r for r in rows if r["significant"] == "NO"],
                    key=lambda r: float(r["mean_diff"]), reverse=True)
rows_sorted = rows_sig + rows_insig

best_disp   = DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)
col_headers = [
    "Baseline Model",
    f"Mean {METRIC.upper()}\n(Baseline)",
    f"Mean {METRIC.upper()}\n({best_disp})",
    f"\u0394 {METRIC.upper()}",
    "p-value",
    f"Significant\n(\u03b1={ALPHA})",
]
col_widths = [0.34, 0.12, 0.14, 0.10, 0.10, 0.10]

cell_text = []
for r in rows_sorted:
    disp = DISPLAY_NAMES.get(r["baseline"], r["baseline"])
    cell_text.append([
        disp,
        r["mean_baseline"],
        r["mean_best"],
        r["mean_diff"],
        r["p_value"],
        r["significant"],
    ])

fig_h = max(6.0, 0.48 * (len(rows_sorted) + 2) + 1.8)
fig, ax = plt.subplots(figsize=(11.5, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText  = cell_text,
    colLabels = col_headers,
    cellLoc   = "center",
    loc       = "center",
    colWidths = col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.55)

# Header row
for j in range(len(col_headers)):
    cell = tbl[0, j]
    cell.set_facecolor("#1A2E5A")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_edgecolor("#FFFFFF")

# Data rows
for i, r in enumerate(rows_sorted):
    p_val = float(r["p_value"])
    diff  = float(r["mean_diff"])
    sig   = r["significant"] == "YES"

    bg = "#C8E6C9" if (sig and diff > 0) else \
         "#FFCDD2" if (sig and diff < 0) else "#F5F5F5"

    for j in range(len(col_headers)):
        cell = tbl[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#DDDDDD")

        if j == 0:
            # Baseline name: bold, colour-coded by family, left-aligned
            fc = get_family_color(r["baseline"])
            cell.set_text_props(color=fc, fontweight="bold",
                                fontsize=9, ha="left")
        elif j == 3:
            # Delta: green if positive, red if negative
            color = "#2E7D32" if diff > 0 else "#B71C1C"
            cell.set_text_props(color=color, fontweight="bold")
        elif j == 5:
            # YES/NO: green for YES, red for NO
            color = "#2E7D32" if sig else "#B71C1C"
            cell.set_text_props(color=color, fontweight="bold")
        else:
            cell.set_text_props(color="#111111")

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#C8E6C9", edgecolor="#AAAAAA",
                   label=f"p < {ALPHA}  —  best model significantly better"),
    mpatches.Patch(facecolor="#F5F5F5", edgecolor="#AAAAAA",
                   label=f"p \u2265 {ALPHA}  —  not significant"),
    mpatches.Patch(facecolor="white", edgecolor="#2E7D32",   linewidth=2.5, label="CNN family"),
    mpatches.Patch(facecolor="white", edgecolor="#1565C0",   linewidth=2.5, label="DeepLabV3+ family"),
    mpatches.Patch(facecolor="white", edgecolor="#E65100",   linewidth=2.5, label="Attn-UNet family"),
    mpatches.Patch(facecolor="white", edgecolor="#6A1B9A",   linewidth=2.5, label="MANet family"),
    mpatches.Patch(facecolor="white", edgecolor="#B71C1C",   linewidth=2.5, label="Diffusion family"),
]
ax.legend(
    handles        = legend_patches,
    loc            = "upper center",
    bbox_to_anchor = (0.5, -0.01),
    ncol           = 4,
    fontsize       = 8,
    frameon        = True,
    edgecolor      = "#CCCCCC",
)

plt.title(
    f"Wilcoxon Signed-Rank Test:  {best_disp}  vs.  All Baselines\n"
    f"Metric: {METRIC.upper()}   |   One-sided (best > baseline)   |   "
    f"\u03b1 = {ALPHA}   |   n = 4,349 images",
    fontsize=10, fontweight="bold", color="#1A2E5A", pad=10,
)
plt.figtext(
    0.01, 0.002,
    "\u2020 Diffusion model did not converge.  "
    "Baseline name colour = architectural family.  "
    "Bold \u0394 = direction of difference.",
    fontsize=7.5, color="#555555",
)

out_png = f"{RESULTS_DIR}/significance_table.png"
plt.tight_layout()
plt.savefig(out_png, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"  Graphic table saved     : {out_png}")
