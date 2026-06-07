"""
Generates figure of normalised confusion matrices for the six
representative models (one per architectural family).

Output: results/confusion_matrix_fig3.png

Usage:
    python confusion_matrix_fig.py

Requires:
    pip install matplotlib numpy seaborn
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

OUTPUT_PATH = "results/confusion_matrix_fig3.png"

# ── Data: six representative models, one per family ───────────────────────────
# Values taken directly from the results table (results_table.png)
# recall   = TP / (TP+FN)   — from Recall column
# mAcc     = (acc_rip + acc_bg) / 2
# acc_bg   = 2*mAcc - recall  (exact derivation from the nine-metric formula)
# All values are normalised rates [0, 1]

MODELS = [
    {
        "family":     "CNN Encoder-Decoder",
        "label":      "UNet / ResNet-34\n(mIoU = 0.6365)",
        "recall":     0.3902,
        "mAcc":       0.6861,
        "color":      "#2E7D32",   # family colour (green)
    },
    {
        "family":     "DeepLabV3+",
        "label":      "DeepLabV3+ / EffNet-B2\n(mIoU = 0.6360)",
        "recall":     0.3978,
        "mAcc":       0.6902,
        "color":      "#1565C0",   # blue
    },
    {
        "family":     "Attention UNet (scSE)",
        "label":      "Attn-UNet / ResNet-50\n(mIoU = 0.6293)",
        "recall":     0.3699,
        "mAcc":       0.6781,
        "color":      "#E65100",   # orange
    },
    {
        "family":     "MANet (Attn-Residual)",
        "label":      "MANet / EfficientNet-B2\n(mIoU = 0.6313)",
        "recall":     0.3643,
        "mAcc":       0.6764,
        "color":      "#6A1B9A",   # purple
    },
    {
        "family":     "SegFormer (Best Model)",
        "label":      "SegFormer / MiT-B2\n(mIoU = 0.6578)",
        "recall":     0.4681,
        "mAcc":       0.7232,
        "color":      "#1A2E5A",   # navy
    },
    {
        "family":     "Diffusion (Negative Finding)",
        "label":      "Diffusion / ResNet-50\u2020\n(mIoU = 0.4712)",
        "recall":     0.0021,
        "mAcc":       0.4995,
        "color":      "#B71C1C",   # red
    },
]

# ── Compute confusion matrix cells from metrics ───────────────────────────────
def compute_cm(recall, mAcc):
    """
    For binary segmentation, the normalised confusion matrix is:

        Predicted Rip    Predicted Background
    Actual Rip  [  recall        1 - recall     ]
    Actual Bg   [  1 - acc_bg    acc_bg         ]

    acc_bg = 2*mAcc - recall
    (derived from mAcc = (recall + acc_bg) / 2)
    """
    acc_bg = np.clip(2 * mAcc - recall, 0, 1)
    cm = np.array([
        [recall,       1.0 - recall],
        [1.0 - acc_bg, acc_bg      ],
    ])
    return cm

# ── Custom colourmap (white → family colour, matching manuscript palette) ─────
def family_cmap(hex_color):
    """Create a light-to-dark colormap from white to the family colour."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return LinearSegmentedColormap.from_list(
        "family_cm",
        [(1, 1, 1),          # white (low values)
         (r, g, b, 0.25),    # light family colour (mid)
         (r, g, b)],         # full family colour (high values)
        N=256,
    )

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.flatten()

CLASS_LABELS = ["Rip\nCurrent", "Background"]

for ax, model in zip(axes, MODELS):
    cm = compute_cm(model["recall"], model["mAcc"])
    cmap = family_cmap(model["color"])

    sns.heatmap(
        cm,
        ax           = ax,
        annot        = False,     # we'll draw custom annotations below
        fmt          = "",
        cmap         = cmap,
        vmin         = 0.0,
        vmax         = 1.0,
        linewidths   = 1.5,
        linecolor    = "white",
        cbar         = True,
        cbar_kws     = {"shrink": 0.72, "pad": 0.04,
                        "format": "%.1f", "ticks": [0, 0.25, 0.5, 0.75, 1.0]},
        xticklabels  = CLASS_LABELS,
        yticklabels  = CLASS_LABELS,
    )

    # ── Cell annotations: value + label ──────────────────────────────────────
    cell_labels = [
        ["True\nPositive\n(Recall)",    "False\nNegative\n(Miss)"],
        ["False\nPositive\n(Alarm)",    "True\nNegative\n(Specificity)"],
    ]
    highlight = [(0, 0), (0, 1)]   # TP and FN cells (safety-critical row)

    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            label = cell_labels[i][j]
            # Choose text colour based on cell brightness
            brightness = val * 0.8  # approx luminance weight
            txt_col = "white" if brightness > 0.45 else "#1A1A1A"

            # Main value (large, bold)
            ax.text(j + 0.5, i + 0.38, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=txt_col, family="DejaVu Sans")

            # Cell label (small, italic below value)
            ax.text(j + 0.5, i + 0.68, label,
                    ha="center", va="center",
                    fontsize=7.5, fontstyle="italic",
                    color=txt_col, family="DejaVu Sans",
                    linespacing=1.3)

    # ── Red border on FN cell (safety-critical) ───────────────────────────────
    # The False Negative (top-right) cell is operationally critical
    fn_val = cm[0, 1]
    rect = plt.Rectangle((1, 0), 1, 1,
                          fill=False, edgecolor="#D32F2F",
                          linewidth=2.5, zorder=5)
    ax.add_patch(rect)

    # ── Axis labels and title ─────────────────────────────────────────────────
    ax.set_xlabel("Predicted Class", fontsize=9, labelpad=4)
    ax.set_ylabel("Actual Class",    fontsize=9, labelpad=4)
    ax.tick_params(axis="both", labelsize=9, length=0)

    # Title: family name coloured, model below in black
    family_parts = model["label"].split("\n")
    title_str    = model["family"]
    ax.set_title(title_str, fontsize=10, fontweight="bold",
                 color=model["color"], pad=7)

    # Subtitle (model name + mIoU)
    sub_str = "\n".join(family_parts)
    ax.text(0.5, 1.005, sub_str,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=8, color="#333333",
            style="italic")

    # Colorbar label
    ax.collections[0].colorbar.set_label("Normalised Rate", fontsize=8)
    ax.collections[0].colorbar.ax.tick_params(labelsize=7)

# ── Figure-level annotations ──────────────────────────────────────────────────
fig.suptitle(
    "Fig. 3.  Normalised Confusion Matrices for Six Representative Models\n"
    "(one per architectural family, evaluated on RipVIS validation test partition, $n$ = 4,349 images)",
    fontsize    = 11,
    fontweight  = "bold",
    color       = "#1A2E5A",
    y           = 0.98,
)

# Legend patches
legend_patches = [
    mpatches.Patch(facecolor="#2E7D32", label="CNN Encoder-Decoder"),
    mpatches.Patch(facecolor="#1565C0", label="DeepLabV3+"),
    mpatches.Patch(facecolor="#E65100", label="Attention UNet (scSE)"),
    mpatches.Patch(facecolor="#6A1B9A", label="MANet (Attn-Residual)"),
    mpatches.Patch(facecolor="#1A2E5A", label="SegFormer"),
    mpatches.Patch(facecolor="#B71C1C", label="Diffusion"),
    mpatches.Patch(facecolor="white", edgecolor="#D32F2F", linewidth=2,
                   label="Red border = False Negative (miss) cell — operationally dangerous"),
]

fig.legend(
    handles        = legend_patches,
    loc            = "lower center",
    bbox_to_anchor = (0.5, 0.005),
    ncol           = 4,
    fontsize       = 8,
    frameon        = True,
    edgecolor      = "#CCCCCC",
    title          = "\u2020 Diffusion model did not converge; near-zero rip detection.  "
                     "All values are normalised rates (row sums to 1.0).",
    title_fontsize = 7.5,
)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

Path("results").mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUTPUT_PATH}")
