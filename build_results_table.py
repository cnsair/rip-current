"""
Reads all aggregate CSVs from the results/ folder and produces:
  1. Terminal printout: mean ± std, best value per column marked [B]
  2. results/results_table.png — publication-quality graphic table
     (screenshot or insert this directly into your manuscript)
  3. results/results_table.csv — clean CSV for pasting into Excel/Word

"""

import csv
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- works on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = "results"
OUTPUT_PNG  = f"{RESULTS_DIR}/results_table.png"
OUTPUT_CSV  = f"{RESULTS_DIR}/results_table.csv"

METRICS = ["miou","iou","dice","recall","precision","f2","biou","aacc","macc"]
METRIC_LABELS = ["mIoU","IoU","Dice","Recall","Precision","F2","bIoU","aAcc","mAcc"]

# Map file-label prefixes to display names and family group
# Edit these to match your actual output file labels
MODEL_ORDER = [
    # (file_label,              display_name,              family)
    ("unet_resnet50",           "UNet / ResNet-50",         "CNN Encoder-Decoder"),
    ("unet_resnet34",           "UNet / ResNet-34",         "CNN Encoder-Decoder"),
    ("unet_mobilenetv2",       "UNet / MobileNet-V2",      "CNN Encoder-Decoder"),
    ("unet_efficientnetb2",    "UNet / EfficientNet-B2",   "CNN Encoder-Decoder"),
    ("unet_swin_tiny",          "UNet / Swin-Tiny",         "CNN Encoder-Decoder"),
    ("unet_maxvit_tiny",        "UNet / MaxViT-Tiny",       "CNN Encoder-Decoder"),
    ("unet_mambaout_tiny",      "UNet / MambaOut-Tiny",     "CNN Encoder-Decoder"),
    ("unet_convnext_tiny",      "UNet / ConvNeXt-Tiny",     "CNN Encoder-Decoder"),
    ("deeplabv3plus_resnet50",  "DeepLabV3+ / ResNet-50",   "DeepLabV3+"),
    ("deeplabv3plus_efficientnetb2", "DeepLabV3+ / EffNet-B2", "DeepLabV3+"),
    ("attention_unet_resnet50", "Attn-UNet / ResNet-50",    "Attention UNet (scSE)"),
    ("attention_unet_swin_tiny","Attn-UNet / Swin-Tiny",    "Attention UNet (scSE)"),
    ("manet_resnet50",          "MANet / ResNet-50",        "MANet (Attn-Residual)"),
    ("manet_resnet34",          "MANet / ResNet-34",        "MANet (Attn-Residual)"),
    ("manet_mobilenetv2",      "MANet / MobileNet-V2",     "MANet (Attn-Residual)"),
    ("manet_efficientnetb2",   "MANet / EfficientNet-B2",  "MANet (Attn-Residual)"),
    ("manet_swin_tiny",         "MANet / Swin-Tiny",        "MANet (Attn-Residual)"),
    ("manet_maxvit_tiny",       "MANet / MaxViT-Tiny",      "MANet (Attn-Residual)"),
    ("manet_mambaout_tiny",     "MANet / MambaOut-Tiny",    "MANet (Attn-Residual)"),
    ("manet_convnext_tiny",     "MANet / ConvNeXt-Tiny",    "MANet (Attn-Residual)"),
    ("segformer_b2",            "SegFormer / MiT-B2",       "SegFormer"),
    ("diffusion_resnet50",      "Diffusion / ResNet-50†",   "Diffusion"),
]

# Family display colours (light background for each group)
FAMILY_COLORS = {
    "CNN Encoder-Decoder":    "#E8F4E8",   # light green
    "DeepLabV3+":             "#E8EEF4",   # light blue-grey
    "Attention UNet (scSE)":  "#F4EDE8",   # light orange
    "MANet (Attn-Residual)":  "#F4E8F4",   # light purple
    "SegFormer":              "#FFFACD",   # light yellow
    "Diffusion":              "#FFE8E8",   # light red
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_results():
    agg_files = {f.stem.replace("_aggregate",""): f
                 for f in Path(RESULTS_DIR).glob("*_aggregate.csv")}
    data = {}
    for label, fpath in agg_files.items():
        with open(fpath) as f:
            reader = csv.DictReader(f)
            data[label] = {row["metric"]: float(row["mean"]) for row in reader}
    return data

# ── Terminal + CSV output ─────────────────────────────────────────────────────
def print_and_save(data):
    # Find best value per metric column
    best = {m: 0.0 for m in METRICS}
    for label, display, family in MODEL_ORDER:
        if label not in data:
            continue
        for m in METRICS:
            v = data[label].get(m, 0.0)
            if v > best[m]:
                best[m] = v

    rows_out = []
    print(f"\n{'Architecture/Backbone':<35}" +
          "".join(f"{ml:>10}" for ml in METRIC_LABELS))
    print("─" * (35 + 10 * len(METRICS)))

    prev_family = None
    for label, display, family in MODEL_ORDER:
        if label not in data:
            continue
        if family != prev_family:
            print(f"\n  [{family}]")
            prev_family = family
        row = [display]
        row_str = f"  {display:<33}"
        for m in METRICS:
            v = data[label].get(m, float("nan"))
            marker = "*" if abs(v - best[m]) < 1e-4 else " "
            row_str += f"{v:>9.4f}{marker}"
            row.append(f"{v:.4f}")
        print(row_str)
        rows_out.append(row)

    # Save clean CSV
    # with open(OUTPUT_CSV, "w", newline="") as f:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture/Backbone"] + METRIC_LABELS)
        writer.writerows(rows_out)
    print(f"\n  CSV saved: {OUTPUT_CSV}")
    return best

# ── Graphic table ─────────────────────────────────────────────────────────────
def save_graphic_table(data, best):
    """
    Renders a colour-coded publication table as a PNG.

    Colour coding:
      - Row background = architectural family colour
      - Cell text = dark green for best value per column, black otherwise
      - Best value per column is also bold
    """
    # Build ordered rows (skip missing models)
    rows      = []
    row_colors= []
    families  = []
    for label, display, family in MODEL_ORDER:
        if label not in data:
            continue
        row = [display]
        for m in METRICS:
            v = data[label].get(m, float("nan"))
            row.append(f"{v:.4f}")
        rows.append(row)
        row_colors.append(FAMILY_COLORS[family])
        families.append(family)

    if not rows:
        print("  No data to render — check that results/ contains aggregate CSVs.")
        return

    n_rows = len(rows)
    n_cols = 1 + len(METRICS)

    # Figure size: wider than tall
    fig_w = 2.2 + len(METRICS) * 0.95
    fig_h = 0.38 * (n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = ["Architecture / Backbone"] + METRIC_LABELS
    col_widths  = [0.23] + [0.077] * len(METRICS)   # fractions of figure width

    tbl = ax.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "center",
        colWidths   = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1A2E5A")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
        cell.set_edgecolor("#FFFFFF")

    # Style data rows
    for i, (row, bg, family) in enumerate(zip(rows, row_colors, families)):
        for j in range(n_cols):
            cell = tbl[i+1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#CCCCCC")
            if j == 0:
                # Architecture name — left-aligned, slightly smaller
                cell.set_text_props(fontsize=8, ha="left")
            else:
                m = METRICS[j-1]
                v = float(row[j])
                if abs(v - best[m]) < 1e-4:
                    # Best in column: bold dark green
                    cell.set_text_props(color="#1B5E20", fontweight="bold",
                                        fontsize=8.5)
                else:
                    cell.set_text_props(color="#111111", fontsize=8.5)

    # Family group separators — thicker top border on first row of each family
    prev_family = None
    for i, family in enumerate(families):
        if family != prev_family:
            for j in range(n_cols):
                tbl[i+1, j].visible_edges = "BRTL"  # keep all edges
                # Draw a thicker top line
                # if i > 0:
                #     tbl[i+1, j].get_ec()  # ensure rendered
        prev_family = family

    # Legend
    legend_patches = [
        mpatches.Patch(color=col, label=fam)
        for fam, col in FAMILY_COLORS.items()
    ]
    ax.legend(
        handles    = legend_patches,
        loc        = "upper center",
        bbox_to_anchor = (0.5, -0.02),
        ncol       = 3,
        fontsize   = 7.5,
        frameon    = True,
        title      = "Architectural Family",
        title_fontsize = 8,
    )

    # plt.title(
    #     "Semantic Rip Current Segmentation — Test-Set Results (RipVIS val, n=4,349)",
    #     fontsize   = 10,
    #     fontweight = "bold",
    #     color      = "#1A2E5A",
    #     pad        = 8,
    # )
    # plt.figtext(0.01, 0.005,
    #     "* Best value per column  |  † Diffusion model did not converge; "
    #     "result reflects incomplete training (22/32 epochs)",
    #     fontsize=7, color="#555555")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Graphic table saved: {OUTPUT_PNG}  (insert into manuscript)")


if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    data = load_results()
    if not data:
        print("No aggregate CSVs found in results/. Run evaluate_test_set.py first.")
    else:
        best = print_and_save(data)
        save_graphic_table(data, best)
