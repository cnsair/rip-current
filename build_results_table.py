"""
build_results_table.py

Reads all aggregate CSVs and prints the formatted table for copy-paste
into the manuscript. Mean ± Std format, best value per column in bold [B].
"""

import csv
from pathlib import Path
import numpy as np

RESULTS_DIR = "results"
METRICS = ["miou","iou","dice","recall","precision","f2","biou","aacc","macc"]
LABELS  = {    # display names for the table rows
    "unet_resnet50":              "UNet / ResNet-50",
    "unet_resnet34":              "UNet / ResNet-34",
    "unet_efficientnet-b2":       "UNet / EfficientNet-B2",
    "unet_mobilenet_v2":          "UNet / MobileNetV2",
    "unetplusplus_efficientnet-b2": "UNet++ / EfficientNet-B2",
    "deeplabv3plus_resnet50":     "DeepLabV3+ / ResNet-50",
    "segformer_b2":               "SegFormer-B2",
    "attention_unet_resnet50":    "Attention UNet / ResNet-50",
    "manet_swin_tiny":            "MAnet / Swin-Tiny",
    "diffusion_resnet50":         "Diffusion Seg. / ResNet-50",
}

agg_files = sorted(Path(RESULTS_DIR).glob("*_aggregate.csv"))
results = {}
for f in agg_files:
    label = f.stem.replace("_aggregate", "")
    with open(f) as cf:
        reader = csv.DictReader(cf)
        results[label] = {row["metric"]: row for row in reader}

# Find best per column
best = {m: 0.0 for m in METRICS}
for label, data in results.items():
    for m in METRICS:
        if m in data:
            val = float(data[m]["mean"])
            if val > best[m]:
                best[m] = val

print("\nResults Table (Mean ± Std) | [B] = best per column\n")
header = f"{'Architecture/Backbone':<35}" + "".join(f"{m:>12}" for m in METRICS)
print(header)
print("─" * len(header))

for label, display in LABELS.items():
    if label not in results:
        continue
    row_str = f"{display:<35}"
    for m in METRICS:
        if m in results[label]:
            mean = float(results[label][m]["mean"])
            std  = float(results[label][m]["std"])
            marker = "[B]" if abs(mean - best[m]) < 1e-4 else "   "
            row_str += f"{mean:.3f}±{std:.3f}{marker}"
        else:
            row_str += f"{'—':>12}"
    print(row_str)