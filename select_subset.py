"""
select_subset.py
================
Selects a balanced, representative subset of images for manual relabelling.

Strategy:
- Takes the audit report CSV as input
- Filters out clearly bad masks (fragmented, too large)
- Selects images that cover different rip sizes (small, medium, large)
- Outputs a folder of images ready to import into Label Studio

Usage:
    python select_subset.py \
        --report  audit_report/report.csv \
        --images  data/train/images \
        --out     subset_to_label \
        --n       3000
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def select_subset(report_csv: str, images_dir: str, out_dir: str, n: int = 300) -> None:
    out_dir    = Path(out_dir)
    images_dir = Path(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(report_csv)
    df["rip_fraction"] = df["rip_fraction"].astype(float)

    # ── Pool 1: Clean annotated images ────────────────────────────────────
    # These already have GT — we include them so the model doesn't lose
    # good training signal. We just need to verify the annotation is tight.
    clean = df[
        (df["issues"] == "OK") &
        (df["rip_fraction"] >= 0.02) &   # not suspiciously tiny
        (df["rip_fraction"] <= 0.45)      # not suspiciously huge
    ].copy()

    # Divide clean images into three rip-size bands so the subset
    # covers small, medium and large rip channels equally
    small  = clean[clean["rip_fraction"] < 0.05]
    medium = clean[(clean["rip_fraction"] >= 0.05) & (clean["rip_fraction"] < 0.15)]
    large  = clean[clean["rip_fraction"] >= 0.15]

    per_band = n // 4   # reserve one quarter for unannotated images below

    selected_clean = pd.concat([
        small.sample( min(per_band, len(small)),  random_state=42),
        medium.sample(min(per_band, len(medium)), random_state=42),
        large.sample( min(per_band, len(large)),  random_state=42),
    ])

    # ── Pool 2: Unannotated images to manually check ──────────────────────
    # We include a sample of EMPTY_MASK images so you can inspect them and
    # add labels if a rip is actually visible — fixing the missing-annotation
    # problem identified in the audit.
    empty = df[df["issues"] == "EMPTY_MASK"]
    selected_empty = empty.sample(min(per_band, len(empty)), random_state=42)

    # ── Combine and copy ──────────────────────────────────────────────────
    final = pd.concat([selected_clean, selected_empty]).drop_duplicates(subset="image")
    print(f"\n  Subset breakdown:")
    print(f"    Clean annotated (small rip)  : {len(small.sample(min(per_band, len(small)), random_state=42))}")
    print(f"    Clean annotated (medium rip) : {len(medium.sample(min(per_band, len(medium)), random_state=42))}")
    print(f"    Clean annotated (large rip)  : {len(large.sample(min(per_band, len(large)), random_state=42))}")
    print(f"    Unannotated (to inspect)     : {len(selected_empty)}")
    print(f"    ─────────────────────────────")
    print(f"    Total selected               : {len(final)}")

    # Copy selected images to the output folder
    missing = 0
    for img_name in final["image"]:
        src = images_dir / img_name
        if src.exists():
            shutil.copy2(src, out_dir / img_name)
        else:
            missing += 1

    print(f"\n  Copied {len(final) - missing} images → {out_dir}")
    if missing:
        print(f"   {missing} source images not found — check your images_dir path")

    # Save the list so you know exactly which images are in the subset
    final[["image", "rip_fraction", "issues"]].to_csv(
        out_dir / "selected_images.csv", index=False
    )
    print(f"  Selection log saved → {out_dir / 'selected_images.csv'}")
    print(f"\nNext step: import the '{out_dir}' folder into Label Studio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",  required=True, help="Path to audit_report/report.csv")
    parser.add_argument("--images",  required=True, help="data/train/images directory")
    parser.add_argument("--out",     required=True, help="Output folder for selected images")
    parser.add_argument("--n",       type=int, default=300, help="Target subset size (default 300)")
    args = parser.parse_args()

    select_subset(args.report, args.images, args.out, args.n)
