"""
gt_audit_and_export.py
======================
Two tools in one file:

PART 1 — GT Quality Audit
    Scans your existing masks and flags images whose ground truth is likely
    noisy or unusable: masks that are empty, suspiciously tiny, or
    implausibly large. Outputs a CSV report and a folder of side-by-side
    image+mask previews for manual inspection.

PART 2 — Label Studio Export → Training Masks
    After you have manually labelled a clean subset in Label Studio and
    exported it as a COCO JSON, this script converts those annotations
    into binary PNG masks in the same format your train_segmentation.py
    expects — so you can drop them straight into data/train/masks/.

Usage
-----
    # Step 1: audit existing GT
    python gt_audit_and_export.py --mode audit \
        --images data/train/images \
        --masks  data/train/masks \
        --out    audit_report

    # Step 2: convert Label Studio COCO export to binary masks
    python gt_audit_and_export.py --mode export \
        --coco   labelstudio_export.json \
        --images data/train/images \
        --out    data/train/masks_clean

Requirements
------------
    pip install pillow numpy pycocotools tqdm pandas opencv-python
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — GT QUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════

# Thresholds for flagging suspicious masks.
# Adjust these based on how your dataset looks.
MIN_RIP_FRACTION = 0.001   # masks with < 0.1% rip pixels → likely missing annotation
MAX_RIP_FRACTION = 0.60    # masks with > 60% rip pixels  → likely annotation error


def compute_mask_stats(mask_arr: np.ndarray) -> dict:
    """
    Compute basic statistics about a binary mask.

    Parameters
    ----------
    mask_arr : np.ndarray (H, W), dtype uint8, values {0, 255}

    Returns
    -------
    dict with:
        total_pixels   — H × W
        rip_pixels     — count of foreground (255) pixels
        rip_fraction   — rip_pixels / total_pixels
        num_components — number of connected rip regions (blobs)
        largest_blob   — pixel area of the largest connected component
    """
    binary = (mask_arr > 127).astype(np.uint8)
    total  = mask_arr.size
    rip    = int(binary.sum())

    # Connected component analysis — a rip current should form 1–3 blobs.
    # Many tiny disconnected blobs suggest noisy annotation.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    # stats rows: [x, y, w, h, area] — label 0 is background
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([0])
    largest_blob    = int(component_areas.max()) if len(component_areas) > 0 else 0
    num_components  = num_labels - 1  # subtract background label

    return dict(
        total_pixels   = total,
        rip_pixels     = rip,
        rip_fraction   = rip / total,
        num_components = num_components,
        largest_blob   = largest_blob,
    )


def flag_issues(stats: dict) -> list[str]:
    """
    Return a list of issue strings for a mask given its stats.
    Empty list means the mask looks clean.
    """
    issues = []
    if stats["rip_pixels"] == 0:
        issues.append("EMPTY_MASK")               # no rip annotation at all
    elif stats["rip_fraction"] < MIN_RIP_FRACTION:
        issues.append("TOO_SMALL")                # annotation is suspiciously tiny
    if stats["rip_fraction"] > MAX_RIP_FRACTION:
        issues.append("TOO_LARGE")                # annotation covers most of image
    if stats["num_components"] > 10:
        issues.append("FRAGMENTED")               # many disconnected blobs → noisy
    return issues


def save_preview(image_path: Path, mask_path: Path, out_path: Path) -> None:
    """
    Save a side-by-side preview: original image | mask overlay.
    The rip current region is highlighted in semi-transparent red.
    Useful for quickly visually verifying annotation quality.
    """
    img  = np.array(Image.open(image_path).convert("RGB"))
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L"))
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Create red overlay where mask > 127
    overlay       = img.copy()
    rip_region    = mask > 127
    overlay[rip_region] = (
        overlay[rip_region] * 0.4 + np.array([220, 50, 50]) * 0.6
    ).astype(np.uint8)

    # Stack original and overlay side by side
    divider   = np.ones((img.shape[0], 4, 3), dtype=np.uint8) * 200  # grey divider
    combined  = np.concatenate([img, divider, overlay], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(out_path)


def run_audit(images_dir: str, masks_dir: str, out_dir: str) -> None:
    """
    Audit all image–mask pairs. Outputs:
      audit_report/report.csv          — full stats for every image
      audit_report/flagged_previews/   — side-by-side previews of flagged images
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir / "flagged_previews"

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"\n🔍  Auditing {len(image_paths)} images...")

    rows         = []
    flagged_count = 0

    for img_path in tqdm(image_paths, desc="  Auditing"):
        mask_path = masks_dir / (img_path.stem + ".png")

        # If mask file is missing, treat as empty
        if not mask_path.exists():
            stats  = dict(total_pixels=0, rip_pixels=0, rip_fraction=0.0,
                          num_components=0, largest_blob=0)
            issues = ["MISSING_MASK"]
        else:
            mask_arr = np.array(Image.open(mask_path).convert("L"))
            stats    = compute_mask_stats(mask_arr)
            issues   = flag_issues(stats)

        issue_str = "|".join(issues) if issues else "OK"
        rows.append({
            "image":          img_path.name,
            "rip_fraction":   f"{stats['rip_fraction']:.4f}",
            "rip_pixels":     stats["rip_pixels"],
            "num_components": stats["num_components"],
            "largest_blob":   stats["largest_blob"],
            "issues":         issue_str,
        })

        # Save a visual preview for every flagged image
        if issues:
            flagged_count += 1
            save_preview(
                img_path, mask_path,
                previews_dir / (img_path.stem + "_preview.jpg"),
            )

    # Write CSV report
    report_path = out_dir / "report.csv"
    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)

    # Summary
    ok_count = len(df[df["issues"] == "OK"])
    print(f"\n  Clean masks : {ok_count}")
    print(f"  Flagged      : {flagged_count}")
    print(f"  Report saved : {report_path}")
    print(f"  Previews in  : {previews_dir}")
    print(f"\n  Open the previews folder and visually inspect flagged images.")
    print(f"  Use Label Studio to re-annotate the ones that look wrong.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — LABEL STUDIO COCO EXPORT → BINARY MASKS
# ══════════════════════════════════════════════════════════════════════════════

def export_labelstudio_masks(
    coco_json_path: str,
    images_dir: str,
    out_masks_dir: str,
) -> None:
    """
    Convert a Label Studio COCO export into binary PNG masks.

    Label Studio exports standard COCO JSON, so this uses the same
    polygon→mask conversion as preprocess_coco_to_mask.py.

    Parameters
    ----------
    coco_json_path : path to the JSON file exported from Label Studio
                     (Export → COCO format)
    images_dir     : directory containing the images you labelled
    out_masks_dir  : where to save the resulting binary PNG masks
    """
    coco_json_path = Path(coco_json_path)
    images_dir     = Path(images_dir)
    out_masks_dir  = Path(out_masks_dir)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading Label Studio export: {coco_json_path}")
    coco = COCO(str(coco_json_path))

    img_ids = coco.getImgIds()
    images  = coco.loadImgs(img_ids)
    print(f"    Found {len(images)} labelled images.")

    for img_info in tqdm(images, desc="  Exporting masks"):
        h, w      = img_info["height"], img_info["width"]
        stem      = Path(img_info["file_name"]).stem
        out_path  = out_masks_dir / (stem + ".png")

        # Start with an all-background mask
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids     = coco.getAnnIds(imgIds=[img_info["id"]])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            if isinstance(seg, list):
                # Polygon format — Label Studio uses this by default
                rles    = maskUtils.frPyObjects(seg, h, w)
                rle     = maskUtils.merge(rles)
                ann_mask = maskUtils.decode(rle)
            elif isinstance(seg, dict):
                # RLE format
                ann_mask = maskUtils.decode(seg)
            else:
                continue

            # OR the annotation into the mask
            mask = np.maximum(mask, (ann_mask > 0).astype(np.uint8))

        # Save as 0/255 grayscale PNG
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

    print(f"\n  Masks saved to: {out_masks_dir}")
    print(f"  Copy these into your data/train/masks/ or data/val/masks/ folder,")
    print(f"  then retrain with train_segmentation.py as normal.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GT audit and Label Studio export tool for RipVIS segmentation."
    )
    parser.add_argument(
        "--mode", required=True, choices=["audit", "export"],
        help="'audit' to check existing GT quality; 'export' to convert Label Studio annotations."
    )
    parser.add_argument("--images", help="Directory containing images.")
    parser.add_argument("--masks",  help="[audit mode] Directory containing existing masks.")
    parser.add_argument("--coco",   help="[export mode] Path to Label Studio COCO JSON export.")
    parser.add_argument("--out",    required=True, help="Output directory.")

    args = parser.parse_args()

    if args.mode == "audit":
        if not args.images or not args.masks:
            parser.error("--images and --masks are required for audit mode.")
        run_audit(args.images, args.masks, args.out)

    elif args.mode == "export":
        if not args.coco or not args.images:
            parser.error("--coco and --images are required for export mode.")
        export_labelstudio_masks(args.coco, args.images, args.out)
