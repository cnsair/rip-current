"""
Converts COCO-format annotations (polygons or RLE masks) into binary PNG masks
suitable for training semantic segmentation models on the RipVIS dataset.

Dataset structure expected (matches HuggingFace RipVIS layout):
    ./train/coco_annotations/train.json
    ./train/sampled_images/images/<filename>.jpg
    ./val/coco_annotations/val.json
    ./val/sampled_images/images/<filename>.jpg

Output structure:
    data/train/images/  — copies of training images
    data/train/masks/   — binary PNG masks (255 = rip current, 0 = background)
    data/val/images/
    data/val/masks/

"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def build_binary_mask(coco_api: COCO, img_info: dict) -> np.ndarray:
    """
    Construct a binary mask for a single image by OR-ing all annotated regions.

    Supports both:
      - Polygon segmentations (list of [x1,y1,x2,y2,...] lists)
      - RLE segmentations (dict with 'counts' and 'size')

    Parameters
    ----------
    coco_api  : initialised COCO object for the split
    img_info  : one entry from coco.loadImgs(...)

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8
        1 where any rip-current annotation exists, 0 elsewhere.
    """
    h, w = img_info["height"], img_info["width"]
    # Start with an all-zero (background) mask
    mask = np.zeros((h, w), dtype=np.uint8)

    ann_ids = coco_api.getAnnIds(imgIds=[img_info["id"]], iscrowd=None)
    annotations = coco_api.loadAnns(ann_ids)

    if not annotations:
        # Image has no annotations → return blank mask (valid "negative" sample)
        return mask

    for ann in annotations:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        if isinstance(seg, list):
            # ── Polygon format ──────────────────────────────────────────────
            # seg is a list of polygons, each polygon is [x1,y1,x2,y2,...].
            # frPyObjects converts each to RLE, then merge combines them into
            # a single RLE before decoding.
            rles = maskUtils.frPyObjects(seg, h, w)
            rle = maskUtils.merge(rles)
            ann_mask = maskUtils.decode(rle)  # (H, W) uint8 {0,1}

        elif isinstance(seg, dict):
            # ── RLE format ─────────────────────────────────────────────────
            # Uncompressed RLE: {'counts': [...], 'size': [H, W]}
            # Compressed RLE:   {'counts': 'ZYX...', 'size': [H, W]}
            ann_mask = maskUtils.decode(seg)   # (H, W) uint8 {0,1}

        else:
            print(f"Unknown segmentation format for ann id {ann['id']}, skipping.")
            continue

        # Combine with existing mask using element-wise maximum (logical OR)
        mask = np.maximum(mask, (ann_mask > 0).astype(np.uint8))

    return mask


# ──────────────────────────────────────────────
# Main processing function
# ──────────────────────────────────────────────

def process_split(
    coco_json_path: Path,
    images_root: Path,
    out_images_dir: Path,
    out_masks_dir: Path,
    copy_images: bool = True,
) -> None:
    """
    Process one dataset split (train or val):
      1. Load COCO annotations.
      2. For each image, build a binary mask from all annotated polygons/RLEs.
      3. Optionally copy the source image to out_images_dir.
      4. Save the mask as a grayscale PNG (0 = background, 255 = rip current).

    Parameters
    ----------
    coco_json_path : path to the COCO JSON file for this split
    images_root    : directory containing the raw source images
    out_images_dir : destination directory for image copies
    out_masks_dir  : destination directory for mask PNG files
    copy_images    : if True, copy (not symlink) images to out_images_dir
    """
    print(f"\n Loading annotations: {coco_json_path}")
    coco = COCO(str(coco_json_path))

    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)

    img_ids = coco.getImgIds()
    images  = coco.loadImgs(img_ids)
    print(f"    Found {len(images)} images in annotation file.")

    skipped = 0
    for img_info in tqdm(images, desc="  Processing"):
        src_path  = Path(images_root) / img_info["file_name"]
        stem      = Path(img_info["file_name"]).stem          # filename without extension
        dst_image = out_images_dir / img_info["file_name"]
        dst_mask  = out_masks_dir  / (stem + ".png")          # masks always saved as PNG

        # ── Skip if source image is missing ───────────────────────────────
        if not src_path.exists():
            print(f"Missing source image: {src_path}")
            skipped += 1
            continue

        # ── Copy image (preserves original format: jpg/png) ───────────────
        if copy_images and not dst_image.exists():
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_image)

        # ── Build and save binary mask ─────────────────────────────────────
        if not dst_mask.exists():
            mask_arr = build_binary_mask(coco, img_info)
            # Scale 0/1 → 0/255 so standard image viewers show it correctly
            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
            mask_img.save(dst_mask)

    print(f"Done. Skipped {skipped} missing images.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    base = Path(".")   # adjust if your dataset is stored elsewhere

    # # ── Training split ────────────────────────────────────────────────────
    # process_split(
    #     coco_json_path = base / "train" / "coco_annotations" / "train.json",
    #     images_root    = base / "train" / "sampled_images"   / "images",
    #     out_images_dir = Path("data_two/train/images"),
    #     out_masks_dir  = Path("data_two/train/masks"),
    # )

    # # ── Validation split ──────────────────────────────────────────────────
    # process_split(
    #     coco_json_path = base / "val" / "coco_annotations" / "val.json",
    #     images_root    = base / "val" / "sampled_images"   / "images",
    #     out_images_dir = Path("data_two/val/images"),
    #     out_masks_dir  = Path("data_two/val/masks"),
    # )
    
        # ── Training split ────────────────────────────────────────────────────
    process_split(
        coco_json_path = base / "datasets" / "RipDetSeg_v1.1.6_train" / "train_annotations.json",
        images_root    = base / "datasets" / "RipDetSeg_v1.1.6_train" / "train_images",
        out_images_dir = Path("data_three/train/images"),
        out_masks_dir  = Path("data_three/train/masks"),
    )

    # ── Validation split ──────────────────────────────────────────────────
    process_split(
        coco_json_path = base / "datasets" / "RipDetSeg_v1.1.6_val" / "val_no_annotations.json",
        images_root    = base / "datasets" / "RipDetSeg_v1.1.6_val" / "val_images",
        out_images_dir = Path("data_three/val/images"),
        out_masks_dir  = Path("data_three/val/masks"),
    )

    print("\n  Preprocessing complete. Outputs in data_three/train/ and data_three/val/")
