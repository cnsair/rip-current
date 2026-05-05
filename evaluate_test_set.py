"""
Batch evaluation over the RipVIS test partition (4,349 images).
Produces:
  results/per_image_metrics.csv    -- one row per image, all 9 metrics
  results/aggregate_metrics.csv    -- mean, std, median, 95% CI per metric
  results/bootstrap_ci.csv         -- bootstrap confidence intervals
  results/significance_tests.csv   -- Wilcoxon signed-rank results vs baseline

Run once per checkpoint:
  python evaluate_test_set.py --checkpoint manet_swin_tiny.pth \
                               --backbone tu-swin_tiny_patch4_window7_224 \
                               --architecture manet
                               

# UNet + ResNet50
python evaluate_test_set.py \
  --checkpoint unet_resnet50.pth \
  --backbone resnet50 \
  --architecture unet \
  --label unet_resnet50

# MAnet + Swin-Tiny
python evaluate_test_set.py \
  --checkpoint manet_swin_tiny.pth \
  --backbone tu-swin_tiny_patch4_window7_224 \
  --architecture manet \
  --label manet_swin_tiny

# SegFormer requires the separate segformer evaluation path in your
# train_segformer.py — adapt similarly with its own checkpoint loader                       
                               
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import stats
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_erosion
from sklearn.metrics import fbeta_score

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ── Config ────────────────────────────────────────────────────────────────────
TEST_IMAGES  = "data_local/ripvis_test/images"
TEST_MASKS   = "data_local/ripvis_test/masks"
RESULTS_DIR  = "results"
IMG_SIZE     = 256
THRESHOLD    = 0.5
DEVICE       = "cuda"
ENCODER_WEIGHTS = None   # weights come from checkpoint

# ── Transform ─────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── Model loading ─────────────────────────────────────────────────────────────
def build_model(backbone, architecture, checkpoint_path):
    arch_map = {
        "unet":           smp.Unet,
        "unetplusplus":   smp.UnetPlusPlus,
        "fpn":            smp.FPN,
        "deeplabv3plus":  smp.DeepLabV3Plus,
        "manet":          smp.MAnet,
    }
    if architecture == "attention_unet":
        model = smp.Unet(encoder_name=backbone, encoder_weights=None,
                         decoder_attention_type="scse",
                         in_channels=3, classes=1, activation=None)
    else:
        model = arch_map[architecture](encoder_name=backbone,
                                       encoder_weights=None,
                                       in_channels=3, classes=1, activation=None)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"  Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, "
          f"train-val mIoU={ckpt.get('val_iou',0):.4f})")
    return model

# ── Per-image metrics ─────────────────────────────────────────────────────────
def get_boundary(mask_np, px=2):
    return mask_np.astype(bool) & ~binary_erosion(mask_np, iterations=px)

def compute_single(pred_bin, gt_bin):
    """All 9 metrics for one image. Both arrays are 0/1 uint8 numpy."""
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, 1 - gt_bin).sum())
    fn = int(np.logical_and(1 - pred_bin, gt_bin).sum())
    tn = int(np.logical_and(1 - pred_bin, 1 - gt_bin).sum())
    total = tp + fp + fn + tn

    iou       = tp / (tp + fp + fn + 1e-6)
    iou_bg    = tn / (tn + fp + fn + 1e-6)
    miou      = (iou + iou_bg) / 2.0
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    aacc      = (tp + tn) / (total + 1e-6)
    macc      = (recall + tn / (tn + fp + 1e-6)) / 2.0
    f2        = fbeta_score(gt_bin.flatten(), pred_bin.flatten(),
                            beta=2, zero_division=0)

    pb = get_boundary(pred_bin)
    gb = get_boundary(gt_bin)
    b_inter = int(np.logical_and(pb, gb).sum())
    b_union = int(np.logical_or(pb, gb).sum())
    biou    = b_inter / (b_union + 1e-6)

    return dict(miou=miou, iou=iou, dice=dice, recall=recall,
                precision=precision, f2=f2, biou=biou, aacc=aacc, macc=macc)

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Return (mean, lower, upper) via bootstrap resampling."""
    values = np.array(values)
    boot_means = [np.random.choice(values, len(values), replace=True).mean()
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
    return values.mean(), lo, hi

# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate(model, out_prefix):
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    image_paths = sorted(Path(TEST_IMAGES).glob("*.jpg")) + \
                  sorted(Path(TEST_IMAGES).glob("*.png"))
    masks_dir   = Path(TEST_MASKS)

    metrics_keys = ["miou","iou","dice","recall","precision",
                    "f2","biou","aacc","macc"]
    all_rows   = []
    per_metric = {k: [] for k in metrics_keys}

    for img_path in tqdm(image_paths, desc="Evaluating", ascii=True):
        mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        img  = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        aug    = transform(image=img)
        tensor = aug["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_bin = (probs >= THRESHOLD).astype("uint8")
        # Resize prediction back to original mask resolution for fair comparison
        pred_full = np.array(Image.fromarray(pred_bin * 255).resize(
            (orig_w, orig_h), Image.NEAREST)) // 255

        gt_raw  = np.array(Image.open(mask_path).convert("L"))
        gt_bin  = (gt_raw > 127).astype("uint8")

        m = compute_single(pred_full, gt_bin)
        row = {"image": img_path.name}
        row.update(m)
        all_rows.append(row)
        for k in metrics_keys:
            per_metric[k].append(m[k])

    # ── Write per-image CSV ───────────────────────────────────────────────────
    per_image_path = f"{RESULTS_DIR}/{out_prefix}_per_image.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image"] + metrics_keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Per-image metrics saved: {per_image_path}")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    agg_path = f"{RESULTS_DIR}/{out_prefix}_aggregate.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric","mean","std","median",
                          "ci95_lo","ci95_hi","min","max"])
        for k in metrics_keys:
            vals = np.array(per_metric[k])
            mean, lo, hi = bootstrap_ci(vals)
            writer.writerow([k, f"{mean:.4f}", f"{vals.std():.4f}",
                              f"{np.median(vals):.4f}",
                              f"{lo:.4f}", f"{hi:.4f}",
                              f"{vals.min():.4f}", f"{vals.max():.4f}"])
    print(f"  Aggregate stats saved:   {agg_path}")

    # Print summary to terminal
    print(f"\n{'─'*65}")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
    print(f"{'─'*65}")
    for k in metrics_keys:
        vals = np.array(per_metric[k])
        mean, lo, hi = bootstrap_ci(vals, n_boot=5000)
        print(f"  {k:<12} {mean:>8.4f} {vals.std():>8.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"{'─'*65}")
    print(f"  N images evaluated: {len(all_rows)}")

    return per_metric

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--backbone",     required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--label",        default=None,
                        help="Short label for output files, e.g. manet_swin")
    args = parser.parse_args()

    label = args.label or f"{args.architecture}_{args.backbone.replace('tu-','').replace('/','_')}"

    print(f"\n{'='*65}")
    print(f"  Evaluating: {label}")
    print(f"  Device:     {DEVICE}")
    print(f"  Test set:   {TEST_IMAGES}")
    print(f"{'='*65}\n")

    model = build_model(args.backbone, args.architecture, args.checkpoint)
    evaluate(model, out_prefix=label)