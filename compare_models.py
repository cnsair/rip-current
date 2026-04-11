"""
compare_models.py
=================
Train multiple segmentation architectures and backbones on the RipVIS dataset,
then produce a professional comparison table showing which performs best.

This is the standard approach in academic segmentation papers — train each
model under identical conditions and compare on the same validation set.

Models compared
---------------
    U-Net       + MobileNetV2   (your current baseline)
    U-Net       + ResNet34      (stronger encoder, same decoder)
    U-Net       + EfficientNet-B0
    FPN         + ResNet34      (different decoder style)
    DeepLabV3+  + ResNet50      (atrous convolutions)
    UNet++      + EfficientNet-B0 (nested skip connections)

All models use identical:
    - Training data
    - Loss function (BCE + Dice)
    - Optimiser (AdamW)
    - Learning rate and scheduler
    - Image size and augmentations
    - Number of epochs

Usage
-----
    python compare_models.py

Results saved to:
    comparison_results.csv   — raw numbers
    comparison_table.png     — publication-ready figure

Requirements
------------
    pip install torch torchvision segmentation-models-pytorch
               albumentations tqdm pandas matplotlib
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — shared across ALL models for a fair comparison
# ══════════════════════════════════════════════════════════════════════════════

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 256
BATCH_SIZE   = 1
NUM_WORKERS  = 0
EPOCHS       = 30        # use same epoch count for every model
LR           = 5e-5
WEIGHT_DECAY = 1e-5
POS_WEIGHT   = 10.0

TRAIN_IMGS   = "data/train/images"
TRAIN_MASKS  = "data/train/masks"
VAL_IMGS     = "data/val/images"
VAL_MASKS    = "data/val/masks"
RESULTS_CSV  = "comparison_results.csv"
RESULTS_PNG  = "comparison_table.png"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — add or remove models here
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (display_name, smp_class, encoder_name)
# smp_class must be a class from segmentation_models_pytorch
MODELS = [
    ("U-Net + MobileNetV2",    smp.Unet,        "mobilenet_v2"),
    ("U-Net + ResNet34",       smp.Unet,        "resnet34"),
    ("U-Net + EfficientNet-B0",smp.Unet,        "efficientnet-b0"),
    ("FPN + ResNet34",         smp.FPN,         "resnet34"),
    ("DeepLabV3+ + ResNet50",  smp.DeepLabV3Plus,"resnet50"),
    ("UNet++ + EfficientNet-B0",smp.UnetPlusPlus,"efficientnet-b0"),
]

# NOTE on Mask R-CNN:
# Mask R-CNN is an instance segmentation model and does not share the same
# API as segmentation_models_pytorch. It requires torchvision and a different
# training loop. It is excluded from this direct comparison but can be added
# separately using torchvision.models.detection.maskrcnn_resnet50_fpn.
# For binary semantic segmentation (rip vs background), U-Net family models
# are the standard comparison baseline in the literature.


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  (same as train_segmentation.py)
# ══════════════════════════════════════════════════════════════════════════════

class RipSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        images_dir = Path(images_dir)
        self.images = sorted(images_dir.glob("*.jpg")) + \
                      sorted(images_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No images in {images_dir}")
        self.masks_dir  = Path(masks_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = (mask > 127).astype("float32")
        if self.transforms:
            out   = self.transforms(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask.unsqueeze(0)


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9,1.1),
                     rotate=(-10,10), mode=0, p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
            A.HueSaturationValue(10, 20, 10, p=0.4),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# LOSS AND METRICS  (identical to train_segmentation.py)
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(logits, targets, eps=1e-6):
    preds = torch.sigmoid(logits)
    inter = (preds * targets).sum(dim=(1,2,3))
    denom = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return 1.0 - ((2*inter + eps) / (denom + eps)).mean()

def combined_loss(logits, targets):
    pw  = torch.tensor([POS_WEIGHT], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    return 0.3 * bce + 0.7 * dice_loss(logits, targets)

@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1-targets)).sum(dim=(1,2,3))
    fn = ((1-preds) * targets).sum(dim=(1,2,3))
    return dict(
        iou       = (tp / (tp+fp+fn+1e-6)).mean().item(),
        dice      = (2*tp / (2*tp+fp+fn+1e-6)).mean().item(),
        precision = (tp / (tp+fp+1e-6)).mean().item(),
        recall    = (tp / (tp+fn+1e-6)).mean().item(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION FOR ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(name, model_class, encoder_name,
                       train_loader, val_loader) -> dict:
    """
    Train one model for EPOCHS epochs and return its best validation metrics.
    Saves the best checkpoint as checkpoints/<safe_name>.pth
    """
    print(f"\n{'═'*60}")
    print(f"  Model : {name}")
    print(f"  Encoder: {encoder_name}  |  Device: {DEVICE}")
    print(f"{'═'*60}")

    # Build model
    model = model_class(
        encoder_name    = encoder_name,
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_iou     = 0.0
    best_metrics = {}
    start_time   = time.time()

    # Safe filename for checkpoint
    safe_name = name.replace(" ", "_").replace("+", "").replace("/", "")
    ckpt_dir  = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{safe_name}.pth"

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        for imgs, masks in tqdm(train_loader,
                                desc=f"  Epoch {epoch:02d}/{EPOCHS} train",
                                leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            combined_loss(model(imgs), masks).backward()
            optimizer.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        accum = dict(iou=0., dice=0., precision=0., recall=0.)
        n = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                m = compute_metrics(model(imgs), masks)
                for k in accum: accum[k] += m[k]
                n += 1
        val = {k: v/max(1,n) for k,v in accum.items()}

        print(f"  Epoch {epoch:02d}  "
              f"IoU={val['iou']:.4f}  Dice={val['dice']:.4f}  "
              f"Recall={val['recall']:.4f}  Precision={val['precision']:.4f}")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["iou"])
        if optimizer.param_groups[0]["lr"] < prev_lr:
            print(f"  LR reduced → {optimizer.param_groups[0]['lr']:.2e}")

        if val["iou"] > best_iou:
            best_iou     = val["iou"]
            best_metrics = val.copy()
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_iou": best_iou, "config": {"encoder": encoder_name}
            }, ckpt_path)
            print(f"  💾  New best IoU={best_iou:.4f} saved → {ckpt_path}")

    elapsed = time.time() - start_time
    best_metrics["training_time_min"] = round(elapsed / 60, 1)
    best_metrics["model"]             = name
    print(f"\n  ✅  Finished in {elapsed/60:.1f} min  |  Best IoU: {best_iou:.4f}")
    return best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE AND FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def save_results_table(results: list[dict]) -> None:
    """
    Save a CSV and a publication-style PNG comparison table.

    The PNG table is the kind you would include in a research paper or
    show your supervisor — rows are models, columns are metrics,
    best value in each column is highlighted in green.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("iou", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df)+1)

    # ── Save CSV ──────────────────────────────────────────────────────────
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n📄  Results saved → {RESULTS_CSV}")

    # ── Build PNG table ───────────────────────────────────────────────────
    display_cols = ["rank", "model", "iou", "dice", "precision",
                    "recall", "training_time_min"]
    col_labels   = ["Rank", "Model", "IoU ↑", "Dice ↑",
                    "Precision ↑", "Recall ↑", "Train Time (min)"]

    table_data = []
    for _, row in df[display_cols].iterrows():
        table_data.append([
            str(int(row["rank"])),
            row["model"],
            f"{row['iou']:.4f}",
            f"{row['dice']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['training_time_min']}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.7 + 2)))
    ax.axis("off")

    tbl = ax.table(
        cellText    = table_data,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best value in each metric column (green) and worst (light red)
    metric_col_indices = [2, 3, 4, 5]   # IoU, Dice, Precision, Recall
    metric_keys        = ["iou", "dice", "precision", "recall"]

    for col_idx, key in zip(metric_col_indices, metric_keys):
        values    = df[key].values
        best_row  = values.argmax() + 1   # +1 because row 0 is header
        worst_row = values.argmin() + 1

        tbl[best_row,  col_idx].set_facecolor("#2ecc71")   # green = best
        tbl[best_row,  col_idx].set_text_props(fontweight="bold")
        tbl[worst_row, col_idx].set_facecolor("#e74c3c")   # red = worst
        tbl[worst_row, col_idx].set_text_props(color="white")

    # Alternating row colours for readability
    for i in range(1, len(results)+1):
        bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            if tbl[i, j].get_facecolor() == (1,1,1,1) or \
               list(tbl[i, j].get_facecolor())[:3] == [1,1,1]:
                tbl[i, j].set_facecolor(bg)

    # Legend
    green_patch = mpatches.Patch(color="#2ecc71", label="Best value")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Worst value")
    ax.legend(handles=[green_patch, red_patch],
              loc="lower right", fontsize=9, framealpha=0.8)

    plt.title(
        "RipVIS Segmentation Model Comparison\n"
        f"(Val set, {IMG_SIZE}×{IMG_SIZE}, {EPOCHS} epochs, CPU={DEVICE=='cpu'})",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️   Comparison table saved → {RESULTS_PNG}")

    # Also print to terminal
    print(f"\n{'─'*60}")
    print(df[display_cols].to_string(index=False))
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"🖥️   Device : {DEVICE}")
    print(f"📐  Size={IMG_SIZE}  Batch={BATCH_SIZE}  Epochs={EPOCHS}  LR={LR}")
    print(f"🔬  Comparing {len(MODELS)} model configurations\n")

    # Build shared dataloaders — same data for every model
    train_ds = RipSegDataset(TRAIN_IMGS, TRAIN_MASKS, get_transforms(True))
    val_ds   = RipSegDataset(VAL_IMGS,   VAL_MASKS,   get_transforms(False))
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE=="cuda"))

    # Train each model and collect results
    all_results = []
    for name, model_class, encoder in MODELS:
        metrics = train_and_evaluate(
            name, model_class, encoder, train_loader, val_loader
        )
        all_results.append(metrics)

        # Save intermediate results after each model
        # so you don't lose data if the script crashes halfway
        pd.DataFrame(all_results).to_csv(RESULTS_CSV + ".partial", index=False)

    # Final table
    save_results_table(all_results)
    print("\n  All models trained. Check comparison_results.csv and comparison_table.png")


if __name__ == "__main__":
    main()