"""
Author: Chisom Nwachukwu
Date: 2026-03-xx

Train a semantic segmentation model to detect rip currents in images
from CCTV cameras, drones, and mobile phones.

Key design decisions
--------------------
* Architecture  : U-Net/FPN with a resnet50 encoder (fast, CPU-friendly).
                  Other options are commented out below (FPN, ResNet50, etc.).
* Loss function : BCE + Dice loss — handles severe class imbalance (rip
                  current pixels are a small fraction of each frame).
* Augmentations : Tuned for beach/ocean imagery (brightness, haze, flips).
* Validation    : Reports IoU (Intersection-over-Union), the standard metric
                  for binary segmentation tasks.
* Checkpointing : Saves the best model (by val IoU) automatically.

Folder structure expected
--------------------------
    data/
      train/
        images/   ← JPG or PNG frames
        masks/    ← PNG binary masks (255 = rip, 0 = background)
      val/
        images/
        masks/

Usage (CPU-optimised defaults)
-------------------------------
    pip install torch torchvision segmentation-models-pytorch albumentations tqdm
    python train_segmentation.py

To run on a GPU just set DEVICE = 'cuda' and increase BATCH_SIZE to 8+.

Datasets Used: 
    1. RipVIS dataset (https://ripvis.ai / arXiv:2504.01128).
    2. https://www.kaggle.com/datasets/harsh1tha/  - ripcurrentdatasetNTIRE_2026_Rip_Current_Detection_and_Segmentation__RipDetSeg__Challenge___Participants_report_template (Unzipped Files)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import time
import random
from sklearn.metrics import fbeta_score as sklearn_fbeta
from scipy.ndimage import binary_erosion

# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION  — edit these to match your hardware and experiment goals
# ══════════════════════════════════════════════════════════════════════════════

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 256       # Resize all images to this square size before training.
                            # 256 is fast on CPU; use 512 for higher detail on GPU.
BATCH_SIZE   = 8         # 1–2 on CPU; 8–16 on GPU with 8 GB+ VRAM.
NUM_WORKERS  = 2         # 0 on CPU / Windows / notebooks; 2–4 on Linux GPU.
EPOCHS       = 50        # Increase to 50–100 for a full training run.
LR           = 5e-5 #1e-4      # AdamW initial learning rate.
WEIGHT_DECAY = 1e-5      # L2 regularisation (prevents over-fitting).

# Class imbalance weight: rip-current pixels are rare, so we penalise
# missing them more heavily than false positives.
# Tune this: if the model predicts mostly background, raise pos_weight.
POS_WEIGHT   = 10.0 #3.0       # weight applied to the positive (rip) class in BCE

TRAIN_IMGS   = "data_local/train_local/images"
TRAIN_MASKS  = "data_local/train_local/masks"
VAL_IMGS     = "data_local/val_local/images"
VAL_MASKS    = "data_local/val_local/masks"
CHECKPOINT   = "deeplabv3plus_resnet50.pth"   # saved when val IoU improves


# ══════════════════════════════════════════════════════════════════════════════
#   DATASET
# ══════════════════════════════════════════════════════════════════════════════

class RipSegDataset(Dataset):
    """
    Loads image–mask pairs for rip current segmentation.

    Images : RGB JPEG or PNG frames from CCTV / drone / mobile cameras.
    Masks  : Grayscale PNGs where pixel value > 127 means "rip current".
              The mask name must match the image name (different extension is OK).
    """

    def __init__(self, images_dir: str, masks_dir: str, transforms=None):
        images_dir = Path(images_dir)
        masks_dir  = Path(masks_dir)

        # Collect all image paths (JPG and PNG)
        self.images = sorted(images_dir.glob("*.jpg")) + \
                      sorted(images_dir.glob("*.png"))

        if len(self.images) == 0:
            raise FileNotFoundError(f" No images found in {images_dir}")

        self.masks_dir  = masks_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path  = self.images[idx]
        # Masks are stored as PNG — strip original extension and add .png
        mask_path = self.masks_dir / (img_path.stem + ".png")

        # ── Load image as RGB numpy array ──────────────────────────────────
        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8

        # ── Load mask as binary float array ───────────────────────────────
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))  # (H, W) uint8
        else:
            # If no mask exists for this image, treat the whole frame as background.
            # This happens for unannotated "negative" frames.
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = (mask > 127).astype("float32")  # binarise: {0.0, 1.0}

        # ── Apply augmentations ───────────────────────────────────────────
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # mask shape after ToTensorV2: (H, W) → we need (1, H, W) for BCE loss
        return image, mask.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
#   AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(train: bool = True, size: int = IMG_SIZE) -> A.Compose:
    """
    Return an Albumentations pipeline.

    Training augmentations are designed for beach / coastal imagery:
      • Horizontal flip  — waves/rips appear on both sides of beaches.
      • Brightness/contrast — handles glare, overcast vs. sunny days.
      • Hue/saturation shift — camera white-balance differences.
      • Gaussian noise   — simulates low-quality CCTV or compressed video.
      • Slight rotation  — handheld mobile camera tilt.
      • Coarse dropout   — forces the model to use context, not shortcuts.

    Validation uses a deterministic resize only (no random transforms).
    """
    if train:
        return A.Compose([
            # Resize to a fixed square — fast and deterministic
            A.Resize(size, size),

            # ── Geometric augmentations ───────────────────────────────────
            # Rip currents can appear anywhere along a beach; flipping is safe.
            A.HorizontalFlip(p=0.5),

            # Small rotations simulate tilted camera mounts or handheld phones.
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=0,
                p=0.5,
            ),

            # ── Colour / appearance augmentations ─────────────────────────
            # These simulate the wide range of lighting conditions seen in
            # CCTV footage, drone shots, and mobile phone videos.
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4,
            ),

            # Simulate haze / atmospheric scattering common in coastal scenes.
            # A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.2),
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.2),

            # Simulate CCTV compression artefacts and sensor noise.
            # A.GaussNoise(var_limit=(10, 40), p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),

            # ── Regularisation ────────────────────────────────────────────
            # Randomly zero out small rectangular patches; forces the model
            # to rely on context rather than isolated texture cues.
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),

            # ── Normalise and convert to tensor ──────────────────────────
            # Uses ImageNet mean/std because the encoder was pretrained on ImageNet.
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Validation: resize only — no random transforms
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ══════════════════════════════════════════════════════════════════════════════
#   LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss — differentiable approximation of the Dice coefficient.

    Dice = 2|A∩B| / (|A| + |B|)

    Why use it alongside BCE?
    --------------------------
    BCE treats every pixel independently and is dominated by the large
    number of background pixels. Dice loss directly optimises the overlap
    between prediction and ground-truth, making it robust to class imbalance.

    Parameters
    ----------
    pred_logits : raw model outputs (before sigmoid), shape (B, 1, H, W)
    targets     : binary ground-truth masks,          shape (B, 1, H, W)
    eps         : small constant to avoid division by zero

    Returns
    -------
    Scalar tensor — mean Dice loss over the batch.
    """
    preds = torch.sigmoid(pred_logits)                    # map to [0, 1]
    inter = (preds * targets).sum(dim=(1, 2, 3))          # |A ∩ B|
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))   # |A| + |B|
    dice  = (2 * inter + eps) / (denom + eps)             # per-sample Dice coeff
    return 1.0 - dice.mean()                              # loss = 1 − Dice


def combined_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = POS_WEIGHT,
    bce_weight: float = 0.3, 
    dice_weight: float = 0.7,
) -> torch.Tensor:
    """
    Weighted sum of BCE (with class-imbalance weight) and soft Dice loss.

    Parameters
    ----------
    pred_logits  : model raw output, shape (B, 1, H, W)
    targets      : binary masks,     shape (B, 1, H, W)
    pos_weight   : penalty multiplier for the positive (rip) class in BCE
    bce_weight   : contribution of BCE to total loss
    dice_weight  : contribution of Dice to total loss

    Returns
    -------
    Scalar tensor.
    """
    # pos_weight tensor must match device of predictions
    pw = torch.tensor([pos_weight], device=pred_logits.device)

    bce  = F.binary_cross_entropy_with_logits(pred_logits, targets, pos_weight=pw)
    dice = dice_loss(pred_logits, targets)

    return bce_weight * bce + dice_weight * dice


# ══════════════════════════════════════════════════════════════════════════════
#   METRICS
# ══════════════════════════════════════════════════════════════════════════════

def get_boundary(mask_np: np.ndarray, erosion_px: int = 2) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask via erosion.
    Boundary = original mask minus its eroded version.
    """
    eroded = binary_erosion(mask_np, iterations=erosion_px)
    return mask_np.astype(bool) & ~eroded

@torch.no_grad()
def compute_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    preds = (torch.sigmoid(pred_logits) > threshold).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    iou       = (tp / (tp + fp + fn + 1e-6)).mean().item()
    dice      = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall    = (tp / (tp + fn + 1e-6)).mean().item()

    # ── F2 and Boundary IoU require numpy ────────────────────────────
    preds_np   = preds.cpu().numpy()    # (B, 1, H, W)
    targets_np = targets.cpu().numpy()

    # F2 score — weights recall twice as heavily as precision
    f2 = sklearn_fbeta(
        targets_np.flatten().astype(int),
        preds_np.flatten().astype(int),
        beta=2,
        zero_division=0,
    )

    # Boundary IoU — evaluates edge accuracy, averaged over batch
    b_ious = []
    for p, t in zip(preds_np[:, 0], targets_np[:, 0]):  # iterate batch dim
        p_boundary = get_boundary(p.astype(bool))
        t_boundary = get_boundary(t.astype(bool))
        inter = (p_boundary & t_boundary).sum()
        union = (p_boundary | t_boundary).sum()
        b_ious.append(inter / (union + 1e-6))
    boundary_iou = float(np.mean(b_ious))

    return dict(
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        f2=f2,
        boundary_iou=boundary_iou,
    )


# ══════════════════════════════════════════════════════════════════════════════
#   MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model() -> torch.nn.Module:
    """
    Why MobileNetV2?
    ----------------
    • Lightweight — runs on CPU within reasonable time.
    • ImageNet-pretrained — the encoder already understands textures, edges,
      and water-like patterns; fine-tuning converges faster.
    • Encoder–decoder skip connections in U-Net preserve fine spatial detail
      needed to accurately outline rip current boundaries.

    Alternative architectures to try on GPU (just change the strings below):
    ─────────────────────────────────────────────────────────────────────────
    Better accuracy, more VRAM:
        smp.Unet(encoder_name='resnet34', ...)
        smp.Unet(encoder_name='resnet50', ...)
        smp.UnetPlusPlus(encoder_name='efficientnet-b2', ...)  # nested skip connections

    Speed / memory trade-off:
        smp.FPN(encoder_name='resnet18', ...)          # Feature Pyramid Network
        smp.DeepLabV3Plus(encoder_name='resnet50', ...) # atrous convolutions

    All models from segmentation_models_pytorch share the same API, so you
    can swap them in with a single line change.
    """
    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet50",  # pretrained encoder
        encoder_weights = "imagenet",      # use ImageNet weights
        in_channels     = 3,              # RGB input
        classes         = 1,             # binary: rip / no-rip
        activation      = None,           # raw logits — we apply sigmoid in loss
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device) -> float:
    """Run one full pass over the training set. Returns mean loss."""
    model.train()
    total_loss = 0.0

    # loop = tqdm(loader, desc="  train", leave=False)
    loop = tqdm(loader, desc="  train", leave=False, ascii=True, dynamic_ncols=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)                              # forward pass
        loss   = combined_loss(logits, masks)               # compute loss
        loss.backward()                                     # backprop
        optimizer.step()                                    # update weights

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Run one full pass over the validation set. Returns dict of metrics."""
    model.eval()
    accum = dict(iou=0.0, dice=0.0, precision=0.0, recall=0.0, f2=0.0, boundary_iou=0.0)
    n = 0

    # for images, masks in tqdm(loader, desc="  val  ", leave=False):
    for images, masks in tqdm(loader, desc="  val  ", leave=False, ascii=True, dynamic_ncols=False):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        batch_metrics = compute_metrics(logits, masks)
        for k in accum:
            accum[k] += batch_metrics[k]
        n += 1

    return {k: v / max(1, n) for k, v in accum.items()}


# ══════════════════════════════════════════════════════════════════════════════
#   CONTROL CONSISTENCY BETWEEN MACHINES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)

    # ── Optimiser ──────────────────────────────────────────────────────────
    # AdamW = Adam with decoupled weight decay — standard choice for vision models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Learning rate scheduler ────────────────────────────────────────────
    # ReduceLROnPlateau: halves the LR if val IoU stops improving.
    # This often gives a free boost without manual tuning.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",       # we want IoU to increase
        factor=0.5,       # multiply LR by 0.5 on plateau
        patience=3,       # wait 3 epochs before reducing
    )

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    train_ds = RipSegDataset(TRAIN_IMGS, TRAIN_MASKS, transforms=get_transforms(train=True))
    val_ds   = RipSegDataset(VAL_IMGS,   VAL_MASKS,   transforms=get_transforms(train=False))

    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
        drop_last   = True,   # prevents the last incomplete batch (batch_size=1)
                              # from crashing BatchNorm inside DeepLabV3+ ASPP
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_iou = 0.0
    best_metrics = {}
    history      = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        epoch_start = time.time()  

        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        # Pretty-print metrics
        print(
            f"  loss={train_loss:.4f}  "
            f"IoU={val_metrics['iou']:.4f}  "
            f"Dice={val_metrics['dice']:.4f}  "
            f"Precision={val_metrics['precision']:.4f}  "
            f"Recall={val_metrics['recall']:.4f}  "
            f"F2={val_metrics['f2']:.4f}  "
            f"BoundaryIoU={val_metrics['boundary_iou']:.4f}"
        )
        epoch_mins = (time.time() - epoch_start) / 60
        print(f"  Epoch time: {epoch_mins:.1f} min") 

        # Recall is especially important for a safety-critical task:
        # a missed rip current (false negative) is more dangerous than
        # a false alarm (false positive).
        if val_metrics["recall"] < 0.3 and epoch > 5:
            print("Low recall — consider raising POS_WEIGHT.")

        # ── Scheduler step ────────────────────────────────────────────────        
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["iou"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

        # ── Save best model ───────────────────────────────────────────────
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_iou":    best_val_iou,
                    "config": {
                        "encoder": "resnet50",
                        "architecture": "deeplabv3plus",
                        "img_size": IMG_SIZE,
                    },
                },
                CHECKPOINT,
            )
            print(f" Saved best model → {CHECKPOINT}  (IoU={best_val_iou:.4f})")

        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f" Training complete.  Best val IoU: {best_val_iou:.4f}")
    if best_metrics:
        print(
            f" Best metrics: IoU={best_metrics['iou']:.4f}, Dice={best_metrics['dice']:.4f}, "
            f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, "
            f"F2={best_metrics['f2']:.4f}, BoundaryIoU={best_metrics['boundary_iou']:.4f}"
        )
    else:
        print(" No best metrics recorded. The validation set may be empty or the metric never improved")
    print(f" Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()
