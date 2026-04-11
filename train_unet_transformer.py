"""
Author: Chisom Nwachukwu
Date: 2026-03-xx

Train a semantic segmentation model to detect rip currents in images
from CCTV cameras, drones, and mobile phones.

Key design decisions
--------------------
* Architecture  : U-Net decoder with a VMamba-Tiny encoder (Mamba-based
                  transformer backbone). VMamba uses a 2-D Selective Scan
                  (SS2D) across four spatial directions, giving long-range
                  spatial awareness at linear (not quadratic) compute cost.
                  The BACKBONE + ARCHITECTURE constants let you swap in
                  ConvNeXt or Swin Transformer with a one-line change.
* Loss function : BCE + Dice loss -- handles severe class imbalance (rip
                  current pixels are a small fraction of each frame).
* Augmentations : Tuned for beach/ocean imagery (brightness, haze, flips).
* Validation    : Reports IoU, Dice, Precision, Recall, F2, BoundaryIoU.
* Checkpointing : Saves the best model (by val IoU) automatically.
* Resumption    : Set RESUME_FROM + RESUME_EPOCH to restart interrupted runs.

Folder structure expected
--------------------------
    data/
      train/
        images/   <- JPG or PNG frames
        masks/    <- PNG binary masks (255 = rip, 0 = background)
      val/
        images/
        masks/

Install
-------
    pip install torch torchvision segmentation-models-pytorch albumentations tqdm
    pip install timm>=1.0.3    # VMamba, ConvNeXt, Swin all come from timm

Datasets Used:
    1. RipVIS dataset (https://ripvis.ai / arXiv:2504.01128).
    2. NTIRE 2026 RipDetSeg Challenge dataset (Kaggle).
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Suppress Albumentations version-check network call that spams the terminal
# with "getaddrinfo failed" warnings every epoch on offline machines.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# timm is required to supply VMamba-Tiny (and ConvNeXt / Swin)
# as encoders.  smp's "tu-" prefix delegates backbone creation to timm,
# so any model in timm's registry works without any other code change.
# Install: pip install timm>=1.0.3
import timm  # noqa: F401 -- imported so smp can discover timm encoders

import time
import random
from sklearn.metrics import fbeta_score as sklearn_fbeta
from scipy.ndimage import binary_erosion


# ==============================================================================
#   CONFIGURATION
# ==============================================================================

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 256       # VMamba works well at 256; raise to 512 for more
                            # spatial detail if VRAM allows (~10 GB for batch=8).
BATCH_SIZE   = 8
NUM_WORKERS  = 2
EPOCHS       = 50
LR           = 5e-5
WEIGHT_DECAY = 1e-5

POS_WEIGHT   = 10.0      # Weight applied to the positive (rip) class in BCE.

# BACKBONE and ARCHITECTURE are the two constants to edit when
# comparing models in an ablation study.  Everything else stays the same.
#
# The "tu-" prefix tells smp to load the encoder from timm instead of its
# own registry.  timm's VMamba uses a pure-PyTorch selective scan -- no
# mamba-ssm CUDA kernel build is required on Windows.
#
# To list available VMamba names on your timm version:
#   python -c "import timm; print([m for m in timm.list_models() if 'vmamba' in m])"
#
# Backbone swap guide  (change BACKBONE + CHECKPOINT, nothing else):
# +--------------------------------------------------+---------+---------------+
# | BACKBONE string                                  | ~Params | Notes         |
# +--------------------------------------------------+---------+---------------+
# | "tu-vmamba_tiny_s2l5"   <- default (this file)  |  ~22 M  | SS2D 4-dir    |
# | "tu-vmamba_small_s2l15"                          |  ~50 M  | more capacity |
# | "tu-convnext_tiny"                               |  ~28 M  | pure CNN fast |
# | "tu-convnext_small"                              |  ~50 M  |               |
# | "tu-swin_tiny_patch4_window7_224"                |  ~28 M  | Swin Transf.  |
# | "tu-swin_small_patch4_window7_224"               |  ~50 M  |               |
# | "resnet50"  (set ENCODER_WEIGHTS="imagenet")     |  ~25 M  | plain CNN     |
# +--------------------------------------------------+---------+---------------+
#
# Architecture swap guide  (change ARCHITECTURE + CHECKPOINT, nothing else):
# "unet"          -> classic encoder-decoder with skip connections (default)
# "unetplusplus"  -> nested dense skip connections, often +1-2 IoU
# "fpn"           -> Feature Pyramid Network, good for multi-scale targets
BACKBONE        = "tu-vmamba_tiny_s2l5"
ARCHITECTURE    = "unet"

# set ENCODER_WEIGHTS to None for timm backbones -- timm loads its
# own ImageNet pretrained weights internally via its pretrained registry.
# For native smp backbones such as "resnet50", use ENCODER_WEIGHTS="imagenet".
ENCODER_WEIGHTS = None

TRAIN_IMGS  = "data_local/train_local/images"
TRAIN_MASKS = "data_local/train_local/masks"
VAL_IMGS    = "data_local/val_local/images"
VAL_MASKS   = "data_local/val_local/masks"

# CHANGE: checkpoint name now reflects the active backbone + architecture so
# multiple experiment checkpoints can coexist on disk without overwriting.
CHECKPOINT  = "unet_vmamba_tiny.pth"

# CHANGE: resume support.  Set RESUME_FROM to the checkpoint file path and
# RESUME_EPOCH to the last fully completed epoch to restart a crashed run.
# Leave RESUME_FROM = None to always start from scratch.
RESUME_FROM  = None   # e.g. "unet_vmamba_tiny.pth"
RESUME_EPOCH = 0      # last fully completed epoch (used only with RESUME_FROM)


# ==============================================================================
#   DATASET  -- unchanged from baseline
# ==============================================================================

class RipSegDataset(Dataset):
    """
    Loads image-mask pairs for rip current segmentation.

    Images : RGB JPEG or PNG frames from CCTV / drone / mobile cameras.
    Masks  : Grayscale PNGs where pixel value > 127 means "rip current".
             The mask name must match the image name (different extension OK).
    """

    def __init__(self, images_dir: str, masks_dir: str, transforms=None):
        images_dir = Path(images_dir)
        masks_dir  = Path(masks_dir)

        self.images = sorted(images_dir.glob("*.jpg")) + \
                      sorted(images_dir.glob("*.png"))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        self.masks_dir  = masks_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path  = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = (mask > 127).astype("float32")   # binarise: {0.0, 1.0}

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.unsqueeze(0)   # mask -> (1, H, W) for BCE loss


# ==============================================================================
#   AUGMENTATIONS  -- unchanged from baseline
# ==============================================================================

def get_transforms(train: bool = True, size: int = IMG_SIZE) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=0,
                p=0.5,
            ),
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
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ==============================================================================
#   LOSS FUNCTION  -- unchanged from baseline
# ==============================================================================

def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    preds = torch.sigmoid(pred_logits)
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice  = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = POS_WEIGHT,
    bce_weight: float = 0.3,
    dice_weight: float = 0.7,
) -> torch.Tensor:
    pw   = torch.tensor([pos_weight], device=pred_logits.device)
    bce  = F.binary_cross_entropy_with_logits(pred_logits, targets, pos_weight=pw)
    dice = dice_loss(pred_logits, targets)
    return bce_weight * bce + dice_weight * dice


# ==============================================================================
#   METRICS  -- unchanged from baseline
# ==============================================================================

def get_boundary(mask_np: np.ndarray, erosion_px: int = 2) -> np.ndarray:
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

    preds_np   = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    f2 = sklearn_fbeta(
        targets_np.flatten().astype(int),
        preds_np.flatten().astype(int),
        beta=2,
        zero_division=0,
    )

    b_ious = []
    for p, t in zip(preds_np[:, 0], targets_np[:, 0]):
        p_boundary = get_boundary(p.astype(bool))
        t_boundary = get_boundary(t.astype(bool))
        inter = (p_boundary & t_boundary).sum()
        union = (p_boundary | t_boundary).sum()
        b_ious.append(inter / (union + 1e-6))
    boundary_iou = float(np.mean(b_ious))

    return dict(iou=iou, dice=dice, precision=precision,
                recall=recall, f2=f2, boundary_iou=boundary_iou)


# ==============================================================================
#   MODEL
# ==============================================================================

def build_model() -> torch.nn.Module:
    """
    Build a segmentation model from BACKBONE + ARCHITECTURE constants.

    Why VMamba-Tiny as the default backbone?
    ----------------------------------------
    VMamba-Tiny (~22 M parameters) replaces the standard CNN encoder with a
    Mamba-based Vision State-Space Model.  Its 2-D Selective Scan (SS2D)
    sweeps the feature map in four directions before merging, giving it
    long-range spatial context that CNNs only reach after many stacked layers.
    Unlike standard Vision Transformers, VMamba's compute scales linearly with
    image size (not quadratically), making it practical at 256x256 on a T4.
    Paired with UNet's skip connections it combines fine-grained boundary
    information from shallow layers with global channel geometry from deep
    layers -- both are important for rip current detection.

    Why "tu-" prefix?
    -----------------
    smp's "tu-" prefix delegates encoder creation to timm.  Any model in
    timm's registry can be used as a backbone with a one-string change.
    timm's VMamba uses a pure-PyTorch selective scan -- no mamba-ssm CUDA
    kernel compilation is required on Windows.

    Swapping the backbone (change BACKBONE constant only):
    ------------------------------------------------------
    "tu-vmamba_tiny_s2l5"             <- VMamba-Tiny  (~22 M)  [default]
    "tu-vmamba_small_s2l15"           <- VMamba-Small (~50 M)
    "tu-convnext_tiny"                <- ConvNeXt-Tiny (~28 M)
    "tu-convnext_small"               <- ConvNeXt-Small (~50 M)
    "tu-swin_tiny_patch4_window7_224" <- Swin-Tiny (~28 M)
    "resnet50"  + ENCODER_WEIGHTS="imagenet"  <- plain ResNet-50

    Swapping the architecture (change ARCHITECTURE constant only):
    --------------------------------------------------------------
    "unet"         -> classic skip-connection encoder-decoder (default)
    "unetplusplus" -> nested dense skip connections, often +1-2 IoU
    "fpn"          -> Feature Pyramid Network
    """
    # architecture is selected via the ARCHITECTURE constant
    # instead of being hardcoded. The dispatch dict maps
    # string keys to smp constructors; adding a new architecture only
    # requires adding one entry here, with no other code changes needed.
    arch_map = {
        "unet":         smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn":          smp.FPN,
    }
    if ARCHITECTURE not in arch_map:
        raise ValueError(
            f"Unknown ARCHITECTURE '{ARCHITECTURE}'. "
            f"Choose from: {list(arch_map.keys())}"
        )

    # encoder_name and encoder_weights now come from the BACKBONE and
    # ENCODER_WEIGHTS constants.  For timm "tu-" backbones ENCODER_WEIGHTS
    # must be None -- timm handles pretrained weight loading internally.
    # For native smp backbones such as "resnet50", set ENCODER_WEIGHTS="imagenet".
    model = arch_map[ARCHITECTURE](
        encoder_name    = BACKBONE,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels     = 3,    # RGB -- unchanged
        classes         = 1,    # binary rip / no-rip -- unchanged
        activation      = None, # raw logits; sigmoid applied in loss/metrics
    )
    return model


# ==============================================================================
#   TRAINING LOOP  -- unchanged from baseline
# ==============================================================================

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="  train", leave=False, ascii=True, dynamic_ncols=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    accum = dict(iou=0.0, dice=0.0, precision=0.0,
                 recall=0.0, f2=0.0, boundary_iou=0.0)
    n = 0
    for images, masks in tqdm(loader, desc="  val  ", leave=False,
                               ascii=True, dynamic_ncols=False):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        batch_metrics = compute_metrics(logits, masks)
        for k in accum:
            accum[k] += batch_metrics[k]
        n += 1
    return {k: v / max(1, n) for k, v in accum.items()}


# ==============================================================================
#   SEED  -- unchanged from baseline
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==============================================================================
#   MAIN
# ==============================================================================

def train() -> None:
    # print backbone + architecture so the terminal log is self-documenting.
    print(f"Device: {DEVICE}")
    print(f"Backbone: {BACKBONE}  |  Architecture: {ARCHITECTURE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    # -- Model -----------------------------------------------------------------
    model = build_model().to(DEVICE)

    # -- Optimiser -------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -- LR scheduler ----------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # -- Datasets & DataLoaders ------------------------------------------------
    train_ds = RipSegDataset(TRAIN_IMGS, TRAIN_MASKS, transforms=get_transforms(train=True))
    val_ds   = RipSegDataset(VAL_IMGS,   VAL_MASKS,   transforms=get_transforms(train=False))
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
    )

    # -- Training state --------------------------------------------------------
    best_val_iou = 0.0
    best_metrics = {}
    history      = []
    start_epoch  = 1    # default: train from scratch

    # resume support.
    # Restores model weights AND optimiser momentum (AdamW m/v buffers) so
    # training continues exactly where it left off with no loss spike.
    # The LR scheduler is NOT saved because ReduceLROnPlateau re-infers its
    # patience counter from the IoU values it sees on subsequent epochs,
    # which is the correct behaviour after a restart.
    # Usage: set RESUME_FROM = "unet_vmamba_tiny.pth" and RESUME_EPOCH = N
    # where N is the last epoch that printed a result before the crash.
    if RESUME_FROM and Path(RESUME_FROM).exists():
        print(f"\nResuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_iou = ckpt["val_iou"]
        start_epoch  = RESUME_EPOCH + 1
        print(f"  Restored epoch {RESUME_EPOCH}  |  best IoU so far: {best_val_iou:.4f}")
    else:
        print("Starting training from scratch.")

    # -- Epoch loop ------------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        epoch_start = time.time()

        train_loss  = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(
            f"  loss={train_loss:.4f}  "
            f"IoU={val_metrics['iou']:.4f}  "
            f"Dice={val_metrics['dice']:.4f}  "
            f"Precision={val_metrics['precision']:.4f}  "
            f"Recall={val_metrics['recall']:.4f}  "
            f"F2={val_metrics['f2']:.4f}  "
            f"BoundaryIoU={val_metrics['boundary_iou']:.4f}"
        )
        print(f"  Epoch time: {(time.time() - epoch_start) / 60:.1f} min")

        if val_metrics["recall"] < 0.3 and epoch > 5:
            print("  Low recall -- consider raising POS_WEIGHT.")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["iou"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_iou":         best_val_iou,
                    # config dict records the live BACKBONE and
                    # ARCHITECTURE constants so every checkpoint is
                    # self-describing -- you can always tell which model
                    # produced it without reading the filename.
                    "config": {
                        "backbone":     BACKBONE,
                        "architecture": ARCHITECTURE,
                        "img_size":     IMG_SIZE,
                    },
                },
                CHECKPOINT,
            )
            print(f"  Saved best model -> {CHECKPOINT}  (IoU={best_val_iou:.4f})")

        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

    # -- Final summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Training complete.  Best val IoU: {best_val_iou:.4f}")
    if best_metrics:
        print(
            f" Best metrics: IoU={best_metrics['iou']:.4f}, "
            f"Dice={best_metrics['dice']:.4f}, "
            f"Precision={best_metrics['precision']:.4f}, "
            f"Recall={best_metrics['recall']:.4f}, "
            f"F2={best_metrics['f2']:.4f}, "
            f"BoundaryIoU={best_metrics['boundary_iou']:.4f}"
        )
    else:
        print(" No best metrics recorded.")
    print(f" Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()