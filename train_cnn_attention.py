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
# IMPORTANT -- verify the exact backbone name on your timm version BEFORE
# setting BACKBONE.  Run:
#   python -c "import timm; print(timm.__version__)"
#   python -c "import timm; print([m for m in timm.list_models() if 'vmamba' in m])"
#   python -c "import timm; print([m for m in timm.list_models() if 'swin' in m][:5])"
#
# If VMamba returns [] your timm is too old -- upgrade with:
#   pip install --upgrade timm
# or use a Swin/ConvNeXt backbone instead (both available in older timm).
#
# Backbone swap guide  (change BACKBONE + CHECKPOINT, nothing else):
# +--------------------------------------------------+---------+---------------+
# | BACKBONE string                                  | ~Params | Notes         |
# +--------------------------------------------------+---------+---------------+
# | "tu-vmamba_tiny_s2l5"                            |  ~22 M  | needs timm>=1.0.3  |
# | "tu-vmamba_small_s2l15"                          |  ~50 M  | needs timm>=1.0.3  |
# | "tu-convnext_tiny"                               |  ~28 M  | pure CNN fast      |
# | "tu-convnext_small"                              |  ~50 M  |               |
# | "tu-swin_tiny_patch4_window7_224"                |  ~28 M  | Swin Transf. (default fallback) |
# | "tu-swin_small_patch4_window7_224"               |  ~50 M  |               |
# | "resnet50"  (set ENCODER_WEIGHTS="imagenet")     |  ~25 M  | plain CNN     |
# +--------------------------------------------------+---------+---------------+
#
# Architecture swap guide  (change ARCHITECTURE + CHECKPOINT, nothing else):
# "unet"             -> classic encoder-decoder with skip connections
# "unetplusplus"     -> nested dense skip connections, often +1-2 IoU
# "fpn"              -> Feature Pyramid Network, good for multi-scale targets
# "attention_unet"   -> UNet with scSE attention gates on skip connections
#                       (Squeeze-and-Excitation spatial + channel attention)
# "manet"            -> Multi-scale Attention Net (Attention Residual):
#                       PAB (global spatial attention) + MFAB (multi-scale
#                       channel attention).  Best choice for rip current edges
#                       -- captures both channel geometry and fine boundaries.
BACKBONE        = "tu-swin_tiny_patch4_window7_224"  # safe default: works on all timm versions
                                                      # change to "tu-vmamba_tiny_s2l5" after
                                                      # confirming VMamba is available via:
                                                      # python -c "import timm; print([m for m in timm.list_models() if 'vmamba' in m])"
ARCHITECTURE    = "manet"       # set to "attention_unet" or "manet" for new architectures

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
CHECKPOINT  = "manet_swin_tiny.pth"   # auto-named: {ARCHITECTURE}_{BACKBONE}

# CHANGE: resume support.  Set RESUME_FROM to the checkpoint file path and
# RESUME_EPOCH to the last fully completed epoch to restart a crashed run.
# Leave RESUME_FROM = None to always start from scratch.
RESUME_FROM  = None   # e.g. "unet_vmamba_tiny.pth"
RESUME_EPOCH = 0      # last fully completed epoch (used only with RESUME_FROM)

# -- Early stopping ------------------------------------------------------------
# CHANGE: Early stopping halts training when mIoU stops meaningfully improving.
#
# Why mIoU as the monitor?
#   The standard segmentation headline metric used by all NTIRE challenge
#   reports and published baselines.  mIoU = (IoU_rip + IoU_background) / 2,
#   so it only improves when the model gets better at BOTH classes — unlike
#   single-class IoU which can increase by simply predicting more rip pixels.
#   Using the same monitor across all pipeline variants (SegFormer, VMamba,
#   ConvNeXt, Swin) also keeps the ablation study's stopping criterion
#   consistent, which is required for a fair comparison.
#
# EARLY_STOP_PATIENCE  : epochs to wait for improvement before stopping.
#                        5 is recommended — long enough to survive the dip
#                        after an LR reduction, short enough to save real
#                        compute on a run that plateaus early.
# EARLY_STOP_MIN_DELTA : minimum mIoU gain counted as real improvement.
#                        0.001 (0.1 pp) filters floating-point noise without
#                        masking genuine late-training gains.
EARLY_STOP_PATIENCE  = 5      # epochs without improvement before stopping
EARLY_STOP_MIN_DELTA = 0.001  # minimum mIoU improvement to reset the counter


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
    # CHANGE: tn (true negatives) is now tracked so that background IoU
    # and mIoU can be computed.  mIoU is the early stopping monitor and
    # the standard headline metric for the ablation table.
    tn = ((1 - preds) * (1 - targets)).sum(dim=(1, 2, 3))

    iou_rip   = (tp / (tp + fp + fn + 1e-6)).mean().item()
    # Background IoU — TN over (TN + FP + FN).  FP and FN swap roles
    # relative to the rip class because we are now scoring the other class.
    iou_bg    = (tn / (tn + fp + fn + 1e-6)).mean().item()
    # CHANGE: mIoU = mean of per-class IoUs.  The early stopping logic,
    # LR scheduler, and best-model checkpoint all use this value.
    miou      = (iou_rip + iou_bg) / 2.0

    dice      = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall    = (tp / (tp + fn + 1e-6)).mean().item()

    # CHANGE: aAcc and mAcc added.  Derived from tp/tn/fp/fn already
    # computed above — no extra GPU operations required.
    #
    # aAcc: fraction of all pixels correctly labelled.  Dominated by the
    #   background class on imbalanced data; must be read alongside mIoU.
    #   Key diagnostic: aAcc ~95% with mIoU ~0.5 signals all-background collapse.
    total = tp + tn + fp + fn
    aacc  = ((tp + tn) / (total + 1e-6)).mean().item()

    # mAcc: mean of per-class pixel recall.  Gives equal weight to rip and
    #   background regardless of pixel count, correcting for class imbalance.
    #   Standard MMSegmentation companion metric to mIoU.
    acc_rip = (tp / (tp + fn + 1e-6)).mean().item()   # identical to recall
    acc_bg  = (tn / (tn + fp + 1e-6)).mean().item()
    macc    = (acc_rip + acc_bg) / 2.0

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

    return dict(
        iou=iou_rip,
        miou=miou,          # CHANGE: added — drives early stopping, scheduler, checkpoint
        aacc=aacc,          # CHANGE: all-pixel accuracy
        macc=macc,          # CHANGE: mean class accuracy
        dice=dice,
        precision=precision,
        recall=recall,
        f2=f2,
        boundary_iou=boundary_iou,
    )


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
    "unet"           -> classic skip-connection encoder-decoder (default)
    "unetplusplus"   -> nested dense skip connections, often +1-2 IoU
    "fpn"            -> Feature Pyramid Network
    "attention_unet" -> UNet + scSE attention gates on every skip connection.
                        Squeeze-and-Excitation recalibrates spatial and channel
                        features independently at each decoder stage.  Low
                        overhead over plain UNet; good first step toward
                        attention-based segmentation.
    "manet"          -> Multi-scale Attention Net (the Attention Residual model).
                        Two dedicated decoder blocks:
                        PAB (Position-wise Attention Block) — global spatial
                        attention, captures full rip channel geometry even when
                        the current extends across the entire image width.
                        MFAB (Multi-scale Fusion Attention Block) — channel
                        attention fused from multiple feature map scales,
                        distinguishing turbid rip water from foam at every
                        resolution simultaneously.
                        Recommended architecture when BACKBONE is a residual
                        encoder (resnet50, efficientnet) to fully exploit the
                        residual + attention combination your supervisor requested.
    """
    # CHANGE: arch_map extended with two attention-based architectures.
    #
    # "attention_unet": smp.Unet with decoder_attention_type="scse" adds
    #   Squeeze-and-Excitation Channel-Spatial (scSE) gates to every decoder
    #   skip connection with no structural change to the backbone — the same
    #   backbone string works as-is.
    #
    # "manet": smp.MAnet is the Multi-scale Attention Net.  It implements
    #   the "Attention Residual" architecture class your supervisor specified:
    #   residual encoder features combined with multi-scale attention decoder.
    #   No extra dependencies; ships with segmentation-models-pytorch.
    arch_map = {
        "unet":           smp.Unet,
        "unetplusplus":   smp.UnetPlusPlus,
        "fpn":            smp.FPN,
        "attention_unet": None,   # handled separately below (requires extra kwarg)
        "manet":          smp.MAnet,
    }
    if ARCHITECTURE not in arch_map:
        raise ValueError(
            f"Unknown ARCHITECTURE '{ARCHITECTURE}'. "
            f"Choose from: {list(arch_map.keys())}"
        )

    # FIX: validate timm backbone availability BEFORE calling smp, so the
    # error message tells you exactly what to do instead of a cryptic
    # RuntimeError buried inside smp/timm internals.
    if BACKBONE.startswith("tu-"):
        backbone_name = BACKBONE[3:]   # strip the "tu-" prefix
        available = timm.list_models(backbone_name)
        if not available:
            # Check what IS available to give a helpful suggestion
            family = backbone_name.split("_")[0]
            suggestions = timm.list_models(f"*{family}*")[:3]
            raise RuntimeError(
                f"\n  Backbone '{BACKBONE}' not found in your timm version "
                f"({timm.__version__}).\n"
                f"  Options:\n"
                f"  1. Upgrade timm:  pip install --upgrade timm\n"
                f"  2. Use an available backbone instead. "
                f"Similar '{family}' models found: {suggestions}\n"
                f"  3. Use 'tu-swin_tiny_patch4_window7_224' -- "
                f"guaranteed available in all timm versions."
            )

    # "attention_unet" uses smp.Unet but passes decoder_attention_type="scse"
    # to activate Squeeze-and-Excitation gates on every skip connection.
    # All other architectures use the standard constructor call.
    if ARCHITECTURE == "attention_unet":
        model = smp.Unet(
            encoder_name          = BACKBONE,
            encoder_weights       = ENCODER_WEIGHTS,
            decoder_attention_type = "scse",   # scSE gates on skip connections
            in_channels           = 3,
            classes               = 1,
            activation            = None,
        )
    else:
        model = arch_map[ARCHITECTURE](
            encoder_name    = BACKBONE,
            encoder_weights = ENCODER_WEIGHTS,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
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
    # CHANGE: miou, aacc, macc added to accumulator.
    accum = dict(iou=0.0, miou=0.0, aacc=0.0, macc=0.0, dice=0.0,
                 precision=0.0, recall=0.0, f2=0.0, boundary_iou=0.0)
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
#   EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Stops training when mIoU fails to improve for EARLY_STOP_PATIENCE
    consecutive epochs.

    FIX: the previous design maintained its own internal best_score and
    compared against it with a min_delta threshold.  This created two
    independent trackers of the same quantity:

        best_val_iou          -- updated when val_metrics["miou"] > best_val_iou
        early_stopping.best_score -- updated when improvement > min_delta

    When improvement was between 0 and min_delta (e.g. 0.0008 < 0.001),
    the checkpoint was saved (strict >) but the early stopping counter
    still incremented (improvement below min_delta).  The two trackers
    diverged silently, producing the confusing output where a new best
    checkpoint was saved AND the patience counter incremented in the same
    epoch.

    The fix: remove internal improvement logic entirely.  step() now
    receives an `improved` boolean from the training loop — the exact
    same condition used to decide whether to save the checkpoint.
    The counter resets if and only if a new checkpoint was saved.
    The two systems are now structurally identical; divergence is impossible.

    min_delta is intentionally removed.  The checkpoint saving criterion
    (strict >) is the single source of truth for what counts as improvement.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE):
        self.patience    = patience
        self.counter     = 0
        self.best_score  = None   # tracked for display only, not for logic
        self.should_stop = False

    def step(self, score: float, improved: bool) -> bool:
        """
        Call once per epoch.

        Parameters
        ----------
        score    : current epoch mIoU (used only for display in the log).
        improved : True if the training loop saved a new best checkpoint
                   this epoch — i.e. val_metrics["miou"] > best_val_iou.
                   This is the ONLY signal used to reset the counter.

        Returns True if training should stop.
        """
        if self.best_score is None:
            # First epoch — initialise display tracker, do not count against patience.
            self.best_score = score
        elif improved:
            # New best checkpoint was saved — reset counter and update display.
            self.best_score = score
            self.counter    = 0
        else:
            # No new checkpoint — increment counter.
            self.counter += 1
            print(
                f"  EarlyStopping: no improvement for {self.counter}/"
                f"{self.patience} epochs  "
                f"(best mIoU={self.best_score:.4f}, current={score:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"  EarlyStopping: patience exhausted after "
                    f"{self.patience} epochs without improvement. "
                    f"Stopping training."
                )
        return self.should_stop

    def state_dict(self) -> dict:
        """Serialise for checkpoint saving."""
        return {"counter": self.counter, "best_score": self.best_score}

    def load_state_dict(self, state: dict) -> None:
        """Restore from a loaded checkpoint."""
        self.counter    = state["counter"]
        self.best_score = state["best_score"]


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

    # CHANGE: EarlyStopping is created once here so its counter and best_score
    # persist for the entire run and can be saved/restored via checkpoint.
    # Unlike the SegFormer script there is no GradScaler in this pipeline
    # (CNN/timm backbones train stably in fp32), so no scaler state is needed.
    early_stopping = EarlyStopping(
        patience = EARLY_STOP_PATIENCE,
    )

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
        # CHANGE: restore early stopping counter so patience already accrued
        # before the crash is not reset to zero on resume.
        if "early_stopping_state" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping_state"])
            print(
                f"  Early stopping counter restored: "
                f"{early_stopping.counter}/{early_stopping.patience}  "
                f"(best mIoU={early_stopping.best_score:.4f})"
            )
        print(f"  Restored epoch {RESUME_EPOCH}  |  best mIoU so far: {best_val_iou:.4f}")
    else:
        print("Starting training from scratch.")

    # -- Epoch loop ------------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        epoch_start = time.time()

        train_loss  = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        # CHANGE: print split across two lines for readability now that
        # aAcc and mAcc are included.  Line 1 = primary comparison metrics.
        # Line 2 = supporting diagnostics.
        print(
            f"  loss={train_loss:.4f}  "
            f"IoU={val_metrics['iou']:.4f}  "
            f"mIoU={val_metrics['miou']:.4f}  "
            f"Dice={val_metrics['dice']:.4f}  "
            f"Recall={val_metrics['recall']:.4f}  "
            f"Precision={val_metrics['precision']:.4f}"
        )
        print(
            f"  aAcc={val_metrics['aacc']:.4f}  "
            f"mAcc={val_metrics['macc']:.4f}  "
            f"F2={val_metrics['f2']:.4f}  "
            f"BoundaryIoU={val_metrics['boundary_iou']:.4f}"
        )
        print(f"  Epoch time: {(time.time() - epoch_start) / 60:.1f} min")

        if val_metrics["recall"] < 0.3 and epoch > 5:
            print("  Low recall -- consider raising POS_WEIGHT.")

        prev_lr = optimizer.param_groups[0]["lr"]
        # CHANGE: scheduler now monitors mIoU instead of single-class IoU,
        # consistent with the early stopping and best-model save logic.
        # All three signals (scheduler, checkpoint, early stop) must track
        # the same metric to avoid conflicting optimisation pressures.
        scheduler.step(val_metrics["miou"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        # FIX: compute `improved` as a boolean BEFORE updating best_val_iou.
        # This single flag is then passed to both the checkpoint save block
        # and early_stopping.step(), guaranteeing they use an identical
        # definition of "improvement" and can never diverge.
        improved = val_metrics["miou"] > best_val_iou

        if improved:
            best_val_iou = val_metrics["miou"]
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state":          model.state_dict(),
                    "optimizer_state":      optimizer.state_dict(),
                    "val_iou":              best_val_iou,
                    "early_stopping_state": early_stopping.state_dict(),
                    "config": {
                        "backbone":     BACKBONE,
                        "architecture": ARCHITECTURE,
                        "img_size":     IMG_SIZE,
                    },
                },
                CHECKPOINT,
            )
            print(f"  Saved best model -> {CHECKPOINT}  (mIoU={best_val_iou:.4f})")

        # Pass the same `improved` flag so early stopping resets if and only
        # if a new checkpoint was just saved.  No separate threshold; no
        # independent best_score comparison.  One condition, two consumers.
        if early_stopping.step(val_metrics["miou"], improved=improved):
            print(f"\n  Training stopped early at epoch {epoch}.")
            break

        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

    # -- Final summary ---------------------------------------------------------
    stopped_early = early_stopping.should_stop
    print(f"\n{'='*60}")
    print(f" Training {'stopped early' if stopped_early else 'complete'}.  "
          f"Best val mIoU: {best_val_iou:.4f}")
    if best_metrics:
        print(
            f" Best metrics:\n"
            f"   mIoU={best_metrics['miou']:.4f}  "
            f"IoU={best_metrics['iou']:.4f}  "
            f"Dice={best_metrics['dice']:.4f}  "
            f"Recall={best_metrics['recall']:.4f}  "
            f"Precision={best_metrics['precision']:.4f}\n"
            f"   aAcc={best_metrics['aacc']:.4f}  "
            f"mAcc={best_metrics['macc']:.4f}  "
            f"F2={best_metrics['f2']:.4f}  "
            f"BoundaryIoU={best_metrics['boundary_iou']:.4f}"
        )
    else:
        print(" No best metrics recorded.")
    print(f" Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()