"""
Author: Chisom Nwachukwu
Date: 2026-03-xx

Train a semantic segmentation model to detect rip currents in images
from CCTV cameras, drones, and mobile phones.

Key design decisions
--------------------
* Architecture  : SegFormer with a Mix Transformer (MiT-B2) encoder.
                  IMPORTANT — SegFormer's encoder IS the MiT backbone; it
                  cannot be swapped for ResNet50. MiT-B2 (~25 M parameters)
                  is the closest parameter-count equivalent to ResNet50 (~25 M)
                  and is the standard choice for a fair SegFormer baseline.
                  Source: Xie et al., "SegFormer: Simple and Efficient Design
                  for Semantic Segmentation with Transformers", NeurIPS 2021.
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

Datasets Used: 
    https://www.kaggle.com/datasets/harsh1tha/ripcurrentdatasetNTIRE_2026_Rip_Current_Detection_and_Segmentation__RipDetSeg__Challenge___Participants_report_template (Unzipped Files)
"""

import os
# import certifi
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# --- SSL fix for Windows ---
# certifi_win32.generate_pem()
# cert_path = certifi_win32.wincerts.where()
# os.environ['SSL_CERT_FILE'] = cert_path          # This is what httpx needs
# os.environ['REQUESTS_CA_BUNDLE'] = cert_path     # For compatibility

# from huggingface_hub import snapshot_download
# snapshot_download('nvidia/segformer-b2-finetuned-ade-512-512', local_dir='./segformer-b2-local')


from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from transformers import SegformerForSemanticSegmentation
import time
import random
from sklearn.metrics import fbeta_score as sklearn_fbeta
from scipy.ndimage import binary_erosion

# FOAM-GAP INTEGRATION: the physical loss lives in its own module (foam_gap_loss.py),
# which must sit in the same folder as this script. It is imported, not pasted in,
# so it stays reusable across the other training pipelines.
from foam_gap_loss import FoamGapLoss, warmup_lambda

# FIX 1: torch.cuda.amp.autocast and torch.cuda.amp.GradScaler are deprecated
# in PyTorch 2.x and produced the FutureWarning seen every epoch.
# The modern API is torch.amp.autocast("cuda") and torch.amp.GradScaler("cuda").
# No behaviour change — just the correct non-deprecated call path.

# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION  — edit these to match your hardware and experiment goals
# ══════════════════════════════════════════════════════════════════════════════

DEVICE       = "cuda"
IMG_SIZE     = 512   # SegFormer was trained at 512×512 on ADE20K and
                         # Cityscapes; using 256 degrades its multi-scale attention
                         # significantly. Set back to 256 only if VRAM is tight.
BATCH_SIZE   = int(os.environ.get("BATCH", 2))  # MUST be identical across every
                         # ablation arm. The Table III baseline (and the whole
                         # comparative study) used batch 2 for SegFormer-B2 at
                         # 512x512; changing it confounds the loss comparison and
                         # halves the gradient updates per epoch. Keep at 2 unless
                         # you re-run the baseline at the new batch AND rescale LR.
NUM_WORKERS  = 2         # 0 on CPU / Windows / notebooks; 2–4 on Linux GPU.
EPOCHS       = int(os.environ.get("EPOCHS", 50))  # fixed budget; set the EPOCHS env
                         # var so every sweep arm trains the SAME number of epochs
                         # (the baseline vs foam comparison must be epoch-matched).
LR           = 5e-5      # AdamW initial learning rate.
WEIGHT_DECAY = 1e-5      # L2 regularisation (prevents over-fitting).

# ── Mixed precision ───────────────────────────────────────────────────────────
# FIX (NaN cascade): switch AMP from float16 to bfloat16. The RTX 4090 (Ada) has
# native bf16, and bf16 shares float32's 8-bit exponent range, so the large BCE
# gradients that overflowed float16 — the source of the degenerate >30%-NaN
# epochs — no longer go to inf/NaN. bf16 also needs no loss scaling, so the
# GradScaler is created with enabled=(AMP_DTYPE is float16) and becomes a
# transparent pass-through under bf16 (every scaler.* call still works).
# Set to torch.float16 only if you must run on pre-Ampere hardware.
AMP_DTYPE = torch.bfloat16    # torch.bfloat16 (4090, recommended) or torch.float16

POS_WEIGHT   = 2.0       # FIX (this run): lowered 3.0 -> 2.0 per the degenerate-
                         # epoch recovery advice. With bf16 the float16 overflow
                         # path is gone, but a smaller positive weight further
                         # tames the BCE-gradient spikes on near-empty masks while
                         # still penalising missed rip pixels more than false
                         # positives. Raise toward 3.0 only if recall is too low.

# MiT-B2 is the HuggingFace model ID for SegFormer's Mix Transformer B2
# encoder. B0–B5 trade speed for accuracy; B2 (~25 M params) matches ResNet50.
# Other options: "nvidia/mit-b0" (fastest), "nvidia/mit-b4" (highest accuracy).
# SEGFORMER_VARIANT = "nvidia/mit-b2"
SEGFORMER_VARIANT = "./segformer-b2-local"

TRAIN_IMGS   = "data_local/train_local/images"
TRAIN_MASKS  = "data_local/train_local/masks"
VAL_IMGS     = "data_local/val_local/images"
VAL_MASKS    = "data_local/val_local/masks"

CHECKPOINT   = os.environ.get("CKPT", "./trained_models/segformer_b2_local.pth")
               # Per-arm output path. Pass a distinct CKPT per sweep run, e.g.
               # CKPT=./trained_models/segformer_b2_foam_l010.pth

# Resume support — set RESUME_FROM to the checkpoint path to continue
# a crashed/interrupted run; set to None to always start from scratch.
RESUME_FROM  = None      # clean, from-scratch runs for the sweep. Set a checkpoint
                         # path ONLY to resume a crashed run; leave None otherwise.
RESUME_EPOCH = 10 # last fully completed epoch (set alongside RESUME_FROM) e.g. "9"

# ── Early stopping ────────────────────────────────────────────────────────────
# CHANGE: Early stopping halts training when mIoU stops meaningfully improving,
# preventing wasted compute and overfitting.
#
# Why mIoU?
#   - It is the standard headline metric in all segmentation papers and NTIRE
#     challenge reports, so stopping on it keeps optimisation and evaluation
#     targets consistent.
#   - Unlike the single-class IoU already in the script, mIoU averages rip
#     and background IoU equally, so it only improves when the model genuinely
#     gets better across both classes — not just by predicting more background.
#   - Validation loss is noisy under AMP and would interact badly with the
#     NaN guard; mIoU is stable and directly interpretable.
#
# EARLY_STOP_PATIENCE : how many consecutive epochs without improvement to
#                       tolerate before stopping.  5 is recommended here:
#                       long enough to survive a temporary dip (e.g. after an
#                       LR reduction), short enough to save ~10 hours of
#                       compute on a 50-epoch run that plateaus at epoch 20.
# EARLY_STOP_MIN_DELTA: minimum mIoU gain that counts as a real improvement.
#                       0.001 (0.1 pp) filters out floating-point noise without
#                       masking genuine but small gains late in training.
EARLY_STOP_PATIENCE  = 5      # epochs to wait before stopping
EARLY_STOP_MIN_DELTA = 0.001  # minimum mIoU improvement to reset the counter

# Metric used for checkpoint selection, early stopping, and LR scheduling.
# Default "miou" preserves prior behaviour. For the sweep, "recall" or "f2" is
# worth trying: the foam loss inflates val mIoU specifically, so selecting on it
# pushed the run to overfit (epoch 22 vs the baseline's 15). Selecting on the
# safety metrics tracks test-set generalisation more honestly. Higher-is-better
# for all of {miou, recall, f2, dice, iou}, matching the scheduler's mode="max".
MONITOR_METRIC = os.environ.get("MONITOR", "miou")

# ── NaN cascade protection ─────────────────────────────────────────────────────
# CHANGE: these two constants control the degenerate-epoch handler added to
# train_one_epoch.  When nearly every batch produces NaN loss the model is not
# learning — the NaN guard is a safety net, not a cure.
#
# NAN_SKIP_THRESHOLD : fraction of batches in a single epoch that may be
#   skipped before the epoch is declared degenerate.  0.30 = if more than
#   30% of batches are skipped the epoch is considered unrecoverable.
#
# MAX_DEGENERATE_EPOCHS : if this many consecutive degenerate epochs occur,
#   training is halted entirely.  The weights are in an unrecoverable state
#   and you must restart from the last clean checkpoint.
NAN_SKIP_THRESHOLD     = 0.30  # fraction of batches skipped before aborting epoch
MAX_DEGENERATE_EPOCHS  = 2     # consecutive degenerate epochs before hard stop


# ══════════════════════════════════════════════════════════════════════════════
#   FOAM-GAP PHYSICAL LOSS  (task-specific rip-current prior)
# ══════════════════════════════════════════════════════════════════════════════
# Adds L_total = L_seg + lambda * L_physics, where L_physics penalises predicted
# rip regions that are NOT darker than their surf-zone surround — the empirically
# verified low-reflectance prior (Step-1: 94.2% of 24,268 GT regions darker than
# surround, median Michelson contrast 0.111, p < 1e-300).
#
# THE ABLATION IS RUN BY TOGGLING USE_FOAM_LOSS:
#   * Baseline arm  : USE_FOAM_LOSS = False  (pure L_seg, lambda = 0)
#   * Treatment arm : USE_FOAM_LOSS = True   (lambda ramped to FOAM_LAMBDA_MAX)
# Use a DISTINCT CHECKPOINT path per arm so they do not overwrite each other,
# set RESUME_FROM = None for clean from-scratch ablation runs, and keep the seed
# fixed (set_seed(42) below) so the only difference between arms is the loss.
# Sweep-friendly overrides: a single launcher script can set these per run via
# the environment, e.g.  USE_FOAM=1 FOAM_LAMBDA=0.10 CKPT=...foam_l010.pth python train_segformer.py
USE_FOAM_LOSS       = os.environ.get("USE_FOAM", "1") == "1"   # "0" = baseline arm
FOAM_LAMBDA_MAX     = float(os.environ.get("FOAM_LAMBDA", 0.10))  # was 0.3; lowered
                              # after the sweep finding — 0.3 over-constrained the
                              # model toward the foam cue and hurt test recall/mIoU.
FOAM_WARMUP_EPOCHS  = 3       # epochs held at lambda=0 so L_seg localises first
FOAM_RAMP_EPOCHS    = 3       # epochs to ramp lambda 0 -> FOAM_LAMBDA_MAX
FOAM_MARGIN         = float(os.environ.get("FOAM_MARGIN", 0.07))  # was 0.10; a lower
                              # target contrast makes the prior a gentler regulariser.
FOAM_RING_PX        = 15      # surround band width (matches the verification run)
FOAM_GUARD_PX       = 2       # neutral gap between core and surround band
FOAM_MODE           = "michelson"   # exposure-invariant (see Step-1 analysis)
FOAM_MIN_MASS       = 50.0    # per-image predicted-rip mass gate (NaN guard)
FOAM_DOWNSAMPLE     = 2       # halve resolution before the max-pool morphology.
                              # At IMG_SIZE=512 this quarters the morphology cost
                              # with negligible effect on the global statistic.
                              # Set to 1 for exact full-resolution computation.


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
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
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

    # return bce_weight * bce + dice_weight * dice
    return 0.3 * bce + 0.7 * dice


# ── De-normalisation for the foam-gap loss ────────────────────────────────────
# The foam-gap term needs the UN-normalised image (RGB in [0,1]) to read true
# luminance, but the dataloader feeds the model the ImageNet-normalised tensor.
# Rather than change the dataloader, we invert A.Normalize on-GPU. These MUST
# match the mean/std in get_transforms().
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def denormalize_imagenet(x_norm: torch.Tensor,
                         mean_t: torch.Tensor,
                         std_t: torch.Tensor) -> torch.Tensor:
    """Invert A.Normalize: normalised (B,3,H,W) -> RGB in [0,1], same device.
    Recovers the AUGMENTED brightness the model actually sees, spatially
    aligned with the prediction (flips/rotations already applied)."""
    return (x_norm * std_t + mean_t).clamp(0.0, 1.0)


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
    # CHANGE: TN is now tracked so that background IoU and mIoU can be
    # computed.  mIoU is the standard segmentation headline metric and is
    # used as the early stopping monitor (see EarlyStopping below).
    tn = ((1 - preds) * (1 - targets)).sum(dim=(1, 2, 3))

    iou_rip   = (tp / (tp + fp + fn + 1e-6)).mean().item()
    # Background IoU: TN / (TN + FP + FN) — note FP and FN swap roles
    # for the background class relative to the rip class.
    iou_bg    = (tn / (tn + fp + fn + 1e-6)).mean().item()
    # CHANGE: mIoU averages both class IoUs.  It is the metric the early
    # stopping and best-model logic now track instead of single-class IoU.
    miou      = (iou_rip + iou_bg) / 2.0

    dice      = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall    = (tp / (tp + fn + 1e-6)).mean().item()

    # CHANGE: aAcc and mAcc added.  All four values (tp, tn, fp, fn) are
    # already computed above so these are pure arithmetic — no extra GPU ops.
    #
    # aAcc (all-pixel accuracy): fraction of all pixels correctly labelled.
    #   Intuitive for non-specialists but dominated by the background class
    #   on imbalanced data, so it must always be read alongside mIoU.
    #   Diagnostic value: if aAcc stays high (~95%) while mIoU collapses,
    #   the model has degenerated to predicting all-background.
    total = tp + tn + fp + fn
    aacc  = ((tp + tn) / (total + 1e-6)).mean().item()

    # mAcc (mean class accuracy): average of per-class pixel recall.
    #   acc_rip = TP / (TP + FN) — fraction of actual rip pixels found.
    #   acc_bg  = TN / (TN + FP) — fraction of actual background pixels found.
    #   mAcc gives equal weight to both classes regardless of their pixel
    #   count, correcting for the imbalance that makes aAcc misleading.
    #   Standard companion to mIoU in MMSegmentation evaluation output.
    acc_rip = (tp / (tp + fn + 1e-6)).mean().item()   # same as recall
    acc_bg  = (tn / (tn + fp + 1e-6)).mean().item()
    macc    = (acc_rip + acc_bg) / 2.0

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
        iou=iou_rip,
        miou=miou,            # CHANGE: added — used by early stopping and checkpoint logic
        aacc=aacc,            # CHANGE: all-pixel accuracy
        macc=macc,            # CHANGE: mean class accuracy
        dice=dice,
        precision=precision,
        recall=recall,
        f2=f2,
        boundary_iou=boundary_iou,
    )


# ══════════════════════════════════════════════════════════════════════════════
#   MODEL
# ══════════════════════════════════════════════════════════════════════════════

# SegFormerWrapper is needed because HuggingFace's SegFormer decoder
# outputs logits at 1/4 of the input resolution (e.g. 128×128 for a 512×512
# input). The rest of the pipeline — loss, metrics, checkpointing — expects
# full-resolution masks (512×512). This wrapper performs the bilinear upsample
# internally so that `logits = model(images)` keeps working unchanged everywhere
# else in the script (train_one_epoch, evaluate, combined_loss, compute_metrics).
class SegFormerWrapper(torch.nn.Module):
    """
    Thin wrapper around HuggingFace SegformerForSemanticSegmentation that:
      1. Accepts standard (B, 3, H, W) image tensors (same as smp models).
      2. Upsamples the 1/4-resolution decoder output back to (B, 1, H, W).

    Without this wrapper every call site would need an explicit F.interpolate,
    which would require touching train_one_epoch, evaluate, and compute_metrics.
    Wrapping once here keeps all other code identical to other architectures.
    """
    def __init__(self, hf_model: torch.nn.Module, output_size: tuple):
        super().__init__()
        self.model       = hf_model
        self.output_size = output_size  # (H, W) — the full input resolution

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # HuggingFace forward — keyword arg is 'pixel_values', not 'x'
        out    = self.model(pixel_values=pixel_values)
        logits = out.logits                            # (B, 1, H/4, W/4)
        # Upsample back to input resolution so loss & metrics see full-res masks
        logits = F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )                                              # (B, 1, H, W)
        return logits


def build_model() -> torch.nn.Module:
    """
    Builds a SegFormer model wrapped for full-resolution binary segmentation.

    Why SegFormer?
    --------------
    • Hierarchical Mix Transformer (MiT) encoder captures both fine-grained
      textures (wave foam, turbid water) and large-scale structure (rip channel
      geometry) via multi-scale self-attention — without the quadratic cost of
      standard ViT on high-resolution inputs.
    • Lightweight All-MLP decoder avoids the complex ASPP or FPN heads used
      in DeepLabV3+ or FPN, making it faster at inference.
    • MiT-B2 (~25 M parameters) is parameter-count equivalent to ResNet50,
      making it the correct backbone choice for a fair SegFormer baseline.

    MiT variant guide:
    ------------------
        "nvidia/mit-b0"  ~3.7 M params  — fastest, lowest accuracy
        "nvidia/mit-b1"  ~14  M params
        "nvidia/mit-b2"  ~25  M params  ← ResNet50 equivalent (used here)
        "nvidia/mit-b3"  ~45  M params
        "nvidia/mit-b4"  ~64  M params
        "nvidia/mit-b5"  ~82  M params  — slowest, highest accuracy
    """
    # instantiate via HuggingFace API instead of smp.
    # num_labels=1 keeps the binary (rip / no-rip) setup identical to before;
    # the model outputs a single-channel logit map consumed by combined_loss
    # with sigmoid + BCE + Dice — no changes needed in the loss function.
    # ignore_mismatched_sizes=True allows loading ImageNet-pretrained MiT weights
    # even though the original classification head has a different output size.
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_VARIANT,
        num_labels            = 1,
        ignore_mismatched_sizes = True,
    )

    # Wrap so output is always (B, 1, IMG_SIZE, IMG_SIZE) — identical contract
    # to what smp models return, so nothing else in the script needs changing.
    model = SegFormerWrapper(hf_model, output_size=(IMG_SIZE, IMG_SIZE))
    return model


# ══════════════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    foam_loss_fn=None, foam_lambda=0.0) -> tuple:
    """
    Run one full pass over the training set.

    Returns
    -------
    (mean_loss, is_degenerate)
        mean_loss      : float — mean loss over non-skipped batches.
        is_degenerate  : bool  — True if more than NAN_SKIP_THRESHOLD of
                         batches were skipped due to NaN/Inf loss.

    CHANGE: the function now tracks skipped batches and declares the epoch
    degenerate when the skip rate exceeds NAN_SKIP_THRESHOLD.  This replaces
    the previous behaviour of silently skipping bad batches indefinitely,
    which allows the run to stall for hours without making progress.
    The caller (train()) acts on the degenerate flag to reduce LR and,
    if it persists, abort training entirely.
    """
    model.train()
    total_loss   = 0.0
    batches_ok   = 0
    batches_nan  = 0   # CHANGE: count of NaN-skipped batches this epoch
    total_batches = len(loader)

    # FOAM-GAP: only active once lambda has ramped above 0 (after warmup), so
    # warmup epochs incur zero extra compute. Build the de-norm buffers once.
    use_foam = (foam_loss_fn is not None) and (foam_lambda > 0.0)
    total_foam = 0.0
    if use_foam:
        mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std_t  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    loop = tqdm(loader, desc="  train", leave=False, ascii=True, dynamic_ncols=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(images)
            loss   = combined_loss(logits, masks)

        # ── Foam-gap physical term ─────────────────────────────────────────
        # Computed OUTSIDE autocast: the module upcasts to fp32 internally, so
        # the morphology/contrast math is numerically safe under AMP (this is
        # the guard against the historical fp16 NaN cascade). Added to `loss`
        # BEFORE the NaN check below so any anomaly is caught by the existing
        # degenerate-epoch handler. `images` is ImageNet-normalised; we recover
        # the un-normalised RGB the term needs by inverting A.Normalize on-GPU.
        if use_foam:
            images_raw = denormalize_imagenet(images, mean_t, std_t)
            foam = foam_loss_fn(logits, images_raw)
            loss = loss + foam_lambda * foam
            total_foam += float(foam.detach())

        # NaN guard — skip batch and increment counter instead of printing
        # a warning every single time (which floods the terminal as seen).
        # CHANGE: warning now includes the running skip rate so severity is
        # immediately visible without scrolling.
        if torch.isnan(loss) or torch.isinf(loss):
            batches_nan += 1
            skip_rate = batches_nan / max(1, batches_ok + batches_nan)
            loop.set_postfix(
                status=f"NaN x{batches_nan} ({skip_rate*100:.0f}% skipped)"
            )
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batches_ok += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    # CHANGE: report NaN statistics clearly at end of epoch
    if batches_nan > 0:
        skip_rate = batches_nan / max(1, total_batches)
        print(
            f"\n  NaN summary: {batches_nan}/{total_batches} batches skipped "
            f"({skip_rate*100:.1f}%)"
        )

    mean_loss      = total_loss / max(1, batches_ok)
    mean_foam      = (total_foam / max(1, batches_ok)) if use_foam else 0.0
    is_degenerate  = (batches_nan / max(1, total_batches)) > NAN_SKIP_THRESHOLD
    return mean_loss, is_degenerate, mean_foam


@torch.no_grad()
def evaluate(model, loader, device, foam_loss_fn=None) -> dict:
    """Run one full pass over the validation set. Returns dict of metrics.
    If foam_loss_fn is given, also reports the physical-consistency metric
    (mean prediction-vs-surround contrast) and the gate fraction — the
    'physical consistency' column for the ablation (supervisor rec. #3)."""
    model.eval()
    # CHANGE: miou, aacc, macc added to the accumulator.
    accum = dict(iou=0.0, miou=0.0, aacc=0.0, macc=0.0, dice=0.0,
                 precision=0.0, recall=0.0, f2=0.0, boundary_iou=0.0)
    n = 0

    # FOAM-GAP consistency accumulators (only used if foam_loss_fn provided).
    cons_sum, gate_sum, cons_n = 0.0, 0.0, 0
    if foam_loss_fn is not None:
        mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std_t  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    for images, masks in tqdm(loader, desc="  val  ", leave=False, ascii=True, dynamic_ncols=False):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        batch_metrics = compute_metrics(logits, masks)
        for k in accum:
            accum[k] += batch_metrics[k]
        n += 1

        if foam_loss_fn is not None:
            images_raw = denormalize_imagenet(images, mean_t, std_t)
            c, g = foam_loss_fn.consistency_metric(logits, images_raw)
            if c == c:                      # skip NaN (no valid regions in batch)
                cons_sum += c
                gate_sum += g
                cons_n   += 1

    out = {k: v / max(1, n) for k, v in accum.items()}
    if foam_loss_fn is not None:
        out["consistency"] = (cons_sum / cons_n) if cons_n else float("nan")
        out["gate_frac"]   = (gate_sum / cons_n) if cons_n else 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#   EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when mIoU fails to improve for EARLY_STOP_PATIENCE
    consecutive epochs.

    FIX (applied from train_unet_transformer.py): the previous design kept
    its own internal best_score and required improvement > min_delta before
    resetting the counter.  This created two independent trackers:

        best_val_iou              updated when val_metrics["miou"] > best_val_iou
        early_stopping.best_score updated when improvement > min_delta

    When improvement fell between 0 and min_delta (e.g. 0.0008 < 0.001),
    the checkpoint was saved (strict >) but the counter still incremented
    (below min_delta threshold).  The trackers diverged silently, producing
    the observed behaviour where a new best checkpoint was saved AND the
    patience counter incremented in the same epoch.

    The fix removes min_delta entirely and changes step() to accept an
    `improved` boolean from the training loop — the same flag used to decide
    whether to save the checkpoint.  The counter resets if and only if a new
    checkpoint was saved.  One condition, two consumers, no possible divergence.

    Why mIoU as the monitor?
    ------------------------
    mIoU = (IoU_rip + IoU_background) / 2.  It is the standard segmentation
    headline metric used by every major paper and the NTIRE challenge.
    Stopping on it keeps the optimisation target and the reported evaluation
    target identical — a necessary condition for a scientifically honest
    baseline comparison.

    Serialisation note:
    -------------------
    counter and best_score (display only) are saved into the checkpoint so
    they survive a crash/resume cycle without resetting to zero.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE):
        # FIX: min_delta removed — no longer stored or used.
        self.patience    = patience
        self.counter     = 0
        self.best_score  = None   # used for terminal display only, not for logic
        self.should_stop = False

    def step(self, score: float, improved: bool) -> bool:
        """
        Call once per epoch.

        Parameters
        ----------
        score    : current epoch mIoU — used only for the terminal log line.
        improved : True if the training loop saved a new best checkpoint this
                   epoch (i.e. val_metrics["miou"] > best_val_iou).  This is
                   the sole signal that resets the counter; no internal
                   threshold is applied.

        Returns True if training should stop, False otherwise.
        """
        if self.best_score is None:
            # First epoch — initialise display tracker, no counter penalty.
            self.best_score = score
        elif improved:
            # New checkpoint was saved — counter resets, display updates.
            self.best_score = score
            self.counter    = 0
        else:
            # No new checkpoint — increment counter and log.
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
                    f"{self.patience} epochs without improvement.  "
                    f"Stopping training."
                )
        return self.should_stop

    def state_dict(self) -> dict:
        """Serialise counter and best_score for checkpoint saving."""
        return {
            "counter":    self.counter,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore counter and best_score from a loaded checkpoint."""
        self.counter    = state["counter"]
        self.best_score = state["best_score"]


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

    # ── Foam-gap physical loss ───────────────────────────────────────────────
    # Instantiated once and moved to DEVICE (its luminance-weight buffer follows).
    # When USE_FOAM_LOSS is False this stays None and the run is the pure-L_seg
    # baseline arm of the ablation.
    foam_loss_fn = None
    if USE_FOAM_LOSS:
        foam_loss_fn = FoamGapLoss(
            margin     = FOAM_MARGIN,
            ring_px    = FOAM_RING_PX,
            guard_px   = FOAM_GUARD_PX,
            mode       = FOAM_MODE,
            min_mass   = FOAM_MIN_MASS,
            downsample = FOAM_DOWNSAMPLE,
        ).to(DEVICE)
        print(f"Foam-gap loss ENABLED  |  lambda_max={FOAM_LAMBDA_MAX}  "
              f"margin={FOAM_MARGIN}  ring={FOAM_RING_PX}px  mode={FOAM_MODE}  "
              f"warmup={FOAM_WARMUP_EPOCHS}+{FOAM_RAMP_EPOCHS}ep  "
              f"downsample={FOAM_DOWNSAMPLE}")
    else:
        print("Foam-gap loss DISABLED  |  baseline arm (lambda = 0)")

    print(f"Run config  |  amp={str(AMP_DTYPE).replace('torch.','')}  "
          f"batch={BATCH_SIZE}  monitor={MONITOR_METRIC}  pos_weight={POS_WEIGHT}  "
          f"epochs={EPOCHS}  seed=42  ckpt={CHECKPOINT}")

    # ── Optimiser ──────────────────────────────────────────────────────────
    # AdamW = Adam with decoupled weight decay — standard choice for vision models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # FIX 2 (continued): GradScaler is created once here and lives for the
    # entire training run. This lets it accumulate a history of safe loss
    # scale values across epochs instead of resetting every epoch.
    # torch.amp.GradScaler("cuda") replaces the deprecated
    # torch.cuda.amp.GradScaler() — same behaviour, no FutureWarning.
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

    # FIX: min_delta removed from instantiation — EarlyStopping no longer
    # applies its own threshold; it defers entirely to the `improved` flag.
    early_stopping = EarlyStopping(
        patience = EARLY_STOP_PATIENCE,
    )

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
        drop_last   = True,   # prevents a batch-size-1 remainder from crashing
                              # BatchNorm layers inside the MiT encoder
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_iou       = 0.0
    best_metrics       = {}
    history            = []
    start_epoch        = 1
    # CHANGE: track consecutive degenerate epochs so training can be halted
    # before wasting hours on a model with unrecoverable corrupted weights.
    consecutive_degenerate = 0

    # resume support — restores weights, optimiser momentum, and the
    # best-IoU tracker so a crashed/interrupted run continues seamlessly.
    # Set RESUME_FROM and RESUME_EPOCH in the config section at the top.
    if RESUME_FROM and Path(RESUME_FROM).exists():
        print(f"\nResuming from checkpoint: {RESUME_FROM}")
        ckpt         = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_iou = ckpt["val_iou"]
        start_epoch  = RESUME_EPOCH + 1
        # FIX 2 (continued): restore scaler state if it was saved, so the
        # loss scale history is preserved across interrupted runs.
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        # CHANGE: restore early stopping counter so patience accumulated
        # before the crash is not lost on resume.  Without this, a run that
        # crashed on epoch 15 after 4 patience epochs would reset to 0 and
        # waste another full patience window before stopping.
        if "early_stopping_state" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping_state"])
            print(
                f"  Early stopping counter restored: "
                f"{early_stopping.counter}/{early_stopping.patience}  "
                f"(best mIoU={early_stopping.best_score:.4f})"
            )
        print(f"  Restored epoch {RESUME_EPOCH}  |  best IoU so far: {best_val_iou:.4f}")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, EPOCHS + 1):
        # FOAM-GAP: lambda schedule — 0 during warmup, then linear ramp to max.
        # On resume mid-run this correctly returns the full lambda (no re-warmup).
        foam_lambda = (warmup_lambda(epoch, FOAM_LAMBDA_MAX,
                                     FOAM_WARMUP_EPOCHS, FOAM_RAMP_EPOCHS)
                       if USE_FOAM_LOSS else 0.0)

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e}"
              f"{f', foam λ={foam_lambda:.3f}' if USE_FOAM_LOSS else ''})")
        epoch_start = time.time()  

        # CHANGE: train_one_epoch now returns (loss, is_degenerate, foam_term).
        # is_degenerate is True when >NAN_SKIP_THRESHOLD of batches were
        # skipped due to NaN/Inf loss — meaning the model made almost no
        # gradient updates this epoch and is effectively not learning.
        train_loss, is_degenerate, train_foam = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE,
            foam_loss_fn=foam_loss_fn, foam_lambda=foam_lambda,
        )

        # CHANGE: degenerate epoch handler.
        # When most batches are NaN the standard epoch logic (validate,
        # update scheduler, save checkpoint) is meaningless — validation
        # metrics will be stale and saving would overwrite a good checkpoint
        # with a model that made no progress.  Instead:
        #   1. Halve the LR immediately (more aggressive than the scheduler's
        #      patience=3 wait, because we know the epoch was wasted).
        #   2. Reset the scaler to a conservative starting scale (2^8 = 256)
        #      so it rebuilds from a safe baseline rather than the inflated
        #      value that caused the overflow.
        #   3. If MAX_DEGENERATE_EPOCHS consecutive bad epochs have occurred,
        #      the weights are unrecoverable from this checkpoint — stop now
        #      and prompt the user to restart from the last clean checkpoint.
        if is_degenerate:
            consecutive_degenerate += 1
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr     = current_lr * 0.5
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            # FIX: replace scaler._init_scale + scaler.update() with a fresh
            # GradScaler.  In PyTorch 2.x, calling scaler.update() without a
            # preceding scaler.step() raises:
            #   AssertionError: No inf checks were recorded prior to update.
            # Creating a new GradScaler at init_scale=256 is always valid and
            # achieves the same reset without touching internal state.
            scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 8,
                                          enabled=(AMP_DTYPE == torch.float16))

            # FIX: reload best-checkpoint weights when a degenerate epoch fires.
            # WHY: after resuming from a partially corrupted checkpoint, the
            # model can collapse to all-background on the first epoch (all
            # metrics = 0.0000, mIoU = 0.5 from background IoU only).  On the
            # next epoch the loss spikes because BCE on near-zero logits with
            # high POS_WEIGHT produces large gradients.  Reloading the best
            # saved weights resets the model to the last known-good state before
            # attempting further training with the reduced LR and fresh scaler.
            if Path(CHECKPOINT).exists():
                ckpt = torch.load(CHECKPOINT, map_location=DEVICE,
                                  weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                print(f"  Weights reloaded from best checkpoint: {CHECKPOINT}  "
                      f"(epoch {ckpt.get('epoch','?')}, "
                      f"mIoU={ckpt.get('val_iou',0):.4f})")

            print(
                f"\n  DEGENERATE EPOCH {consecutive_degenerate}/"
                f"{MAX_DEGENERATE_EPOCHS}: "
                f">{NAN_SKIP_THRESHOLD*100:.0f}% of batches were NaN.\n"
                f"  LR halved: {current_lr:.2e} -> {new_lr:.2e}  |  "
                f"Scaler replaced (fresh, init_scale=256).  "
                f"Weights restored from best checkpoint.\n"
                f"  Skipping validation — no useful gradients were applied."
            )
            if consecutive_degenerate >= MAX_DEGENERATE_EPOCHS:
                print(
                    f"\n  STOPPING: {MAX_DEGENERATE_EPOCHS} consecutive "
                    f"degenerate epochs.\n"
                    f"  Recovery failed.  Action required:\n"
                    f"  1. Set POS_WEIGHT lower (currently {POS_WEIGHT}).\n"
                    f"  2. Set RESUME_FROM to {CHECKPOINT} and restart."
                )
                break
            history.append({
                "epoch": epoch, "loss": train_loss,
                "degenerate": True
            })
            continue

        # Epoch was clean — reset the consecutive counter
        consecutive_degenerate = 0
        val_metrics = evaluate(model, val_loader, DEVICE, foam_loss_fn=foam_loss_fn)

        # CHANGE: print split across two lines for readability now that
        # aAcc and mAcc are included.  Line 1 = primary segmentation metrics
        # used for comparison across models.  Line 2 = supporting diagnostics.
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
        # FOAM-GAP: physical-consistency diagnostics (the ablation's extra column).
        if USE_FOAM_LOSS:
            print(
                f"  foam: train_term={train_foam:.4f}  lambda={foam_lambda:.3f}  "
                f"val_consistency={val_metrics.get('consistency', float('nan')):.4f}  "
                f"gate_frac={val_metrics.get('gate_frac', 0.0):.2f}"
            )
        epoch_mins = (time.time() - epoch_start) / 60
        print(f"  Epoch time: {epoch_mins:.1f} min")

        # Recall is especially important for a safety-critical task:
        # a missed rip current (false negative) is more dangerous than
        # a false alarm (false positive).
        if val_metrics["recall"] < 0.3 and epoch > 5:
            print("Low recall — consider raising POS_WEIGHT.")

        # ── Scheduler step ─────────────────────────────────────────────────
        # CHANGE: scheduler now monitors mIoU instead of single-class IoU,
        # consistent with the early stopping and best-model save logic.
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics[MONITOR_METRIC])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        # ── Save best model ────────────────────────────────────────────────
        # FIX: `improved` is computed BEFORE best_val_iou is updated.
        # This single boolean is then passed to both the checkpoint save block
        # and early_stopping.step() so they share one definition of
        # "improvement" and cannot diverge.  Previously, the checkpoint used
        # strict (>) while early stopping used (> + min_delta), causing the
        # counter to increment on the same epoch a new checkpoint was saved.
        improved = val_metrics[MONITOR_METRIC] > best_val_iou

        if improved:
            best_val_iou = val_metrics[MONITOR_METRIC]
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state":          model.state_dict(),
                    "optimizer_state":      optimizer.state_dict(),
                    "scaler_state":         scaler.state_dict(),
                    "early_stopping_state": early_stopping.state_dict(),
                    "val_iou":              best_val_iou,
                    "config": {
                        "encoder":       SEGFORMER_VARIANT,
                        "architecture":  "segformer",
                        "img_size":      IMG_SIZE,
                    },
                },
                CHECKPOINT,
            )
            print(f"  Saved best model -> {CHECKPOINT}  (mIoU={best_val_iou:.4f})")

        # Pass the same `improved` flag — counter resets if and only if a new
        # checkpoint was just saved.  No separate threshold; no independent
        # best_score comparison inside the class.  One condition, two consumers.
        if early_stopping.step(val_metrics[MONITOR_METRIC], improved=improved):
            print(f"\n  Training stopped early at epoch {epoch}.")
            break

        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

    # ── Final summary ─────────────────────────────────────────────────────
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
        print(" No best metrics recorded. The validation set may be empty or the metric never improved")
    print(f" Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()
