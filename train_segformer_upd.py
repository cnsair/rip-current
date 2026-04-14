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
BATCH_SIZE   = 2         # 1–2 on CPU; 8–16 on GPU with 8 GB+ VRAM.
NUM_WORKERS  = 2         # 0 on CPU / Windows / notebooks; 2–4 on Linux GPU.
EPOCHS       = 50        # Increase to 50–100 for a full training run.
LR           = 5e-5      # AdamW initial learning rate.
WEIGHT_DECAY = 1e-5      # L2 regularisation (prevents over-fitting).

POS_WEIGHT   = 5.0       # CHANGE: reduced from 10.0 to 5.0.
                         # POS_WEIGHT=10 amplifies the BCE term to a point
                         # where float16 (AMP) overflows and produces NaN loss.
                         # 5.0 still penalises missed rip pixels heavily enough
                         # for the imbalanced dataset while keeping loss values
                         # comfortably within float16 range (~65504 max).

# MiT-B2 is the HuggingFace model ID for SegFormer's Mix Transformer B2
# encoder. B0–B5 trade speed for accuracy; B2 (~25 M params) matches ResNet50.
# Other options: "nvidia/mit-b0" (fastest), "nvidia/mit-b4" (highest accuracy).
# SEGFORMER_VARIANT = "nvidia/mit-b2"
SEGFORMER_VARIANT = "./segformer-b2-local"

TRAIN_IMGS   = "data_local/train_local/images"
TRAIN_MASKS  = "data_local/train_local/masks"
VAL_IMGS     = "data_local/val_local/images"
VAL_MASKS    = "data_local/val_local/masks"

CHECKPOINT   = "segformer_b2_local.pth"

# Resume support — set RESUME_FROM to the checkpoint path to continue
# a crashed/interrupted run; set to None to always start from scratch.
RESUME_FROM  = "segformer_b2_local.pth" # None      # e.g. "segformer_b2_local.pth"
RESUME_EPOCH = 10    # 0     # last fully completed epoch (set alongside RESUME_FROM) e.g. "9"

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

def train_one_epoch(model, loader, optimizer, scaler, device) -> tuple:
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

    loop = tqdm(loader, desc="  train", leave=False, ascii=True, dynamic_ncols=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss   = combined_loss(logits, masks)

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
    is_degenerate  = (batches_nan / max(1, total_batches)) > NAN_SKIP_THRESHOLD
    return mean_loss, is_degenerate


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Run one full pass over the validation set. Returns dict of metrics."""
    model.eval()
    # CHANGE: miou, aacc, macc added to the accumulator.
    accum = dict(iou=0.0, miou=0.0, aacc=0.0, macc=0.0, dice=0.0,
                 precision=0.0, recall=0.0, f2=0.0, boundary_iou=0.0)
    n = 0

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

    # ── Optimiser ──────────────────────────────────────────────────────────
    # AdamW = Adam with decoupled weight decay — standard choice for vision models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # FIX 2 (continued): GradScaler is created once here and lives for the
    # entire training run. This lets it accumulate a history of safe loss
    # scale values across epochs instead of resetting every epoch.
    # torch.amp.GradScaler("cuda") replaces the deprecated
    # torch.cuda.amp.GradScaler() — same behaviour, no FutureWarning.
    scaler = torch.amp.GradScaler("cuda")

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
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        epoch_start = time.time()  

        # CHANGE: train_one_epoch now returns (loss, is_degenerate).
        # is_degenerate is True when >NAN_SKIP_THRESHOLD of batches were
        # skipped due to NaN/Inf loss — meaning the model made almost no
        # gradient updates this epoch and is effectively not learning.
        train_loss, is_degenerate = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE
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
            # Reset scaler to a safe conservative scale
            scaler._init_scale = 2.0 ** 8
            scaler.update()
            print(
                f"\n  DEGENERATE EPOCH {consecutive_degenerate}/"
                f"{MAX_DEGENERATE_EPOCHS}: "
                f">{NAN_SKIP_THRESHOLD*100:.0f}% of batches were NaN.\n"
                f"  LR halved: {current_lr:.2e} -> {new_lr:.2e}  |  "
                f"Scaler reset to scale=256.\n"
                f"  Skipping validation — no useful gradients were applied."
            )
            if consecutive_degenerate >= MAX_DEGENERATE_EPOCHS:
                print(
                    f"\n  STOPPING: {MAX_DEGENERATE_EPOCHS} consecutive "
                    f"degenerate epochs.\n"
                    f"  The weights loaded from '{RESUME_FROM}' are "
                    f"unrecoverable.\n"
                    f"  Action required: set RESUME_FROM to your last CLEAN "
                    f"checkpoint\n"
                    f"  (the most recent epoch BEFORE NaN warnings appeared)\n"
                    f"  and restart with POS_WEIGHT <= {POS_WEIGHT}."
                )
                break
            # Skip validation and checkpoint logic for this epoch
            history.append({
                "epoch": epoch, "loss": train_loss,
                "degenerate": True
            })
            continue

        # Epoch was clean — reset the consecutive counter
        consecutive_degenerate = 0
        val_metrics = evaluate(model, val_loader, DEVICE)

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
        scheduler.step(val_metrics["miou"])
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
        improved = val_metrics["miou"] > best_val_iou

        if improved:
            best_val_iou = val_metrics["miou"]
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
        if early_stopping.step(val_metrics["miou"], improved=improved):
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
