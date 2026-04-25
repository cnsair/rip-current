"""
Author: Chisom Nwachukwu
Date: 2026-04-xx

Diffusion-based semantic segmentation for rip current detection.

Architecture
------------
Based on SegDiff (Amit et al., 2021) and DDP (Ji et al., ICCV 2023).
The model conditions a denoising UNet on image features extracted by a
frozen ResNet-50 encoder, iteratively refining a noisy mask toward the
ground truth over T diffusion timesteps.

Why diffusion for rip current segmentation?
-------------------------------------------
1. Probabilistic output: running inference K times produces K plausible
   masks. Their mean is the prediction; their variance is an uncertainty
   map -- directly useful for a safety-critical system where boundary
   confidence matters as much as the binary decision.
2. Iterative refinement: each denoising step corrects boundary errors from
   the previous step, producing sharper rip channel outlines than a single
   feed-forward pass.
3. Data augmentation via generation: the trained model can synthesise
   labelled training pairs (image, mask) by conditioning on real images,
   addressing the dataset scarcity that limits rip current research.

Pipeline difference from CNN/Transformer baselines
---------------------------------------------------
The training loop is structurally different from train_unet_transformer.py:
  FORWARD PASS  : add Gaussian noise to the ground truth mask at a random
                  timestep t, then ask the model to predict that noise.
  LOSS          : MSE between predicted noise and true noise (not BCE+Dice).
  INFERENCE     : start from pure Gaussian noise, denoise T times using the
                  learned reverse process, threshold the result.
This is why diffusion cannot be expressed as an ARCHITECTURE entry in
train_unet_transformer.py -- the training loop itself changes, not just
the model constructor.

Install
-------
    pip install torch torchvision segmentation-models-pytorch albumentations tqdm
    (no extra packages required -- diffusion components are implemented here)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import time
import random
from sklearn.metrics import fbeta_score as sklearn_fbeta
from scipy.ndimage import binary_erosion


# ==============================================================================
#   CONFIGURATION
# ==============================================================================

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 256
BATCH_SIZE   = 4        # smaller than CNN baselines -- diffusion UNet is heavier
NUM_WORKERS  = 2
EPOCHS       = 100
LR           = 1e-4     # diffusion models typically need a higher LR than transformers
WEIGHT_DECAY = 1e-5

# -- Diffusion hyperparameters -------------------------------------------------
# T      : total number of diffusion timesteps.  100 is sufficient for binary
#          masks; published SegDiff uses 1000 but that is for natural images.
# BETA_1 : noise level at the first timestep (very small).
# BETA_T : noise level at the final timestep (close to 1 -- pure noise).
# K_SAMPLES : number of independent reverse-diffusion runs at inference time.
#   The K mask predictions are averaged into a final soft mask.
#   More K = better uncertainty estimate but slower inference.
T         = 100
BETA_1    = 1e-4
BETA_T    = 0.02
K_SAMPLES = 5

# -- Image encoder backbone ----------------------------------------------------
# The image encoder is frozen after initialisation -- only the denoising UNet
# is trained.  This is the DDP approach: use rich pre-trained features to
# condition the diffusion process rather than training the encoder from scratch.
IMG_ENCODER  = "resnet50"
ENCODER_WGTS = "imagenet"

# -- Paths & checkpointing -----------------------------------------------------
TRAIN_IMGS   = "data_local/train_local/images"
TRAIN_MASKS  = "data_local/train_local/masks"
VAL_IMGS     = "data_local/val_local/images"
VAL_MASKS    = "data_local/val_local/masks"
CHECKPOINT   = "diffusion_seg_resnet50.pth"
RESUME_FROM  = None
RESUME_EPOCH = 0

EARLY_STOP_PATIENCE = 5


# ==============================================================================
#   NOISE SCHEDULE  (linear beta schedule)
# ==============================================================================

def make_noise_schedule(T: int, beta_1: float, beta_T: float) -> dict:
    """
    Pre-compute all quantities needed for forward and reverse diffusion.

    Linear beta schedule: betas increase linearly from beta_1 to beta_T.
    Alphas and their cumulative products are derived from betas.
    All tensors are created on CPU.  Call schedule_to_device() in train()
    to move them to the GPU once before the epoch loop so that CUDA-tensor
    indexing in q_sample works correctly.
    """
    betas  = torch.linspace(beta_1, beta_T, T)          # (T,)
    alphas = 1.0 - betas                                 # (T,)
    alpha_bar = torch.cumprod(alphas, dim=0)             # (T,) cumulative product
    sqrt_alpha_bar     = alpha_bar.sqrt()
    sqrt_one_minus_ab  = (1.0 - alpha_bar).sqrt()

    return {
        "betas":             betas,
        "alphas":            alphas,
        "alpha_bar":         alpha_bar,
        "sqrt_ab":           sqrt_alpha_bar,
        "sqrt_1m_ab":        sqrt_one_minus_ab,
    }

SCHEDULE = make_noise_schedule(T, BETA_1, BETA_T)


def schedule_to_device(schedule: dict, device: str) -> dict:
    """
    Move every tensor in the noise schedule to `device` and return a new dict.

    WHY THIS IS NECESSARY
    ---------------------
    SCHEDULE is built at module import time on CPU (torch.linspace lives on CPU
    by default).  During training, the timestep tensor `t` is created on the
    GPU via torch.randint(..., device=DEVICE).

    PyTorch requires the index tensor and the tensor being indexed to be on the
    SAME device.  Indexing a CPU schedule tensor with a CUDA `t` therefore
    raises:
        RuntimeError: indices should be either on cpu or on the same device
        as the indexed tensor (cpu)

    Calling this function once in train(), before the epoch loop, moves the
    entire schedule to the GPU so that schedule["sqrt_ab"][t] works for any
    device `t` lives on.  The per-element .to(device) calls in q_sample and
    p_sample_loop are then removed -- they become no-ops anyway and add
    unnecessary overhead per batch / per reverse-diffusion step.
    """
    return {k: v.to(device) for k, v in schedule.items()}


def q_sample(mask: torch.Tensor, t: torch.Tensor, schedule: dict,
             noise: torch.Tensor = None) -> tuple:
    """
    Forward diffusion: add noise to `mask` at timestep t.

    mask     : (B, 1, H, W) float32 binary mask, values {0, 1}
    t        : (B,) int64 timestep indices — must be on the same device as
               schedule tensors.  Call schedule_to_device() in train() first.
    schedule : noise schedule dict already moved to the correct device.

    Returns (noisy_mask, noise) where noise is what the model must predict.
    """
    if noise is None:
        noise = torch.randn_like(mask)

    # FIX: schedule tensors are now on the same device as t (moved in train()
    # via schedule_to_device()).  The previous .to(mask.device) calls are
    # removed — they were the wrong solution: they moved the *result* after
    # indexing, but the crash happened *during* indexing when t was on CUDA
    # and schedule["sqrt_ab"] was still on CPU.
    sqrt_ab   = schedule["sqrt_ab"][t].view(-1, 1, 1, 1)
    sqrt_1mab = schedule["sqrt_1m_ab"][t].view(-1, 1, 1, 1)

    noisy = sqrt_ab * mask + sqrt_1mab * noise
    return noisy, noise


# ==============================================================================
#   DENOISING UNET  (time-conditioned)
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timestep t.
    Maps scalar t -> embedding vector of size `dim`.
    Identical to the positional encoding in the original Transformer paper
    but applied to the time dimension instead of sequence position.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """
    Return the largest divisor of `channels` that is <= max_groups.
    GroupNorm requires num_channels % num_groups == 0.
    With a hardcoded value of 8 this fails when channels=1 (the noisy mask
    input to enc1).  This helper ensures a valid group count for any channel
    size, including 1 (which maps to num_groups=1, i.e. LayerNorm behaviour).
    """
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1   # fallback -- always valid


class TimeCondResBlock(nn.Module):
    """
    Residual block that injects time embedding via scale-shift (AdaGN).
    Standard in DDPM UNet implementations.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        # FIX: use _num_groups() instead of hardcoded 8.
        # enc1 receives in_ch=1 (the noisy mask); GroupNorm(8, 1) raises
        # ValueError because 1 is not divisible by 8.
        self.norm1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)  # scale + shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # inject time via scale-shift
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DenoisingUNet(nn.Module):
    """
    Lightweight UNet that predicts the noise added to the mask at timestep t,
    conditioned on both t and image features from the frozen encoder.

    Input channels per level = noisy_mask_features + image_features.
    The image features are concatenated at the bottleneck to provide
    semantic context that guides denoising toward the rip region.
    """
    def __init__(self, img_feat_ch: int = 2048, time_dim: int = 128,
                 base_ch: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder: process noisy mask (1 channel)
        self.enc1 = TimeCondResBlock(1,          base_ch,     time_dim)
        self.enc2 = TimeCondResBlock(base_ch,    base_ch * 2, time_dim)
        self.enc3 = TimeCondResBlock(base_ch * 2, base_ch * 4, time_dim)

        self.down = nn.MaxPool2d(2)

        # Bottleneck: fuse mask features with image features
        # img_feat_ch is the number of image encoder output channels
        self.bottleneck_proj = nn.Conv2d(img_feat_ch, base_ch * 4, 1)
        self.bottleneck = TimeCondResBlock(base_ch * 4 + base_ch * 4,
                                           base_ch * 4, time_dim)

        # Decoder
        # ── Channel accounting (with base_ch=64, IMG_SIZE=256) ───────────────
        # up3  input : b      (B, base_ch*4=256, H/8, W/8)
        # up3 output :        (B, base_ch*2=128, H/4, W/4)
        # skip  e3   :        (B, base_ch*4=256, H/4, W/4)  ← enc3 output
        # cat result :        (B, base_ch*6=384, H/4, W/4)  ← dec3 in_ch
        # dec3 output:        (B, base_ch*2=128, H/4, W/4)
        #
        # up2  input : d3     (B, base_ch*2=128, H/4, W/4)
        # up2 output :        (B, base_ch  = 64, H/2, W/2)
        # skip  e2   :        (B, base_ch*2=128, H/2, W/2)  ← enc2 output
        # cat result :        (B, base_ch*3=192, H/2, W/2)  ← dec2 in_ch
        # dec2 output:        (B, base_ch  = 64, H/2, W/2)
        #
        # up1  input : d2     (B, base_ch  = 64, H/2, W/2)
        # up1 output :        (B, base_ch  = 64, H,   W  )
        # skip  e1   :        (B, base_ch  = 64, H,   W  )  ← enc1 output
        # cat result :        (B, base_ch*2=128, H,   W  )  ← dec1 in_ch
        # dec1 output:        (B, base_ch  = 64, H,   W  )
        # ────────────────────────────────────────────────────────────────────
        self.up3  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        # FIX: dec3 in_ch = up3_out(base_ch*2) + skip_e3(base_ch*4) = base_ch*6
        # Previous value base_ch*4=256 caused GroupNorm crash: weight [256] vs
        # input [B, 384, H/4, W/4] — the cat of 128+256 channels is 384, not 256.
        self.dec3 = TimeCondResBlock(base_ch * 6, base_ch * 2, time_dim)

        self.up2  = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        # FIX: dec2 in_ch = up2_out(base_ch) + skip_e2(base_ch*2) = base_ch*3
        # Previous value base_ch*2=128 would crash: cat of 64+128=192 channels.
        self.dec2 = TimeCondResBlock(base_ch * 3, base_ch,     time_dim)

        self.up1  = nn.ConvTranspose2d(base_ch,     base_ch,     2, stride=2)
        # dec1 in_ch = up1_out(base_ch) + skip_e1(base_ch) = base_ch*2  ← correct
        self.dec1 = TimeCondResBlock(base_ch * 2, base_ch,     time_dim)

        self.out  = nn.Conv2d(base_ch, 1, 1)  # predict noise (single channel)

    def forward(self, noisy_mask: torch.Tensor, t: torch.Tensor,
                img_features: torch.Tensor) -> torch.Tensor:
        """
        noisy_mask   : (B, 1, H, W)  noisy version of the ground truth mask
        t            : (B,)           timestep indices
        img_features : (B, C, H', W') feature map from frozen image encoder
        Returns predicted noise (B, 1, H, W).
        """
        t_emb = self.time_embed(t)

        # Encoder
        e1 = self.enc1(noisy_mask, t_emb)                 # (B, 64,  H,   W  )
        e2 = self.enc2(self.down(e1), t_emb)               # (B, 128, H/2, W/2)
        e3 = self.enc3(self.down(e2), t_emb)               # (B, 256, H/4, W/4)

        # Bottleneck -- fuse with image features
        # FIX: compute down_e3 FIRST, then pool img_features to that spatial
        # size.  The previous code pooled img_features to e3.shape[-2:] (H/4)
        # but then concatenated with self.down(e3) which is H/8 -- a size
        # mismatch that caused:
        #   RuntimeError: Sizes of tensors must match except in dimension 1.
        #   Expected size 32 but got size 64 for tensor number 1 in the list.
        down_e3  = self.down(e3)                                # (B, 256, H/8, W/8)
        img_feat = F.adaptive_avg_pool2d(img_features,
                                         down_e3.shape[-2:])   # (B, C,   H/8, W/8)
        img_feat = self.bottleneck_proj(img_feat)               # (B, 256, H/8, W/8)
        b = torch.cat([down_e3, img_feat], dim=1)              # (B, 512, H/8, W/8)
        b = self.bottleneck(b, t_emb)

        # Decoder with skip connections
        # up3(b)=128ch + e3=256ch → cat=384ch → dec3 → 128ch
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), t_emb)
        # up2(d3)=64ch + e2=128ch → cat=192ch → dec2 → 64ch
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        # up1(d2)=64ch + e1=64ch  → cat=128ch → dec1 → 64ch
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out(d1)


# ==============================================================================
#   IMAGE ENCODER  (frozen ResNet-50 -- DDP approach)
# ==============================================================================

class FrozenImageEncoder(nn.Module):
    """
    Frozen ResNet-50 encoder that extracts semantic image features.

    Frozen because we want the denoising UNet to learn all task-specific
    adaptation.  Using a frozen encoder also halves GPU memory compared
    to training both networks jointly and enables larger batch sizes.
    The encoder provides rich ImageNet features that guide the diffusion
    process toward the rip current region without needing rip-specific
    supervision during encoder training.
    """
    def __init__(self, encoder_name: str = "resnet50",
                 encoder_weights: str = "imagenet"):
        super().__init__()
        # Build a dummy smp model just to get the encoder
        dummy = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = encoder_weights,
            in_channels     = 3,
            classes         = 1,
        )
        self.encoder = dummy.encoder
        # Freeze all parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the deepest feature map from the encoder (B, C, H', W')."""
        features = self.encoder(x)   # list of feature maps at each stage
        return features[-1]          # take the deepest (most semantic) features


# ==============================================================================
#   DATASET & AUGMENTATIONS  (identical to train_unet_transformer.py)
# ==============================================================================

class RipSegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transforms=None):
        images_dir = Path(images_dir)
        masks_dir  = Path(masks_dir)
        self.images = sorted(images_dir.glob("*.jpg")) + \
                      sorted(images_dir.glob("*.png"))
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")
        self.masks_dir  = masks_dir
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
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = (mask > 127).astype("float32")
        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask.unsqueeze(0)


def get_transforms(train: bool = True, size: int = IMG_SIZE) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.6),
            A.HueSaturationValue(p=0.4),
            A.RandomFog(fog_coef_range=(0.05, 0.15), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
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
#   METRICS  (identical to train_unet_transformer.py)
# ==============================================================================

def get_boundary(mask_np: np.ndarray, erosion_px: int = 2) -> np.ndarray:
    eroded = binary_erosion(mask_np, iterations=erosion_px)
    return mask_np.astype(bool) & ~eroded


@torch.no_grad()
def compute_metrics(pred_mask: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    preds = (pred_mask > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(1, 2, 3))

    iou_rip = (tp / (tp + fp + fn + 1e-6)).mean().item()
    iou_bg  = (tn / (tn + fp + fn + 1e-6)).mean().item()
    miou    = (iou_rip + iou_bg) / 2.0
    dice    = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    precision = (tp / (tp + fp + 1e-6)).mean().item()
    recall    = (tp / (tp + fn + 1e-6)).mean().item()
    total = tp + tn + fp + fn
    aacc  = ((tp + tn) / (total + 1e-6)).mean().item()
    macc  = ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6))).mean().item() / 2.0

    preds_np   = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    f2 = sklearn_fbeta(
        targets_np.flatten().astype(int),
        preds_np.flatten().astype(int), beta=2, zero_division=0)
    b_ious = []
    for p, t in zip(preds_np[:, 0], targets_np[:, 0]):
        pb = get_boundary(p.astype(bool))
        tb = get_boundary(t.astype(bool))
        inter = (pb & tb).sum()
        union = (pb | tb).sum()
        b_ious.append(inter / (union + 1e-6))

    return dict(iou=iou_rip, miou=miou, aacc=aacc, macc=macc,
                dice=dice, precision=precision, recall=recall,
                f2=f2, boundary_iou=float(np.mean(b_ious)))


# ==============================================================================
#   INFERENCE  (reverse diffusion)
# ==============================================================================

@torch.no_grad()
def p_sample_loop(denoiser: nn.Module, img_features: torch.Tensor,
                  shape: tuple, schedule: dict,
                  device: str) -> torch.Tensor:
    """
    Reverse diffusion: denoise from pure Gaussian noise over T steps.

    schedule : noise schedule dict already on `device` (moved via
               schedule_to_device() in train()).
    Returns a soft mask in [0, 1] (not yet thresholded).
    """
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        pred_noise = denoiser(x, t, img_features)
        # FIX: .to(device) calls removed — schedule is already on the correct
        # device, so these were redundant overhead on every reverse step.
        alpha     = schedule["alphas"][i]
        alpha_bar = schedule["alpha_bar"][i]
        beta      = schedule["betas"][i]
        coeff     = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean      = (1 / alpha.sqrt()) * (x - coeff * pred_noise)
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + beta.sqrt() * noise
        else:
            x = mean
    return torch.sigmoid(x)


@torch.no_grad()
def diffusion_predict(denoiser: nn.Module, encoder: nn.Module,
                      images: torch.Tensor, schedule: dict,
                      device: str, k: int = K_SAMPLES) -> torch.Tensor:
    """
    Run K independent reverse-diffusion chains and average the results.
    Returns a soft mask (B, 1, H, W) in [0, 1].
    """
    img_features = encoder(images)
    shape = (images.shape[0], 1, images.shape[2], images.shape[3])
    masks = torch.stack([
        p_sample_loop(denoiser, img_features, shape, schedule, device)
        for _ in range(k)
    ]).mean(dim=0)
    return masks


# ==============================================================================
#   EARLY STOPPING  (same design as train_unet_transformer.py)
# ==============================================================================

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE):
        self.patience    = patience
        self.counter     = 0
        self.best_score  = None
        self.should_stop = False

    def step(self, score: float, improved: bool) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}  "
                  f"(best mIoU={self.best_score:.4f}, current={score:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  EarlyStopping: patience exhausted. Stopping.")
        return self.should_stop

    def state_dict(self):
        return {"counter": self.counter, "best_score": self.best_score}

    def load_state_dict(self, state):
        self.counter    = state["counter"]
        self.best_score = state["best_score"]


# ==============================================================================
#   SEED
# ==============================================================================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==============================================================================
#   MAIN
# ==============================================================================

def train() -> None:
    print(f"Device         : {DEVICE}")
    print(f"Image encoder  : {IMG_ENCODER} (frozen)")
    print(f"Diffusion T    : {T}  |  K samples : {K_SAMPLES}")
    print(f"Image size     : {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}")

    # -- Models ----------------------------------------------------------------
    encoder  = FrozenImageEncoder(IMG_ENCODER, ENCODER_WGTS).to(DEVICE)
    # Determine encoder output channels by a single forward pass
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        feat_ch   = encoder(dummy_img).shape[1]
    print(f"Encoder output channels: {feat_ch}")

    denoiser = DenoisingUNet(img_feat_ch=feat_ch).to(DEVICE)

    # FIX: move the noise schedule to the GPU once here, before the epoch
    # loop.  SCHEDULE is built at import time on CPU.  During training, the
    # timestep tensor `t` is created on DEVICE (CUDA).  PyTorch forbids
    # indexing a CPU tensor with a CUDA index tensor, producing:
    #   RuntimeError: indices should be either on cpu or on the same device
    #   as the indexed tensor (cpu)
    # Moving the schedule to DEVICE once here costs ~5 KB of VRAM and
    # eliminates the crash.  All downstream functions (q_sample,
    # p_sample_loop, diffusion_predict) accept schedule as a parameter and
    # will now receive this GPU copy.
    device_schedule = schedule_to_device(SCHEDULE, DEVICE)

    # -- Optimiser: only denoiser parameters are trained ---------------------
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # -- Data ----------------------------------------------------------------
    train_ds = RipSegDataset(TRAIN_IMGS, TRAIN_MASKS,
                             transforms=get_transforms(train=True))
    val_ds   = RipSegDataset(VAL_IMGS,   VAL_MASKS,
                             transforms=get_transforms(train=False))
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE == "cuda"))

    # -- State ---------------------------------------------------------------
    best_val_iou = 0.0
    best_metrics = {}
    start_epoch  = 1

    if RESUME_FROM and Path(RESUME_FROM).exists():
        print(f"\nResuming from: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        denoiser.load_state_dict(ckpt["denoiser_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_iou = ckpt["val_iou"]
        start_epoch  = RESUME_EPOCH + 1
        if "early_stopping_state" in ckpt:
            early_stopping.load_state_dict(ckpt["early_stopping_state"])
        print(f"  Restored epoch {RESUME_EPOCH}  |  best mIoU: {best_val_iou:.4f}")
    else:
        print("Starting training from scratch.")

    # -- Epoch loop ----------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        epoch_start = time.time()

        # ---- TRAIN ----------------------------------------------------------
        denoiser.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc="  train", leave=False,
                    ascii=True, dynamic_ncols=False)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            # Sample random timesteps for this batch
            t = torch.randint(0, T, (images.shape[0],), device=DEVICE)

            # Forward diffusion: corrupt the mask
            # device_schedule is already on DEVICE so t (CUDA tensor) can
            # safely index it without the RuntimeError.
            noisy_mask, noise = q_sample(masks, t, device_schedule)

            # Get frozen image features (no grad)
            with torch.no_grad():
                img_feat = encoder(images)

            # Predict the added noise
            optimizer.zero_grad()
            pred_noise = denoiser(noisy_mask, t, img_feat)

            # MSE loss on noise prediction (standard DDPM objective)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, len(train_loader))

        # ---- VALIDATE -------------------------------------------------------
        denoiser.eval()
        accum = dict(iou=0.0, miou=0.0, aacc=0.0, macc=0.0, dice=0.0,
                     precision=0.0, recall=0.0, f2=0.0, boundary_iou=0.0)
        n = 0
        for images, masks in tqdm(val_loader, desc="  val  ", leave=False,
                                   ascii=True, dynamic_ncols=False):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            # Run reverse diffusion to get soft mask
            soft_mask = diffusion_predict(denoiser, encoder, images,
                                          device_schedule, DEVICE, k=K_SAMPLES)
            m = compute_metrics(soft_mask, masks)
            for k in accum:
                accum[k] += m[k]
            n += 1
        val_metrics = {k: v / max(1, n) for k, v in accum.items()}

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
            print("  Low recall -- consider lowering T or increasing K_SAMPLES.")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["miou"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        improved = val_metrics["miou"] > best_val_iou
        if improved:
            best_val_iou = val_metrics["miou"]
            best_metrics = val_metrics.copy()
            torch.save(
                {
                    "epoch":                epoch,
                    "denoiser_state":       denoiser.state_dict(),
                    "optimizer_state":      optimizer.state_dict(),
                    "early_stopping_state": early_stopping.state_dict(),
                    "val_iou":              best_val_iou,
                    "config": {
                        "img_encoder": IMG_ENCODER,
                        "T":           T,
                        "K_samples":   K_SAMPLES,
                        "img_size":    IMG_SIZE,
                    },
                },
                CHECKPOINT,
            )
            print(f"  Saved best model -> {CHECKPOINT}  (mIoU={best_val_iou:.4f})")

        if early_stopping.step(val_metrics["miou"], improved=improved):
            print(f"\n  Training stopped early at epoch {epoch}.")
            break

    # -- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Training {'stopped early' if early_stopping.should_stop else 'complete'}.  "
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
    print(f" Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()
