"""
Inference + postprocessing + metrics for rip current segmentation.

Supports:
  - Single image inference
  - Folder batch inference
  - Live video overlay (press q to quit)
  - Video heatmap (temporal accumulation)

Architecture compatibility
--------------------------
Set MODEL_FAMILY + BACKBONE + ARCHITECTURE in the CONFIG section.

  MODEL_FAMILY = "smp"        -> all CNN and transformer backbones via
                                 segmentation-models-pytorch + timm:
                                 UNet, UNet++, FPN, DeepLabV3+,
                                 Attention UNet (scSE), MAnet (Attention
                                 Residual) with resnet50, resnet34,
                                 efficientnet-b2, convnext_tiny,
                                 swin_tiny, mambaout_tiny, etc.

  MODEL_FAMILY = "segformer"  -> HuggingFace SegFormer (MiT encoder).
                                 Set SEGFORMER_LOCAL_PATH to the folder
                                 you downloaded with snapshot_download.

  MODEL_FAMILY = "diffusion"  -> Diffusion-based segmentation (SegDiff /
                                 DDP style).  Requires TWO checkpoint
                                 components: a frozen ResNet-50 image
                                 encoder and a time-conditioned denoising
                                 UNet.  Both are stored in the single
                                 checkpoint file produced by
                                 train_diffusion_seg.py.
                                 Inference runs K reverse-diffusion chains
                                 and averages them into a soft mask.
                                 Additionally produces a pixel-level
                                 UNCERTAINTY MAP (variance across K chains)
                                 saved alongside the prediction overlay.

The config block is the only thing that changes between experiments.
Everything else -- preprocessing, postprocessing, metrics, video -- is
identical for all three families.
"""

import os
import math
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# CHANGE: suppress Albumentations version-check network call.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# CHANGE: timm is imported so smp can discover "tu-" timm-backed encoders
# (VMamba, ConvNeXt, Swin).  No explicit timm calls are made here.
try:
    import timm  # noqa: F401
except ImportError:
    pass  # timm is only needed for "tu-" backbones; plain smp backbones work without it

# scipy is used for boundary IoU erosion
from scipy.ndimage import binary_erosion

# sklearn is used for F2 score
from sklearn.metrics import fbeta_score as sklearn_fbeta


# ==============================================================================
#   CONFIGURATION -- edit these constants to switch between experiments
# ==============================================================================

# CHANGE: GPU is now the default; falls back to CPU if no CUDA device is found.
# The original script hardcoded DEVICE = "cpu".
DEVICE    = "cuda"

IMG_SIZE  = 256       # must match the size used during training
THRESHOLD = 0.5       # sigmoid threshold for binary prediction
MIN_AREA  = 500       # minimum rip blob area in pixels (removes noise)

# -- Model family -------------------------------------------------------------
# CHANGE: MODEL_FAMILY selects the loading strategy.
#   "smp"       -> build model via smp (UNet, UNet++, FPN, DeepLabV3+)
#   "segformer" -> build model via HuggingFace transformers + SegFormerWrapper
MODEL_FAMILY = "smp"

# -- smp family settings (used when MODEL_FAMILY = "smp") --------------------
# CHANGE: BACKBONE and ARCHITECTURE replace the hardcoded
# smp.Unet(encoder_name="resnet34") from the original script.
#
# Backbone quick-reference:
#   "resnet34"                          plain CNN, fast
#   "resnet50"                          plain CNN, +accuracy
#   "efficientnet-b2"                   efficient CNN
#   "tu-convnext_tiny"                  ConvNeXt (timm)
#   "tu-swin_tiny_patch4_window7_224"   Swin Transformer (timm)
#   "tu-vmamba_tiny_s2l5"               VMamba (timm, needs timm>=1.0.3)
#
# Architecture quick-reference:
#   "unet"           -> classic skip-connection encoder-decoder
#   "unetplusplus"   -> nested dense skip connections
#   "fpn"            -> Feature Pyramid Network
#   "deeplabv3plus"  -> ASPP multi-scale CNN decoder
#   "attention_unet" -> UNet + scSE channel+spatial attention gates on skips
#   "manet"          -> Multi-scale Attention Net (Attention Residual):
#                       PAB global spatial attention + MFAB multi-scale
#                       channel attention.  Use with resnet50 backbone.
BACKBONE        = "efficientnet-b2"  # smp encoder name (e.g. "resnet34", "efficientnet-b2", "tu-convnext_tiny")
ARCHITECTURE    = "unet"
ENCODER_WEIGHTS = "imagenet"   # None at inference -- weights come from checkpoint

# -- SegFormer settings (used when MODEL_FAMILY = "segformer") ---------------
# Path to the folder downloaded with:
#   snapshot_download("nvidia/segformer-b2-finetuned-ade-512-512",
#                     local_dir="./segformer-b2-local")
SEGFORMER_LOCAL_PATH = "./segformer-b2-local"

# -- Diffusion settings (used when MODEL_FAMILY = "diffusion") ---------------
# CHANGE: diffusion inference requires its own constants that match the values
# used in train_diffusion_seg.py exactly.  If any of these differ from the
# training run, the loaded weights will produce incorrect results because the
# denoising UNet's architecture and the reverse-diffusion loop both depend on
# these values.
#
# T, BETA_1, BETA_T: must match train_diffusion_seg.py (100, 1e-4, 0.02).
# DIFF_IMG_ENCODER : must match IMG_ENCODER in train_diffusion_seg.py.
# DIFF_K_SAMPLES   : number of reverse chains averaged at inference.
#                    Use K=5 for final evaluation (full uncertainty map).
#                    Use K=1 for quick preview (no uncertainty map).
DIFF_T           = 100
DIFF_BETA_1      = 1e-4
DIFF_BETA_T      = 0.02
DIFF_IMG_ENCODER = "resnet50"   # frozen image encoder backbone
DIFF_K_SAMPLES   = 5            # chains to average at inference (5 = full eval)

# -- File paths ---------------------------------------------------------------
CHECKPOINT    = "unet_efficientnet-b2.pth"

SINGLE_IMAGE  = "data_local/test_local/val/images/RipVIS-046_01026.jpg" # RipVIS-007_00037
VAL_IMGS_DIR  = "data_local/test_local/val/images"
VAL_MASKS_DIR = "data_local/test_local/val/masks"

# SINGLE_IMAGE  = "data_local/val_local/images/RipDetSeg-0gjloo1io99u.jpg"
# VAL_IMGS_DIR  = "data_local/val_local/images"
# VAL_MASKS_DIR = "data_local/val_local/masks"

VIDEO_IN      = "data_local/test_local/videos/RipVIS-025.mp4"
VIDEO_OUT     = "out_infer/RipVIS-025_overlay.mp4"
OUT_DIR       = "out_infer"


# ==============================================================================
#   SEGFORMER WRAPPER
#   Identical to the one in train_segformer.py -- must match exactly so the
#   checkpoint state dict loads without key mismatches.
# ==============================================================================

# CHANGE: SegFormerWrapper is included here so this single inference file
# can load both smp and SegFormer checkpoints.  The wrapper upsamples
# SegFormer's 1/4-resolution decoder output back to full image resolution,
# giving it the same output contract as smp models.
class SegFormerWrapper(torch.nn.Module):
    def __init__(self, hf_model: torch.nn.Module, output_size: tuple):
        super().__init__()
        self.model       = hf_model
        self.output_size = output_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out    = self.model(pixel_values=pixel_values)
        logits = out.logits                          # (B, 1, H/4, W/4)
        return F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )                                            # (B, 1, H, W)



# ==============================================================================
#   DIFFUSION MODEL COMPONENTS
#   Copied verbatim from train_diffusion_seg.py.
#   These classes MUST be identical to their training counterparts so that
#   the checkpoint state_dict keys match exactly when loaded at inference.
#   Any difference — even a renamed layer — will cause load_state_dict to
#   raise a key mismatch error.
# ==============================================================================

def _diff_num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of channels that is <= max_groups (for GroupNorm)."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep t."""
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
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _TimeCondResBlock(nn.Module):
    """Residual block with AdaGN time-step conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1     = nn.GroupNorm(_diff_num_groups(in_ch),  in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(_diff_num_groups(out_ch), out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class _DenoisingUNet(nn.Module):
    """
    Time-conditioned UNet that predicts added noise at timestep t.
    Architecture must be byte-for-byte identical to DenoisingUNet in
    train_diffusion_seg.py — any layer name change breaks checkpoint loading.
    """
    def __init__(self, img_feat_ch: int = 2048, time_dim: int = 128,
                 base_ch: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.enc1 = _TimeCondResBlock(1,           base_ch,     time_dim)
        self.enc2 = _TimeCondResBlock(base_ch,     base_ch * 2, time_dim)
        self.enc3 = _TimeCondResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.down = nn.MaxPool2d(2)

        self.bottleneck_proj = nn.Conv2d(img_feat_ch, base_ch * 4, 1)
        self.bottleneck = _TimeCondResBlock(base_ch * 4 + base_ch * 4,
                                            base_ch * 4, time_dim)

        self.up3  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = _TimeCondResBlock(base_ch * 6, base_ch * 2, time_dim)
        self.up2  = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec2 = _TimeCondResBlock(base_ch * 3, base_ch,     time_dim)
        self.up1  = nn.ConvTranspose2d(base_ch,     base_ch,     2, stride=2)
        self.dec1 = _TimeCondResBlock(base_ch * 2, base_ch,     time_dim)
        self.out  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, noisy_mask: torch.Tensor, t: torch.Tensor,
                img_features: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        e1 = self.enc1(noisy_mask, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)
        down_e3  = self.down(e3)
        img_feat = F.adaptive_avg_pool2d(img_features, down_e3.shape[-2:])
        img_feat = self.bottleneck_proj(img_feat)
        b = torch.cat([down_e3, img_feat], dim=1)
        b = self.bottleneck(b, t_emb)
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out(d1)


class _FrozenImageEncoder(nn.Module):
    """Frozen ResNet-50 image encoder (feature extractor only, not trained)."""
    def __init__(self, encoder_name: str = "resnet50"):
        super().__init__()
        dummy = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = None,   # weights loaded from checkpoint below
            in_channels     = 3,
            classes         = 1,
        )
        self.encoder = dummy.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)[-1]


def _make_diff_schedule(T: int, beta_1: float, beta_T: float,
                        device: str) -> dict:
    """Build and immediately move the noise schedule to `device`."""
    betas     = torch.linspace(beta_1, beta_T, T)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas":     betas.to(device),
        "alphas":    alphas.to(device),
        "alpha_bar": alpha_bar.to(device),
    }


@torch.no_grad()
def _p_sample_loop(denoiser: nn.Module, img_features: torch.Tensor,
                   shape: tuple, schedule: dict, T: int,
                   device: str) -> torch.Tensor:
    """One full reverse-diffusion chain: pure noise → soft mask."""
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t     = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps   = denoiser(x, t, img_features)
        alpha     = schedule["alphas"][i]
        alpha_bar = schedule["alpha_bar"][i]
        beta      = schedule["betas"][i]
        coeff = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean  = (1 / alpha.sqrt()) * (x - coeff * eps)
        x     = mean + (beta.sqrt() * torch.randn_like(x) if i > 0 else 0)
    return torch.sigmoid(x)


@torch.no_grad()
def predict_diffusion(encoder: nn.Module, denoiser: nn.Module,
                      image_tensor: torch.Tensor, schedule: dict,
                      T: int, device: str,
                      k: int = 5) -> tuple:
    """
    Diffusion inference: run K reverse chains, return mean mask + uncertainty.

    CHANGE: this is the unique capability of the diffusion model that no
    other MODEL_FAMILY can provide.  By running the stochastic reverse
    process K times independently, we obtain K plausible segmentation masks.
    Their mean is the final prediction.  Their pixel-wise variance is an
    uncertainty map: high variance = the model is unsure about that pixel,
    low variance = confident prediction.

    For a safety-critical rip current system, the uncertainty map can flag
    regions where the boundary estimate is unreliable, enabling downstream
    systems to issue conservative alerts.

    Parameters
    ----------
    image_tensor : (1, 3, H, W) normalised float tensor on `device`.
    k            : number of reverse chains.  k=1 is fast but no uncertainty.
                   k=5 gives a reliable uncertainty estimate.

    Returns
    -------
    mean_mask    : (H, W) float32 numpy in [0, 1] -- final soft prediction
    uncertainty  : (H, W) float32 numpy in [0, 1] -- pixel-wise variance
                   normalised to [0, 1].  None when k=1.
    """
    img_features = encoder(image_tensor)
    shape = (image_tensor.shape[0], 1,
             image_tensor.shape[2], image_tensor.shape[3])

    # Incremental accumulation to avoid holding K chains simultaneously
    acc  = torch.zeros(shape, device=device)
    acc2 = torch.zeros(shape, device=device)   # sum of squares for variance

    for _ in range(k):
        chain = _p_sample_loop(denoiser, img_features, shape, schedule, T, device)
        acc  += chain
        acc2 += chain ** 2
        del chain
        torch.cuda.empty_cache()

    mean_mask = (acc / k)[0, 0].cpu().numpy()

    if k > 1:
        variance    = (acc2 / k) - (acc / k) ** 2
        var_np      = variance[0, 0].cpu().numpy()
        uncertainty = (var_np - var_np.min()) / (var_np.max() - var_np.min() + 1e-8)
    else:
        uncertainty = None

    return mean_mask, uncertainty


# ==============================================================================
#   1  MODEL BUILDING
# ==============================================================================

def _build_smp_model() -> torch.nn.Module:
    """
    Construct an smp model from BACKBONE + ARCHITECTURE constants.

    CHANGE: arch_map extended with two attention-based architectures:

    "attention_unet" -- smp.Unet with decoder_attention_type="scse".
        Adds Squeeze-and-Excitation Channel-Spatial (scSE) gates to every
        decoder skip connection.  scSE recalibrates both which channels and
        which spatial locations matter at each decoder stage independently.
        Handled via a special branch because the extra kwarg
        decoder_attention_type cannot be passed through the generic
        arch_map[arch](...) call.

    "manet" -- smp.MAnet, the Multi-scale Attention Net.
        Implements the Attention Residual architecture:
          PAB  (Position-wise Attention Block)  -- global spatial attention
               across all positions in the feature map, capturing full rip
               channel geometry regardless of how far the channel extends.
          MFAB (Multi-scale Fusion Attention Block) -- channel attention
               fused from multiple encoder stages simultaneously, combining
               fine texture and coarse structure cues.
        Consistent with train_unet_transformer.py so that a checkpoint
        saved by that script loads correctly here without key mismatches.
    """
    arch_map = {
        "unet":           smp.Unet,
        "unetplusplus":   smp.UnetPlusPlus,
        "fpn":            smp.FPN,
        "deeplabv3plus":  smp.DeepLabV3Plus,
        "attention_unet": None,    # handled separately — needs extra kwarg
        "manet":          smp.MAnet,
    }
    if ARCHITECTURE not in arch_map:
        raise ValueError(
            f"Unknown ARCHITECTURE '{ARCHITECTURE}'. "
            f"Choose from: {list(arch_map.keys())}"
        )

    # CHANGE: validate timm backbone availability before passing to smp so
    # the error message is clear and actionable rather than a cryptic
    # RuntimeError buried inside smp/timm internals.
    if BACKBONE.startswith("tu-"):
        try:
            import timm as _timm
            backbone_name = BACKBONE[3:]
            available = _timm.list_models(backbone_name)
            if not available:
                family = backbone_name.split("_")[0]
                suggestions = _timm.list_models(f"*{family}*")[:3]
                raise RuntimeError(
                    f"\n  Backbone '{BACKBONE}' not found in timm "
                    f"({_timm.__version__}).\n"
                    f"  Run: pip install --upgrade timm\n"
                    f"  Or use one of: {suggestions}"
                )
        except ImportError:
            raise ImportError(
                "timm is required for 'tu-' backbones. "
                "Install with: pip install timm"
            )

    # CHANGE: attention_unet requires decoder_attention_type="scse" as an
    # extra keyword argument that the standard arch_map[arch](...) call
    # cannot pass.  All other architectures use the standard path.
    if ARCHITECTURE == "attention_unet":
        return smp.Unet(
            encoder_name           = BACKBONE,
            encoder_weights        = ENCODER_WEIGHTS,
            decoder_attention_type = "scse",
            in_channels            = 3,
            classes                = 1,
            activation             = None,
        )

    return arch_map[ARCHITECTURE](
        encoder_name    = BACKBONE,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )


def _build_diffusion_model() -> tuple:
    """
    Construct the two-component diffusion model: (encoder, denoiser).

    CHANGE: diffusion is the only MODEL_FAMILY that returns TWO model objects
    instead of one.  This is why it needs its own MODEL_FAMILY value rather
    than being an ARCHITECTURE entry inside "smp".

    The encoder is a frozen ResNet-50 feature extractor.
    The denoiser is the time-conditioned UNet trained by train_diffusion_seg.py.

    Both are instantiated here with no pretrained weights (weights_only=False
    is set in load_model where the checkpoint is actually read).  The encoder
    channel count is determined by a dummy forward pass, exactly as in
    train_diffusion_seg.py, so the denoiser is built with the correct
    img_feat_ch even if the encoder is changed in the future.
    """
    encoder = _FrozenImageEncoder(DIFF_IMG_ENCODER)
    # Determine encoder output channels via a dummy forward pass on CPU
    # (same approach used in train_diffusion_seg.py)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        feat_ch = encoder(dummy).shape[1]
    denoiser = _DenoisingUNet(img_feat_ch=feat_ch)
    return encoder, denoiser
    """Construct a SegFormerWrapper from SEGFORMER_LOCAL_PATH."""
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        raise ImportError(
            "transformers package is required for SegFormer. "
            "Install with: pip install transformers"
        )
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_LOCAL_PATH,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    return SegFormerWrapper(hf_model, output_size=(IMG_SIZE, IMG_SIZE))


# ==============================================================================
#   2  MODEL LOADING
# ==============================================================================

def load_model(checkpoint_path: str, device: str = DEVICE) -> torch.nn.Module:
    """
    Load a segmentation model from a checkpoint file.

    CHANGE: the original script hardcoded smp.Unet(encoder_name="resnet34").
    This version builds the architecture from MODEL_FAMILY / BACKBONE /
    ARCHITECTURE constants, so any trained model can be loaded by updating
    those three config values -- no code change required.

    Also reads backbone + architecture from the checkpoint's config dict
    (if present) and prints them so you can verify the loaded model matches
    what you expect.

    Handles both checkpoint formats:
      Format A  {"model_state": ..., "epoch": ..., "val_iou": ..., "config": ...}
      Format B  bare state_dict  (older training scripts)
    """
def load_model(checkpoint_path: str, device: str = DEVICE):
    """
    Load a segmentation model from a checkpoint file.

    Returns
    -------
    For MODEL_FAMILY "smp" and "segformer" : a single torch.nn.Module.
    For MODEL_FAMILY "diffusion"           : a tuple (encoder, denoiser).
        Both must be passed to predict_diffusion() together.

    CHANGE: diffusion adds a third branch.  The checkpoint saved by
    train_diffusion_seg.py uses key "denoiser_state" (not "model_state")
    because the denoiser is the only trained component.  The encoder weights
    are the original ImageNet ResNet-50 weights — they are stored separately
    and loaded here by rebuilding the FrozenImageEncoder with ENCODER_WGTS.
    The encoder state is NOT in the diffusion checkpoint because it was
    frozen during training and never updated.
    """
    if MODEL_FAMILY == "smp":
        model = _build_smp_model()
    elif MODEL_FAMILY == "segformer":
        model = _build_segformer_model()
    elif MODEL_FAMILY == "diffusion":
        encoder, denoiser = _build_diffusion_model()
        encoder.to(device)
        denoiser.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict):
            cfg = checkpoint.get("config", {})
            print(f"  Checkpoint info:")
            print(f"    Epoch          : {checkpoint.get('epoch', 'unknown')}")
            print(f"    Best val mIoU  : {checkpoint.get('val_iou', 0.0):.4f}")
            print(f"    Img encoder    : {cfg.get('img_encoder', 'unknown')}")
            print(f"    Diffusion T    : {cfg.get('T', 'unknown')}")
            print(f"    K train        : {cfg.get('K_train', 'unknown')}")
            # Load denoiser weights only — encoder was frozen and not saved
            if "denoiser_state" in checkpoint:
                denoiser.load_state_dict(checkpoint["denoiser_state"])
            else:
                raise KeyError(
                    f"Checkpoint '{checkpoint_path}' has no 'denoiser_state' key. "
                    f"Ensure it was saved by train_diffusion_seg.py."
                )
        else:
            raise ValueError(
                f"Diffusion checkpoint '{checkpoint_path}' is not a dict. "
                f"Expected the format saved by train_diffusion_seg.py."
            )

        # Load ImageNet weights into the frozen encoder separately.
        # These are the same weights used during training; the encoder
        # was never updated so no task-specific weights exist for it.
        try:
            import segmentation_models_pytorch as _smp
            dummy = _smp.Unet(
                encoder_name    = DIFF_IMG_ENCODER,
                encoder_weights = "imagenet",
                in_channels     = 3,
                classes         = 1,
            )
            encoder.encoder.load_state_dict(dummy.encoder.state_dict())
            del dummy
            print(f"  Encoder ImageNet weights loaded for '{DIFF_IMG_ENCODER}'.")
        except Exception as e:
            print(f"  Warning: could not reload encoder ImageNet weights: {e}")
            print(f"  Proceeding with randomly initialised encoder.")

        encoder.eval()
        denoiser.eval()
        print(f"  Diffusion model loaded from : {checkpoint_path}")
        print(f"  Running on                  : {device}")
        return encoder, denoiser
    else:
        raise ValueError(
            f"Unknown MODEL_FAMILY '{MODEL_FAMILY}'. "
            f"Use 'smp', 'segformer', or 'diffusion'."
        )

    model.to(device)

    # -- Load raw checkpoint --------------------------------------------------
    # CHANGE: weights_only=False is set explicitly for PyTorch >= 2.0 compat.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # -- Extract state dict ---------------------------------------------------
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            cfg = checkpoint.get("config", {})
            print(f"  Checkpoint info:")
            print(f"    Epoch          : {checkpoint.get('epoch', 'unknown')}")
            print(f"    Best val IoU   : {checkpoint.get('val_iou', 0.0):.4f}")
            print(f"    Backbone       : {cfg.get('backbone', cfg.get('encoder', 'unknown'))}")
            print(f"    Architecture   : {cfg.get('architecture', 'unknown')}")
            print(f"    Img size       : {cfg.get('img_size', 'unknown')}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint   # Format B: bare state dict
    else:
        state_dict = checkpoint       # Very old PyTorch OrderedDict saves

    # -- Strip DataParallel prefix if present ---------------------------------
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded from  : {checkpoint_path}")
    print(f"  Running on         : {device}")
    return model


# ==============================================================================
#   3  PREPROCESSING
# ==============================================================================

def make_transform(size: int = IMG_SIZE) -> Compose:
    """Resize + ImageNet normalise + to tensor.  Matches the val transform."""
    return Compose([
        Resize(size, size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def preprocess_bgr(img_bgr: np.ndarray, transform: Compose) -> torch.Tensor:
    """Convert a BGR numpy image (from cv2) to a (1, 3, H, W) float tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(image=img_rgb)["image"]
    return tensor.unsqueeze(0)


# ==============================================================================
#   4  PREDICTION
# ==============================================================================

def predict(
    model:     torch.nn.Module,
    tensor:    torch.Tensor,
    device:    str   = DEVICE,
    threshold: float = THRESHOLD,
) -> tuple:
    """
    Forward pass.  Returns (prob_map, bin_mask) both as float32 numpy (H, W).

    CHANGE: .to(device, non_blocking=True) is used instead of plain .to(device)
    so that GPU transfer overlaps with CPU work when pin_memory=True is set on
    the DataLoader (not applicable here for single images, but consistent with
    the training pipeline convention and costs nothing for single images).
    """
    with torch.no_grad():
        logits = model(tensor.to(device, non_blocking=True))   # (1, 1, H, W)
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()    # (H, W)
    return probs, (probs >= threshold).astype("uint8")


# ==============================================================================
#   5  POSTPROCESSING
# ==============================================================================

def postprocess(
    binary_mask:  np.ndarray,
    min_area:     int  = MIN_AREA,
    keep_largest: bool = False,
) -> np.ndarray:
    """
    Clean the raw binary prediction mask.
      1. Morphological open  -- removes isolated speckle noise
      2. Morphological close -- fills small holes inside rip regions
      3. Remove blobs smaller than min_area pixels
      4. Optionally keep only the single largest blob
    """
    m      = (binary_mask > 0).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m      = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=1)
    m      = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out   = np.zeros_like(m)
    blobs = []

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lbl] = 255
            blobs.append((lbl, area))

    if keep_largest and blobs:
        best = max(blobs, key=lambda x: x[1])[0]
        out  = (labels == best).astype("uint8") * 255

    return (out > 0).astype("uint8")


# ==============================================================================
#   6  METRICS
# ==============================================================================

def _get_boundary(mask: np.ndarray, erosion_px: int = 2) -> np.ndarray:
    """Return boundary pixels (mask minus its morphological erosion)."""
    eroded = binary_erosion(mask, iterations=erosion_px)
    return mask.astype(bool) & ~eroded


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute a full suite of binary segmentation metrics.

    Both inputs must be binary numpy arrays (values 0 or 1), same shape.

    CHANGE: the original script only returned IoU, Dice, Precision, Recall.
    The following metrics are now added:

    aAcc  -- all-pixel accuracy: correctly classified pixels / total pixels.
             Intuitive for non-specialists but dominated by background on
             class-imbalanced data, so always report alongside mIoU.

    mAcc  -- mean class accuracy: average of per-class pixel accuracies.
             Gives equal weight to the rare rip class and the dominant
             background class, correcting for the imbalance that makes
             aAcc misleading.

    mIoU  -- mean IoU: average of per-class IoU scores.
             The standard headline metric in segmentation papers.  Penalises
             both under-segmentation and over-segmentation across both classes.
             Every published segmentation paper (DeepLab, SegFormer, etc.)
             reports this number -- without it results cannot be compared.

    F2    -- F-beta score with beta=2, weighting Recall twice as heavily as
             Precision.  Justified for rip current detection because a missed
             rip (false negative) is more dangerous than a false alarm.

    BoundaryIoU -- IoU computed only on a narrow band around the mask boundary.
             Rewards models that delineate rip current edges accurately, not
             just those that get the bulk of the region roughly right.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    # -- Per-pixel confusion values -------------------------------------------
    tp = np.logical_and( pred,  gt).sum()
    fp = np.logical_and( pred, ~gt).sum()
    fn = np.logical_and(~pred,  gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    total = tp + fp + fn + tn

    # -- Class-level metrics (rip = class 1, background = class 0) -----------
    iou_rip  = tp / (tp + fp + fn + 1e-6)
    iou_bg   = tn / (tn + fp + fn + 1e-6)   # background IoU
    dice     = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    # -- Dataset-level aggregate metrics -------------------------------------
    # aAcc: fraction of all pixels correctly labelled
    aAcc = (tp + tn) / (total + 1e-6)

    # mAcc: mean of per-class pixel recall
    acc_rip = tp / (tp + fn + 1e-6)   # = recall for rip class
    acc_bg  = tn / (tn + fp + 1e-6)   # background pixel accuracy
    mAcc    = (acc_rip + acc_bg) / 2.0

    # mIoU: mean of per-class IoU (the standard segmentation headline number)
    mIoU = (iou_rip + iou_bg) / 2.0

    # F2: recall-weighted F score (beta=2 => recall weighted 2x over precision)
    f2 = sklearn_fbeta(
        gt.flatten().astype(int),
        pred.flatten().astype(int),
        beta=2,
        zero_division=0,
    )

    # BoundaryIoU: IoU restricted to a dilation band around the mask edge
    pred_boundary = _get_boundary(pred)
    gt_boundary   = _get_boundary(gt)
    b_inter       = (pred_boundary & gt_boundary).sum()
    b_union       = (pred_boundary | gt_boundary).sum()
    boundary_iou  = b_inter / (b_union + 1e-6)

    return dict(
        IoU         = float(iou_rip),
        Dice        = float(dice),
        Precision   = float(precision),
        Recall      = float(recall),
        aAcc        = float(aAcc),
        mAcc        = float(mAcc),
        mIoU        = float(mIoU),
        F2          = float(f2),
        BoundaryIoU = float(boundary_iou),
    )


# CHANGE: print_metrics updated to display the five new metrics.
def print_metrics(metrics: dict, image_name: str = "") -> None:
    """Pretty-print the full metrics dict."""
    label = f"  [{image_name}]" if image_name else ""
    print(
        f"{label}\n"
        f"    IoU={metrics['IoU']:.4f}  "
        f"Dice={metrics['Dice']:.4f}  "
        f"Precision={metrics['Precision']:.4f}  "
        f"Recall={metrics['Recall']:.4f}\n"
        f"    aAcc={metrics['aAcc']:.4f}  "
        f"mAcc={metrics['mAcc']:.4f}  "
        f"mIoU={metrics['mIoU']:.4f}  "
        f"F2={metrics['F2']:.4f}  "
        f"BoundaryIoU={metrics['BoundaryIoU']:.4f}"
    )


# ==============================================================================
#   7  VISUALISATION
# ==============================================================================

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 0, 220), alpha: float = 0.4) -> np.ndarray:
    """Semi-transparent colour overlay where mask==1, plus white contour."""
    overlay      = img_bgr.copy()
    colour_layer = np.zeros_like(img_bgr)
    colour_layer[mask == 1] = color
    cv2.addWeighted(colour_layer, alpha, overlay, 1 - alpha, 0, overlay)
    contours, _ = cv2.findContours(
        (mask * 255).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


# ==============================================================================
#   8  HIGH-LEVEL: SINGLE IMAGE
# ==============================================================================

def process_image(
    model,
    image_path:   str,
    transform:    Compose,
    device:       str   = DEVICE,
    threshold:    float = THRESHOLD,
    min_area:     int   = MIN_AREA,
    keep_largest: bool  = False,
    out_dir:      str   = None,
    gt_masks_dir: str   = VAL_MASKS_DIR,
) -> dict:
    """
    End-to-end inference on a single image file.

    CHANGE: `model` is now either a single nn.Module (smp / segformer) or a
    (encoder, denoiser) tuple (diffusion).  The function detects which case
    applies and routes to the appropriate prediction function.

    For diffusion, an additional uncertainty map PNG is saved alongside the
    standard overlay — named <stem>_uncertainty.png.
    """
    image_path = Path(image_path)
    img_bgr    = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    tensor         = preprocess_bgr(img_bgr, transform)
    uncertainty_map = None

    if isinstance(model, tuple):
        # Diffusion family: model = (encoder, denoiser)
        encoder, denoiser = model
        schedule = _make_diff_schedule(DIFF_T, DIFF_BETA_1, DIFF_BETA_T, device)
        probs, uncertainty_map = predict_diffusion(
            encoder, denoiser,
            tensor.to(device, non_blocking=True),
            schedule, DIFF_T, device, k=DIFF_K_SAMPLES,
        )
        raw_bin = (probs >= threshold).astype("uint8")
    else:
        probs, raw_bin = predict(model, tensor, device, threshold)

    clean_bin = postprocess(raw_bin, min_area, keep_largest)
    mask_full = cv2.resize(
        clean_bin.astype("uint8"), (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = overlay_mask(img_bgr, mask_full)

    metrics = None
    gt_path = Path(gt_masks_dir) / (image_path.stem + ".png")
    if gt_path.exists():
        gt_raw     = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_raw, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)
        gt_bin     = (gt_resized > 127).astype("uint8")
        metrics    = compute_metrics(mask_full, gt_bin)
        print_metrics(metrics, image_path.name)
    else:
        print(f"  [{image_path.name}]  No GT mask found -- metrics skipped.")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / (image_path.stem + "_mask.png")),
                    (mask_full * 255).astype("uint8"))
        cv2.imwrite(str(out_dir / (image_path.stem + "_overlay.png")), overlay)

        # CHANGE: save uncertainty map for diffusion inference when k > 1
        if uncertainty_map is not None:
            unc_vis = cv2.applyColorMap(
                (uncertainty_map * 255).astype("uint8"), cv2.COLORMAP_HOT
            )
            cv2.imwrite(
                str(out_dir / (image_path.stem + "_uncertainty.png")), unc_vis
            )
            print(f"  Uncertainty map saved -> "
                  f"{out_dir / (image_path.stem + '_uncertainty.png')}")

    return dict(probs=probs, raw_mask=raw_bin, clean_mask=mask_full,
                overlay=overlay, metrics=metrics,
                uncertainty=uncertainty_map)


# ==============================================================================
#   9  HIGH-LEVEL: FOLDER BATCH
# ==============================================================================

def process_folder(
    model,
    images_dir:   str,
    transform:    Compose,
    out_dir:      str,
    device:       str   = DEVICE,
    threshold:    float = THRESHOLD,
    min_area:     int   = MIN_AREA,
    gt_masks_dir: str   = VAL_MASKS_DIR,
) -> None:
    """
    Run inference on every image in images_dir.
    Prints per-image metrics when GT masks are available, then a summary.
    """
    images = sorted(Path(images_dir).glob("*.jpg")) + \
             sorted(Path(images_dir).glob("*.png"))
    print(f"\n  Processing {len(images)} images from {images_dir} ...")

    all_metrics = []
    for img_path in images:
        result = process_image(
            model, img_path, transform,
            device=device, threshold=threshold, min_area=min_area,
            out_dir=out_dir, gt_masks_dir=gt_masks_dir,
        )
        if result["metrics"]:
            all_metrics.append(result["metrics"])

    # CHANGE: aggregate summary now covers all nine metrics including the
    # five new ones (aAcc, mAcc, mIoU, F2, BoundaryIoU).
    if all_metrics:
        print(f"\n{'─'*65}")
        print(f"  Summary over {len(all_metrics)} images with GT masks:")
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        print_metrics(avg, "DATASET AVERAGE")
        print(f"{'─'*65}")


# ==============================================================================
#   10  VIDEO: LIVE OVERLAY
# ==============================================================================

def process_video_overlay(
    model:        torch.nn.Module,
    video_path:   str,
    transform:    Compose,
    out_path:     str   = None,
    device:       str   = DEVICE,
    threshold:    float = THRESHOLD,
    min_area:     int   = MIN_AREA,
    keep_largest: bool  = False,
    stride:       int   = 1,
) -> None:
    """
    Play a video with per-frame rip overlay.  Press 'q' to quit.
    Optionally saves the overlaid video to out_path.

    CHANGE: stride default changed from 2 to 1 (every frame) since the
    script now targets GPU by default.  On CPU, set stride=2 or stride=4
    to reduce processing load.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    last_overlay = None
    frame_idx    = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                tensor         = preprocess_bgr(frame, transform)
                probs, raw_bin = predict(model, tensor, device, threshold)
                clean_bin      = postprocess(raw_bin, min_area, keep_largest)
                mask_full      = cv2.resize(
                    clean_bin.astype("uint8"),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                last_overlay = overlay_mask(frame, mask_full)
            display = last_overlay if last_overlay is not None else frame
            cv2.imshow("Rip Detection (q to quit)", display)
            if writer:
                writer.write(display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


# ==============================================================================
#   11  VIDEO: TEMPORAL HEATMAP
# ==============================================================================

def process_video_heatmap(
    model:      torch.nn.Module,
    video_path: str,
    transform:  Compose,
    out_path:   str,
    device:     str   = DEVICE,
    threshold:  float = THRESHOLD,
    stride:     int   = 1,
) -> np.ndarray:
    """
    Accumulate per-frame probability maps into a temporal heatmap.
    Regions frequently predicted as rip appear hot (red) in the output.
    Saves a COLORMAP_JET PNG to out_path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    heatmap   = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            tensor     = preprocess_bgr(frame, transform)
            probs, _   = predict(model, tensor, device, threshold)
            probs_full = cv2.resize(probs, (frame.shape[1], frame.shape[0]))
            if heatmap is None:
                heatmap = np.zeros_like(probs_full, dtype=np.float32)
            heatmap += probs_full
        frame_idx += 1
    cap.release()

    if heatmap is None:
        raise RuntimeError("No frames were processed.")

    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    vis  = cv2.applyColorMap((norm * 255).astype("uint8"), cv2.COLORMAP_JET)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"  Heatmap saved -> {out_path}")
    return vis


# ==============================================================================
#   ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print(f"  Device       : {DEVICE}")
    print(f"  Model family : {MODEL_FAMILY}")
    if MODEL_FAMILY == "smp":
        print(f"  Backbone     : {BACKBONE}")
        print(f"  Architecture : {ARCHITECTURE}")
    elif MODEL_FAMILY == "segformer":
        print(f"  SegFormer    : {SEGFORMER_LOCAL_PATH}")
    else:
        print(f"  Diffusion encoder : {DIFF_IMG_ENCODER}")
        print(f"  Diffusion T       : {DIFF_T}  |  K chains: {DIFF_K_SAMPLES}")
    print(f"  Checkpoint   : {CHECKPOINT}\n")

    model     = load_model(CHECKPOINT, device=DEVICE)
    transform = make_transform(size=IMG_SIZE)

    # -- 1) Single image with metrics -----------------------------------------
    if Path(SINGLE_IMAGE).exists():
        print(f"\n-- Single image --")
        process_image(
            model, SINGLE_IMAGE, transform,
            device=DEVICE, threshold=THRESHOLD, min_area=MIN_AREA,
            out_dir=OUT_DIR, gt_masks_dir=VAL_MASKS_DIR,
        )
    else:
        print(f"Image not found: {SINGLE_IMAGE}")

    # -- 2) Batch folder with aggregate metrics -------------------------------
    # process_folder(
    #     model, VAL_IMGS_DIR, transform,
    #     out_dir=OUT_DIR, device=DEVICE,
    #     gt_masks_dir=VAL_MASKS_DIR,
    # )

    # -- 3) Live video overlay ------------------------------------------------
    # if Path(VIDEO_IN).exists():
    #     print(f"\n-- Video overlay --")
    #     print(f"   Press 'q' to stop.")
    #     process_video_overlay(
    #         model, VIDEO_IN, transform,
    #         out_path=VIDEO_OUT, device=DEVICE,
    #         threshold=THRESHOLD, min_area=MIN_AREA,
    #         stride=1,   # every frame on GPU; raise to 2-4 on CPU
    #     )
    # else:
    #     print(f"Video not found: {VIDEO_IN}")

    # -- 4) Temporal heatmap --------------------------------------------------
    # if Path(VIDEO_IN).exists():
    #     process_video_heatmap(
    #         model, VIDEO_IN, transform,
    #         out_path="out_infer/heatmap.png",
    #         device=DEVICE, stride=1,
    #     )
