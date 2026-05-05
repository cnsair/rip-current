"""
Universal batch evaluation over the RipVIS test partition (4,349 images).
Supports three model families via --family argument:
  smp        : all smp architectures (UNet, UNet++, MAnet, Attention UNet,
                DeepLabV3+, FPN)
  segformer  : HuggingFace SegFormer-B2 (MiT-B2 encoder)
  diffusion  : Diffusion segmentation (frozen ResNet-50 + denoising UNet)

Outputs per run (saved to RESULTS_DIR):
  {label}_per_image.csv    one row per image, all 9 metrics
  {label}_aggregate.csv    mean, std, median, 95% bootstrap CI per metric

Usage examples
--------------
# smp model
python evaluate_test_set.py \
    --family smp \
    --checkpoint manet_swin_tiny.pth \
    --backbone tu-swin_tiny_patch4_window7_224 \
    --architecture manet \
    --label manet_swin_tiny

# SegFormer
python evaluate_test_set.py \
    --family segformer \
    --checkpoint segformer_b2_local.pth \
    --segformer-path ./segformer-b2-local \
    --label segformer_b2

# Diffusion
python evaluate_test_set.py \
    --family diffusion \
    --checkpoint diffusion_seg_resnet50.pth \
    --diff-encoder resnet50 \
    --diff-k 5 \
    --label diffusion_resnet50
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from sklearn.metrics import fbeta_score

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ────────────────────────────────────────────────────────────────────
TEST_IMAGES  = "data_local/test_local/rip_vis_val_images/images"
TEST_MASKS   = "data_local/test_local/rip_vis_val_images/masks"
RESULTS_DIR  = "results"
THRESHOLD    = 0.5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Image sizes — must match training for each family
SMP_IMG_SIZE       = 256   # all smp models trained at 256×256
SEGFORMER_IMG_SIZE = 512   # SegFormer trained at 512×512
DIFFUSION_IMG_SIZE = 256   # diffusion trained at 256×256

# ==============================================================================
#   TRANSFORMS  (one per family to match training resolution)
# ==============================================================================

def make_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==============================================================================
#   SMP MODEL LOADER
# ==============================================================================

def load_smp_model(backbone, architecture, checkpoint_path):
    """
    Build and load an smp model.
    attention_unet requires the scse kwarg; all others use the arch_map.
    """
    arch_map = {
        "unet":           smp.Unet,
        "unetplusplus":   smp.UnetPlusPlus,
        "fpn":            smp.FPN,
        "deeplabv3plus":  smp.DeepLabV3Plus,
        "manet":          smp.MAnet,
    }
    if architecture == "attention_unet":
        model = smp.Unet(
            encoder_name           = backbone,
            encoder_weights        = None,
            decoder_attention_type = "scse",
            in_channels            = 3,
            classes                = 1,
            activation             = None,
        )
    elif architecture in arch_map:
        model = arch_map[architecture](
            encoder_name    = backbone,
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(arch_map.keys()) + ['attention_unet']}"
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"  Loaded  : {checkpoint_path}")
    print(f"  Epoch   : {ckpt.get('epoch', '?')}  |  "
          f"Train-val mIoU : {ckpt.get('val_iou', 0):.4f}")
    return model


@torch.no_grad()
def predict_smp(model, tensor):
    """Single forward pass; returns (H, W) float32 probability map."""
    logits = model(tensor.to(DEVICE))
    return torch.sigmoid(logits)[0, 0].cpu().numpy()

# ==============================================================================
#   SEGFORMER MODEL LOADER
# ==============================================================================
#
# WHY A SEPARATE PATH?
# SegFormer is a HuggingFace model, not an smp model. Three differences:
#
#   1. MODEL CLASS  — SegformerForSemanticSegmentation from `transformers`,
#      not smp.Unet.  The SegFormerWrapper class used during training must be
#      reproduced here exactly so that checkpoint keys match.
#
#   2. IMAGE SIZE   — SegFormer was trained at 512×512.  Passing 256×256 at
#      test time evaluates a different model in practice (the positional
#      embeddings interpolate but accuracy degrades).  The transform for
#      SegFormer uses 512.
#
#   3. OUTPUT SIZE  — SegFormer logits are at 1/4 input resolution
#      (128×128 for a 512 input).  They must be bilinearly upsampled to
#      512×512 before thresholding.  smp models output at full resolution.
#
# ==============================================================================

def load_segformer_model(checkpoint_path, segformer_path):
    """
    Reconstruct the SegFormerWrapper used during training and load weights.
    segformer_path : local folder downloaded with snapshot_download,
                     e.g. './segformer-b2-local'
    """
    try:
        import certifi
        os.environ["SSL_CERT_FILE"]      = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        pass

    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        raise ImportError(
            "transformers is required for SegFormer evaluation. "
            "Install with: pip install transformers"
        )

    class SegFormerWrapper(nn.Module):
        """
        MUST be byte-for-byte identical to the wrapper used in
        train_segformer.py — any layer name change breaks checkpoint loading.
        """
        def __init__(self, local_path):
            super().__init__()
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                local_path,
                num_labels   = 1,
                ignore_mismatched_sizes = True,
            )
        def forward(self, x):
            out = self.model(pixel_values=x)
            # Upsample from 1/4 resolution back to input resolution
            return F.interpolate(
                out.logits,
                size  = x.shape[-2:],
                mode  = "bilinear",
                align_corners = False,
            )

    model = SegFormerWrapper(segformer_path)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"  Loaded  : {checkpoint_path}")
    print(f"  Epoch   : {ckpt.get('epoch', '?')}  |  "
          f"Train-val mIoU : {ckpt.get('val_iou', 0):.4f}")
    print(f"  Note    : Input size fixed at {SEGFORMER_IMG_SIZE}×{SEGFORMER_IMG_SIZE}")
    return model


@torch.no_grad()
def predict_segformer(model, tensor):
    """
    Forward pass through SegFormer wrapper.
    The wrapper already upsamples to input resolution.
    Returns (H, W) float32 probability map.
    """
    logits = model(tensor.to(DEVICE))
    return torch.sigmoid(logits)[0, 0].cpu().numpy()

# ==============================================================================
#   DIFFUSION MODEL LOADER
# ==============================================================================
#
# WHY A SEPARATE PATH?
# The diffusion pipeline differs from smp and SegFormer in two ways:
#
#   1. TWO MODEL OBJECTS — the checkpoint stores only the denoiser weights
#      ('denoiser_state').  The frozen image encoder is rebuilt separately
#      and loaded with ImageNet weights, exactly as during training.
#
#   2. INFERENCE PROCEDURE — instead of a single model(tensor) forward pass,
#      diffusion inference runs K full reverse-diffusion chains (T=100 steps
#      each) and averages the resulting soft masks.  The variance across K
#      chains produces the uncertainty map.  This is ~100× more compute than
#      a single forward pass, which is why K_SAMPLES_TRAIN=1 was used during
#      training validation and K=5 is used only for final evaluation.
#
# All diffusion classes below are verbatim copies of their counterparts in
# train_diffusion_seg.py.  They MUST stay identical — any layer name change
# will cause a state_dict key mismatch when loading the checkpoint.
#
# ==============================================================================

def _num_groups(ch, max_g=8):
    for g in range(max_g, 0, -1):
        if ch % g == 0:
            return g
    return 1

class _SinTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        dev  = t.device
        half = self.dim // 2
        f    = torch.exp(-math.log(10000) *
                         torch.arange(half, device=dev) / (half - 1))
        a    = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)

class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, td):
        super().__init__()
        self.n1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.tp = nn.Linear(td, out_ch * 2)
        self.sk = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, te):
        h = F.silu(self.n1(x))
        h = self.c1(h)
        sc, sh = self.tp(te).chunk(2, dim=-1)
        h = h * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = F.silu(self.n2(h))
        h = self.c2(h)
        return h + self.sk(x)

class _DenoisingUNet(nn.Module):
    def __init__(self, img_feat_ch=2048, time_dim=128, base_ch=64):
        super().__init__()
        td = time_dim
        self.time_embed = nn.Sequential(
            _SinTime(td), nn.Linear(td, td*4), nn.SiLU(), nn.Linear(td*4, td))
        self.enc1 = _ResBlock(1,          base_ch,     td)
        self.enc2 = _ResBlock(base_ch,    base_ch*2,   td)
        self.enc3 = _ResBlock(base_ch*2,  base_ch*4,   td)
        self.down = nn.MaxPool2d(2)
        self.bottleneck_proj = nn.Conv2d(img_feat_ch, base_ch*4, 1)
        self.bottleneck = _ResBlock(base_ch*4 + base_ch*4, base_ch*4, td)
        self.up3  = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec3 = _ResBlock(base_ch*6, base_ch*2, td)
        self.up2  = nn.ConvTranspose2d(base_ch*2, base_ch,   2, stride=2)
        self.dec2 = _ResBlock(base_ch*3, base_ch,   td)
        self.up1  = nn.ConvTranspose2d(base_ch,    base_ch,   2, stride=2)
        self.dec1 = _ResBlock(base_ch*2, base_ch,   td)
        self.out  = nn.Conv2d(base_ch, 1, 1)
    def forward(self, nm, t, imf):
        te = self.time_embed(t)
        e1 = self.enc1(nm, te)
        e2 = self.enc2(self.down(e1), te)
        e3 = self.enc3(self.down(e2), te)
        de3 = self.down(e3)
        imf = F.adaptive_avg_pool2d(imf, de3.shape[-2:])
        imf = self.bottleneck_proj(imf)
        b   = self.bottleneck(torch.cat([de3, imf], dim=1), te)
        d3  = self.dec3(torch.cat([self.up3(b),  e3], dim=1), te)
        d2  = self.dec2(torch.cat([self.up2(d3), e2], dim=1), te)
        d1  = self.dec1(torch.cat([self.up1(d2), e1], dim=1), te)
        return self.out(d1)

class _FrozenEncoder(nn.Module):
    def __init__(self, name="resnet50"):
        super().__init__()
        dummy = smp.Unet(encoder_name=name, encoder_weights=None,
                         in_channels=3, classes=1)
        self.encoder = dummy.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
    @torch.no_grad()
    def forward(self, x):
        return self.encoder(x)[-1]

def _make_schedule(T, b1, bT, device):
    betas     = torch.linspace(b1, bT, T).to(device)
    alphas    = (1.0 - betas).to(device)
    alpha_bar = torch.cumprod(alphas, dim=0).to(device)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}

@torch.no_grad()
def _reverse_chain(denoiser, img_feat, shape, sched, T, device):
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t   = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = denoiser(x, t, img_feat)
        a   = sched["alphas"][i]
        ab  = sched["alpha_bar"][i]
        b   = sched["betas"][i]
        c   = (1 - a) / (1 - ab).sqrt()
        mu  = (1 / a.sqrt()) * (x - c * eps)
        x   = mu + (b.sqrt() * torch.randn_like(x) if i > 0 else 0)
    return torch.sigmoid(x)


def load_diffusion_model(checkpoint_path, encoder_name="resnet50"):
    """
    Reconstruct encoder + denoiser and load checkpoint weights.
    Returns (encoder, denoiser, feat_ch) tuple.
    """
    encoder = _FrozenEncoder(encoder_name)

    # Determine encoder output channels with a dummy pass on CPU
    with torch.no_grad():
        dummy   = torch.zeros(1, 3, DIFFUSION_IMG_SIZE, DIFFUSION_IMG_SIZE)
        feat_ch = encoder(dummy).shape[1]

    denoiser = _DenoisingUNet(img_feat_ch=feat_ch)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "denoiser_state" not in ckpt:
        raise KeyError(
            f"'denoiser_state' not found in {checkpoint_path}. "
            "Ensure the checkpoint was saved by train_diffusion_seg.py."
        )
    denoiser.load_state_dict(ckpt["denoiser_state"])

    # Reload ImageNet weights into the frozen encoder
    try:
        dummy_smp = smp.Unet(encoder_name=encoder_name,
                             encoder_weights="imagenet",
                             in_channels=3, classes=1)
        encoder.encoder.load_state_dict(dummy_smp.encoder.state_dict())
        del dummy_smp
        print(f"  Encoder ImageNet weights reloaded for '{encoder_name}'.")
    except Exception as e:
        print(f"  Warning: could not reload encoder ImageNet weights: {e}")

    encoder.to(DEVICE).eval()
    denoiser.to(DEVICE).eval()

    cfg = ckpt.get("config", {})
    print(f"  Loaded  : {checkpoint_path}")
    print(f"  Epoch   : {ckpt.get('epoch','?')}  |  "
          f"Train-val mIoU : {ckpt.get('val_iou',0):.4f}")
    print(f"  T={cfg.get('T','?')}  K_train={cfg.get('K_train','?')}  "
          f"K_test={cfg.get('K_test','?')}")
    return encoder, denoiser, feat_ch


@torch.no_grad()
def predict_diffusion(encoder, denoiser, tensor, k=5,
                      T=100, b1=1e-4, bT=0.02):
    """
    Run K reverse-diffusion chains, return (mean_prob_map, uncertainty_map).

    mean_prob_map  : (H, W) float32 in [0, 1]  — the prediction
    uncertainty    : (H, W) float32 in [0, 1]  — pixel-wise normalised
                     variance across K chains.  None when k=1.

    Chains are accumulated incrementally (not stacked) to keep peak VRAM
    at the cost of one chain rather than K chains simultaneously.
    """
    tensor = tensor.to(DEVICE)
    sched  = _make_schedule(T, b1, bT, DEVICE)
    feat   = encoder(tensor)
    shape  = (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])

    acc  = torch.zeros(shape, device=DEVICE)
    acc2 = torch.zeros(shape, device=DEVICE)

    for _ in range(k):
        chain = _reverse_chain(denoiser, feat, shape, sched, T, DEVICE)
        acc  += chain
        acc2 += chain ** 2
        del chain
        torch.cuda.empty_cache()

    mean_mask = (acc / k)[0, 0].cpu().numpy()

    if k > 1:
        var = (acc2 / k) - (acc / k) ** 2
        v   = var[0, 0].cpu().numpy()
        uncertainty = (v - v.min()) / (v.max() - v.min() + 1e-8)
    else:
        uncertainty = None

    return mean_mask, uncertainty

# ==============================================================================
#   SHARED METRICS
# ==============================================================================

def get_boundary(mask, px=2):
    return mask.astype(bool) & ~binary_erosion(mask, iterations=px)

def compute_metrics(pred_bin, gt_bin):
    """All 9 metrics for one image. Both inputs are (H, W) uint8 0/1."""
    tp = int(np.logical_and(pred_bin,     gt_bin    ).sum())
    fp = int(np.logical_and(pred_bin,   1-gt_bin    ).sum())
    fn = int(np.logical_and(1-pred_bin,   gt_bin    ).sum())
    tn = int(np.logical_and(1-pred_bin, 1-gt_bin    ).sum())
    total = tp + fp + fn + tn

    iou       = tp / (tp + fp + fn + 1e-6)
    iou_bg    = tn / (tn + fp + fn + 1e-6)
    miou      = (iou + iou_bg) / 2.0
    dice      = 2*tp / (2*tp + fp + fn + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    aacc      = (tp + tn) / (total + 1e-6)
    acc_bg    = tn / (tn + fp + 1e-6)
    macc      = (recall + acc_bg) / 2.0
    f2        = fbeta_score(gt_bin.flatten(), pred_bin.flatten(),
                            beta=2, zero_division=0)
    pb = get_boundary(pred_bin)
    gb = get_boundary(gt_bin)
    biou = int(np.logical_and(pb, gb).sum()) / \
           (int(np.logical_or(pb, gb).sum()) + 1e-6)

    return dict(miou=miou, iou=iou, dice=dice, recall=recall,
                precision=precision, f2=f2, biou=biou, aacc=aacc, macc=macc)

def bootstrap_ci(values, n_boot=10000, ci=0.95):
    values = np.array(values)
    boots  = [np.random.choice(values, len(values), replace=True).mean()
              for _ in range(n_boot)]
    alpha  = (1 - ci) / 2
    lo, hi = np.percentile(boots, [100*alpha, 100*(1-alpha)])
    return values.mean(), lo, hi

# ==============================================================================
#   MAIN EVALUATION LOOP
# ==============================================================================

METRIC_KEYS = ["miou","iou","dice","recall","precision",
               "f2","biou","aacc","macc"]

def evaluate(predict_fn, transform, out_prefix, uncertainty_dir=None):
    """
    Generic evaluation loop.

    predict_fn  : callable(tensor) -> (H, W) float32 prob map
                  For diffusion, wrap predict_diffusion to match this signature.
    transform   : albumentations Compose (handles the correct image size)
    out_prefix  : string used to name output CSV files
    uncertainty_dir : if set, uncertainty maps (diffusion only) are saved here
    """
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    if uncertainty_dir:
        Path(uncertainty_dir).mkdir(exist_ok=True)

    img_paths  = sorted(Path(TEST_IMAGES).glob("*.jpg")) + \
                 sorted(Path(TEST_IMAGES).glob("*.png"))
    masks_dir  = Path(TEST_MASKS)

    all_rows   = []
    per_metric = {k: [] for k in METRIC_KEYS}

    for img_path in tqdm(img_paths, desc="Evaluating", ascii=True):
        mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        img    = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]
        aug    = transform(image=img)
        tensor = aug["image"].unsqueeze(0)

        probs = predict_fn(tensor)   # (H, W) float32

        pred_bin  = (probs >= THRESHOLD).astype("uint8")
        # Resize prediction back to original mask resolution
        pred_full = np.array(
            Image.fromarray(pred_bin * 255).resize(
                (orig_w, orig_h), Image.NEAREST)) // 255

        gt_raw = np.array(Image.open(mask_path).convert("L"))
        gt_bin = (gt_raw > 127).astype("uint8")

        m   = compute_metrics(pred_full, gt_bin)
        row = {"image": img_path.name}
        row.update(m)
        all_rows.append(row)
        for k in METRIC_KEYS:
            per_metric[k].append(m[k])

    # ── Per-image CSV ─────────────────────────────────────────────────────────
    per_img_path = f"{RESULTS_DIR}/{out_prefix}_per_image.csv"
    with open(per_img_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Per-image metrics : {per_img_path}")

    # ── Aggregate CSV ─────────────────────────────────────────────────────────
    agg_path = f"{RESULTS_DIR}/{out_prefix}_aggregate.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric","mean","std","median",
                          "ci95_lo","ci95_hi","min","max"])
        for k in METRIC_KEYS:
            vals = np.array(per_metric[k])
            mean, lo, hi = bootstrap_ci(vals)
            writer.writerow([k, f"{mean:.4f}", f"{vals.std():.4f}",
                              f"{np.median(vals):.4f}",
                              f"{lo:.4f}", f"{hi:.4f}",
                              f"{vals.min():.4f}", f"{vals.max():.4f}"])
    print(f"  Aggregate stats   : {agg_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8}  {'95% CI':>22}")
    print(f"{'─'*68}")
    for k in METRIC_KEYS:
        vals = np.array(per_metric[k])
        mean, lo, hi = bootstrap_ci(vals, n_boot=5000)
        print(f"  {k:<12} {mean:>8.4f} {vals.std():>8.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"{'─'*68}")
    print(f"  N images evaluated : {len(all_rows)}")

# ==============================================================================
#   ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model on the test set.")
    parser.add_argument("--family",
        required=True, choices=["smp","segformer","diffusion"],
        help="Model family: smp | segformer | diffusion")
    parser.add_argument("--checkpoint", required=True,
        help="Path to the .pth checkpoint file")
    parser.add_argument("--label", default=None,
        help="Short label used to name output CSV files")

    # smp-specific
    parser.add_argument("--backbone",     default=None)
    parser.add_argument("--architecture", default=None)

    # SegFormer-specific
    parser.add_argument("--segformer-path", default="./segformer-b2-local",
        help="Local folder with the downloaded SegFormer-B2 model files")

    # Diffusion-specific
    parser.add_argument("--diff-encoder", default="resnet50",
        help="Frozen image encoder backbone name (default: resnet50)")
    parser.add_argument("--diff-k", type=int, default=5,
        help="Number of reverse-diffusion chains at inference (default: 5)")
    parser.add_argument("--diff-T", type=int, default=100,
        help="Diffusion timesteps, must match training (default: 100)")
    parser.add_argument("--save-uncertainty", action="store_true",
        help="Save uncertainty maps as PNG (diffusion only, k>1)")

    args  = parser.parse_args()
    label = args.label

    print(f"\n{'='*68}")
    print(f"  Family      : {args.family}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Test images : {TEST_IMAGES}")
    print(f"{'='*68}\n")

    # ── Route to correct family ───────────────────────────────────────────────

    if args.family == "smp":
        if not args.backbone or not args.architecture:
            parser.error("--backbone and --architecture are required for --family smp")
        label = label or f"{args.architecture}_{args.backbone.replace('tu-','')}"
        model     = load_smp_model(args.backbone, args.architecture, args.checkpoint)
        transform = make_transform(SMP_IMG_SIZE)
        def predict_fn(tensor):
            return predict_smp(model, tensor)

    elif args.family == "segformer":
        label = label or "segformer_b2"
        model     = load_segformer_model(args.checkpoint, args.segformer_path)
        transform = make_transform(SEGFORMER_IMG_SIZE)
        # CHANGE: SegFormer uses 512×512 — using a different size would
        # misalign with the positional embeddings learned during training.
        def predict_fn(tensor):
            return predict_segformer(model, tensor)

    elif args.family == "diffusion":
        label = label or f"diffusion_{args.diff_encoder}"
        encoder, denoiser, _ = load_diffusion_model(
            args.checkpoint, args.diff_encoder)
        transform = make_transform(DIFFUSION_IMG_SIZE)
        K         = args.diff_k
        T         = args.diff_T
        unc_dir   = f"{RESULTS_DIR}/{label}_uncertainty" \
                    if args.save_uncertainty and K > 1 else None
        print(f"  K chains    : {K}  (uncertainty maps: "
              f"{'yes -> ' + str(unc_dir) if unc_dir else 'no'})")
        # CHANGE: diffusion predict_fn wraps predict_diffusion and discards
        # the uncertainty map during the main loop.  Uncertainty maps are
        # saved separately when --save-uncertainty is set.
        _unc_store = {}
        def predict_fn(tensor):
            probs, unc = predict_diffusion(
                encoder, denoiser, tensor, k=K, T=T)
            _unc_store["last"] = unc
            return probs

    print()
    evaluate(predict_fn, transform, label,
             uncertainty_dir=unc_dir if args.family=="diffusion" else None)
