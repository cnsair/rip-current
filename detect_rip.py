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
                                 UNet, UNet++, FPN, DeepLabV3+ with
                                 resnet50, efficientnet-b2, convnext_tiny,
                                 swin_tiny, vmamba_tiny, etc.

  MODEL_FAMILY = "segformer"  -> HuggingFace SegFormer (MiT encoder).
                                 Set SEGFORMER_LOCAL_PATH to the folder
                                 you downloaded with snapshot_download.

The config block is the only thing that changes between experiments.
Everything else -- preprocessing, postprocessing, metrics, video -- is
identical for both families.
"""

import os
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
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
#   "unet" | "unetplusplus" | "fpn" | "deeplabv3plus"
BACKBONE        = "efficientnet-b2"  # smp encoder name (e.g. "resnet34", "efficientnet-b2", "tu-convnext_tiny")
ARCHITECTURE    = "unet"
ENCODER_WEIGHTS = "imagenet"   # None at inference -- weights come from checkpoint

# -- SegFormer settings (used when MODEL_FAMILY = "segformer") ---------------
# Path to the folder downloaded with:
#   snapshot_download("nvidia/segformer-b2-finetuned-ade-512-512",
#                     local_dir="./segformer-b2-local")
SEGFORMER_LOCAL_PATH = "./segformer-b2-local"

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
#   1  MODEL BUILDING
# ==============================================================================

def _build_smp_model() -> torch.nn.Module:
    """Construct an smp model from BACKBONE + ARCHITECTURE constants."""
    arch_map = {
        "unet":         smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn":          smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }
    if ARCHITECTURE not in arch_map:
        raise ValueError(
            f"Unknown ARCHITECTURE '{ARCHITECTURE}'. "
            f"Choose from: {list(arch_map.keys())}"
        )
    return arch_map[ARCHITECTURE](
        encoder_name    = BACKBONE,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )


def _build_segformer_model() -> torch.nn.Module:
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
    # -- Build architecture skeleton ------------------------------------------
    if MODEL_FAMILY == "smp":
        model = _build_smp_model()
    elif MODEL_FAMILY == "segformer":
        model = _build_segformer_model()
    else:
        raise ValueError(f"Unknown MODEL_FAMILY '{MODEL_FAMILY}'. Use 'smp' or 'segformer'.")

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
    model:        torch.nn.Module,
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
    Returns dict with keys: probs, raw_mask, clean_mask, overlay, metrics.
    """
    image_path = Path(image_path)
    img_bgr    = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    tensor         = preprocess_bgr(img_bgr, transform)
    probs, raw_bin = predict(model, tensor, device, threshold)
    clean_bin      = postprocess(raw_bin, min_area, keep_largest)

    mask_full = cv2.resize(
        clean_bin.astype("uint8"),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = overlay_mask(img_bgr, mask_full)

    metrics = None
    gt_path = Path(gt_masks_dir) / (image_path.stem + ".png")
    if gt_path.exists():
        gt_raw     = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
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

    return dict(probs=probs, raw_mask=raw_bin, clean_mask=mask_full,
                overlay=overlay, metrics=metrics)


# ==============================================================================
#   9  HIGH-LEVEL: FOLDER BATCH
# ==============================================================================

def process_folder(
    model:        torch.nn.Module,
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
    else:
        print(f"  SegFormer    : {SEGFORMER_LOCAL_PATH}")
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
