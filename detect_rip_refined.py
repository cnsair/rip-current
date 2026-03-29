"""
Inference + postprocessing + metrics for rip current segmentation.

Supports:
  - Single image inference
  - Folder batch inference
  - Live video overlay (press q to quit)
  - Video heatmap (temporal accumulation)

"""

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT   = "best_unet_three.pth"   # path to your saved .pth checkpoint
IMG_SIZE     = 256                   # must match the size used during training
DEVICE       = "cpu"                 # "cuda" if you have a GPU
THRESHOLD    = 0.5                   # sigmoid threshold for binary prediction
MIN_AREA     = 500                   # minimum rip blob area in pixels (removes noise)

# Paths for the example runs at the bottom
SINGLE_IMAGE = "val/sampled_images/images/RipVIS-046_00048.jpg"
VAL_MASKS_DIR = "data/val/masks"     # used to compute metrics when GT exists
VIDEO_IN     = "test/videos/RipVIS-025.mp4"
VIDEO_OUT    = "out_infer/RipVIS-025_overlay.mp4"
OUT_DIR      = "out_infer"


# ══════════════════════════════════════════════════════════════════════════════
# 1  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load a U-Net from a checkpoint file.

    Handles two checkpoint formats automatically:

    Format A — produced by the IMPROVED train_segmentation.py:
        torch.save({
            "epoch":           N,
            "model_state":     model.state_dict(),   ← weights HERE
            "optimizer_state": ...,
            "val_iou":         ...,
            "config":          {...},
        }, path)

    Format B — produced by the ORIGINAL train script:
        torch.save(model.state_dict(), path)          ← bare state dict

    The original detect_rip.py only handled Format B, which caused the
    RuntimeError: "Unexpected key(s) in state_dict: epoch, model_state ..."
    when loading a Format A checkpoint.
    """
    # Build model architecture — must match training exactly.
    # encoder_weights=None avoids an unnecessary ImageNet download at inference time;
    # the checkpoint overwrites these weights anyway.
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    model.to(device)

    # Load the raw checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ── Detect and extract the actual state dict ──────────────────────────
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            # Format A: improved train_segmentation.py
            state_dict = checkpoint["model_state"]
            print(f"      Checkpoint info:")
            print(f"      Saved at epoch : {checkpoint.get('epoch', 'unknown')}")
            print(f"      Best val IoU   : {checkpoint.get('val_iou', 'unknown'):.4f}")
            print(f"      Encoder        : {checkpoint.get('config', {}).get('encoder', 'unknown')}")
        elif "state_dict" in checkpoint:
            # Alternative common pattern used by some training frameworks
            state_dict = checkpoint["state_dict"]
        else:
            # Format B: bare state dict (original training script)
            state_dict = checkpoint
    else:
        # Checkpoint is already a raw OrderedDict (very old PyTorch saves)
        state_dict = checkpoint

    # ── Handle DataParallel prefix ────────────────────────────────────────
    # Models saved with DataParallel have keys prefixed with "module."
    # Strip that prefix so single-GPU/CPU loading works.
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded from {checkpoint_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2  PREPROCESSING  —  must match training transforms exactly
# ══════════════════════════════════════════════════════════════════════════════

def make_transform(size: int = IMG_SIZE) -> Compose:
    """
    Inference transform: resize + ImageNet normalise + to tensor.
    Must be identical to the validation transform used in training.
    """
    return Compose([
        Resize(size, size),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def preprocess_bgr(img_bgr: np.ndarray, transform: Compose) -> torch.Tensor:
    """
    Convert a BGR numpy image (from cv2) into a normalised float tensor.
    Returns shape (1, 3, H, W) — the batch dimension is added here.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(image=img_rgb)["image"]
    return tensor.unsqueeze(0)   # add batch dim → (1, 3, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# 3  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    model:     torch.nn.Module,
    tensor:    torch.Tensor,
    device:    str   = DEVICE,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a forward pass and return:
      prob_map  — float32 array (H, W) with sigmoid probabilities in [0, 1]
      bin_mask  — uint8   array (H, W) with values {0, 1}
    """
    with torch.no_grad():
        logits = model(tensor.to(device))            # (1, 1, H, W)
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()   # (H, W)
    return probs, (probs >= threshold).astype("uint8")


# ══════════════════════════════════════════════════════════════════════════════
# 4  POSTPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def postprocess(
    binary_mask:  np.ndarray,
    min_area:     int  = MIN_AREA,
    keep_largest: bool = False,
) -> np.ndarray:
    """
    Clean the raw binary prediction mask:
      1. Morphological open  — removes isolated speckle noise
      2. Morphological close — fills small holes inside rip regions
      3. Remove blobs smaller than min_area pixels
      4. Optionally keep only the single largest blob

    Parameters
    ----------
    binary_mask  : uint8 (H, W), values {0, 1}
    min_area     : blobs with fewer pixels than this are discarded
    keep_largest : if True, discard all but the largest blob

    Returns
    -------
    uint8 (H, W), values {0, 1}
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


# ══════════════════════════════════════════════════════════════════════════════
# 5  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute binary segmentation metrics given prediction and ground-truth masks.

    Both inputs must be binary numpy arrays (values 0 or 1), same shape.

    Returns
    -------
    dict with keys: IoU, Dice, Precision, Recall
    
    Recall is the most safety-critical metric for rip detection:
    a missed rip current (false negative) is more dangerous than a false alarm.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = np.logical_and(pred,  gt ).sum()
    fp = np.logical_and(pred,  ~gt).sum()
    fn = np.logical_and(~pred, gt ).sum()

    iou       = tp / (tp + fp + fn + 1e-6)
    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    return dict(IoU=iou, Dice=dice, Precision=precision, Recall=recall)


def print_metrics(metrics: dict, image_name: str = "") -> None:
    """Pretty-print a metrics dict."""
    label = f"  [{image_name}]" if image_name else ""
    print(
        f"{label}  "
        f"IoU={metrics['IoU']:.4f}  "
        f"Dice={metrics['Dice']:.4f}  "
        f"Precision={metrics['Precision']:.4f}  "
        f"Recall={metrics['Recall']:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 0, 220), alpha: float = 0.4) -> np.ndarray:
    """
    Draw a semi-transparent coloured overlay where mask == 1, plus a white contour.

    Parameters
    ----------
    img_bgr : original BGR image, shape (H, W, 3)
    mask    : binary mask,        shape (H, W), values {0, 1}
    color   : BGR colour for the overlay  (default: red)
    alpha   : opacity of the overlay (0 = invisible, 1 = opaque)
    """
    overlay      = img_bgr.copy()
    colour_layer = np.zeros_like(img_bgr)
    colour_layer[mask == 1] = color
    cv2.addWeighted(colour_layer, alpha, overlay, 1 - alpha, 0, overlay)

    contours, _ = cv2.findContours(
        (mask * 255).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# 7  HIGH-LEVEL: SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

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

    Steps:
      1. Load image
      2. Preprocess (resize + normalise)
      3. Predict (forward pass)
      4. Postprocess (denoise mask)
      5. Resize mask back to original image dimensions
      6. Compute metrics if a GT mask exists in gt_masks_dir
      7. Save outputs if out_dir is provided

    Returns dict with keys: probs, raw_mask, clean_mask, overlay, metrics (or None)
    """
    image_path = Path(image_path)
    img_bgr    = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Predict ───────────────────────────────────────────────────────────
    tensor             = preprocess_bgr(img_bgr, transform)
    probs, raw_bin     = predict(model, tensor, device, threshold)
    clean_bin          = postprocess(raw_bin, min_area, keep_largest)

    # Resize mask from model resolution back to original image size
    mask_full = cv2.resize(
        clean_bin.astype("uint8"),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = overlay_mask(img_bgr, mask_full)

    # ── Metrics (if GT available) ─────────────────────────────────────────
    metrics = None
    gt_path = Path(gt_masks_dir) / (image_path.stem + ".png")
    if gt_path.exists():
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(
            gt_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        gt_bin  = (gt_resized > 127).astype("uint8")
        metrics = compute_metrics(mask_full, gt_bin)
        print_metrics(metrics, image_path.name)
    else:
        print(f"  [{image_path.name}]  No GT mask found — metrics skipped.")

    # ── Save outputs ──────────────────────────────────────────────────────
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / (image_path.stem + "_mask.png")),
                    (mask_full * 255).astype("uint8"))
        cv2.imwrite(str(out_dir / (image_path.stem + "_overlay.png")), overlay)

    return dict(probs=probs, raw_mask=raw_bin, clean_mask=mask_full,
                overlay=overlay, metrics=metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 8  HIGH-LEVEL: FOLDER BATCH
# ══════════════════════════════════════════════════════════════════════════════

def process_folder(
    model:       torch.nn.Module,
    images_dir:  str,
    transform:   Compose,
    out_dir:     str,
    device:      str   = DEVICE,
    threshold:   float = THRESHOLD,
    min_area:    int   = MIN_AREA,
    gt_masks_dir: str  = VAL_MASKS_DIR,
) -> None:
    """
    Run inference on every image in images_dir.
    Prints per-image metrics when GT masks are available, and a summary at the end.
    """
    images = sorted(Path(images_dir).glob("*.jpg")) + \
             sorted(Path(images_dir).glob("*.png"))
    print(f"\n🖼️   Processing {len(images)} images from {images_dir} ...")

    all_metrics = []
    for img_path in images:
        result = process_image(
            model, img_path, transform,
            device=device, threshold=threshold, min_area=min_area,
            out_dir=out_dir, gt_masks_dir=gt_masks_dir,
        )
        if result["metrics"]:
            all_metrics.append(result["metrics"])

    # ── Aggregate summary ─────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'─'*60}")
        print(f"📊  Summary over {len(all_metrics)} images with GT:")
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        print_metrics(avg, "AVERAGE")
        print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 9  VIDEO: LIVE OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

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
    Play a video with per-frame rip overlay.
    Press 'q' to quit. Optionally saves the overlaid video to out_path.

    stride: run inference every N frames (stride=2 halves CPU usage).
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


# ══════════════════════════════════════════════════════════════════════════════
# 10  VIDEO: TEMPORAL HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

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
    Accumulate per-frame probability maps across the whole video into a heatmap.
    Regions that are frequently predicted as rip appear hot (red) in the output.
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
            tensor = preprocess_bgr(frame, transform)
            probs, _ = predict(model, tensor, device, threshold)
            probs_full = cv2.resize(probs, (frame.shape[1], frame.shape[0]))
            if heatmap is None:
                heatmap = np.zeros_like(probs_full, dtype=np.float32)
            heatmap += probs_full
        frame_idx += 1
    cap.release()

    if heatmap is None:
        raise RuntimeError("No frames were processed.")

    # Normalise to 0–255 and apply colour map
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    vis  = cv2.applyColorMap((norm * 255).astype("uint8"), cv2.COLORMAP_JET)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"  Heatmap saved → {out_path}")
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"🖥️   Device : {DEVICE}")
    print(f"Loading : {CHECKPOINT}\n")

    model     = load_model(CHECKPOINT, device=DEVICE)
    transform = make_transform(size=IMG_SIZE)

    # ── 1) Single image with metrics ──────────────────────────────────────
    # if Path(SINGLE_IMAGE).exists():
    #     print(f"\n── Single image ──────────────────────────────────────────")
    #     process_image(
    #         model, SINGLE_IMAGE, transform,
    #         device=DEVICE, threshold=THRESHOLD, min_area=MIN_AREA,
    #         out_dir=OUT_DIR, gt_masks_dir=VAL_MASKS_DIR,
    #     )
    #     print(f"Outputs saved to {OUT_DIR}/")
    # else:
    #     print(f"Image not found: {SINGLE_IMAGE}")

    # ── 2) Batch folder with aggregate metrics ────────────────────────────
    # Uncomment to run on all val images:
    # process_folder(
    #     model, "data/val/images", transform,
    #     out_dir=OUT_DIR, device=DEVICE,
    #     gt_masks_dir=VAL_MASKS_DIR,
    # )

    # ── 3) Live video overlay ─────────────────────────────────────────────
    if Path(VIDEO_IN).exists():
        print(f"\n── Video overlay ─────────────────────────────────────────")
        print(f"   Press 'q' to stop.")
        process_video_overlay(
            model, VIDEO_IN, transform,
            out_path=VIDEO_OUT, device=DEVICE,
            threshold=THRESHOLD, min_area=MIN_AREA,
            stride=2,   # process every 2nd frame to reduce CPU load
        )
    else:
        print(f"Video not found: {VIDEO_IN}")

    # ── 4) Temporal heatmap ───────────────────────────────────────────────
    # Uncomment to generate a heatmap from the video:
    # if Path(VIDEO_IN).exists():
    #     process_video_heatmap(
    #         model, VIDEO_IN, transform,
    #         out_path="out_infer/heatmap.png",
    #         device=DEVICE, stride=2,
    #     )
