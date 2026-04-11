# Inference + simple postprocessing for rip current segmentation using your saved U-Net.
# Usage examples at bottom.

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# -------------------------
# 1) MODEL LOADING
# -------------------------
def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load the U-Net model architecture and weights.

    Why:
    - Must instantiate exactly the same architecture used in training (encoder name, in_channels, classes).
    - Use encoder_weights=None to avoid trying to download imagenet weights at inference time;
      we will overwrite weights from the checkpoint anyway.
    """
    model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, in_channels=3, classes=1)
    model.to(device)
    # load state dict (assumes checkpoint contains state_dict)
    state = torch.load(checkpoint_path, map_location=device)
    # If you saved the 'model.state_dict()', state will be a dict; otherwise handle full checkpoints.
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        # handle DataParallel saved models
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # fallback: if checkpoint is a dict with 'state_dict' key (common pattern)
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            raise
    model.eval()
    return model

# -------------------------
# 2) PREPROCESSING (same as training)
# -------------------------
def make_transform(size=256):
    """
    Build the albumentations transform used for inference.

    Why:
    - Must match the preprocessing used for training (resize + Normalize + ToTensor).
    - Using the same normalization ensures the model input distribution matches training.
    """
    return Compose([
        Resize(size, size),
        Normalize(), # albumentations defaults to ImageNet mean/std if no args (consistent with training)
        ToTensorV2(),
    ])

def preprocess_image_bgr(img_bgr: np.ndarray, transform):
    """
    Convert BGR (cv2) image -> transformed torch tensor batch.
    Returns: tensor shaped [1,3,H,W], dtype float32 on CPU
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)
    tensor = augmented['image'].unsqueeze(0)  # add batch dim
    return tensor  # torch.Tensor

# -------------------------
# 3) PREDICTION & THRESHOLDING
# -------------------------
def predict_mask(model, tensor, device="cpu", threshold=0.5):
    """
    Run forward pass and return binary mask + probability map.

    Returns:
      prob_map: numpy array HxW of [0..1] floats
      bin_mask: numpy array HxW bool (True = rip)
    """
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)  # shape [B,1,H,W]
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # take first batch, channel 0
    bin_mask = (probs >= threshold).astype('uint8')
    return probs, bin_mask

# -------------------------
# 4) POSTPROCESSING (cleaning masks)
# -------------------------
def postprocess_mask(binary_mask, min_area=500, keep_largest=False):
    """
    Clean predicted binary mask:
      - remove small connected components (noise)
      - optionally keep only largest connected component

    Why:
    - Predictions often have speckle noise; morphological cleaning reduces false positives.
    - min_area should be tuned to expected rip size in pixels (after resizing).
    """
    # ensure binary 0/1 uint8
    m = (binary_mask > 0).astype('uint8') * 255

    # morphological open + close to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # connected components -> remove small areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    areas = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lbl] = 255
            areas.append((lbl, area))
    # optionally keep only the largest (if you expect one rip)
    if keep_largest and len(areas) > 0:
        largest_label = max(areas, key=lambda x: x[1])[0]
        out = (labels == largest_label).astype('uint8') * 255

    return (out > 0).astype('uint8')  # return 0/1 mask

# -------------------------
# 5) VISUALIZATION / OVERLAY
# -------------------------
def overlay_mask_on_image(img_bgr, mask, color=(0,0,255), alpha=0.4):
    """
    Draw semi-transparent mask overlay on BGR image.

    mask: binary 0/1 numpy array same HxW as img.
    color: BGR tuple for mask color (red default).
    """
    overlay = img_bgr.copy()
    colored_mask = np.zeros_like(img_bgr, dtype=np.uint8)
    colored_mask[mask == 1] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    # also draw mask contour for clarity
    contours, _ = cv2.findContours((mask*255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255,255,255), 1)  # white contour
    return overlay

# -------------------------
# 6) METRICS (optional, if GT exists)
# -------------------------
def compute_metrics(pred_mask, gt_mask):
    """
    Compute IoU, Dice, Precision, Recall.
    All inputs are binary numpy arrays 0/1 (same HxW).
   
    pred_mask : predicted binary mask (0 or 1)
    gt_mask   : ground truth binary mask (0 or 1)
    """

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()

    # IoU
    iou = TP / (TP + FP + FN + 1e-6)

    # Dice
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    # Precision
    precision = TP / (TP + FP + 1e-6)

    # Recall
    recall = TP / (TP + FN + 1e-6)

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall
    }
    
    # pred = pred_mask.astype(bool)
    # gt = gt_mask.astype(bool)
    # intersection = np.logical_and(pred, gt).sum()
    # union = np.logical_or(pred, gt).sum()
    # iou = intersection / (union + 1e-8)
    # dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    # tp = intersection
    # fp = np.logical_and(pred, ~gt).sum()
    # fn = np.logical_and(~pred, gt).sum()
    # precision = tp / (tp + fp + 1e-8)
    # recall = tp / (tp + fn + 1e-8)
    # return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}

# -------------------------
# 7) HIGH-LEVEL FUNCTIONS
# -------------------------
def process_image_file(model, image_path, out_dir=None, transform=None, device='cpu',
                       threshold=0.5, min_area=500, keep_largest=False, save_overlay=True):
    """
    End-to-end processing for a single image file.
    Returns: (prob_map, raw_bin, cleaned_bin, overlay_image (BGR numpy) or None)
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read {image_path}")
    tensor = preprocess_image_bgr(img_bgr, transform)
    # run model prediction
    probs, raw_bin = predict_mask(model, tensor, device=device, threshold=threshold)
    # clean prediction
    clean_bin = postprocess_mask(raw_bin, min_area=min_area, keep_largest=keep_largest)
    
    ### overlay = overlay_mask_on_image(img_bgr, clean_bin) if save_overlay else None
    ### IMPORTANT: resize mask to match original image
    mask_resized = cv2.resize(
        clean_bin.astype("uint8"),
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Overlay rip detection
    overlay = overlay_mask_on_image(img_bgr, mask_resized) if save_overlay else None
    
    # -----------------------------
    # COMPUTE METRICS IF GT EXISTS
    # -----------------------------
    gt_mask_path = Path("data/val/masks") / (Path(image_path).stem + ".png")
    if gt_mask_path.exists():
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(
            gt_mask,
            (mask_resized.shape[1], mask_resized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        gt_mask = (gt_mask > 127).astype("uint8")
        metrics = compute_metrics(mask_resized, gt_mask)
        # print(f"Metrics for {image_path.name}: {metrics}")
        print(f"Metrics for {Path(image_path).name}: {metrics}")

    # optional save outputs
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # save mask as png
        mask_path = out_dir / (Path(image_path).stem + "_mask.png")
        cv2.imwrite(str(mask_path), (clean_bin*255).astype('uint8'))
        if overlay is not None:
            overlay_path = out_dir / (Path(image_path).stem + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlay)
    return probs, raw_bin, clean_bin, overlay

def process_folder(model, images_dir, out_dir, transform, device='cpu',
                   threshold=0.5, min_area=500, keep_largest=False):
    """
    Process every image in images_dir and produce masks/overlays to out_dir.
    """
    images = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    print(f"Found {len(images)} images in {images_dir}")
    for p in images:
        print("Processing", p.name)
        process_image_file(model, p, out_dir=out_dir, transform=transform, device=device,
                           threshold=threshold, min_area=min_area, keep_largest=keep_largest)

# -------------------------
# 8) VIDEO MODE (temporal aggregation / heatmap)
# -------------------------
def process_video_heatmap(model, video_path, out_heatmap_path, transform, device='cpu',
                          threshold=0.5, min_area=500, stride=1):
    """
    Run inference on every (or every `stride`-th) frame and accumulate probability maps into a heatmap.
    The resulting heatmap highlights regions frequently predicted as rip across the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")

    heatmap_acc = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            tensor = preprocess_image_bgr(frame, transform)
            probs, _ = predict_mask(model, tensor, device=device, threshold=threshold)
            # probs is on resized space (size x size). resize back to original frame
            probs_resized = cv2.resize(probs, (frame.shape[1], frame.shape[0]))
            # probs_resized = cv2.resize(
            #     probs,
            #     (img_bgr.shape[1], img_bgr.shape[0])
            # )
            if heatmap_acc is None:
                heatmap_acc = np.zeros_like(probs_resized, dtype=np.float32)
            heatmap_acc += probs_resized
        frame_idx += 1

    cap.release()
    # normalize heatmap to 0..255
    if heatmap_acc is None:
        raise RuntimeError("No frames processed")
    heatmap_norm = (heatmap_acc - heatmap_acc.min()) / (heatmap_acc.max() - heatmap_acc.min() + 1e-8)
    heatmap_vis = (heatmap_norm * 255).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_heatmap_path), heatmap_color)
    return heatmap_color

    """ 
    Video detection + heatmap aggregation:
    - Run inference on each frame (or every N-th frame for speed). 
    - Optional: Save the raw heatmap (grayscale)
    """
def process_video_overlay(model, video_path, out_video_path=None, transform=None, device='cpu',
                          threshold=0.5, min_area=500, keep_largest=False, stride=1,
                          window_name="Rip Overlay (press q to quit)"):
    """
    Play a video with per-frame rip-overlay (live). Optionally save the overlayed video.
    - model: loaded model
    - video_path: input video file path
    - out_video_path: if provided, will save an output mp4 with overlays
    - transform: preprocessing transform (make_transform)
    - stride: process every `stride`-th frame to save CPU
    - keep_largest/min_area: postprocessing parameters
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")

    # Prepare writer if saving
    writer = None
    if out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    last_overlay = None
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Optionally do inference only every `stride` frames
            if frame_idx % stride == 0:
                # preprocess and predict
                tensor = preprocess_image_bgr(frame, transform)
                probs, raw_bin = predict_mask(model, tensor, device=device, threshold=threshold)
                clean_bin = postprocess_mask(raw_bin, min_area=min_area, keep_largest=keep_largest)

                # resize clean binary mask back to frame size
                mask_resized = cv2.resize(
                    clean_bin.astype("uint8"),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                # create overlay for display
                last_overlay = overlay_mask_on_image(frame, mask_resized)
            else:
                # reuse last overlay if we skip inference on this frame
                if last_overlay is None:
                    # if no overlay yet, create an empty overlay (just the frame)
                    last_overlay = frame.copy()

            # display overlay
            cv2.imshow(window_name, last_overlay)
            # save if requested
            if writer is not None:
                writer.write(last_overlay)

            # allow user to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested quit.")
                break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

# -------------------------
# 9) CLI / Example usage
# -------------------------
if __name__ == "__main__":
    device = 'cpu'
    model = load_model("best_unet_two.pth", device=device)
    transform = make_transform(size=256)

    # ---------------------------------------------------
    # 1) Single image
    # ---------------------------------------------------
    # img_path = "data/val/images/RipVIS-146_00420.jpg"
    # if Path(img_path).exists():
    #     probs, raw_bin, clean_bin, overlay = process_image_file(
    #         model, img_path, out_dir="out_infer",
    #         transform=transform, device=device,
    #         threshold=0.5, min_area=500, keep_largest=False)
    #     print("Saved outputs to out_infer/. To visualize overlay open the PNG file.")

    # ---------------------------------------------------
    # 2) Play video with live overlay (press 'q' to quit)
    #    - This will show overlay on screen and optionally save output
    # ---------------------------------------------------
    video_in = "test/videos/RipVIS-008.mp4"
    video_out = "out_infer/RipVIS-008_overlay_new.mp4"
    if Path(video_in).exists():
        print("Starting video overlay. Press 'q' to stop.")
        process_video_overlay(
            model, video_in, out_video_path=video_out, transform=transform,
            device=device, threshold=0.5, min_area=500, keep_largest=False, stride=2)

    # ---------------------------------------------------
    # 3) Build video heatmap (temporal aggregation) — independent function
    # ---------------------------------------------------
    # Uncomment to run after video overlay (or instead of overlay).
    # process_video_heatmap(model, "test/videos/RipVIS-008.mp4", "heatmap.png", transform,
    #                       device=device, threshold=0.5, stride=2)


