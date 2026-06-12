#!/usr/bin/env python3
"""
Measure the Foam-Gap Consistency metric on the test set for a trained checkpoint.

WHY THIS EXISTS
    evaluate_test_set.py reports mIoU / IoU / recall / precision / F2 / ... but not
    the Foam-Gap Consistency (the mean prediction-vs-surround Michelson contrast).
    That metric is a property of the PREDICTIONS and the IMAGE, so it is defined for
    EVERY arm — including the lambda=0 baseline, which never used the loss in
    training. This script computes it on the 4,349-image test set so you can fill the
    "Foam-Gap Consistency" column for the baseline row (and every foam row) with a
    number measured identically across arms.

HOW IT STAYS FAITHFUL
    It imports your training script and reuses build_model(), get_transforms(),
    denormalize, FoamGapLoss, and the foam geometry constants — so the model,
    preprocessing, and metric definition match your runs exactly. The foam geometry
    (ring/guard/mode/min_mass/downsample) is fixed in the script (not env-driven), so
    the metric is computed the same way regardless of which arm produced the weights.
    Note: the value reported during training was on the VALIDATION split; the table
    needs the TEST-set value, which is what this produces.

USAGE  (run from the project folder, with `ripseg` active):
    python measure_foam_consistency.py --checkpoint ./trained_models/segformer_b2_loss_algo.pth --label baseline
    python measure_foam_consistency.py --checkpoint ./trained_models/segformer_b2_foam_l010.pth --label foam_l010

Run once per checkpoint. The printed "Foam-Gap Consistency" mean is the table value;
rows are also appended to results/foam_consistency.csv for easy comparison.
"""
import os
import sys
import glob
import argparse
import importlib

import numpy as np
import torch
from PIL import Image


def load_train_module():
    """Import the training script (reuses its model + preprocessing + loss)."""
    # Make sure the folder you run from (the project folder) is importable,
    # so this works whether you call it by name or by full path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    name = os.environ.get("TRAIN_MODULE", "train_segformer_foam_gap")
    for candidate in (name, "train_segformer"):
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    sys.exit("ERROR: could not import the training module. Run this from the folder "
             "that contains train_segformer_foam_gap.py (and foam_gap_loss.py).")


def main():
    ap = argparse.ArgumentParser(description="Measure Foam-Gap Consistency on the test set")
    ap.add_argument("--checkpoint", required=True, help="path to a trained .pth")
    ap.add_argument("--images-dir",
                    default="data_local/test_local/rip_vis_val_images/images",
                    help="folder of test images (masks not needed)")
    ap.add_argument("--label", default="model", help="row label written to the CSV")
    ap.add_argument("--out", default="results/foam_consistency.csv")
    args = ap.parse_args()

    T = load_train_module()
    dev = T.DEVICE

    # ── Build the model exactly as in training and load the trained weights ──────
    model = T.build_model().to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded {args.checkpoint}  (epoch {epoch})  on {dev}")

    # ── Foam-gap metric with the SAME geometry used during the foam runs ─────────
    foam = T.FoamGapLoss(
        margin     = getattr(T, "FOAM_MARGIN", 0.07),   # unused by the metric
        ring_px    = T.FOAM_RING_PX,
        guard_px   = T.FOAM_GUARD_PX,
        mode       = T.FOAM_MODE,
        min_mass   = T.FOAM_MIN_MASS,
        downsample = T.FOAM_DOWNSAMPLE,
    ).to(dev)

    # ── Validation-style preprocessing (resize to IMG_SIZE + ImageNet normalise) ─
    tf   = T.get_transforms(train=False)
    mean = torch.tensor(T._IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std  = torch.tensor(T._IMAGENET_STD,  device=dev).view(1, 3, 1, 1)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    paths = sorted(sum([glob.glob(os.path.join(args.images_dir, e)) for e in exts], []))
    if not paths:
        sys.exit(f"No images found in {args.images_dir}")
    print(f"Scoring {len(paths)} test images ...")

    c_sum, n_gated, n = 0.0, 0, 0
    with torch.no_grad():
        for i, pth in enumerate(paths):
            img = np.array(Image.open(pth).convert("RGB"))
            x = tf(image=img)["image"].unsqueeze(0).to(dev)   # (1,3,IMG,IMG) normalised
            logits = model(x)
            raw = (x * std + mean).clamp(0.0, 1.0)             # un-normalised RGB
            c, _ = foam.consistency_metric(logits, raw)
            n += 1
            if c == c:        # not NaN => image was gated (enough predicted rip mass)
                c_sum += float(c)
                n_gated += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(paths)} ...")

    consistency = (c_sum / n_gated) if n_gated else float("nan")
    gate_frac   = (n_gated / n) if n else 0.0
    print("=" * 60)
    print(f"  Foam-Gap Consistency (mean Michelson contrast) : {consistency:.4f}")
    print(f"  Gate fraction (images with predicted rip mass) : {gate_frac:.3f}")
    print(f"  Images gated / total                           : {n_gated} / {n}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if new:
            f.write("label,consistency,gate_frac,n_gated,n_total\n")
        f.write(f"{args.label},{consistency:.4f},{gate_frac:.4f},{n_gated},{n}\n")
    print(f"Appended '{args.label}' row to {args.out}")


if __name__ == "__main__":
    main()
