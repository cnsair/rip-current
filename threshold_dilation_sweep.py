#!/usr/bin/env python3
"""
threshold_dilation_sweep.py
===========================
Answers reviewer comment 5: is the Recall gain of SAWI simply a more aggressive
output distribution, reproducible from the anchor by rethresholding, recalibrating
or dilating?

What this computes
------------------
  threshold sweep   per-image metrics at every threshold in a grid, averaged
                    over images exactly as evaluate_test_set.py does. Traces the
                    full Precision-Recall and Recall-mIoU curves.

  dilation sweep    per-image metrics after binary dilation of the 0.5-threshold
                    mask with square kernels of increasing size.

Why calibration baselines are NOT computed here
-----------------------------------------------
Temperature scaling, Platt scaling and isotonic regression are all monotone
pixelwise maps of the logit. A monotone map cannot reorder pixels by score, so
it leaves the precision-recall curve invariant and only moves the operating
point along it: thresholding sigmoid(z/T) at 0.5 is identical to thresholding
sigmoid(z) at sigmoid(logit(0.5)/... ) i.e. at some other fixed value. The
threshold sweep therefore contains every operating point any of those methods
can reach. Dilation is spatial rather than pixelwise and is genuinely distinct,
so it is swept separately.

Metric definitions replicate compute_metrics() in evaluate_test_set.py exactly,
including the 1e-6 epsilons and the zero_division=0 convention for F-beta, so
the threshold=0.5 row must reproduce the published aggregates. The script
checks this and prints the comparison.

Usage
-----
  python threshold_dilation_sweep.py \
      --checkpoint ./trained_models/wise_swad/segformer_b2_wise_a070.pth \
      --images data_local/test_local/rip_vis_val_images/images \
      --masks  data_local/test_local/rip_vis_val_images/masks \
      --label sawi_a070_test --out-dir results_sweep
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DETAIL", "0")

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import SegformerForSemanticSegmentation

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = [round(x, 3) for x in np.concatenate([
    np.arange(0.02, 0.20, 0.02), np.arange(0.20, 0.85, 0.05),
    np.arange(0.85, 0.99, 0.02)])]
DILATIONS = [0, 3, 5, 7, 9, 11, 15, 21, 31]


class Wrapper(torch.nn.Module):
    def __init__(self, hf, size):
        super().__init__()
        self.model, self.output_size = hf, size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            self.model(pixel_values=x).logits, size=self.output_size,
            mode="bilinear", align_corners=False)


def metrics_from_counts(tp, fp, fn, tn):
    """Exactly compute_metrics() in evaluate_test_set.py, minus BoundaryIoU.
    F-beta is computed from counts, which is identical to sklearn's
    fbeta_score(beta=2, zero_division=0) for the binary case."""
    iou = tp / (tp + fp + fn + 1e-6)
    iou_bg = tn / (tn + fp + fn + 1e-6)
    miou = (iou + iou_bg) / 2.0
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    aacc = (tp + fp + fn + tn) and (tp + tn) / (tp + fp + fn + tn + 1e-6)
    acc_bg = tn / (tn + fp + 1e-6)
    macc = (recall + acc_bg) / 2.0
    den = 5 * tp + 4 * fn + fp
    f2 = (5 * tp / den) if den > 0 else 0.0
    return dict(miou=miou, iou=iou, dice=dice, recall=recall,
                precision=precision, f2=f2, aacc=aacc, macc=macc)


def counts(pred, gt):
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))
    return tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--segformer-path", default="./segformer-b2-local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results_sweep")
    ap.add_argument("--skip-dilation", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)), ToTensorV2()])
    rs = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

    hf = SegformerForSemanticSegmentation.from_pretrained(
        a.segformer_path, num_labels=1, ignore_mismatched_sizes=True)
    model = Wrapper(hf, (IMG_SIZE, IMG_SIZE))
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu",
                                     weights_only=False)["model_state"],
                          strict=True)
    model.to(DEVICE).eval()

    files = sorted(f for f in os.listdir(a.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(files)} images | {len(THRESHOLDS)} thresholds"
          f"{'' if a.skip_dilation else f' | {len(DILATIONS)} dilations'}")

    acc_t = {t: [] for t in THRESHOLDS}
    acc_d = {k: [] for k in DILATIONS}
    n_eval = 0

    with torch.no_grad():
        for i, fn in enumerate(files):
            mp = Path(a.masks) / (Path(fn).stem + ".png")
            if not mp.exists():
                continue
            img = np.array(Image.open(Path(a.images) / fn).convert("RGB"))
            gtr = np.array(Image.open(mp).convert("L"))
            gt = rs(image=img, mask=gtr)["mask"] > 127

            x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
            n_eval += 1

            for t in THRESHOLDS:
                acc_t[t].append(metrics_from_counts(*counts(prob >= t, gt)))

            if not a.skip_dilation:
                base = (prob >= 0.5).astype(np.uint8)
                for k in DILATIONS:
                    pm = base if k == 0 else cv2.dilate(
                        base, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
                    acc_d[k].append(metrics_from_counts(*counts(pm.astype(bool), gt)))

            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    def mean_rows(acc, pname):
        rows = []
        for p, lst in acc.items():
            if not lst:
                continue
            r = {"op": pname, "param": p, "n": len(lst)}
            for k in lst[0]:
                r[k] = float(np.mean([d[k] for d in lst]))
            rows.append(r)
        return rows

    rows = mean_rows(acc_t, "threshold")
    if not a.skip_dilation:
        rows += mean_rows(acc_d, "dilation")

    fp_out = out / f"{a.label}_sweep.csv"
    with open(fp_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    base = [r for r in rows if r["op"] == "threshold" and abs(r["param"] - 0.50) < 1e-9]
    print("\n" + "=" * 66)
    print(f"  images evaluated: {n_eval}")
    if base:
        b = base[0]
        print("  SELF-CHECK at threshold 0.50 — must match the published "
              "aggregate for this checkpoint:")
        for k in ("recall", "f2", "miou", "precision"):
            print(f"    {k:<10} {b[k]:.4f}")
        print("  If these differ from the corresponding *_aggregate.csv, the "
              "sweep is not comparable and must be reconciled before use.")
    print(f"  wrote {fp_out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
