#!/usr/bin/env python3
"""
evaluate_ripaid.py
==================
One-time evaluation on RipAID (Soriano-González et al., 2025, Zenodo
10.5281/zenodo.15082427), the independently curated corpus used to answer
reviewer comment 3.

RipAID annotates rip currents with ORIENTED BOUNDING BOXES, not masks. A box is
a superset of the rip it contains, so pixel-level overlap metrics (mIoU, pixel
Recall, BoundaryIoU) are not interpretable against it and are deliberately NOT
computed here. Evaluation is at detection granularity, matching the protocol of
Section IV-F.

Metrics
-------
  image-level detection rate   an image counts as detected if the prediction
                               contains a connected component of at least
                               MIN_COMPONENT_PX pixels. Denominator: images
                               carrying at least one rip_current box.

  box-level detection rate     a ground-truth box is hit if at least
                               MIN_COMPONENT_PX predicted pixels fall inside
                               its oriented polygon. Denominator: rip_current
                               boxes.

  localisation precision       share of predicted foreground pixels that fall
                               inside some rip_current box. Doubt regions are
                               excluded from both numerator and denominator, so
                               predictions there are neither credited nor
                               penalised.

  image false-alarm rate       share of clean-negative images (no rip_current
                               and no doubt) producing any component of at
                               least MIN_COMPONENT_PX.

Doubt handling (pre-registered)
-------------------------------
  * images with doubt but no rip_current are EXCLUDED entirely
  * on rip images that also carry doubt, doubt regions are masked out of the
    precision computation
  * the negative set is images carrying neither label
  * --doubt-as-rip runs the sensitivity variant in which doubt boxes count as
    positives, giving an upper bound on detection rate

Annotations are read from the Ultralytics YOLO-OBB export, which stores four
explicit corner points per box and therefore avoids any assumption about CVAT's
rotation convention. The script prints box counts so the class mapping can be
verified against the published totals (1,959 rip_current and 915 doubt).

Usage
-----
  python evaluate_ripaid.py \
      --checkpoint ./trained_models/wise_swad/segformer_b2_wise_a070.pth \
      --images  RipAID_v1.0.0/images/default \
      --labels  RipAID_v1.0.0/RipAID_v1.0.0_yolo-obb/labels/train \
      --label sawi_a070 --out-dir results_ripaid
"""

import argparse
import csv
import os
from collections import Counter
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
THRESHOLD = 0.5
MIN_COMPONENT_PX = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CLASS_MAP = {0: "rip_current", 1: "doubt"}


class Wrapper(torch.nn.Module):
    def __init__(self, hf, size):
        super().__init__()
        self.model, self.output_size = hf, size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            self.model(pixel_values=x).logits, size=self.output_size,
            mode="bilinear", align_corners=False)


def read_obb(path, class_map):
    """YOLO-OBB: 'cls x1 y1 x2 y2 x3 y3 x4 y4', coordinates normalised [0,1].
    Returns {class_name: [ (4,2) float arrays in normalised coords ]}."""
    out = {v: [] for v in class_map.values()}
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 9:
            continue
        c = class_map.get(int(float(p[0])))
        if c is None:
            continue
        out[c].append(np.array([float(v) for v in p[1:9]],
                               dtype=np.float32).reshape(4, 2))
    return out


def polys_to_mask(polys, size):
    m = np.zeros((size, size), np.uint8)
    for q in polys:
        pts = np.round(q * size).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="YOLO-OBB labels directory")
    ap.add_argument("--segformer-path", default="./segformer-b2-local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results_ripaid")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--min-px", type=int, default=MIN_COMPONENT_PX)
    ap.add_argument("--doubt-as-rip", action="store_true",
                    help="sensitivity variant: count doubt boxes as positives")
    ap.add_argument("--rip-class", type=int, default=0)
    ap.add_argument("--doubt-class", type=int, default=1)
    a = ap.parse_args()

    cmap = {a.rip_class: "rip_current", a.doubt_class: "doubt"}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)), ToTensorV2()])

    hf = SegformerForSemanticSegmentation.from_pretrained(
        a.segformer_path, num_labels=1, ignore_mismatched_sizes=True)
    model = Wrapper(hf, (IMG_SIZE, IMG_SIZE))
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu",
                                     weights_only=False)["model_state"],
                          strict=True)
    model.to(DEVICE).eval()

    files = sorted(f for f in os.listdir(a.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"{len(files)} images in {a.images}")

    rows = []
    tally = Counter()
    n_rip_boxes = n_doubt_boxes = 0

    with torch.no_grad():
        for i, fn in enumerate(files):
            lab = read_obb(Path(a.labels) / (Path(fn).stem + ".txt"), cmap)
            rip_polys = lab["rip_current"]
            doubt_polys = lab["doubt"]
            if a.doubt_as_rip:
                rip_polys = rip_polys + doubt_polys
                doubt_polys = []
            n_rip_boxes += len(lab["rip_current"])
            n_doubt_boxes += len(lab["doubt"])

            has_rip = len(rip_polys) > 0
            has_doubt = len(doubt_polys) > 0
            # pre-registered exclusion: doubt without rip is ambiguous
            if (not has_rip) and has_doubt:
                tally["excluded_doubt_only"] += 1
                continue

            img = np.array(Image.open(Path(a.images) / fn).convert("RGB"))
            x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
            pred = prob >= a.threshold

            n_lbl, lbl = cv2.connectedComponents(pred.astype(np.uint8), 8)
            comp_sizes = [int((lbl == j).sum()) for j in range(1, n_lbl)]
            detected_img = any(s >= a.min_px for s in comp_sizes)

            rip_mask = polys_to_mask(rip_polys, IMG_SIZE)
            doubt_mask = polys_to_mask(doubt_polys, IMG_SIZE)

            hits = 0
            for q in rip_polys:
                bm = polys_to_mask([q], IMG_SIZE)
                if int((bm & pred).sum()) >= a.min_px:
                    hits += 1

            pred_valid = pred & ~doubt_mask          # doubt regions neutral
            n_pred = int(pred_valid.sum())
            n_in = int((pred_valid & rip_mask).sum())

            rows.append({
                "image": fn, "has_rip": int(has_rip),
                "n_boxes": len(rip_polys), "boxes_hit": hits,
                "pred_px": int(pred.sum()), "pred_px_valid": n_pred,
                "pred_px_in_box": n_in,
                "largest_component": max(comp_sizes) if comp_sizes else 0,
                "detected": int(detected_img),
            })
            tally["rip_images" if has_rip else "negative_images"] += 1
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    print(f"\nbox counts read: rip_current={n_rip_boxes}  doubt={n_doubt_boxes}")
    print("published totals: rip_current=1959  doubt=915  "
          "-> if these disagree, the --rip-class/--doubt-class mapping is wrong")

    rip = [r for r in rows if r["has_rip"]]
    neg = [r for r in rows if not r["has_rip"]]
    tot_boxes = sum(r["n_boxes"] for r in rip)
    tot_hits = sum(r["boxes_hit"] for r in rip)
    px_all = sum(r["pred_px_valid"] for r in rip)
    px_in = sum(r["pred_px_in_box"] for r in rip)

    s = {
        "label": a.label, "threshold": a.threshold, "min_px": a.min_px,
        "doubt_as_rip": int(a.doubt_as_rip),
        "images_evaluated": len(rows),
        "images_excluded_doubt_only": tally["excluded_doubt_only"],
        "rip_images": len(rip), "negative_images": len(neg),
        "rip_boxes": tot_boxes,
        "image_detection_rate": (len([r for r in rip if r["detected"]]) / len(rip)) if rip else float("nan"),
        "box_detection_rate": (tot_hits / tot_boxes) if tot_boxes else float("nan"),
        "localisation_precision": (px_in / px_all) if px_all else float("nan"),
        "image_false_alarm_rate": (len([r for r in neg if r["detected"]]) / len(neg)) if neg else float("nan"),
        "mean_pred_px_rip_images": float(np.mean([r["pred_px"] for r in rip])) if rip else float("nan"),
        "silent_rip_images": (len([r for r in rip if r["pred_px"] == 0]) / len(rip)) if rip else float("nan"),
    }

    with open(out / f"{a.label}_ripaid_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out / f"{a.label}_ripaid_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        for k, v in s.items():
            w.writerow([k, v])

    print("\n" + "=" * 64)
    for k, v in s.items():
        print(f"  {k:<30} {v}")
    print("=" * 64)
    print("  NOTE: mIoU, pixel Recall and BoundaryIoU are deliberately not")
    print("  reported; RipAID annotates with boxes, which are supersets of the")
    print("  rip and cannot support pixel-overlap metrics.")


if __name__ == "__main__":
    main()
