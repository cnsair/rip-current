#!/usr/bin/env python3
"""
evaluate_ripaid_tiled.py
========================
Replacement for evaluate_ripaid.py. Same metrics, corrected geometry.

Why the whole-frame version was wrong
-------------------------------------
RipAID frames are 1280x960 oblique panoramas covering between 65 m and 580 m of
coastline depending on camera focal length. Resizing them to 512x512 for
inference compressed rips to a median minor axis of 27-31 px on the wide-field
cameras, against 45.6 px for the training corpus. Detection rate correlated with
metres of coastline in frame at Spearman rho = -0.81 (p = 0.015): the model was
not failing to generalise, it was being shown rips at a spatial scale it was
never trained on.

What this does instead
----------------------
Inference is run on TILES cut from the frame at native resolution, so a rip
subtends roughly the pixel extent it would in training. Each tile is resized
from TILE_H x TILE_W to the network's 512x512 input, which for the default
512x384 tile is close to an identity resampling. Tile probability maps are
stitched back into a full-frame map by taking the per-pixel maximum over
overlapping tiles, and every metric is then computed on the full-frame map at
native 1280x960 resolution, where the ground-truth boxes live.

Tiling is a deployment-realistic choice as well as a fair one: a fixed camera
covering 580 m of beach would in practice be processed in sections.

Metrics are unchanged from evaluate_ripaid.py and remain detection-level only.
RipAID annotates with oriented boxes, which are supersets of the rip, so
pixel-overlap metrics (mIoU, pixel Recall, BoundaryIoU) are not computed.

Usage
-----
  python evaluate_ripaid_tiled.py \
      --checkpoint ./trained_models/segformer_b2_local.pth \
      --images data_local/indep_local/ripaid/images/default \
      --labels data_local/indep_local/ripaid/yolo-obb/labels/train \
      --label theta0_tiled --out-dir results_ripaid_tiled

  # diagnostic: also write the whole-frame result for direct comparison
  python evaluate_ripaid_tiled.py ... --also-wholeframe
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

NET_SIZE = 512          # network input
TILE_H, TILE_W = 480, 640   # native-resolution tile; 2x3 grid over 1280x960
OVERLAP = 0.25          # fraction of tile size overlapped between neighbours
THRESHOLD = 0.5
MIN_COMPONENT_PX_512 = 50   # pre-registered on the 512 grid; rescaled below
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Wrapper(torch.nn.Module):
    def __init__(self, hf, size):
        super().__init__()
        self.model, self.output_size = hf, size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            self.model(pixel_values=x).logits, size=self.output_size,
            mode="bilinear", align_corners=False)


def read_obb(path, class_map):
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


def polys_to_mask(polys, w, h):
    m = np.zeros((h, w), np.uint8)
    for q in polys:
        pts = np.round(q * np.array([w, h])).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def tile_origins(full, tile, overlap):
    """Start coordinates covering `full` with `tile`-sized windows overlapping
    by `overlap` fraction. The last window is flush with the far edge."""
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(full - tile, 0) + 1, step))
    if not xs or xs[-1] != full - tile:
        xs.append(max(full - tile, 0))
    return sorted(set(xs))


def predict_tiled(model, img, tf, tile_h, tile_w, overlap):
    """Full-frame probability map at native resolution, per-pixel max over tiles."""
    H, W = img.shape[:2]
    prob = np.zeros((H, W), np.float32)
    for y in tile_origins(H, min(tile_h, H), overlap):
        for x in tile_origins(W, min(tile_w, W), overlap):
            crop = img[y:y + tile_h, x:x + tile_w]
            t = tf(image=crop)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
            p = cv2.resize(p, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            sl = prob[y:y + crop.shape[0], x:x + crop.shape[1]]
            np.maximum(sl, p, out=sl)
    return prob


def predict_whole(model, img, tf):
    H, W = img.shape[:2]
    t = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def score(pred, rip_polys, doubt_polys, W, H, min_px):
    n_lbl, lbl = cv2.connectedComponents(pred.astype(np.uint8), 8)
    comps = [int((lbl == j).sum()) for j in range(1, n_lbl)]
    detected = any(c >= min_px for c in comps)
    rip_mask = polys_to_mask(rip_polys, W, H)
    doubt_mask = polys_to_mask(doubt_polys, W, H)
    hits = sum(1 for q in rip_polys
               if int((polys_to_mask([q], W, H) & pred).sum()) >= min_px)
    valid = pred & ~doubt_mask
    return dict(detected=int(detected), boxes_hit=hits,
                pred_px=int(pred.sum()), pred_px_valid=int(valid.sum()),
                pred_px_in_box=int((valid & rip_mask).sum()),
                largest_component=max(comps) if comps else 0)


def summarise(rows, label, args, excluded, key=""):
    rip = [r for r in rows if r["has_rip"]]
    neg = [r for r in rows if not r["has_rip"]]
    tb = sum(r["n_boxes"] for r in rip)
    th = sum(r[f"boxes_hit{key}"] for r in rip)
    pa = sum(r[f"pred_px_valid{key}"] for r in rip)
    pi = sum(r[f"pred_px_in_box{key}"] for r in rip)
    return {
        "label": label, "threshold": args.threshold,
        "tile": f"{args.tile_h}x{args.tile_w}", "overlap": args.overlap,
        "min_px_native": args.min_px_native,
        "images_evaluated": len(rows), "images_excluded_doubt_only": excluded,
        "rip_images": len(rip), "negative_images": len(neg), "rip_boxes": tb,
        "image_detection_rate": len([r for r in rip if r[f"detected{key}"]]) / len(rip) if rip else float("nan"),
        "box_detection_rate": th / tb if tb else float("nan"),
        "localisation_precision": pi / pa if pa else float("nan"),
        "image_false_alarm_rate": len([r for r in neg if r[f"detected{key}"]]) / len(neg) if neg else float("nan"),
        "mean_pred_px_rip_images": float(np.mean([r[f"pred_px{key}"] for r in rip])) if rip else float("nan"),
        "silent_rip_images": len([r for r in rip if r[f"pred_px{key}"] == 0]) / len(rip) if rip else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--segformer-path", default="./segformer-b2-local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results_ripaid_tiled")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--tile-h", type=int, default=TILE_H)
    ap.add_argument("--tile-w", type=int, default=TILE_W)
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    ap.add_argument("--min-px-native", type=int, default=None,
                    help="minimum component size in NATIVE pixels; default "
                         "rescales the pre-registered 50 px from the 512 grid")
    ap.add_argument("--doubt-as-rip", action="store_true")
    ap.add_argument("--also-wholeframe", action="store_true",
                    help="also score the 512x512 whole-frame prediction, for "
                         "a like-for-like comparison with the original run")
    ap.add_argument("--rip-class", type=int, default=0)
    ap.add_argument("--doubt-class", type=int, default=1)
    a = ap.parse_args()

    cmap = {a.rip_class: "rip_current", a.doubt_class: "doubt"}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    tf = A.Compose([A.Resize(NET_SIZE, NET_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)), ToTensorV2()])

    hf = SegformerForSemanticSegmentation.from_pretrained(
        a.segformer_path, num_labels=1, ignore_mismatched_sizes=True)
    model = Wrapper(hf, (NET_SIZE, NET_SIZE))
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu",
                                     weights_only=False)["model_state"],
                          strict=True)
    model.to(DEVICE).eval()

    files = sorted(f for f in os.listdir(a.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"{len(files)} images | tile {a.tile_h}x{a.tile_w} overlap {a.overlap}")

    rows, tally = [], Counter()
    nrip = ndbt = 0

    for i, fn in enumerate(files):
        lab = read_obb(Path(a.labels) / (Path(fn).stem + ".txt"), cmap)
        rp, dp = lab["rip_current"], lab["doubt"]
        nrip += len(rp); ndbt += len(dp)
        if a.doubt_as_rip:
            rp, dp = rp + dp, []
        if not rp and dp:
            tally["excluded"] += 1
            continue

        img = np.array(Image.open(Path(a.images) / fn).convert("RGB"))
        H, W = img.shape[:2]
        if a.min_px_native is None:
            # keep the pre-registered area threshold in relative terms
            a.min_px_native = int(round(MIN_COMPONENT_PX_512 * (H * W) / (NET_SIZE ** 2)))

        pt = predict_tiled(model, img, tf, a.tile_h, a.tile_w, a.overlap) >= a.threshold
        rec = {"image": fn, "has_rip": int(bool(rp)), "n_boxes": len(rp)}
        rec.update(score(pt, rp, dp, W, H, a.min_px_native))

        if a.also_wholeframe:
            pw = predict_whole(model, img, tf) >= a.threshold
            for k, v in score(pw, rp, dp, W, H, a.min_px_native).items():
                rec[k + "_wf"] = v

        rows.append(rec)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    print(f"\nbox counts read: rip_current={nrip}  doubt={ndbt}"
          f"   (published: 1959 / 915)")
    print(f"min component size: {a.min_px_native} native px "
          f"(= 50 px on the 512 grid)")

    s = summarise(rows, a.label, a, tally["excluded"])
    with open(out / f"{a.label}_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out / f"{a.label}_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        for k, v in s.items():
            w.writerow([k, v])

    print("\n" + "=" * 64)
    print("  TILED (native resolution)")
    for k, v in s.items():
        print(f"    {k:<28} {v}")
    if a.also_wholeframe:
        sw = summarise(rows, a.label + "_wholeframe", a, tally["excluded"], key="_wf")
        with open(out / f"{a.label}_wholeframe_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["metric", "value"])
            for k, v in sw.items():
                w.writerow([k, v])
        print("\n  WHOLE-FRAME 512x512 (for comparison)")
        for k in ("image_detection_rate", "box_detection_rate",
                  "localisation_precision", "image_false_alarm_rate",
                  "silent_rip_images"):
            print(f"    {k:<28} {sw[k]}")
    print("=" * 64)


if __name__ == "__main__":
    main()
