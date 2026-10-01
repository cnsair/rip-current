#!/usr/bin/env python3
"""
video_colour_stats.py
=====================
Quantifies the colour of the water in each transfer-partition video, so that a
visual impression ("the undetected sequences look bluish-green") can be tested
against the detection outcome rather than asserted.

Method
------
For each video, a fixed number of frames is sampled at even stride. Rip pixels
are excluded using the ground-truth mask, so the statistics describe the
SURROUNDING water rather than the rip itself. Frames are converted to HSV and
the median hue, saturation and value of the non-rip region are recorded, along
with mean R, G, B and two ratios that separate blue-green water from the
brown/grey sediment-laden water typical of many beach scenes:

    green_blue  = mean(G) / mean(B)      < 1 indicates blue-dominant
    sat_ratio   = median(S) / 255        higher = more saturated colour

Only the lower 60% of the frame is used by default, on the assumption that sky
and beach occupy the upper portion in most orientations. Disable with
--full-frame if that assumption fails for your data.

Output: one row per video, joined to detection outcome if a per-frame CSV from
evaluate_event_level.py is supplied.

Usage
-----
  python video_colour_stats.py \
      --images data_local/test_local/rip_vis_val_images/images \
      --masks  data_local/test_local/rip_vis_val_images/masks \
      --per-frame results_eventlevel/theta_per_frame.csv \
      --out video_colour_stats.csv
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAME_RE = re.compile(r"^(RipVIS-(?:NR-)?\d+)_(\d+)\.", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--per-frame", default=None,
                    help="per_frame CSV from evaluate_event_level.py, to join "
                         "the silent-frame fraction onto each video")
    ap.add_argument("--samples", type=int, default=12,
                    help="frames sampled per video")
    ap.add_argument("--full-frame", action="store_true",
                    help="use the whole frame instead of the lower 60%%")
    ap.add_argument("--out", default="video_colour_stats.csv")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    byvid = defaultdict(list)
    for f in files:
        m = NAME_RE.match(f)
        if m:
            byvid[m.group(1)].append(f)
    print(f"{len(files)} frames across {len(byvid)} videos")

    silent = {}
    if a.per_frame:
        import collections
        acc = collections.defaultdict(lambda: [0, 0])
        with open(a.per_frame) as fh:
            for row in csv.DictReader(fh):
                if row["has_rip"] != "1":
                    continue
                acc[row["video"]][0] += int(int(row["pred_px"]) == 0)
                acc[row["video"]][1] += 1
        silent = {v: s / n for v, (s, n) in acc.items() if n}

    rows = []
    for vid, fl in sorted(byvid.items()):
        fl = sorted(fl)
        step = max(1, len(fl) // a.samples)
        picks = fl[::step][:a.samples]
        H, S, V, R, G, B = [], [], [], [], [], []
        for fn in picks:
            img = cv2.imread(str(Path(a.images) / fn), cv2.IMREAD_COLOR)
            if img is None:
                continue
            mp = Path(a.masks) / (Path(fn).stem + ".png")
            gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if gt is not None and gt.shape != img.shape[:2]:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            water = np.ones(img.shape[:2], bool)
            if gt is not None:
                water &= gt <= 127                      # exclude rip pixels
            if not a.full_frame:
                water[: int(0.4 * img.shape[0]), :] = False   # drop sky/beach
            if water.sum() < 500:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H.append(np.median(hsv[..., 0][water]))
            S.append(np.median(hsv[..., 1][water]))
            V.append(np.median(hsv[..., 2][water]))
            B.append(img[..., 0][water].mean())
            G.append(img[..., 1][water].mean())
            R.append(img[..., 2][water].mean())
        if not H:
            print(f"  !! no usable frames for {vid}")
            continue
        r, g, b = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
        rows.append({
            "video": vid, "n_frames": len(fl), "n_sampled": len(H),
            "hue_med": round(float(np.median(H)), 2),
            "sat_med": round(float(np.median(S)), 2),
            "val_med": round(float(np.median(V)), 2),
            "R": round(r, 1), "G": round(g, 1), "B": round(b, 1),
            "green_blue": round(g / b, 4) if b else None,
            "red_blue": round(r / b, 4) if b else None,
            "silent_frac": round(silent.get(vid, float("nan")), 4),
        })
        print(f"  {vid}: hue={rows[-1]['hue_med']:.0f} sat={rows[-1]['sat_med']:.0f} "
              f"G/B={rows[-1]['green_blue']} silent={rows[-1]['silent_frac']}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {a.out}")
    print("Note: OpenCV hue is 0-179. Blue-green water sits near 90-105; "
          "brown/sediment water near 10-25.")


if __name__ == "__main__":
    main()
