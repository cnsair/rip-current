#!/usr/bin/env python3
"""
Derive the crop window for each sequence in the qualitative figure from its
ground-truth annotation rather than by eye, and write the crops.csv that
make_qualitative_figure.py consumes.

Why this is not done visually
-----------------------------
A crop chosen by hand can be accused of favouring one row over another. A crop
derived from the annotation by a stated rule is reproducible from the released
data, and the rule can be given in the caption.

The rule
--------
Take the tight bounding box of the ground-truth mask, expand it by a fixed
fraction of its own size in every direction, then adjust the window to the
target aspect ratio by growing the shorter axis only. Growing rather than
cropping guarantees the whole annotation stays inside the window.

Uniform panels
--------------
Every panel in the figure is drawn at the same size, so the windows must share
one aspect ratio or the panels will be distorted. That is enforced here, not
left to chance: --aspect fixes the height-to-width ratio for every sequence
(default 0.75, i.e. 4:3). Windows differ in position and scale between
sequences, which is correct — each frames its own rip — but never in shape.

Pass --equal-scale to additionally force one window SIZE across all sequences,
taking the largest required. Panels are then directly comparable in scale, at
the cost of extra margin around the smaller rips.

Usage
  # masks named to match the sequence folders under panels/
  python make_crops.py \
      --masks  data_local/test_local/masks \
      --frames ./panels/frames.csv \
      --margin 0.25 \
      --out    ./panels/crops.csv
      
python make_crops.py \
    --masks  data_local/test_local/masks \
    --frames ./panels/frames.csv \
    --margin 0.25 \
    --out    crops.csv

  frames.csv maps each sequence to the frame being shown:
      RipVIS-121,RipVIS-121_00000.jpg
      RipVIS-066,RipVIS-066_00040.jpg
      RipVIS-108,RipVIS-108_00112.jpg
      
    Case 1 = RipVIS-121,RipVIS-121_00210
    Case 2 = RipVIS-066,RipVIS-066_00085
    Case 3 = RipVIS-108,RipVIS-108_01998

Depends on: numpy, opencv-python.
"""

import argparse
import os
import sys
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Requires opencv-python:  pip install opencv-python-headless")


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8 and m.max() <= 1.0:
        m = (m * 255).astype(np.uint8)
    return (m > 127)


def bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1     # x0,y0,x1,y1 px


def expand(b, margin, W, H):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    return (x0 - margin * w, y0 - margin * h,
            x1 + margin * w, y1 + margin * h)


def fit_aspect(win, aspect, W, H):
    """Grow the shorter axis so height/width == aspect. Never shrinks, so the
    annotation always stays inside."""
    x0, y0, x1, y1 = win
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if h / w < aspect:                 # too wide -> grow height
        h = w * aspect
    else:                              # too tall -> grow width
        w = h / aspect
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def clamp(win, W, H, aspect):
    """Shift the window inside the image; if it cannot fit, scale it down
    about its centre while holding the aspect ratio."""
    x0, y0, x1, y1 = win
    w, h = x1 - x0, y1 - y0
    if w > W:
        s = W / w; w, h = W, h * s
    if h > H:
        s = H / h; h, w = H, w * s
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx = min(max(cx, w / 2), W - w / 2)
    cy = min(max(cy, h / 2), H - h / 2)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="ground-truth mask directory")
    ap.add_argument("--panels", default=None,
                    help="panels root; sequence folder names are read from here")
    ap.add_argument("--frames", required=True,
                    help="CSV: sequence,mask_filename  (one line per sequence)")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="expansion as a fraction of bbox size, per side (default 0.25)")
    ap.add_argument("--aspect", type=float, default=0.75,
                    help="height/width for every window (default 0.75 = 4:3)")
    ap.add_argument("--equal-scale", action="store_true",
                    help="use one window size for all sequences (the largest required)")
    ap.add_argument("--out", default="crops.csv")
    args = ap.parse_args()

    def clean(v):
        """Strip whitespace, CR, and stray quotes that survive shell quoting
        or Windows editors."""
        return v.strip().strip('\r').strip().strip('"').strip("'").strip()

    pairs = []
    for ln in open(args.frames, encoding="utf-8-sig"):
        ln = ln.strip().strip('\r')
        if not ln or ln.startswith("#"):
            continue
        p = [clean(x) for x in ln.split(",")]
        if len(p) != 2 or not p[0] or not p[1]:
            sys.exit(f"Bad frames line (need  sequence,mask_filename ): {ln!r}")
        pairs.append(tuple(p))
    if not pairs:
        sys.exit("No sequences read from --frames")

    if args.panels and os.path.isdir(args.panels):
        have = {d for d in os.listdir(args.panels)
                if os.path.isdir(os.path.join(args.panels, d))}
        miss = [s for s, _ in pairs if s not in have]
        if miss:
            print(f"  ! sequences in --frames with no panel folder: {miss}")

    raw = []
    for seq, fn in pairs:
        mp = os.path.join(args.masks, fn)
        mask = read_mask(mp)
        if mask is None:
            # try the other common extensions before giving up
            stem = os.path.splitext(fn)[0]
            alt = None
            for e in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                cand = os.path.join(args.masks, stem + e)
                if os.path.exists(cand):
                    alt = cand
                    break
            if alt:
                print(f"  ! {fn} not found; using {os.path.basename(alt)} instead")
                mp = alt
                mask = read_mask(mp)
        if mask is None:
            near = [f for f in os.listdir(args.masks)
                    if os.path.splitext(fn)[0].split(".")[0][:18] in f][:5]
            sys.exit(f"Could not read mask: {mp}\n"
                     f"  Files in {args.masks} matching that stem: {near or 'none'}")
        H, W = mask.shape
        b = bbox(mask)
        if b is None:
            sys.exit(f"{seq}: mask {fn} has no foreground; choose a frame "
                     f"containing the annotation.")
        win = fit_aspect(expand(b, args.margin, W, H), args.aspect, W, H)
        raw.append([seq, fn, W, H, b, win])
        bw, bh = b[2] - b[0], b[3] - b[1]
        print(f"{seq:14s} image {W}x{H}  bbox {bw}x{bh} px")

    if args.equal_scale:
        ww = max(w[5][2] - w[5][0] for w in raw)
        hh = max(w[5][3] - w[5][1] for w in raw)
        hh = max(hh, ww * args.aspect); ww = hh / args.aspect
        print(f"\nequal-scale window: {ww:.0f} x {hh:.0f} px for every sequence")
        for r in raw:
            x0, y0, x1, y1 = r[5]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r[5] = (cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)

    lines = []
    print()
    for seq, fn, W, H, b, win in raw:
        x0, y0, x1, y1 = clamp(win, W, H, args.aspect)
        fx0, fy0, fx1, fy1 = x0 / W, y0 / H, x1 / W, y1 / H
        got = (y1 - y0) / (x1 - x0)
        lines.append(f"{seq},{fx0:.4f},{fy0:.4f},{fx1:.4f},{fy1:.4f}")
        print(f"{seq:14s} window {int(x1-x0)}x{int(y1-y0)} px  "
              f"aspect {got:.3f}  fractions "
              f"[{fx0:.3f},{fy0:.3f},{fx1:.3f},{fy1:.3f}]")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# derived by make_crops.py: ground-truth bbox expanded by "
                f"{args.margin:.0%} per side, fitted to aspect {args.aspect}"
                + (", equal scale across sequences" if args.equal_scale else "") + "\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    print("\nCaption sentence:")
    print(f'  "Each panel is cropped to a window derived from the ground-truth '
          f'bounding box of the displayed frame, expanded by {args.margin:.0%} of '
          f'its own size in each direction and fitted to a common aspect ratio'
          + (", with one window size used across all sequences." if args.equal_scale
             else ".") + '"')


if __name__ == "__main__":
    main()