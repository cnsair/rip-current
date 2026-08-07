#!/usr/bin/env python3
"""
Build a contact sheet for labelling each RipVIS video sequence by acquisition
modality, and emit a template CSV to fill in.

Why three frames per video
--------------------------
The static/mobile distinction cannot be judged from a single frame. A fixed
installation holds its background constant; drone, handheld and body-worn
acquisition all show viewpoint motion. This script therefore shows the FIRST,
MIDDLE and LAST frame of every sequence side by side, so that camera motion is
visible at a glance:

  static  - horizon, shoreline and any fixed structures stay in place across
            the three frames
  mobile  - the framing shifts, rotates, or changes scale between frames

Usage
  python make_modality_contact_sheet.py \
      --lookup  frame_to_video.csv \
      --images  data_local/test_local/images \
      --out     modality_labelling

Outputs
  contact_sheet_1.png ...   one or more sheets, 36 videos across them
  video_modality_TEMPLATE.csv   video_id, n_frames, has_rip, modality (blank)

Fill the modality column with 'static' or 'mobile', save as
video_modality.csv, and pass it to stratify_by_modality.py.

Depends on: numpy, pandas, matplotlib, pillow.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle
except ImportError:
    sys.exit("Requires matplotlib:  pip install matplotlib")


def frame_index(fn):
    """Trailing integer of a RipVIS filename, e.g. RipVIS-014_00177.jpg -> 177."""
    stem = os.path.splitext(fn)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True,
                    help="frame_to_video.csv (file_name, video_id, is_NR, has_rip)")
    ap.add_argument("--images", required=True, help="RipVIS test image directory")
    ap.add_argument("--out", default="modality_labelling")
    ap.add_argument("--per-sheet", type=int, default=36,
                    help="videos per contact sheet (default 36)")
    ap.add_argument("--cols", type=int, default=4, help="video blocks per row")
    ap.add_argument("--include-nr", action="store_true",
                    help="also show the no-rip (-NR-) sequences")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lk = pd.read_csv(args.lookup)
    if not args.include_nr and "is_NR" in lk.columns:
        lk = lk[lk.is_NR == 0]

    vids = sorted(lk.video_id.unique())
    if not len(vids):
        sys.exit("No videos found in the lookup after filtering.")
    print(f"{len(vids)} video sequences to label")

    # first / middle / last frame per video
    picks = {}
    for v in vids:
        f = lk[lk.video_id == v].file_name.tolist()
        f.sort(key=frame_index)
        if len(f) >= 3:
            picks[v] = [f[0], f[len(f) // 2], f[-1]]
        else:
            picks[v] = (f * 3)[:3]

    # ---- template CSV ----
    rows = []
    for v in vids:
        g = lk[lk.video_id == v]
        rows.append({"video_id": v, "n_frames": len(g),
                     "has_rip": int(g.has_rip.sum()) if "has_rip" in g.columns else "",
                     "modality": ""})
    tpl = os.path.join(args.out, "video_modality_TEMPLATE.csv")
    pd.DataFrame(rows).to_csv(tpl, index=False)
    print(f"template written: {tpl}")

    # ---- contact sheets ----
    COLS = args.cols
    TW, TH = 0.56, 0.42          # thumbnail size, inches
    BW = 3 * TW + 0.06           # block width
    BH = TH + 0.20               # block height incl. label
    GX, GY = 0.14, 0.10

    chunks = [vids[i:i + args.per_sheet] for i in range(0, len(vids), args.per_sheet)]
    for si, chunk in enumerate(chunks, 1):
        nrow = int(np.ceil(len(chunk) / COLS))
        W = 0.20 + COLS * BW + (COLS - 1) * GX + 0.20
        H = 0.42 + nrow * BH + (nrow - 1) * GY + 0.16
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
        ax.text(W / 2, H - 0.20,
                f"RipVIS sequences — first / middle / last frame   (sheet {si} of {len(chunks)})",
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(W / 2, H - 0.36,
                "static = framing identical across the three frames    |    "
                "mobile = framing shifts, rotates or changes scale",
                ha="center", va="center", fontsize=7.5, style="italic", color="#555555")

        missing = 0
        for k, v in enumerate(chunk):
            r, c = divmod(k, COLS)
            bx = 0.20 + c * (BW + GX)
            by = H - 0.50 - (r + 1) * BH - r * GY
            for j, fn in enumerate(picks[v]):
                px = bx + j * (TW + 0.03)
                p = os.path.join(args.images, fn)
                if os.path.exists(p):
                    im = mpimg.imread(p)
                    ax.imshow(im, extent=[px, px + TW, by + 0.18, by + 0.18 + TH],
                              aspect="auto", zorder=3)
                else:
                    missing += 1
                    ax.add_patch(Rectangle((px, by + 0.18), TW, TH,
                                 fc="#eeeeee", ec="#bbbbbb", lw=0.6, zorder=3))
                ax.add_patch(Rectangle((px, by + 0.18), TW, TH,
                             fc="none", ec="#444444", lw=0.6, zorder=4))
            ax.text(bx + BW / 2 - 0.03, by + 0.09, v, ha="center", va="center",
                    fontsize=7.2, fontweight="bold")

        sp = os.path.join(args.out, f"contact_sheet_{si}.png")
        fig.savefig(sp, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"  {sp}   ({len(chunk)} videos"
              + (f", {missing} frames not found)" if missing else ")"))

    print("\nNext: fill the modality column with 'static' or 'mobile',")
    print("save as video_modality.csv, then run stratify_by_modality.py")


if __name__ == "__main__":
    main()
