#!/usr/bin/env python3
"""
Assemble the Section IV-E qualitative comparison as a portrait grid: three
sequence columns by seven condition rows, at IEEE single-column width.

Reading down a column shows how the predicted rip extent changes between
conditions with the boundary held in the same horizontal position, which is
easier to judge than reading across a row.

INPUT LAYOUT
------------
Put one folder per sequence under --root. Inside each, one image per condition,
named by its key:

    panels/
      RipVIS-121/  input.png  gt.png  theta0.png  ft.png  dual.png  photo.png  sawi.png
      RipVIS-066/  input.png  gt.png  ...
      RipVIS-108/  input.png  gt.png  ...

Folder names become the column headers. Any extension matplotlib can read
works (.png, .jpg). A missing file renders as a labelled grey placeholder, so
the figure can be built and checked before every panel exists.

CROPPING
--------
All panels are cropped to a common window per sequence so the comparison is
like-for-like. Supply the window as fractions of image width and height in
--crops (one line per sequence):

    RipVIS-121,0.10,0.35,0.95,0.70      # seq,x0,y0,x1,y1  (fractions)

Derive it from the ground-truth bounding box expanded by a fixed margin rather
than by eye, so it is reproducible and cannot favour any row. Sequences absent
from the file are drawn uncropped. Pass --no-crop-input to leave the input row
uncropped, showing the whole scene for context.

USAGE
python make_qualitative_figure.py \
    --root   ./qualitative_results/panels \
    --crops  ./qualitative_results/panels/crops.csv \
    --rows   ./qualitative_results/panels/rows.txt \
    --cols   ./qualitative_results/panels/cols.txt \
    --out    ./qualitative_results/panels/fig_qualitative \
    --width  3.5 \
    --no-crop-input

  # double-column version
  python make_qualitative_figure.py --root panels --out fig_qual_wide --width 7.16

Depends on: numpy, matplotlib.
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# key -> (filename stem, short row label, long name for the caption)
CONDITIONS = [
    ("input",  "input frame",            "input frame"),
    ("gt",     "ground truth",           "ground-truth annotation"),
    ("theta0", r"Baseline $\theta_0$",   "early-stopped baseline"),
    ("ft",     "Arm 1a",                 "extended fine-tuning"),
    ("dual",   "Arm 1b",                 "dual-branch detail fusion"),
    ("photo",  "Arm 2b",                 "photometric augmentation"),
    ("sawi10",   r"SAWI $\alpha$=0.10",  "SAWI, primary safety-recommended operating point"),
    ("sawi70",   r"SAWI $\alpha$=0.70",  "SAWI, secondary safety-recommended operating point"),
]
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_rows(path):
    """Optional rows file overriding CONDITIONS. One row per line:

        stem | short row label | long name for the caption

    Order in the file is the order of rows in the figure. Blank lines and
    lines beginning with # are ignored. Labels may contain mathtext, e.g.
    $\\theta_0$ or SAWI $\\alpha$=0.70.
    """
    if not path:
        return CONDITIONS
    if not os.path.exists(path):
        sys.exit(f"--rows file not found: {path}")
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == 2:
            parts.append(parts[1])
        if len(parts) != 3:
            sys.exit(f"Bad row line (need  stem | short | long ): {ln}")
        out.append(tuple(parts))
    if not out:
        sys.exit(f"No rows read from {path}")
    return out


def load_cols(path):
    """Optional columns file. One per line:

        folder_name | column header | bottom label

    The bottom label is optional. Order in the file is the order of columns."""
    if not path:
        return None
    if not os.path.exists(path):
        sys.exit(f"--cols file not found: {path}")
    out = []
    for ln in open(path, encoding="utf-8-sig"):
        ln = ln.strip().strip("\r")
        if not ln or ln.startswith("#"):
            continue
        parts = [q.strip().strip('"').strip("'").strip() for q in ln.split("|")]
        out.append((parts[0],
                    parts[1] if len(parts) > 1 and parts[1] else parts[0],
                    parts[2] if len(parts) > 2 else ""))
    return out or None


def find(folder, stem):
    for e in EXTS:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def load_crops(path):
    """seq,x0,y0,x1,y1 as fractions. Returns {} if no file given."""
    if not path:
        return {}
    if not os.path.exists(path):
        sys.exit(f"--crops file not found: {path}")
    out = {}
    for ln in open(path, encoding="utf-8-sig"):
        ln = ln.strip().strip("\r")
        if not ln or ln.startswith("#"):
            continue
        # accept comma, tab or semicolon: spreadsheets often rewrite the delimiter
        for sep in (",", "\t", ";"):
            parts = [q.strip().strip('"').strip("'").strip() for q in ln.split(sep)]
            if len(parts) == 5:
                break
        if len(parts) != 5:
            sys.exit(f"Bad crop line (need  seq,x0,y0,x1,y1 ): {ln!r}\n"
                     f"  If this file was opened in a spreadsheet it may have been "
                     f"saved tab-delimited; rewrite it with commas.")
        try:
            vals = tuple(float(v) for v in parts[1:])
        except ValueError:
            sys.exit(f"Non-numeric crop value in: {ln!r}")
        if not all(0.0 <= v <= 1.0 for v in vals):
            sys.exit(f"Crop values must be fractions in [0,1]: {ln!r}")
        out[parts[0]] = vals
    if not out:
        sys.exit(f"No usable crop lines read from {path}")
    return out


def crop(img, box):
    if box is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    a, b = int(round(x0 * w)), int(round(x1 * w))
    c, d = int(round(y0 * h)), int(round(y1 * h))
    a, b = max(0, a), min(w, b)
    c, d = max(0, c), min(h, d)
    if b <= a or d <= c:
        return img
    return img[c:d, a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder of per-sequence subfolders")
    ap.add_argument("--out", default="fig_qualitative")
    ap.add_argument("--crops", default=None)
    ap.add_argument("--width", type=float, default=3.5,
                    help="figure width in inches (3.5 single column, 7.16 double)")
    ap.add_argument("--no-crop-input", action="store_true",
                    help="leave the input row uncropped to show the whole scene")
    ap.add_argument("--label-width", type=float, default=0.60,
                    help="width of the left label column, inches")
    ap.add_argument("--gap", type=float, default=0.035, help="gap between panels, inches")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--rows", default=None,
                    help="file of  stem | short label | long name  lines, one per row")
    ap.add_argument("--cols", default=None,
                    help="file of  folder | column header  lines, one per column")
    ap.add_argument("--write-templates", action="store_true",
                    help="write rows.txt and cols.txt from the current defaults, then exit")
    args = ap.parse_args()

    global CONDITIONS
    CONDITIONS = load_rows(args.rows)

    if args.write_templates:
        with open("rows.txt", "w", encoding="utf-8") as f:
            f.write("# stem | short row label | long name for the caption\n")
            for k, sh, lo in CONDITIONS:
                f.write(f"{k} | {sh} | {lo}\n")
        folders = sorted(d for d in os.listdir(args.root)
                         if os.path.isdir(os.path.join(args.root, d)))
        with open("cols.txt", "w", encoding="utf-8") as f:
            f.write("# folder | column header | bottom label\n")
            hints = ["best case", "typical case", "failure case"]
            for i_, d_ in enumerate(folders):
                f.write(f"{d_} | {d_} | {hints[i_] if i_ < len(hints) else ''}\n")
        print("wrote rows.txt and cols.txt — edit and pass with --rows / --cols")
        return

    colspec = load_cols(args.cols)
    if colspec:
        missing_dirs = [d for d, _, _ in colspec
                        if not os.path.isdir(os.path.join(args.root, d))]
        if missing_dirs:
            sys.exit(f"Folders listed in --cols not found: {missing_dirs}")
        seqs = [c[0] for c in colspec]
        headers = [c[1] for c in colspec]
        footers = [c[2] for c in colspec]
    else:
        seqs = sorted(d for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d)))
        headers = list(seqs)
        footers = ["" for _ in seqs]
    if not seqs:
        sys.exit(f"No sequence subfolders found in {args.root}")
    print(f"{len(seqs)} sequences: {', '.join(seqs)}")
    crops = load_crops(args.crops)
    if args.crops:
        unmatched = [k for k in crops if k not in seqs]
        nocrop = [s_ for s_ in seqs if s_ not in crops]
        full = [k for k, v in crops.items()
                if v[0] <= 1e-6 and v[1] <= 1e-6 and v[2] >= 1 - 1e-6 and v[3] >= 1 - 1e-6]
        print(f"crops: {len(crops)} window(s) read from {args.crops}")
        if unmatched:
            print(f"  ! crop entries with no matching sequence folder: {unmatched}")
        if nocrop:
            print(f"  ! sequences with no crop window (drawn uncropped): {nocrop}")
        if full:
            print(f"  ! windows covering the whole frame, so no crop applied: {full}")
            print(f"    rerun make_crops.py with a smaller --margin if this is unintended")

    # panel geometry from the first available image's aspect ratio
    aspect = 3 / 4
    for s in seqs:
        p = find(os.path.join(args.root, s), "input") or find(os.path.join(args.root, s), "gt")
        if p:
            im = mpimg.imread(p)
            box = crops.get(s)
            if box and not args.no_crop_input:
                im = crop(im, box)
            aspect = im.shape[0] / im.shape[1]
            break

    NC, NR = len(seqs), len(CONDITIONS)
    G = args.gap

    # size the label column to the widest row label so nothing is clipped
    probe = plt.figure(figsize=(4, 1)); pax = probe.add_axes([0, 0, 1, 1])
    pax.set_xlim(0, 4); pax.set_ylim(0, 1); pax.set_aspect("equal"); pax.axis("off")
    probe.canvas.draw(); prend = probe.canvas.get_renderer()
    need = 0.0
    for _k, short, _l in CONDITIONS:
        t = pax.text(0, 0, short, fontsize=6.6, fontweight="bold")
        bb = t.get_window_extent(renderer=prend).transformed(pax.transData.inverted())
        need = max(need, bb.width); t.remove()
    plt.close(probe)
    LW = max(args.label_width, need + 0.10)
    if LW > args.label_width:
        print(f"label column widened to {LW:.2f} in to fit '{max((c[1] for c in CONDITIONS), key=len)}'")
    PW = (args.width - LW - (NC - 1) * G - 0.04) / NC
    PH = PW * aspect
    HDR = 0.17
    FTR = 0.16 if any(f for f in footers) else 0.0
    H = 0.06 + HDR + NR * PH + (NR - 1) * G + FTR + 0.06

    fig = plt.figure(figsize=(args.width, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, args.width); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"})

    # column headers
    def fit(cx, cy, txt, maxw, size, **kw):
        while size > 4.2:
            t = ax.text(cx, cy, txt, ha="center", va="center", fontsize=size, **kw)
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer()
                                     ).transformed(ax.transData.inverted())
            if bb.width <= maxw:
                return size
            t.remove(); size -= 0.1
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=size, **kw)
        return size

    fig.canvas.draw()
    hsz = 7.0
    for hdr in headers:                      # one size for all, set by the longest
        hsz = min(hsz, fit(-99, -99, hdr, PW - 0.03, 7.0, fontweight="bold"))
    for t in list(ax.texts):                 # clear the probes
        if t.get_position()[0] == -99:
            t.remove()
    for j, hdr in enumerate(headers):
        cx = LW + j * (PW + G) + PW / 2
        ax.text(cx, H - 0.06 - HDR / 2, hdr, ha="center", va="center",
                fontsize=hsz, fontweight="bold")

    missing = []
    for i, (key, short, _long) in enumerate(CONDITIONS):
        ytop = H - 0.06 - HDR - i * (PH + G)
        ax.text(LW - 0.05, ytop - PH / 2, short, ha="right", va="center",
                fontsize=6.6, fontweight="bold")
        for j, s in enumerate(seqs):
            x0 = LW + j * (PW + G)
            p = find(os.path.join(args.root, s), key)
            if p:
                im = mpimg.imread(p)
                box = crops.get(s)
                if box and not (key == "input" and args.no_crop_input):
                    im = crop(im, box)
                cm = "gray" if im.ndim == 2 else None
                ax.imshow(im, extent=[x0, x0 + PW, ytop - PH, ytop],
                          aspect="auto", zorder=3, cmap=cm)
            else:
                missing.append(f"{s}/{key}")
                ax.add_patch(Rectangle((x0, ytop - PH), PW, PH, fc="#f2f2f2",
                             ec="#cccccc", lw=0.5, zorder=3))
                ax.text(x0 + PW / 2, ytop - PH / 2, key, ha="center", va="center",
                        fontsize=5.6, color="#999999", zorder=4)
            ax.add_patch(Rectangle((x0, ytop - PH), PW, PH, fc="none",
                         ec="#444444", lw=0.55, zorder=5))

    if FTR:
        ybot = H - 0.06 - HDR - (NR - 1) * (PH + G) - PH
        fsz = 6.8
        for ftr in footers:
            if ftr:
                fsz = min(fsz, fit(-99, -99, ftr, PW - 0.03, 6.8, fontweight="bold"))
        for t in list(ax.texts):
            if t.get_position()[0] == -99:
                t.remove()
        for j, ftr in enumerate(footers):
            if not ftr:
                continue
            cx = LW + j * (PW + G) + PW / 2
            ax.text(cx, ybot - FTR / 2, ftr, ha="center", va="center",
                    fontsize=fsz, fontweight="bold")

    for ext in ("png", "pdf"):
        fp = f"{args.out}.{ext}"
        fig.savefig(fp, dpi=args.dpi if ext == "png" else None, facecolor="white")
        print(f"wrote {fp}")
    print(f"figure size: {args.width:.2f} x {H:.2f} in   panel: {PW:.2f} x {PH:.2f} in")
    if missing:
        print(f"\n{len(missing)} panels missing (drawn as placeholders):")
        for m in missing[:14]:
            print("   ", m)

    # caption skeleton with the long names in row order
    cap = ("Fig. N. Qualitative comparison on three RipVIS transfer sequences. "
           "Rows, top to bottom: " + "; ".join(l for _, _, l in CONDITIONS) + ". "
           "Columns: " + "; ".join(headers) + ". Predictions are shown as overlays at a "
           "fixed threshold, and all panels are cropped to a common window per "
           "sequence, derived from the ground-truth bounding box. The foam-gap arm "
           "is omitted as its output is indistinguishable from the fine-tuning "
           "control (Section IV-B).")
    open(f"{args.out}_caption.txt", "w").write(cap + "\n")
    print(f"wrote {args.out}_caption.txt")


if __name__ == "__main__":
    main()
