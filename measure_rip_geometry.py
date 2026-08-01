#!/usr/bin/env python3
"""
Measure the geometry of annotated rip currents from ground-truth masks, so
that the neck-width premise of the manuscript is stated as a measurement
rather than as an assertion.

Why this is needed
------------------
Section IV-A currently states that "rip channels spanning 4-8 pixels at the
lower resolution occupy 8-16 pixels at the higher one". That figure is
asserted, and the argument that predicted rip extent is weakly constrained by
the training objective depends on it. This script measures the distribution
directly over every annotated region in a mask directory.

Three width estimators are computed per connected region, because no single
one is correct for an amorphous elongated structure:

  width_inscribed  = 2 x max(distance transform)
                     Diameter of the largest circle that fits inside the
                     region. For a channel this is the width at its WIDEST
                     point, so it is an upper estimate of neck width.

  width_mean       = area / skeleton_length
                     Mean width along the medial axis. This is the most
                     representative single number for an elongated structure
                     and is the one to quote for "neck width".

  width_minrect    = shorter side of the minimum-area enclosing rectangle
                     Exact for a straight channel, an over-estimate for a
                     curved or branching one. Useful as a cross-check.

Also reported: area in pixels, area as a fraction of the image, elongation
(longer/shorter side of the min-area rectangle), and the number of regions
per image.

Masks are assumed binary-ish: any pixel that is neither 0 nor the ignore
value counts as foreground. Set --fg-value if your masks use 255 for rip.

Usage
  python measure_rip_geometry.py \
      --masks   data_local/test_local/masks \
      --out     geometry_test \
      --min-area 200 \
      --working-res 512

  # if masks are not at the resolution the model trained on, --working-res
  # rescales every width to that longest-edge size so the numbers are
  # comparable to the manuscript's 512x512 figures.
  
  On --working-res 512

    It is only a reporting scale factor, not a measurement choice. The script 
    measures in native mask pixels and multiplies by working_res / max(H, W). 
    Since the scaling is linear, you never need to re-run for another 
    resolution — just multiply:

    256 → measured × 0.5
    224 → measured × 0.4375

    That's exactly how I produced the conversion table.

    I chose 512 in the command for one reason: SegFormer-B2 trains at 512, it is 
    the baseline, and every SAWI number in the paper comes from it. So 512 is 
    the resolution at which your central claims are actually measured. Reporting 
    there keeps the geometry commensurate with the results.

Outputs
  regions.csv    one row per connected region
  per_image.csv  one row per image (aggregated)
  summary.txt    the distribution statistics to quote in the manuscript

Depends on: numpy, opencv-python, scikit-image, pandas.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    sys.exit('Requires opencv-python:  pip install opencv-python-headless')
try:
    from skimage.morphology import skeletonize
except ImportError:
    sys.exit('Requires scikit-image:  pip install scikit-image')

MASK_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_masks(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(MASK_EXT))


def region_geometry(comp, scale):
    """comp: bool array of one connected region (already cropped is fine).
    scale: multiply pixel distances by this to reach working resolution."""
    m = comp.astype(np.uint8)
    area = int(m.sum())

    # inscribed width: 2 x max distance to background
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    width_inscribed = 2.0 * float(dt.max())

    # mean width along the medial axis
    skel = skeletonize(comp)
    skel_len = int(skel.sum())
    width_mean = (area / skel_len) if skel_len > 0 else np.nan

    # minimum-area enclosing rectangle
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
        else:
            x, y, w, h = cv2.boundingRect(c)
        short, long_ = (min(w, h), max(w, h))
    else:
        short = long_ = np.nan
    elong = (long_ / short) if (short and short > 0) else np.nan

    return {
        'area_px': area,
        'width_inscribed': width_inscribed * scale,
        'width_mean': width_mean * scale if width_mean == width_mean else np.nan,
        'width_minrect': short * scale if short == short else np.nan,
        'length_minrect': long_ * scale if long_ == long_ else np.nan,
        'elongation': elong,
        'skeleton_px': skel_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-area', type=int, default=200,
                    help='discard regions smaller than this many pixels '
                         '(default 200, matching the foam-gap study)')
    ap.add_argument('--fg-value', type=int, default=None,
                    help='pixel value denoting rip. Default: any value that '
                         'is neither 0 nor 255 is treated as foreground, and '
                         '255 is also treated as foreground unless '
                         '--ignore-255 is set.')
    ap.add_argument('--ignore-255', action='store_true',
                    help='treat 255 as an ignore/void label rather than rip')
    ap.add_argument('--working-res', type=int, default=None,
                    help='longest-edge resolution the model trains at, e.g. '
                         '512. Widths are rescaled to this so they are '
                         'comparable to the manuscript figures. Omit to '
                         'report in native mask pixels.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = list_masks(args.masks)
    if not names:
        sys.exit(f'No masks found in {args.masks}')

    rows, per_img, n_empty, n_small = [], [], 0, 0
    for k, n in enumerate(names):
        m = cv2.imread(os.path.join(args.masks, n), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        H, W = m.shape

        if args.fg_value is not None:
            fg = (m == args.fg_value)
        elif args.ignore_255:
            fg = (m > 0) & (m != 255)
        else:
            fg = (m > 0)

        scale = (args.working_res / max(H, W)) if args.working_res else 1.0

        n_lab, lab = cv2.connectedComponents(fg.astype(np.uint8)) \
            if hasattr(cv2, 'connectedComponents') else \
            cv2.connectedComponents(fg.astype(np.uint8))
        # cv2.connectedComponents returns (retval, labels)
        img_regions = []
        for lid in range(1, n_lab):
            comp = (lab == lid)
            if comp.sum() < args.min_area:
                n_small += 1
                continue
            g = region_geometry(comp, scale)
            g.update({'file_name': n, 'region_id': lid,
                      'image_h': H, 'image_w': W,
                      'area_frac': g['area_px'] / (H * W)})
            rows.append(g)
            img_regions.append(g)

        if not img_regions:
            n_empty += 1
            per_img.append({'file_name': n, 'n_regions': 0,
                            'width_mean_min': np.nan, 'width_mean_median': np.nan,
                            'area_px_total': 0, 'area_frac_total': 0.0})
        else:
            wm = [r['width_mean'] for r in img_regions if r['width_mean'] == r['width_mean']]
            per_img.append({
                'file_name': n,
                'n_regions': len(img_regions),
                'width_mean_min': float(np.min(wm)) if wm else np.nan,
                'width_mean_median': float(np.median(wm)) if wm else np.nan,
                'area_px_total': int(sum(r['area_px'] for r in img_regions)),
                'area_frac_total': float(sum(r['area_frac'] for r in img_regions)),
            })

        if (k + 1) % 500 == 0:
            print(f'  {k+1}/{len(names)}')

    reg = pd.DataFrame(rows)
    pim = pd.DataFrame(per_img)
    reg.to_csv(os.path.join(args.out, 'regions.csv'), index=False)
    pim.to_csv(os.path.join(args.out, 'per_image.csv'), index=False)

    unit = f'px at longest edge {args.working_res}' if args.working_res else 'native mask px'
    lines = []
    lines.append(f'Masks scanned            : {len(names)}')
    lines.append(f'Images with no region    : {n_empty}')
    lines.append(f'Regions kept (>= {args.min_area} px) : {len(reg)}')
    lines.append(f'Regions discarded (small): {n_small}')
    lines.append(f'Units                    : {unit}')
    lines.append('')
    if len(reg):
        for col, label in [('width_mean', 'MEAN WIDTH (area / skeleton length)'),
                           ('width_inscribed', 'INSCRIBED WIDTH (2 x max DT)'),
                           ('width_minrect', 'MIN-RECT SHORT SIDE'),
                           ('elongation', 'ELONGATION (long/short)'),
                           ('area_frac', 'AREA FRACTION OF IMAGE')]:
            s = reg[col].dropna()
            if not len(s):
                continue
            lines.append(f'{label}')
            lines.append(f'  n={len(s)}  mean={s.mean():.3f}  median={s.median():.3f}')
            lines.append(f'  p5={s.quantile(.05):.3f}  p25={s.quantile(.25):.3f}  '
                         f'p75={s.quantile(.75):.3f}  p95={s.quantile(.95):.3f}')
            lines.append(f'  min={s.min():.3f}  max={s.max():.3f}')
            lines.append('')
        s = reg['width_mean'].dropna()
        if len(s):
            lines.append('SENTENCE FOR THE MANUSCRIPT (substitute the measured values):')
            lines.append(f'  "Across {len(s)} annotated rip regions, the mean width along the')
            lines.append(f'   medial axis has median {s.median():.1f} {unit} '
                         f'(interquartile range {s.quantile(.25):.1f}-{s.quantile(.75):.1f}),')
            lines.append(f'   with 5th and 95th percentiles of {s.quantile(.05):.1f} and '
                         f'{s.quantile(.95):.1f}."')
            lines.append('')
            lines.append('  Compare against the value currently asserted in Section IV-A.')
            lines.append('  If the measured median falls outside 8-16, correct the manuscript.')

    txt = '\n'.join(lines)
    open(os.path.join(args.out, 'summary.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'\nWrote {args.out}/regions.csv, per_image.csv, summary.txt')


if __name__ == '__main__':
    main()
