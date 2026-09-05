#!/usr/bin/env python3
"""
reconcile_hashes.py
-------------------
Identify which dhash/phash convention reproduces the original near-duplicate
audit, so that one hash definition can be used consistently throughout the paper.

`flagged_pairs.csv` records, for 5,537 specific (val, train) pairs, the exact
`dhash_dist` and `phash_dist` produced by the original audit script. Those
columns are ground truth. This script re-hashes the images in a sample of those
pairs under every plausible convention and reports which combination reproduces
the recorded distances.

Swept dimensions
----------------
  pre-resize     : hash the raw image, or a copy first resized to 512x512
                   (relevant if the original audit hashed training-pipeline
                   copies rather than originals)
  resize filter  : LANCZOS / BICUBIC / BILINEAR / BOX / NEAREST
  phash median   : median over the full 8x8 DCT block (imagehash convention)
                   or over the block excluding the DC term

dhash and phash are scored independently, because they may disagree for
different reasons.

Usage
-----
  python reconcile_hashes.py \
      --flagged-pairs flagged_pairs.csv \
      --train-dir ./data_local/train_local/images \
      --val-dir   ./data_local/val_local/images \
      --sample 800
"""

import argparse
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy.fftpack import dct as _dct
except ImportError:
    _dct = None

FILTERS = {
    "LANCZOS": Image.LANCZOS,
    "BICUBIC": Image.BICUBIC,
    "BILINEAR": Image.BILINEAR,
    "BOX": Image.BOX,
    "NEAREST": Image.NEAREST,
}

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dct2(a):
    if _dct is not None:
        return _dct(_dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    n = a.shape[0]
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    m[0, :] /= np.sqrt(2)
    m *= np.sqrt(2.0 / n)
    return m @ a @ m.T


def hamming(a, b):
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def load_gray(path, pre_resize):
    with Image.open(path) as im:
        im.load()
        g = im.convert("L")
        if pre_resize:
            g = g.resize((pre_resize, pre_resize), Image.BILINEAR)
        return g.copy()


def dhash_bits(gray, filt, size=8):
    a = np.asarray(gray.resize((size + 1, size), filt), dtype=np.float32)
    return np.packbits(a[:, 1:] > a[:, :-1])


def phash_block(gray, filt, size=8, highfreq=4):
    n = size * highfreq
    a = np.asarray(gray.resize((n, n), filt), dtype=np.float32)
    return _dct2(a)[:size, :size]


def phash_bits(block, include_dc):
    flat = block.flatten()
    med = np.median(flat) if include_dc else np.median(flat[1:])
    return np.packbits(block > med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged-pairs", default="flagged_pairs.csv")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--sample", type=int, default=800,
                    help="number of pairs to test (0 = all 5,537)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre-resize", type=str, default="none,512",
                    help="comma-separated: 'none' and/or a pixel size")
    a = ap.parse_args()

    fp = pd.read_csv(a.flagged_pairs)
    if a.sample and a.sample < len(fp):
        fp = fp.sample(a.sample, random_state=a.seed).reset_index(drop=True)
    print(f"Testing {len(fp)} pairs from {a.flagged_pairs}\n")

    pres = [None if p.strip().lower() == "none" else int(p)
            for p in a.pre_resize.split(",")]

    # Hash every unique image once per (pre_resize, filter) configuration.
    need = {}
    for _, r in fp.iterrows():
        need[r.val_image] = a.val_dir
        need[r.nearest_train_image] = a.train_dir
    print(f"Hashing {len(need)} unique images under "
          f"{len(pres)} x {len(FILTERS)} configurations...")

    dh_tab, ph_tab, bad = {}, {}, set()
    for i, (name, d) in enumerate(need.items()):
        path = os.path.join(d, name)
        for pr in pres:
            try:
                g = load_gray(path, pr)
            except Exception as e:
                if name not in bad:
                    print(f"  WARNING: unreadable, excluded: {name} ({e})",
                          file=sys.stderr)
                    bad.add(name)
                continue
            for fname, filt in FILTERS.items():
                dh_tab[(name, pr, fname)] = dhash_bits(g, filt)
                ph_tab[(name, pr, fname)] = phash_block(g, filt)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(need)}", flush=True)

    keep = fp[~fp.val_image.isin(bad) & ~fp.nearest_train_image.isin(bad)]
    if len(keep) < len(fp):
        print(f"\n{len(fp) - len(keep)} pairs dropped (unreadable images); "
              f"{len(keep)} scored.")
    if keep.empty:
        sys.exit("ERROR: no scorable pairs.")

    vi = keep.val_image.tolist()
    ti = keep.nearest_train_image.tolist()
    d_true = keep.dhash_dist.to_numpy()
    p_true = keep.phash_dist.to_numpy()

    # ---- dhash sweep ----
    print("\n" + "=" * 62)
    print("dhash — reproduction of the recorded dhash_dist column")
    print("=" * 62)
    print(f"{'pre-resize':>11} {'filter':>10} {'exact':>9} {'mean signed err':>17}")
    print("-" * 62)
    best_d = None
    for pr, fname in product(pres, FILTERS):
        got = np.array([hamming(dh_tab[(v, pr, fname)], dh_tab[(t, pr, fname)])
                        for v, t in zip(vi, ti)])
        acc = float((got == d_true).mean())
        err = float((got - d_true).mean())
        print(f"{str(pr):>11} {fname:>10} {acc:>8.1%} {err:>17.3f}")
        if best_d is None or acc > best_d[0]:
            best_d = (acc, pr, fname, err)

    # ---- phash sweep ----
    print("\n" + "=" * 62)
    print("phash — reproduction of the recorded phash_dist column")
    print("=" * 62)
    print(f"{'pre-resize':>11} {'filter':>10} {'median':>10} {'exact':>9} "
          f"{'mean signed err':>17}")
    print("-" * 62)
    best_p = None
    for pr, fname, dc in product(pres, FILTERS, [True, False]):
        got = np.array([
            hamming(phash_bits(ph_tab[(v, pr, fname)], dc),
                    phash_bits(ph_tab[(t, pr, fname)], dc))
            for v, t in zip(vi, ti)
        ])
        acc = float((got == p_true).mean())
        err = float((got - p_true).mean())
        lbl = "with DC" if dc else "no DC"
        print(f"{str(pr):>11} {fname:>10} {lbl:>10} {acc:>8.1%} {err:>17.3f}")
        if best_p is None or acc > best_p[0]:
            best_p = (acc, pr, fname, dc, err)

    print("\n" + "=" * 62)
    print(f"Best dhash : pre-resize={best_d[1]}, filter={best_d[2]} "
          f"-> {best_d[0]:.1%} exact (mean signed error {best_d[3]:+.3f})")
    print(f"Best phash : pre-resize={best_p[1]}, filter={best_p[2]}, "
          f"median={'with DC' if best_p[3] else 'no DC'} "
          f"-> {best_p[0]:.1%} exact (mean signed error {best_p[4]:+.3f})")
    print("=" * 62)
    print(
        "\nReading the result:\n"
        "  >99% exact on both  -> convention identified; patch build_val_grouped.py\n"
        "                         and the recomputed leakage figure will match 69.2%.\n"
        "  High on one, low on the other -> only that hash differs; the other is\n"
        "                         already correct.\n"
        "  Both below ~80%     -> the difference is not in these dimensions\n"
        "                         (candidates: grayscale conversion, hash size,\n"
        "                         highfreq factor, EXIF auto-rotation). Send the\n"
        "                         original audit script.\n"
    )


if __name__ == "__main__":
    main()
