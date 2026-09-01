#!/usr/bin/env python3
"""
build_val_grouped.py
--------------------
Construct a GROUP-DISJOINT validation subset (`val_grouped`) for RipDetSeg.

Motivation
----------
An image-level deduplicated validation set removes val images that closely
resemble some train image. It does NOT guarantee that a val image belongs to a
video/acquisition sequence which is absent from training: a frame may sit far
from every train frame in pairwise hash distance while still being transitively
linked to them through intermediate frames of the same clip.

This script therefore:
  1. hashes every train and val image (dhash + phash, 64-bit each),
  2. builds a near-duplicate graph over train u val with an edge wherever
     combined = max(dhash_dist, phash_dist) <= LINK_THRESHOLD,
  3. extracts connected components (union-find)  ~= pseudo-videos,
  4. emits val_grouped = { val images whose component contains NO train image }.

No training data is modified. No model is retrained. Hashing is CPU-only.

Modes
-----
  --mode verify      Recompute val->train nearest-neighbour distances and print
                     the cumulative table. Use this FIRST to confirm the hash
                     implementation reproduces the audit (expect 69.15% at d<=2).
  --mode calibrate   Sweep LINK_THRESHOLD and report component statistics so the
                     operating point can be chosen by a stated criterion
                     (largest L before the giant component percolates).
  --mode build       Emit val_grouped (manifest + optional symlink/copy tree).

Usage
-----
  python build_val_grouped.py --mode verify    --train-dir ... --val-dir ...
  python build_val_grouped.py --mode calibrate --train-dir ... --val-dir ...
  python build_val_grouped.py --mode build     --train-dir ... --val-dir ... \
      --link-threshold 6 --out-dir ./val_grouped --copy-mode symlink

Dependencies: numpy, pillow, scipy (scipy only for the DCT in phash).
"""

import argparse
import csv
import os
import shutil
import sys

import numpy as np
from PIL import Image

try:
    from scipy.fftpack import dct as _dct
except ImportError:  # pragma: no cover
    _dct = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------------


def _dct2(a):
    """2-D DCT-II. Falls back to a matrix-multiply implementation if scipy is
    unavailable, so the script runs in a bare environment."""
    if _dct is not None:
        return _dct(_dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    n = a.shape[0]
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    m[0, :] = m[0, :] / np.sqrt(2)
    m *= np.sqrt(2.0 / n)
    return m @ a @ m.T


def dhash(img, size=8):
    """Difference hash: 64 bits from horizontal gradient sign."""
    g = np.asarray(
        img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.float32
    )
    return np.packbits(g[:, 1:] > g[:, :-1])


def phash(img, size=8, highfreq=4):
    """Perceptual hash: 64 bits from the low-frequency DCT block, thresholded at
    the median with the DC term excluded."""
    n = size * highfreq
    g = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float32)
    d = _dct2(g)[:size, :size]
    med = np.median(d.flatten()[1:])  # exclude DC
    return np.packbits(d > med)


def hash_directory(directory, label, verbose=True):
    """Return (names, dhash_array, phash_array, labels) for one image folder."""
    files = sorted(
        f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        sys.exit(f"ERROR: no images found in {directory}")
    dh = np.zeros((len(files), 8), dtype=np.uint8)
    ph = np.zeros((len(files), 8), dtype=np.uint8)
    kept = []
    for i, f in enumerate(files):
        try:
            with Image.open(os.path.join(directory, f)) as im:
                im.load()
                dh[len(kept)] = dhash(im)
                ph[len(kept)] = phash(im)
            kept.append(f)
        except Exception as e:  # unreadable / truncated file
            print(f"  WARNING: skipping {f} ({e})", file=sys.stderr)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(files)} in {label}", flush=True)
    k = len(kept)
    if verbose:
        print(f"  {label}: {k} images hashed", flush=True)
    return kept, dh[:k], ph[:k], np.full(k, label, dtype=object)


# ----------------------------------------------------------------------------
# Hamming distance
# ----------------------------------------------------------------------------

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_block(a, b):
    """Pairwise Hamming distance between packed-bit arrays a (n,8) and b (m,8).
    Returns an (n, m) uint8 matrix."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POPCOUNT[x].sum(axis=2).astype(np.uint8)


def combined_block(dh_a, ph_a, dh_b, ph_b):
    """combined = max(dhash_dist, phash_dist), matching the original audit."""
    return np.maximum(hamming_block(dh_a, dh_b), hamming_block(ph_a, ph_b))


# ----------------------------------------------------------------------------
# Union-find
# ----------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)
        self.r = np.zeros(n, dtype=np.int32)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------


def mode_verify(dh_v, ph_v, dh_t, ph_t, chunk):
    """Recompute val->train nearest-neighbour distance and print the cumulative
    table. This must reproduce the original audit before anything downstream is
    trusted (expected: 29.82% at d=0, 44.63% at d<=1, 69.15% at d<=2)."""
    n = dh_v.shape[0]
    mind = np.full(n, 255, dtype=np.uint8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = combined_block(dh_v[s:e], ph_v[s:e], dh_t, ph_t)
        mind[s:e] = d.min(axis=1)
        print(f"  verified {e}/{n}", end="\r", flush=True)
    print()
    print(f"\nValidation images: {n}")
    print(f"{'threshold':>10} {'flagged':>9} {'share':>9}")
    for t in range(0, 9):
        c = int((mind <= t).sum())
        print(f"{'d <= ' + str(t):>10} {c:>9} {100 * c / n:>8.2f}%")
    np.save("min_distances_recomputed.npy", mind)
    print("\nWrote min_distances_recomputed.npy")
    print(
        "Compare the d<=2 row against the manuscript's 69.2%. A mismatch means "
        "the hash implementation here differs from the original audit script; "
        "reconcile before proceeding."
    )
    return mind


def build_components(dh, ph, is_train, thr, chunk, verbose=True):
    """Connected components over the near-duplicate graph at threshold `thr`."""
    n = dh.shape[0]
    uf = UnionFind(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = combined_block(dh[s:e], ph[s:e], dh, ph)
        rows, cols = np.nonzero(d <= thr)
        for r, c in zip(rows, cols):
            gi = s + r
            if gi < c:  # each undirected edge once, skip self-loops
                uf.union(gi, c)
        if verbose:
            print(f"  linking {e}/{n} at L={thr}", end="\r", flush=True)
    if verbose:
        print()
    roots = np.array([uf.find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def component_stats(comp, is_train):
    n = len(comp)
    ncomp = comp.max() + 1
    sizes = np.bincount(comp)
    has_train = np.zeros(ncomp, dtype=bool)
    np.logical_or.at(has_train, comp, is_train)
    val_idx = np.nonzero(~is_train)[0]
    clean = val_idx[~has_train[comp[val_idx]]]
    return {
        "n_components": int(ncomp),
        "largest_share": float(sizes.max() / n),
        "singletons": int((sizes == 1).sum()),
        "median_size": float(np.median(sizes)),
        "n_val": int(len(val_idx)),
        "n_val_grouped": int(len(clean)),
        "clean_idx": clean,
    }


def mode_calibrate(dh, ph, is_train, chunk, thresholds):
    print("\nPercolation sweep — choose the largest L before the giant component")
    print("takes off, then verify |val_grouped| is large enough to select on.\n")
    hdr = f"{'L':>4} {'components':>12} {'largest share':>15} {'median size':>12} {'|val_grouped|':>15}"
    print(hdr)
    print("-" * len(hdr))
    for thr in thresholds:
        comp = build_components(dh, ph, is_train, thr, chunk, verbose=False)
        s = component_stats(comp, is_train)
        print(
            f"{thr:>4} {s['n_components']:>12} {s['largest_share']:>14.2%} "
            f"{s['median_size']:>12.1f} {s['n_val_grouped']:>15}"
        )
    print(
        "\nIf |val_grouped| falls below ~300 at the chosen L, report all "
        "selection results on it with bootstrap confidence intervals, and "
        "report the deduplicated subset alongside as a larger-but-weaker set."
    )


def mode_build(names, dh, ph, is_train, thr, chunk, out_dir, val_dir, copy_mode):
    comp = build_components(dh, ph, is_train, thr, chunk)
    s = component_stats(comp, is_train)
    clean = s["clean_idx"]

    print(f"\nLINK_THRESHOLD          : {thr}")
    print(f"components              : {s['n_components']}")
    print(f"largest component share : {s['largest_share']:.2%}")
    print(f"validation images       : {s['n_val']}")
    print(f"val_grouped             : {s['n_val_grouped']} "
          f"({100 * s['n_val_grouped'] / s['n_val']:.1f}% of val)")

    with open("val_grouped_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "component_id", "component_size"])
        sizes = np.bincount(comp)
        for i in clean:
            w.writerow([names[i], int(comp[i]), int(sizes[comp[i]])])
    print("\nWrote val_grouped_manifest.csv")

    with open("component_assignment_full.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split", "component_id"])
        for i, nm in enumerate(names):
            w.writerow([nm, "train" if is_train[i] else "val", int(comp[i])])
    print("Wrote component_assignment_full.csv (audit trail for the response letter)")

    if out_dir:
        img_out = os.path.join(out_dir, "images")
        os.makedirs(img_out, exist_ok=True)
        for i in clean:
            src = os.path.join(val_dir, names[i])
            dst = os.path.join(img_out, names[i])
            if copy_mode == "copy":
                shutil.copy2(src, dst)
            else:
                if os.path.lexists(dst):
                    os.remove(dst)
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)  # Windows without developer mode
        print(f"Wrote {len(clean)} images to {img_out} ({copy_mode})")
        print(
            "NOTE: masks are not copied. Point your evaluation script at the "
            "existing mask directory and filter by the manifest, or replicate "
            "this loop for the mask folder."
        )


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["verify", "calibrate", "build"], required=True)
    ap.add_argument("--train-dir", required=True, help="RipDetSeg training images")
    ap.add_argument("--val-dir", required=True, help="RipDetSeg validation images")
    ap.add_argument("--link-threshold", type=int, default=6,
                    help="combined-distance edge threshold L for grouping (build mode)")
    ap.add_argument("--sweep", type=str, default="0,1,2,3,4,6,8,10,12,14,16",
                    help="comma-separated L values for calibrate mode")
    ap.add_argument("--chunk", type=int, default=256, help="rows per distance block")
    ap.add_argument("--out-dir", default=None, help="write val_grouped image tree here")
    ap.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--cache", default="hash_cache.npz",
                    help="cache hashes so repeated runs skip re-hashing")
    a = ap.parse_args()

    if os.path.exists(a.cache):
        print(f"Loading hashes from {a.cache}")
        z = np.load(a.cache, allow_pickle=True)
        names, dh, ph, is_train = (
            list(z["names"]), z["dh"], z["ph"], z["is_train"].astype(bool),
        )
    else:
        print("Hashing training images...")
        tn, tdh, tph, _ = hash_directory(a.train_dir, "train")
        print("Hashing validation images...")
        vn, vdh, vph, _ = hash_directory(a.val_dir, "val")
        names = tn + vn
        dh = np.vstack([tdh, vdh])
        ph = np.vstack([tph, vph])
        is_train = np.concatenate(
            [np.ones(len(tn), bool), np.zeros(len(vn), bool)]
        )
        np.savez_compressed(a.cache, names=np.array(names, dtype=object),
                            dh=dh, ph=ph, is_train=is_train)
        print(f"Cached hashes to {a.cache}")

    print(f"\nTotal images: {len(names)}  "
          f"(train {int(is_train.sum())}, val {int((~is_train).sum())})")

    if a.mode == "verify":
        mode_verify(dh[~is_train], ph[~is_train], dh[is_train], ph[is_train], a.chunk)
    elif a.mode == "calibrate":
        mode_calibrate(dh, ph, is_train, a.chunk,
                       [int(x) for x in a.sweep.split(",")])
    else:
        mode_build(names, dh, ph, is_train, a.link_threshold, a.chunk,
                   a.out_dir, a.val_dir, a.copy_mode)


if __name__ == "__main__":
    main()
