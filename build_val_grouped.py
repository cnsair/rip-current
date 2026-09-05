#!/usr/bin/env python3
"""
-----------------------
Construct a GROUP-DISJOINT validation subset for RipDetSeg.

Supersedes build_val_grouped.py. Changes from v1, and why:

  1. HASHING NOW MATCHES audit_near_duplicates.py EXACTLY.
     v1 used PIL with LANCZOS resizing; the original audit used OpenCV with
     INTER_AREA. This version uses cv2.imread -> COLOR_BGR2GRAY ->
     cv2.resize(INTER_AREA) -> cv2.dct, identical to the audit, so every
     distance in this script is directly comparable to flagged_pairs.csv.
     A PIL/BOX fallback is retained for environments without OpenCV, but it
     is NOT bit-identical and warns accordingly.

  2. TWO COMBINING RULES, EXPLICITLY SEPARATED.
     The original audit computes  max(min_j dHash, min_k pHash), taking each
     hash's minimum over potentially DIFFERENT training images. That is the
     rule behind the manuscript's 69.2% figure. It is not a per-pair distance
     and cannot define graph edges.
       --combine audit    reproduces the original rule (verify mode only)
       --combine pairwise min_j max(dHash_ij, pHash_ij), the per-pair rule
                          required for grouping (default)
     Verify mode reports BOTH so the 69.15% figure can be reproduced and the
     pairwise figure (~67.3%) documented alongside it.

  3. UNREADABLE IMAGES ARE EXCLUDED, NOT ZEROED.
     In the original hash_dir, an unreadable image keeps hash 0, and all-zero
     is a valid hash, so such images appear spuriously close to one another.
     Here they are dropped from the index entirely and listed at the end.

  4. WINDOWS-SAFE MATERIALISATION.
     Ported from build_clean_val_subset.py: symlink -> hard link -> copy,
     since Windows blocks symlinks without Developer Mode (WinError 1314).

Modes
-----
  --mode verify      Reproduce the val->train nearest-neighbour distribution
                     under both combining rules. Run this first.
  --mode calibrate   Sweep the linkage threshold L and report component
                     statistics, so L is chosen by a stated criterion.
  --mode build       Emit val_grouped (manifest + materialised tree).

No training data is modified and no model is retrained.

Usage
-----
  python build_val_grouped.py --mode verify \
      --train-dir data_local/train_local/images \
      --val-dir   data_local/val_local/images

  python build_val_grouped.py --mode calibrate --train-dir ... --val-dir ...

  python build_val_grouped.py --mode build --train-dir ... --val-dir ... \
      --link-threshold 6 --masks data_local/val_local/masks \
      --out-dir data_local/val_grouped --materialize

Depends on: numpy, opencv-python (preferred), pillow+scipy (fallback only).
"""

import argparse
import csv
import os
import shutil
import sys

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    from PIL import Image
    try:
        from scipy.fftpack import dct as _sdct
    except ImportError:
        _sdct = None

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
POPC = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


# ---------------------------------------------------------------------------
# Hashing — identical to audit_near_duplicates.py when OpenCV is present
# ---------------------------------------------------------------------------


def _dhash_cv(gray):
    r = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((r[:, 1:] > r[:, :-1]).flatten()).view(np.uint64)[0]


def _phash_cv(gray):
    r = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(r)[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])          # exclude DC, as in the original audit
    return np.packbits(flat > med).view(np.uint64)[0]


def _dct2_np(a):
    if _sdct is not None:
        return _sdct(_sdct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    n = a.shape[0]
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    m[0, :] /= np.sqrt(2)
    m *= np.sqrt(2.0 / n)
    return m @ a @ m.T


def _hash_pil(path):
    with Image.open(path) as im:
        im.load()
        g = im.convert("L")
    a = np.asarray(g.resize((9, 8), Image.BOX), dtype=np.float32)
    dh = np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(np.uint64)[0]
    b = np.asarray(g.resize((32, 32), Image.BOX), dtype=np.float32)
    d = _dct2_np(b)[:8, :8].flatten()
    ph = np.packbits(d > np.median(d[1:])).view(np.uint64)[0]
    return dh, ph


def hash_dir(directory, label, verbose=True):
    """Return (names, dhash, phash, unreadable). Unreadable images are EXCLUDED
    from the returned arrays rather than left as zero hashes."""
    files = list_images(directory)
    if not files:
        sys.exit(f"ERROR: no images in {directory}")
    dh = np.zeros(len(files), dtype=np.uint64)
    ph = np.zeros(len(files), dtype=np.uint64)
    kept, bad = [], []
    for i, f in enumerate(files):
        p = os.path.join(directory, f)
        try:
            if HAVE_CV2:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError("cv2.imread returned None")
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                d, q = _dhash_cv(g), _phash_cv(g)
            else:
                d, q = _hash_pil(p)
        except Exception as e:
            bad.append((f, str(e)))
            continue
        dh[len(kept)], ph[len(kept)] = d, q
        kept.append(f)
        if verbose and (i + 1) % 4000 == 0:
            print(f"  {label}: {i + 1}/{len(files)}", flush=True)
    if verbose:
        print(f"  {label}: {len(kept)} hashed, {len(bad)} unreadable", flush=True)
    return kept, dh[:len(kept)], ph[:len(kept)], bad


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def hamming_block(a, b):
    """(N,) uint64 vs (M,) uint64 -> (N, M) uint8 Hamming distances."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POPC[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(
        axis=2).astype(np.uint8)


def min_distance_single(qh, rh, chunk=2048):
    """Per-hash minimum over the reference set — the original audit's rule."""
    best = np.full(len(qh), 64, dtype=np.uint8)
    arg = np.zeros(len(qh), dtype=np.int64)
    for s in range(0, len(rh), chunk):
        d = hamming_block(qh, rh[s:s + chunk])
        m, a = d.min(axis=1), d.argmin(axis=1) + s
        upd = m < best
        best[upd], arg[upd] = m[upd], a[upd]
    return best, arg


def min_distance_pairwise(qd, qp, rd, rp, chunk=2048):
    """min_j max(dHash_ij, pHash_ij) — the per-pair rule needed for grouping."""
    best = np.full(len(qd), 64, dtype=np.uint8)
    arg = np.zeros(len(qd), dtype=np.int64)
    for s in range(0, len(rd), chunk):
        c = np.maximum(hamming_block(qd, rd[s:s + chunk]),
                       hamming_block(qp, rp[s:s + chunk]))
        m, a = c.min(axis=1), c.argmin(axis=1) + s
        upd = m < best
        best[upd], arg[upd] = m[upd], a[upd]
    return best, arg


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------


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


def build_components(dh, ph, thr, chunk, verbose=True):
    n = len(dh)
    uf = UnionFind(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c = np.maximum(hamming_block(dh[s:e], dh), hamming_block(ph[s:e], ph))
        rows, cols = np.nonzero(c <= thr)
        for r, col in zip(rows, cols):
            gi = s + int(r)
            if gi < col:
                uf.union(gi, int(col))
        if verbose:
            print(f"  linking {e}/{n} at L={thr}", end="\r", flush=True)
    if verbose:
        print()
    roots = np.array([uf.find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def component_stats(comp, is_train):
    n = len(comp)
    sizes = np.bincount(comp)
    has_train = np.zeros(comp.max() + 1, dtype=bool)
    np.logical_or.at(has_train, comp, is_train)
    val_idx = np.nonzero(~is_train)[0]
    clean = val_idx[~has_train[comp[val_idx]]]
    return {"n_components": int(len(sizes)),
            "largest_share": float(sizes.max() / n),
            "median_size": float(np.median(sizes)),
            "n_val": int(len(val_idx)),
            "n_val_grouped": int(len(clean)),
            "sizes": sizes,
            "clean_idx": clean}


# ---------------------------------------------------------------------------
# Materialisation (symlink -> hard link -> copy; ported from
# build_clean_val_subset.py for Windows compatibility)
# ---------------------------------------------------------------------------


def link_or_copy(src, dst, mode, state):
    if os.path.lexists(dst):
        return True
    order = {"auto": ["symlink", "hardlink", "copy"], "symlink": ["symlink"],
             "hardlink": ["hardlink"], "copy": ["copy"]}[mode]
    if state.get("method"):
        order = [state["method"]]
    for m in order:
        try:
            if m == "symlink":
                os.symlink(src, dst)
            elif m == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            if not state.get("method"):
                state["method"] = m
                if m != "symlink":
                    print(f"  (using {m}s — symlinks unavailable on this system)")
            return True
        except (OSError, NotImplementedError):
            continue
    return False


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_verify(vd, vp, td, tp, chunk, out):
    n = len(vd)
    print("\nComputing per-hash minima (original audit rule)...")
    bd, ba = min_distance_single(vd, td, chunk)
    bp, _ = min_distance_single(vp, tp, chunk)
    audit = np.maximum(bd, bp)
    print("Computing joint per-pair minima (grouping rule)...")
    pair, pa = min_distance_pairwise(vd, vp, td, tp, chunk)

    print(f"\nValidation images: {n}")
    print(f"{'threshold':>10} {'audit rule':>22} {'pairwise rule':>22}")
    print("-" * 56)
    for t in range(0, 7):
        ca, cp = int((audit <= t).sum()), int((pair <= t).sum())
        print(f"{'d <= ' + str(t):>10} {ca:>10} {100 * ca / n:>10.2f}% "
              f"{cp:>10} {100 * cp / n:>10.2f}%")
    print("\nThe audit-rule d<=2 row is the manuscript's 69.2% figure and should")
    print("reproduce it. The pairwise column is the stricter per-pair reading")
    print("used for graph construction; a ~2 pp gap is expected and correct.")
    np.save(os.path.join(out, "min_distances_audit_rule.npy"), audit)
    np.save(os.path.join(out, "min_distances_pairwise.npy"), pair)
    print(f"\nWrote both distance arrays to {out}/")
    return audit, pair


def mode_calibrate(dh, ph, is_train, chunk, thresholds):
    print("\nPercolation sweep. Choose the largest L before the giant component")
    print("takes off, then confirm |val_grouped| is large enough to select on.\n")
    hdr = (f"{'L':>4} {'components':>12} {'largest share':>15} "
           f"{'median size':>12} {'|val_grouped|':>15}")
    print(hdr)
    print("-" * len(hdr))
    for thr in thresholds:
        s = component_stats(build_components(dh, ph, thr, chunk, verbose=False),
                            is_train)
        print(f"{thr:>4} {s['n_components']:>12} {s['largest_share']:>14.2%} "
              f"{s['median_size']:>12.1f} {s['n_val_grouped']:>15}")
    print("\nIf |val_grouped| falls below ~300 at the chosen L, report selection")
    print("results on it with bootstrap CIs and present the deduplicated subset")
    print("(distance > 6, 976 images) alongside as the larger, weaker set.")


def mode_build(names, dh, ph, is_train, thr, chunk, args):
    comp = build_components(dh, ph, thr, chunk)
    s = component_stats(comp, is_train)
    clean = s["clean_idx"]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nlinkage threshold L     : {thr}")
    print(f"components              : {s['n_components']}")
    print(f"largest component share : {s['largest_share']:.2%}")
    print(f"median component size   : {s['median_size']:.1f}")
    print(f"validation images       : {s['n_val']}")
    print(f"val_grouped             : {s['n_val_grouped']} "
          f"({100 * s['n_val_grouped'] / s['n_val']:.1f}% of val)")

    man = os.path.join(args.out_dir, "val_grouped_manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "component_id", "component_size"])
        for i in clean:
            w.writerow([names[i], int(comp[i]), int(s["sizes"][comp[i]])])
    full = os.path.join(args.out_dir, "component_assignment_full.csv")
    with open(full, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split", "component_id"])
        for i, nm in enumerate(names):
            w.writerow([nm, "train" if is_train[i] else "val", int(comp[i])])
    print(f"\nWrote {man}")
    print(f"Wrote {full}  (audit trail for the response letter)")

    if args.materialize:
        img_dir = os.path.join(args.out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        msk_dir = None
        if args.masks:
            msk_dir = os.path.join(args.out_dir, "masks")
            os.makedirs(msk_dir, exist_ok=True)
        state, n_i, n_m, failed = {}, 0, 0, []
        for i in clean:
            nm = names[i]
            src = os.path.abspath(os.path.join(args.val_dir, nm))
            if link_or_copy(src, os.path.join(img_dir, nm), args.link_mode, state):
                n_i += 1
            else:
                failed.append(nm)
            if msk_dir:
                stem = os.path.splitext(nm)[0]
                ext = args.mask_ext or os.path.splitext(nm)[1]
                ms = os.path.abspath(os.path.join(args.masks, stem + ext))
                if os.path.exists(ms) and link_or_copy(
                        ms, os.path.join(msk_dir, stem + ext), args.link_mode, state):
                    n_m += 1
        print(f"\nMaterialised {n_i} images at {img_dir} "
              f"(method: {state.get('method', 'n/a')})")
        if failed:
            print(f"  ! {len(failed)} failed, e.g. {failed[:3]}")
        if msk_dir:
            print(f"Materialised {n_m} masks at {msk_dir}")
            if n_m != n_i:
                print(f"  ! {n_i - n_m} masks missing — check --mask-ext")
        print("\nNext: re-evaluate EXISTING checkpoints on this directory.")
        print("No retraining is required.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["verify", "calibrate", "build"], required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--mask-ext", default=None)
    ap.add_argument("--link-threshold", type=int, default=6)
    ap.add_argument("--sweep", default="0,1,2,3,4,6,8,10,12,14,16")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out-dir", default="val_grouped_out")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--link-mode", choices=["auto", "symlink", "hardlink", "copy"],
                    default="auto")
    ap.add_argument("--cache", default="hash_cache_v2.npz")
    args = ap.parse_args()

    if not HAVE_CV2:
        print("WARNING: OpenCV not found. Falling back to PIL/BOX hashing, which\n"
              "         is NOT bit-identical to audit_near_duplicates.py. Install\n"
              "         opencv-python before reporting any reconciled figure.\n")

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.cache):
        print(f"Loading hashes from {args.cache}")
        z = np.load(args.cache, allow_pickle=True)
        names = list(z["names"])
        dh, ph, is_train = z["dh"], z["ph"], z["is_train"].astype(bool)
    else:
        print("Hashing training images...")
        tn, tdh, tph, tbad = hash_dir(args.train_dir, "train")
        print("Hashing validation images...")
        vn, vdh, vph, vbad = hash_dir(args.val_dir, "val")
        if tbad or vbad:
            print(f"\n! UNREADABLE: {len(tbad)} train, {len(vbad)} val — excluded.")
            for f, e in (tbad + vbad)[:10]:
                print(f"    {f}: {e}")
            print("  Restore these from your archive; every reported count "
                  "should be over the full partition.")
        names = tn + vn
        dh, ph = np.concatenate([tdh, vdh]), np.concatenate([tph, vph])
        is_train = np.concatenate([np.ones(len(tn), bool), np.zeros(len(vn), bool)])
        np.savez_compressed(args.cache, names=np.array(names, dtype=object),
                            dh=dh, ph=ph, is_train=is_train)
        print(f"Cached hashes to {args.cache}")

    print(f"\nTotal: {len(names)} images "
          f"(train {int(is_train.sum())}, val {int((~is_train).sum())})")

    if args.mode == "verify":
        mode_verify(dh[~is_train], ph[~is_train], dh[is_train], ph[is_train],
                    args.chunk, args.out_dir)
    elif args.mode == "calibrate":
        mode_calibrate(dh, ph, is_train, args.chunk,
                       [int(x) for x in args.sweep.split(",")])
    else:
        mode_build(names, dh, ph, is_train, args.link_threshold, args.chunk, args)


if __name__ == "__main__":
    main()
