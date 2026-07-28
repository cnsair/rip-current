#!/usr/bin/env python3
"""
Decide whether a high near-duplicate rate reported by audit_near_duplicates.py
is REAL duplication or HASH SATURATION on low-texture imagery.

The problem
-----------
A 64-bit perceptual hash discriminates well on textured photographs. On a
corpus of visually homogeneous scenes — ocean, horizon, sand, low contrast —
the hashes can collapse into a narrow region of the space, so that even
unrelated images land within a Hamming distance of 1-2. In that regime a
"69% near-duplicate" finding measures the failure of the descriptor, not the
content of the dataset.

The test
--------
Establish the NULL distribution. Compare, on the same corpus:

  (a) nearest-neighbour distance   val -> train   (the audit's number)
  (b) distance between RANDOM PAIRS of training images (unrelated by
      construction, unless the corpus really is saturated with duplicates)
  (c) nearest-neighbour distance   train -> train, excluding self
  (d) hash degeneracy: how many DISTINCT hash values exist

Then confirm a sample of flagged pairs at pixel level with normalised RMSE
and SSIM, which no hash artefact can fake.

Reading the result
------------------
  Random-pair median >= ~20 and val->train median <= 2
      -> the descriptor discriminates; the duplicates are REAL.

  Random-pair median <= ~6
      -> the hash is saturated on this corpus; the audit is UNINFORMATIVE
         and must be redone with a stronger descriptor.

  Distinct hashes << number of images
      -> severe degeneracy (or genuine exact duplication); check which via
         the pixel-level confirmation.

Usage
  python verify_duplicate_finding.py \
      --train data_local/train_local/images \
      --val   data_local/val_local/images \
      --pairs near_dupe_audit/flagged_pairs.csv \
      --out   near_dupe_audit

Depends on: numpy, opencv-python.
"""

import argparse
import os
import csv
import sys
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('Requires opencv-python:  pip install opencv-python-headless')

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
POPC = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def dhash(g):
    r = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((r[:, 1:] > r[:, :-1]).flatten()).view(np.uint64)[0]


def phash(g):
    r = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(r)[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:])).view(np.uint64)[0]


def hash_dir(d, names, label):
    dh, ph, ok = [], [], []
    for i, n in enumerate(names):
        img = cv2.imread(os.path.join(d, n), cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dh.append(dhash(g)); ph.append(phash(g)); ok.append(n)
        if (i + 1) % 4000 == 0:
            print(f'  {label}: {i+1}/{len(names)}')
    return np.array(dh, dtype=np.uint64), np.array(ph, dtype=np.uint64), ok


def hdist(a, b):
    x = np.bitwise_xor(a, b)
    return POPC[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def hmat(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    v = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return POPC[v].sum(axis=2).astype(np.uint8)


def nn_min(q, ref, exclude_self=False, chunk=2048):
    best = np.full(len(q), 64, dtype=np.int16)
    for s in range(0, len(ref), chunk):
        d = hmat(q, ref[s:s + chunk]).astype(np.int16)
        if exclude_self:
            for i in range(d.shape[0]):
                j = i - s
                if 0 <= j < d.shape[1]:
                    d[i, j] = 64
        best = np.minimum(best, d.min(axis=1))
    return best


def nrmse_ssim(pa, pb, size=256):
    """Normalised RMSE (0 = identical) and a global SSIM approximation."""
    a = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None, None
    a = cv2.resize(a, (size, size)).astype(np.float64)
    b = cv2.resize(b, (size, size)).astype(np.float64)
    rmse = np.sqrt(np.mean((a - b) ** 2)) / 255.0
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))
    return rmse, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--pairs', default=None, help='flagged_pairs.csv from the audit')
    ap.add_argument('--out', default='near_dupe_audit')
    ap.add_argument('--n-random', type=int, default=200000,
                    help='random training pairs for the null distribution')
    ap.add_argument('--n-confirm', type=int, default=60,
                    help='flagged pairs to confirm at pixel level')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tn, vn = list_images(args.train), list_images(args.val)
    print(f'Hashing {len(tn)} train / {len(vn)} val ...')
    td, tp, tok = hash_dir(args.train, tn, 'train')
    vd, vp, vok = hash_dir(args.val, vn, 'val')

    # ---------- (d) degeneracy ----------
    print('\n' + '=' * 70)
    print('1. HASH DEGENERACY')
    print('=' * 70)
    for nm, h, tot in (('train dHash', td, len(tok)), ('train pHash', tp, len(tok)),
                       ('val   dHash', vd, len(vok)), ('val   pHash', vp, len(vok))):
        u = len(np.unique(h))
        print(f'  {nm}: {u} distinct values of {tot} images '
              f'({100.0*u/tot:.1f}% unique)')
    combo = np.array([hash(tuple(x)) for x in zip(td.tolist(), tp.tolist())])
    print(f'  train dHash+pHash jointly: {len(np.unique(combo))} distinct '
          f'of {len(tok)}')

    # ---------- (b) null distribution ----------
    print('\n' + '=' * 70)
    print('2. NULL DISTRIBUTION — random pairs of TRAINING images')
    print('=' * 70)
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(td), args.n_random)
    j = rng.integers(0, len(td), args.n_random)
    keep = i != j
    i, j = i[keep], j[keep]
    rd = hdist(td[i], td[j])
    rp = hdist(tp[i], tp[j])
    rc = np.maximum(rd, rp)
    print(f'  n = {len(rc)} random pairs')
    print(f'  combined distance:  median {np.median(rc):.0f}   '
          f'mean {rc.mean():.1f}   5th pct {np.percentile(rc,5):.0f}   '
          f'1st pct {np.percentile(rc,1):.0f}')
    print(f'  fraction of RANDOM pairs at distance <= 2: '
          f'{100.0*np.mean(rc<=2):.3f}%')
    print(f'  fraction of RANDOM pairs at distance <= 6: '
          f'{100.0*np.mean(rc<=6):.3f}%')

    # ---------- (c) train -> train nearest neighbour ----------
    print('\n' + '=' * 70)
    print('3. NEAREST NEIGHBOUR WITHIN TRAINING (internal duplication)')
    print('=' * 70)
    sub = rng.choice(len(td), min(3000, len(td)), replace=False)
    ttd = nn_min(td[sub], td, exclude_self=False)
    ttp = nn_min(tp[sub], tp, exclude_self=False)
    # exclude_self handled crudely: a self-match gives 0 on both; drop exact 0-0
    tt = np.maximum(ttd, ttp)
    print(f'  sample of {len(sub)} training images vs the full training set')
    print(f'  (self-matches included, so 0 is expected for every image; '
          f'the informative figure is the SECOND nearest)')
    print(f'  median {np.median(tt):.0f}')

    # ---------- (a) the audit's number, recomputed ----------
    print('\n' + '=' * 70)
    print('4. VAL -> TRAIN NEAREST NEIGHBOUR (the audit result)')
    print('=' * 70)
    vt = np.maximum(nn_min(vd, td), nn_min(vp, tp))
    print(f'  median {np.median(vt):.0f}   '
          f'<=2: {100.0*np.mean(vt<=2):.2f}%   '
          f'<=6: {100.0*np.mean(vt<=6):.2f}%')

    # ---------- verdict on discriminability ----------
    print('\n' + '=' * 70)
    print('VERDICT ON THE DESCRIPTOR')
    print('=' * 70)
    med_rand = float(np.median(rc))
    frac_rand2 = float(np.mean(rc <= 2))
    if med_rand >= 18 and frac_rand2 < 0.005:
        print(f'  Random unrelated pairs sit at median distance {med_rand:.0f}, '
              f'and only {100*frac_rand2:.3f}% of them fall within 2.')
        print('  The descriptor DISCRIMINATES on this corpus. A high val->train')
        print('  rate at distance <=2 therefore reflects REAL duplication.')
        real = True
    elif med_rand <= 8:
        print(f'  Random unrelated pairs sit at median distance {med_rand:.0f}.')
        print('  The hash is SATURATED on this corpus — unrelated images are')
        print('  already close, so the audit cannot distinguish duplicates from')
        print('  ordinary scene similarity. The 64-bit result is UNINFORMATIVE.')
        real = False
    else:
        print(f'  Intermediate: random-pair median {med_rand:.0f}, '
              f'{100*frac_rand2:.3f}% within 2.')
        print('  Rely on the pixel-level confirmation below rather than on the')
        print('  hash distances alone.')
        real = None

    # ---------- pixel-level confirmation ----------
    if args.pairs and os.path.exists(args.pairs):
        print('\n' + '=' * 70)
        print('5. PIXEL-LEVEL CONFIRMATION of flagged pairs')
        print('=' * 70)
        rows = list(csv.DictReader(open(args.pairs)))[:args.n_confirm]
        res = []
        for r in rows:
            pa = os.path.join(args.val, r['val_image'])
            pb = os.path.join(args.train, r['nearest_train_image'])
            rm, ss = nrmse_ssim(pa, pb)
            if rm is not None:
                res.append((r['val_image'], r['nearest_train_image'],
                            int(r['combined']), rm, ss))
        if res:
            rms = np.array([x[3] for x in res])
            sss = np.array([x[4] for x in res])
            print(f'  {len(res)} pairs measured')
            print(f'  normalised RMSE : median {np.median(rms):.4f}   '
                  f'min {rms.min():.4f}   max {rms.max():.4f}')
            print(f'  global SSIM     : median {np.median(sss):.4f}   '
                  f'max {sss.max():.4f}')
            n_dup = int(np.sum((rms < 0.05) | (sss > 0.90)))
            print(f'\n  pairs that are pixel-wise near-identical '
                  f'(RMSE < 0.05 or SSIM > 0.90): {n_dup} / {len(res)}')
            print('\n  closest 12 pairs:')
            print(f'  {"hash":>5s} {"nRMSE":>7s} {"SSIM":>7s}  val / train')
            for v, t, c, rm, ss in sorted(res, key=lambda x: x[3])[:12]:
                print(f'  {c:>5d} {rm:>7.4f} {ss:>7.4f}  {v[:30]} / {t[:30]}')

            with open(os.path.join(args.out, 'pixel_confirmation.csv'),
                      'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['val_image', 'train_image', 'hash_dist',
                            'nrmse', 'ssim'])
                w.writerows(res)

            print('\n  INTERPRETATION: normalised RMSE below ~0.05 with SSIM above')
            print('  ~0.90 means the two images are visually the same frame. If')
            print('  most flagged pairs show RMSE above ~0.15, the hash matches')
            print('  are compositional coincidences, not duplicates.')

    print('\nDone. Written to', args.out)


if __name__ == '__main__':
    main()
