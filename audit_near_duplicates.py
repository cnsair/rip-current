#!/usr/bin/env python3
"""
audit_near_duplicates.py
========================
Test whether the RipDetSeg image-level 80/20 split leaks near-duplicate
images between the training and validation partitions.

Why this replaces the video-level split for RipDetSeg
------------------------------------------------------
RipDetSeg filenames are randomised hashes (RipDetSeg-<12 chars>.jpg). They
carry no video or sequence identifier, so a video-level regrouping is not
possible and, if the corpus really is a collection of independent stills,
not necessary either.

But RipDetSeg extends RipVIS, which IS video-based. If any RipDetSeg images
were sampled from video, near-duplicate frames could straddle the train/val
boundary while the randomised filenames conceal it. Absence of video IDs is
not evidence of independence.

This script tests independence directly on pixel content. It computes two
64-bit perceptual hashes per image (dHash and pHash), then for every
validation image finds its closest training image by Hamming distance.

Interpreting the output
-----------------------
  distance 0-2    near-certain duplicate (same frame, or consecutive frames)
  distance 3-6    very similar (same scene/burst, likely same video)
  distance 7-12   similar composition, plausibly independent
  distance >12    unrelated

A clean corpus of independent stills shows almost nothing below ~10.
A video-derived corpus shows a heavy spike at 0-6.

If clean, this is a positive result you can report: it rebuts the concern
that the image-level split leaks, with evidence rather than assertion.

Usage
  python audit_near_duplicates.py \
      --train data_local/train_local/images \
      --val   data_local/val_local/images \
      --out   near_dupe_audit

Depends on: numpy, opencv-python.
"""

import argparse
import os
import sys
import csv
import hashlib
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('Requires opencv-python:  pip install opencv-python-headless')

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def dhash(gray):
    """64-bit difference hash: horizontal gradient on a 9x8 thumbnail."""
    r = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = r[:, 1:] > r[:, :-1]
    return np.packbits(bits.flatten()).view(np.uint64)[0]


def phash(gray):
    """64-bit perceptual hash: low-frequency DCT coefficients."""
    r = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(r)[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])          # exclude DC term
    bits = flat > med
    return np.packbits(bits).view(np.uint64)[0]


def hash_dir(d, names, label):
    dh = np.zeros(len(names), dtype=np.uint64)
    ph = np.zeros(len(names), dtype=np.uint64)
    fh = []
    bad = []
    for i, n in enumerate(names):
        p = os.path.join(d, n)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            bad.append(n)
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dh[i] = dhash(g)
        ph[i] = phash(g)
        with open(p, 'rb') as f:
            fh.append(hashlib.md5(f.read()).hexdigest())
        if (i + 1) % 2000 == 0:
            print(f'  {label}: {i+1}/{len(names)}')
    return dh, ph, fh, bad


POPC = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hamming_matrix(a, b_chunk):
    """Hamming distance between uint64 vector a (N,) and b_chunk (M,)
    -> (N, M) uint8. Uses byte-wise popcount lookup."""
    x = np.bitwise_xor(a[:, None], b_chunk[None, :])          # (N, M) uint64
    bytes_ = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return POPC[bytes_].sum(axis=2).astype(np.uint8)


def min_distance(val_h, train_h, chunk=2048):
    """For each val hash, min Hamming distance to any train hash,
    plus the index of the closest train image."""
    n = len(val_h)
    best = np.full(n, 64, dtype=np.uint8)
    arg = np.zeros(n, dtype=np.int64)
    for s in range(0, len(train_h), chunk):
        blk = train_h[s:s + chunk]
        d = hamming_matrix(val_h, blk)
        m = d.min(axis=1)
        a = d.argmin(axis=1) + s
        upd = m < best
        best[upd] = m[upd]
        arg[upd] = a[upd]
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', default='near_dupe_audit')
    ap.add_argument('--flag-threshold', type=int, default=6,
                    help='report val images whose nearest train image is '
                         'within this Hamming distance (default 6). Distance '
                         '<=2 is the high-confidence duplicate band; 3-6 '
                         'includes genuine matches but also scenes that merely '
                         'share composition, so verify visually.')
    ap.add_argument('--contact-sheet', type=int, default=30,
                    help='write a side-by-side JPEG of the N closest flagged '
                         'pairs for visual verification (0 to disable)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr_names = list_images(args.train)
    va_names = list_images(args.val)
    if not tr_names or not va_names:
        sys.exit('One of the directories contains no images.')

    print(f'Hashing {len(tr_names)} train and {len(va_names)} val images...')
    tr_d, tr_p, tr_f, tr_bad = hash_dir(args.train, tr_names, 'train')
    va_d, va_p, va_f, va_bad = hash_dir(args.val, va_names, 'val')
    if tr_bad or va_bad:
        print(f'  ! unreadable: {len(tr_bad)} train, {len(va_bad)} val')

    # exact byte-level duplicates
    exact = set(tr_f) & set(va_f)
    print(f'\nExact file duplicates across the split: {len(exact)}')

    print('Computing nearest-neighbour distances (dHash)...')
    bd, ba = min_distance(va_d, tr_d)
    print('Computing nearest-neighbour distances (pHash)...')
    bp, _ = min_distance(va_p, tr_p)

    # combined: an image is suspicious if BOTH hashes agree it is close
    combined = np.maximum(bd, bp)

    bands = [(0, 2, 'near-certain duplicate'),
             (3, 6, 'very similar (likely same video/burst)'),
             (7, 12, 'similar composition'),
             (13, 64, 'unrelated')]
    print(f'\nDistribution of each validation image\'s nearest training image')
    print(f'(max of dHash and pHash distance; n = {len(va_names)})\n')
    print(f'{"distance":>10s} {"count":>8s} {"pct":>7s}   interpretation')
    print('-' * 72)
    for lo, hi, desc in bands:
        c = int(((combined >= lo) & (combined <= hi)).sum())
        print(f'{lo:>4d}-{hi:<5d} {c:>8d} {100.0*c/len(va_names):>6.2f}%   {desc}')

    print(f'\nmedian distance: {int(np.median(combined))}   '
          f'min: {int(combined.min())}   '
          f'5th pct: {int(np.percentile(combined, 5))}')

    flagged = np.where(combined <= args.flag_threshold)[0]
    out_csv = os.path.join(args.out, 'flagged_pairs.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['val_image', 'nearest_train_image', 'dhash_dist',
                    'phash_dist', 'combined'])
        for i in flagged[np.argsort(combined[flagged])]:
            w.writerow([va_names[i], tr_names[ba[i]], int(bd[i]),
                        int(bp[i]), int(combined[i])])

    np.save(os.path.join(args.out, 'min_distances.npy'), combined)

    # contact sheet: side-by-side pairs, closest first, for eyeball verification
    if args.contact_sheet and len(flagged):
        order = flagged[np.argsort(combined[flagged])][:args.contact_sheet]
        tiles = []
        for i in order:
            a = cv2.imread(os.path.join(args.val, va_names[i]))
            b = cv2.imread(os.path.join(args.train, tr_names[ba[i]]))
            if a is None or b is None:
                continue
            H = 150
            a = cv2.resize(a, (int(a.shape[1] * H / a.shape[0]), H))
            b = cv2.resize(b, (int(b.shape[1] * H / b.shape[0]), H))
            sep = np.full((H, 6, 3), 255, np.uint8)
            row = np.hstack([a, sep, b])
            bar = np.full((22, row.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f'd={int(combined[i])}  VAL | TRAIN', (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tiles.append(np.vstack([bar, row]))
        if tiles:
            w = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)),
                            constant_values=255) for t in tiles]
            sheet = np.vstack(tiles)
            sp = os.path.join(args.out, 'flagged_contact_sheet.jpg')
            cv2.imwrite(sp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f'Contact sheet ({len(tiles)} pairs): {sp}')

    strict = int((combined <= 2).sum())
    pct = 100.0 * len(flagged) / len(va_names)
    pct_strict = 100.0 * strict / len(va_names)
    print(f'\nHigh-confidence duplicates (distance <= 2): '
          f'{strict} / {len(va_names)}  ({pct_strict:.2f}%)')
    print(f'Flagged for review (distance <= {args.flag_threshold}): '
          f'{len(flagged)} / {len(va_names)}  ({pct:.2f}%)')
    print(f'Pair list: {out_csv}')

    print('\n' + '=' * 72)
    if len(exact) > 0:
        print('VERDICT: EXACT duplicates found across the split. The partitions '
              'overlap and must be rebuilt.')
    elif pct_strict < 0.5:
        print('VERDICT: CLEAN. Near-duplicate contamination is negligible, which '
              'is consistent with a corpus of independent still images. The '
              'image-level split is defensible and can be reported as such, '
              'citing this audit.')
    elif pct_strict < 5:
        print('VERDICT: MINOR contamination. Inspect the flagged pairs; if they '
              'are genuine duplicates, remove the affected validation images '
              'and re-report. State the figure in the manuscript.')
    else:
        print('VERDICT: SUBSTANTIAL contamination. A meaningful fraction of the '
              'validation set has near-identical counterparts in training. '
              'Validation scores are optimistic and the partitions must be '
              'rebuilt by grouping near-duplicates (see --help).')
    print('=' * 72)
    print('\nALWAYS eyeball the top ~20 flagged pairs before concluding. '
          'Beach imagery is visually repetitive, so low-distance pairs can be '
          'different scenes that merely share composition (horizon, sand, surf).')


if __name__ == '__main__':
    main()
