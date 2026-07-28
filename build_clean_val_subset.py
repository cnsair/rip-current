#!/usr/bin/env python3
"""
Construct a leak-free validation subset by DELETION, using the output of
audit_near_duplicates.py. No retraining is involved: every existing
checkpoint can simply be re-evaluated on the smaller set.

Rationale
---------
The RipDetSeg 80/20 split leaks: a large fraction of validation images have a
near-duplicate in training. Rebuilding the split properly would require
retraining every configuration, which is not affordable. But the leak can be
neutralised for REPORTING by removing the contaminated validation images and
re-evaluating the already-trained models on what remains. Evaluation is
inference-only.

What this does and does not fix
-------------------------------
FIXES   : the reported in-domain figures, and the ability to re-check that the
          alpha selection rules pick the same value on uncontaminated data.
DOES NOT: the early-stopping decision that produced theta_0, which was taken
          on the contaminated set and cannot be undone without retraining.
          That must be disclosed. Note however that the paper's evidence for
          theta_0 being transfer-robust comes from the RipVIS measurements,
          which the leak does not touch.

Known bias, state it in the manuscript
--------------------------------------
The retained images are by construction those LEAST similar to the training
distribution. The clean subset is therefore a conservative, slightly
pessimistic in-domain estimate rather than an unbiased one. That is the safe
direction to err in, but it must be stated.

Usage
  python build_clean_val_subset.py \
      --val        data_local/val_local/images \
      --distances  near_dupe_audit/min_distances.npy \
      --threshold  6 \
      --out        data_local/val_clean \
      --masks      data_local/val_local/masks \
      --materialize

Depends on: numpy (opencv not required).
"""

import argparse
import os
import csv
import sys
import numpy as np

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def link_or_copy(src, dst, mode, state):
    """Materialise src at dst. Tries, in order: symlink, hard link, copy.
    Windows blocks symlinks without Administrator or Developer Mode
    (WinError 1314); hard links work on NTFS with ordinary permissions and
    consume no extra disk space. `state` records which method succeeded so
    the fallback is reported once rather than per file."""
    import shutil
    if os.path.lexists(dst):
        return True
    order = {'auto': ['symlink', 'hardlink', 'copy'],
             'symlink': ['symlink'], 'hardlink': ['hardlink'],
             'copy': ['copy']}[mode]
    if state.get('method'):                  # already settled on a method
        order = [state['method']]
    for m in order:
        try:
            if m == 'symlink':
                os.symlink(src, dst)
            elif m == 'hardlink':
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            if not state.get('method'):
                state['method'] = m
                if m != 'symlink':
                    print(f'  (using {m}s — symlinks unavailable on this system)')
            return True
        except (OSError, NotImplementedError):
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True,
                    help='validation image directory used for the audit')
    ap.add_argument('--distances', required=True,
                    help='min_distances.npy written by audit_near_duplicates.py')
    ap.add_argument('--threshold', type=int, default=6,
                    help='KEEP images whose nearest training neighbour is '
                         'STRICTLY FURTHER than this Hamming distance '
                         '(default 6; use 2 for a permissive subset)')
    ap.add_argument('--masks', default=None)
    ap.add_argument('--mask-ext', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--materialize', action='store_true',
                    help='create images/ and masks/ trees ready for the '
                         'existing evaluation script')
    ap.add_argument('--link-mode', choices=['auto', 'symlink', 'hardlink', 'copy'],
                    default='auto',
                    help='how to materialise files. auto tries symlink, then '
                         'hard link, then copy. On Windows without Developer '
                         'Mode, symlinks fail (WinError 1314) and hardlink is '
                         'used automatically.')
    args = ap.parse_args()

    names = list_images(args.val)
    dist = np.load(args.distances)
    if len(dist) != len(names):
        sys.exit(f'Mismatch: {len(names)} images in {args.val} but '
                 f'{len(dist)} distances in {args.distances}. The audit must '
                 f'have been run on this exact directory.')

    keep_mask = dist > args.threshold
    keep = [n for n, k in zip(names, keep_mask) if k]
    drop = [n for n, k in zip(names, keep_mask) if not k]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'val_clean_files.txt'), 'w') as f:
        f.write('\n'.join(keep) + '\n')
    with open(os.path.join(args.out, 'val_removed_files.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'nearest_train_distance'])
        for n, d in zip(names, dist):
            if d <= args.threshold:
                w.writerow([n, int(d)])

    print(f'Validation images total   : {len(names)}')
    print(f'Removed (distance <= {args.threshold:>2d}) : {len(drop)} '
          f'({100.0*len(drop)/len(names):.2f}%)')
    print(f'Retained (clean subset)   : {len(keep)} '
          f'({100.0*len(keep)/len(names):.2f}%)')
    print(f'Retained distance range   : {int(dist[keep_mask].min())} to '
          f'{int(dist[keep_mask].max())}' if len(keep) else '')

    if len(keep) < 300:
        print('\nWARNING: the clean subset is small. Report its size '
              'explicitly and treat per-image significance on it with '
              'caution. Consider also reporting the threshold-2 subset as a '
              'sensitivity check.')

    # sensitivity: how the subset size varies with threshold
    print('\nSubset size by threshold (for a sensitivity table):')
    for t in (2, 4, 6, 8, 10, 12):
        n = int((dist > t).sum())
        print(f'  keep distance > {t:>2d} : {n:>6d} images '
              f'({100.0*n/len(names):>5.2f}%)')

    if args.materialize:
        img_dir = os.path.join(args.out, 'images')
        os.makedirs(img_dir, exist_ok=True)
        if args.masks:
            msk_dir = os.path.join(args.out, 'masks')
            os.makedirs(msk_dir, exist_ok=True)
        state, n_i, n_m, failed = {}, 0, 0, []
        for x in keep:
            src = os.path.abspath(os.path.join(args.val, x))
            dst = os.path.join(img_dir, x)
            if link_or_copy(src, dst, args.link_mode, state):
                n_i += 1
            else:
                failed.append(x)
            if args.masks:
                stem = os.path.splitext(x)[0]
                ext = args.mask_ext or os.path.splitext(x)[1]
                ms = os.path.abspath(os.path.join(args.masks, stem + ext))
                md = os.path.join(msk_dir, stem + ext)
                if os.path.exists(ms) and link_or_copy(ms, md, args.link_mode, state):
                    n_m += 1
        print(f'\nMaterialised {n_i} images at {img_dir} '
              f'(method: {state.get("method", "n/a")})')
        if failed:
            print(f'  ! {len(failed)} failed, e.g. {failed[:3]}')
        if args.masks:
            print(f'Materialised {n_m} masks at {msk_dir}')
            if n_m != n_i:
                print(f'  ! {n_i-n_m} masks missing — check --mask-ext '
                      f'(masks may be .png while images are .jpg)')
        print('\nNext: re-evaluate the EXISTING checkpoints on this directory.')
        print('  EVAL_IMAGES=%s/images EVAL_MASKS=%s/masks \\' % (args.out, args.out))
        print('  EVAL_RESULTS=results_val_clean python evaluate_test_set.py ...')
        print('\nNo retraining is required for any of this.')


if __name__ == '__main__':
    main()