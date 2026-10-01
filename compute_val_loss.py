#!/usr/bin/env python3
"""
compute_val_loss.py
===================
Computes the validation loss of every per-epoch checkpoint of a run, writing a
CSV trace that build_weight_baselines.py consumes to select the SWAD averaging
window (t_s, t_e).

Why this exists: SWAD (Cha et al., NeurIPS 2021) selects its averaging window by
monitoring VALIDATION LOSS. The training pipeline records validation metrics but
never the validation loss, so the trace has to be reconstructed after the fact.

The loss, dataset and evaluation transform are IMPORTED from
train_segformer_dual_branch.py rather than reimplemented, so the values here are
by construction the same quantity the training loop minimises
(0.3 x BCE[pos_weight=2.0] + 0.7 x soft Dice).

A fixed subset of the validation partition is used by default: evaluating all
6,513 images for all 30 checkpoints of all 5 runs would take ~50 GPU-hours,
whereas 1,500 images takes ~3 hours in total and is ample for locating a window
boundary. The subset is chosen by a fixed stride so it is deterministic and
identical across runs.

Usage
-----
  python compute_val_loss.py --epoch-dir trained_models/determinism_check/epochs/segformer_b2_ft_bf16_seed42
  python compute_val_loss.py --epoch-dir ... --subset 0        # full partition
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import os
# USE_DETAIL_BRANCH is read at import time and defaults to 1. Every checkpoint
# in this campaign is a plain SegFormerWrapper, so force the baseline arm before
# importing, rather than relying on the caller to set DETAIL=0.
os.environ["DETAIL"] = "0"
import train_segformer_dual_branch as T   # __main__-guarded: safe to import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-dir", required=True,
                    help="directory of epochNNN.pth snapshots for ONE run")
    ap.add_argument("--val-images", default="data_local/val_local/images")
    ap.add_argument("--val-masks", default="data_local/val_local/masks")
    ap.add_argument("--subset", type=int, default=1500,
                    help="evaluate a fixed-stride subset of this size (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <epoch-dir>/val_loss_trace.csv)")
    a = ap.parse_args()

    epoch_dir = Path(a.epoch_dir)
    ckpts = sorted(epoch_dir.glob("*epoch*.pth"))
    if not ckpts:
        raise SystemExit(f"No epoch checkpoints in {epoch_dir}")
    out = Path(a.out) if a.out else epoch_dir / "val_loss_trace.csv"

    ds = T.RipSegDataset(a.val_images, a.val_masks,
                         transforms=T.get_transforms(train=False))
    if a.subset and a.subset < len(ds):
        stride = len(ds) / a.subset
        idx = [int(i * stride) for i in range(a.subset)]
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(T.DEVICE == "cuda"))
    print(f"Validation loss trace over {len(ds)} images, "
          f"{len(ckpts)} checkpoints in {epoch_dir.name}")

    model = T.build_model() if hasattr(T, "build_model") else None
    if model is None:
        from transformers import SegformerForSemanticSegmentation
        hf = SegformerForSemanticSegmentation.from_pretrained(
            T.SEGFORMER_VARIANT, num_labels=1, ignore_mismatched_sizes=True)
        model = T.SegFormerWrapper(hf, output_size=(T.IMG_SIZE, T.IMG_SIZE))
    model.to(T.DEVICE)

    rows = []
    for ci, cp in enumerate(ckpts):
        c = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model_state"], strict=True)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(T.DEVICE, non_blocking=True)
                masks = masks.to(T.DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=T.AMP_DTYPE):
                    logits = model(images)
                    loss = T.combined_loss(logits, masks)
                tot += float(loss) * images.size(0)
                n += images.size(0)
        rows.append({"epoch": int(c["epoch"]),
                     "val_loss": tot / n,
                     "val_iou": float(c.get("val_iou", float("nan"))),
                     "checkpoint": cp.name})
        print(f"  [{ci+1:>2}/{len(ckpts)}] epoch {c['epoch']:>2}  "
              f"val_loss={tot/n:.5f}  val_iou={c.get('val_iou', float('nan')):.5f}",
              flush=True)

    rows.sort(key=lambda r: r["epoch"])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "val_loss", "val_iou", "checkpoint"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
