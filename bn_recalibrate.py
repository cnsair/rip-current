"""
bn_recalibrate.py
=================
Fixes the WiSE-FT interpolated checkpoints by re-estimating the decode head's
BatchNorm running statistics (the MiT encoder is LayerNorm and unaffected).

Why this is needed: linearly blending two models' weights does NOT produce a
model whose activation statistics are the blend of the parents' statistics.
The single BatchNorm2d after linear_fuse therefore normalises with wrong
running_mean / running_var at eval time, which saturates the classifier and
causes the predict-rip-everywhere behaviour observed at alpha=0.5. This is
the standard weight-averaging gotcha that torch.optim.swa_utils.update_bn
exists to fix: reset the BN buffers, then run gradient-free forward passes
over training-distribution images so the buffers re-estimate the TRUE
statistics of the blended network.

What it does, per checkpoint in trained_models/wise_ft/:
  1. Build the plain SegFormerWrapper model and load the blended weights.
  2. update_bn over N batches of training images (eval-style preprocessing:
     resize 512 + ImageNet normalise — matches what the BN layer sees at
     inference; no augmentation, no labels needed, no gradients).
  3. Overwrite model_state in place and set  "bn_recalibrated": True.
     (Originals are regenerable in seconds from the parents, so in-place
     is safe; the flag records provenance.)

Usage:
    python bn_recalibrate.py                          # all alphas in wise_ft/
    python bn_recalibrate.py --only a050              # just alpha = 0.5
    python bn_recalibrate.py --train-images data_local/train_local/images
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

# ── Match these to your training config ─────────────────────────────────────
SEGFORMER_VARIANT = "./segformer-b2-local"
IMG_SIZE          = 512
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegFormerWrapper(torch.nn.Module):
    """Minimal copy of the training-script wrapper (same key layout)."""
    def __init__(self, hf_model, output_size):
        super().__init__()
        self.model = hf_model
        self.output_size = output_size

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=self.output_size,
                             mode="bilinear", align_corners=False)


class ImageFolderDataset(Dataset):
    """Images only — BN re-estimation needs no masks."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str):
        self.paths = sorted(p for p in Path(root).iterdir()
                            if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise SystemExit(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr).permute(2, 0, 1)


def build_model() -> SegFormerWrapper:
    hf = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_VARIANT, num_labels=1, ignore_mismatched_sizes=True)
    return SegFormerWrapper(hf, output_size=(IMG_SIZE, IMG_SIZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wise-dir",     default="./trained_models/wise_ft")
    ap.add_argument("--train-images", default="data_local/train_local/images",
                    help="Folder of TRAINING-split images (statistics source)")
    ap.add_argument("--batches", type=int, default=400,
                    help="Forward-pass batches for re-estimation (400x8 = "
                         "3,200 images — ample for one BN layer)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="Substring filter, e.g. 'a050' for alpha=0.5 only")
    args = ap.parse_args()

    ckpts = sorted(Path(args.wise_dir).glob("segformer_b2_wise_a*.pth"))
    if args.only:
        ckpts = [c for c in ckpts if args.only in c.name]
    if not ckpts:
        raise SystemExit(f"No matching checkpoints in {args.wise_dir}")

    ds = ImageFolderDataset(args.train_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(DEVICE == "cuda"),
                        drop_last=True)
    print(f"Statistics source: {len(ds)} images from {args.train_images}  |  "
          f"using {args.batches} batches of {args.batch_size}")

    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("bn_recalibrated"):
            print(f"  {path.name}: already recalibrated — skipped")
            continue

        model = build_model()
        missing, unexpected = model.load_state_dict(ckpt["model_state"],
                                                    strict=True), None
        model.to(DEVICE)

        # update_bn resets every BN layer's running stats, switches them to
        # cumulative-average mode, and re-estimates over the loader.
        capped = itertools.islice(iter(loader), args.batches)
        with torch.no_grad():
            torch.optim.swa_utils.update_bn(capped, model, device=DEVICE)

        ckpt["model_state"]     = {k: v.cpu() for k, v in
                                   model.state_dict().items()}
        ckpt["bn_recalibrated"] = True
        torch.save(ckpt, path)
        print(f"  {path.name}: BN re-estimated and saved")

    print("\nDone. Re-run the evaluations for the recalibrated checkpoints — "
          "previous results from these files are invalid.")


if __name__ == "__main__":
    main()
