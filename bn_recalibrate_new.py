"""
bn_recalibrate.py  (revised)

Re-estimates the decode head's BatchNorm running statistics for interpolated
checkpoints (the MiT encoder is LayerNorm and unaffected).

Why this is needed: linearly blending two models' weights does NOT produce a
model whose activation statistics are the blend of the parents' statistics.
The single BatchNorm2d after linear_fuse therefore normalises with wrong
running_mean / running_var at eval time, which saturates the classifier and
causes the predict-rip-everywhere behaviour observed at alpha=0.5. This is
the standard weight-averaging gotcha that torch.optim.swa_utils.update_bn
exists to fix: reset the BN buffers, then run gradient-free forward passes
over training-distribution images so the buffers re-estimate the TRUE
statistics of the blended network.

Revisions from the original, and why each matters
-------------------------------------------------
1. PREPROCESSING NOW MATCHES evaluate_test_set.py EXACTLY.
   The original resized with PIL Image.BILINEAR; evaluate_test_set.py resizes
   with A.Resize, which dispatches to cv2.INTER_LINEAR. Pillow scales its
   filter support to the reduction factor and therefore antialiases on
   downscale; OpenCV does not. Source frames are well above 512 px, so every
   image took a real downscale and the BN buffers were being fitted to a
   different input distribution than the one they normalise at inference.
   This version imports the same albumentations pipeline, so the two are
   identical by construction rather than by assertion.

2. DEFAULT --wise-dir CORRECTED to trained_models/wise_swad.
   The SAWI interpolation checkpoints live there. The previous default pointed
   at wise_ft/, so an invocation without an explicit --wise-dir silently
   recalibrated the wrong arm and left the SAWI checkpoints saturated.

3. GUARD AGAINST A SILENT NO-OP.
   torch.optim.swa_utils.update_bn returns immediately if it finds no BN
   layers. The original would then stamp bn_recalibrated=True on a checkpoint
   that had not been touched. This version counts BN layers first and aborts
   if there are none.

4. DETERMINISTIC STATISTICS SOURCE.
   The DataLoader shuffled unseeded, so each invocation drew a different
   sample and produced different running statistics — and therefore different
   predictions from the same checkpoint. The generator is now seeded and the
   seed is recorded in the checkpoint.

5. ATOMIC SAVE.
   Writes to a temporary file and renames, so an interrupted save cannot
   truncate the checkpoint it is replacing.

6. --audit MODE.
   Reports the bn_recalibrated flag for every checkpoint without modifying
   anything. Run this before evaluating.

Usage:
    python bn_recalibrate.py --audit
    python bn_recalibrate.py                                # wise_swad/
    python bn_recalibrate.py --wise-dir ./trained_models/wise_ft
    python bn_recalibrate.py --only a050
"""

import argparse
import itertools
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

# ── Match these to evaluate_test_set.py ─────────────────────────────────────
SEGFORMER_VARIANT = "./segformer-b2-local"
IMG_SIZE          = 512          # SEGFORMER_IMG_SIZE in evaluate_test_set.py
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"


def make_transform(size):
    """Byte-for-byte the transform used by evaluate_test_set.py. A.Resize
    dispatches to cv2.INTER_LINEAR; A.Normalize divides by max_pixel_value=255
    before standardising. Do not substitute a PIL resize here."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


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

    def __init__(self, root: str, transform):
        self.paths = sorted(p for p in Path(root).iterdir()
                            if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise SystemExit(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        return self.transform(image=img)["image"]


def build_model() -> SegFormerWrapper:
    hf = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_VARIANT, num_labels=1, ignore_mismatched_sizes=True)
    return SegFormerWrapper(hf, output_size=(IMG_SIZE, IMG_SIZE))


def count_bn(model) -> int:
    return sum(1 for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def audit(ckpts):
    print(f"{'checkpoint':<44} {'recalibrated':>13} {'seed':>6}")
    print("-" * 65)
    stale = 0
    for p in ckpts:
        c = torch.load(p, map_location="cpu", weights_only=False)
        flag = bool(c.get("bn_recalibrated", False))
        seed = c.get("bn_seed", "-")
        stale += (not flag)
        print(f"{p.name:<44} {str(flag):>13} {str(seed):>6}")
    print("-" * 65)
    if stale:
        print(f"{stale} checkpoint(s) NOT recalibrated. Evaluating these will "
              f"reproduce the saturation failure. Re-run without --audit.")
    else:
        print("All checkpoints recalibrated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wise-dir", default="./trained_models/wise_swad",
                    help="Directory of interpolated checkpoints. Defaults to "
                         "the SAWI arm; pass ./trained_models/wise_ft for the "
                         "plain WiSE-FT arm.")
    ap.add_argument("--train-images", default="data_local/train_local/images",
                    help="Folder of TRAINING-split images (statistics source). "
                         "Never point this at validation data.")
    ap.add_argument("--batches", type=int, default=400,
                    help="Forward-pass batches for re-estimation (400x8 = "
                         "3,200 images — ample for one BN layer)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed for the statistics source. Recorded in "
                         "the checkpoint as bn_seed.")
    ap.add_argument("--only", default=None,
                    help="Substring filter, e.g. 'a050' for alpha=0.5 only")
    ap.add_argument("--audit", action="store_true",
                    help="Report the bn_recalibrated flag for each checkpoint "
                         "and exit without modifying anything")
    ap.add_argument("--force", action="store_true",
                    help="Recalibrate even if already flagged (e.g. after "
                         "changing the preprocessing or the seed)")
    args = ap.parse_args()

    ckpts = sorted(Path(args.wise_dir).glob("segformer_b2_wise_a*.pth"))
    if args.only:
        ckpts = [c for c in ckpts if args.only in c.name]
    if not ckpts:
        raise SystemExit(f"No matching checkpoints in {args.wise_dir}")

    if args.audit:
        audit(ckpts)
        return

    transform = make_transform(IMG_SIZE)
    ds = ImageFolderDataset(args.train_images, transform)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(DEVICE == "cuda"),
                        drop_last=True, generator=gen)
    print(f"Statistics source: {len(ds)} images from {args.train_images}")
    print(f"Sampling         : {args.batches} batches of {args.batch_size}, "
          f"seed {args.seed}")
    print(f"Preprocessing    : A.Resize({IMG_SIZE}) + A.Normalize "
          f"(matches evaluate_test_set.py)\n")

    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("bn_recalibrated") and not args.force:
            print(f"  {path.name}: already recalibrated — skipped "
                  f"(--force to redo)")
            continue

        model = build_model()
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)

        n_bn = count_bn(model)
        if n_bn == 0:
            raise SystemExit(
                f"{path.name}: no BatchNorm layers found. update_bn would "
                f"silently no-op and the checkpoint would be falsely flagged "
                f"as recalibrated. Check the wrapper nesting before "
                f"proceeding.")

        # update_bn resets every BN layer's running stats, switches them to
        # cumulative-average mode, and re-estimates over the loader.
        capped = itertools.islice(iter(loader), args.batches)
        with torch.no_grad():
            torch.optim.swa_utils.update_bn(capped, model, device=DEVICE)

        ckpt["model_state"]     = {k: v.cpu() for k, v in
                                   model.state_dict().items()}
        ckpt["bn_recalibrated"] = True
        ckpt["bn_seed"]         = args.seed
        ckpt["bn_batches"]      = args.batches
        ckpt["bn_source"]       = str(args.train_images)

        tmp = path.with_suffix(".pth.tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)
        print(f"  {path.name}: {n_bn} BN layer(s) re-estimated and saved")

    print("\nDone. Re-run the evaluations for the recalibrated checkpoints — "
          "previous results from these files are invalid.")


if __name__ == "__main__":
    main()
