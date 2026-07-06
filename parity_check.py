"""
parity_check.py
===============
Run ONCE before training the dual-branch model (Step 5 of the guide).

Verifies three things:
  1. The baseline checkpoint loads into DualBranchSegFormer with NO
     unexpected keys (i.e. every baseline weight is restored) and the only
     missing keys belong to the new detail/fusion/aux modules.
  2. Zero-init parity: in eval mode, DualBranchSegFormer output is
     bit-identical (within float tolerance) to the baseline SegFormerWrapper
     on the same random input — i.e. training starts at mIoU 0.6505 exactly.
  3. Parameter overhead of the new modules.

Usage (conda env `ripseg`, same folder as the training script):
    python parity_check.py
"""

import torch
from transformers import SegformerForSemanticSegmentation

from dual_branch_segformer import DualBranchSegFormer

# ── Match these three lines to your training-script config ──────────────────
SEGFORMER_VARIANT = "./segformer-b2-local"
IMG_SIZE          = 512
BASELINE_CKPT     = "./trained_models/segformer_b2_local.pth"   # Table III arm
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def build_hf():
    return SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_VARIANT, num_labels=1, ignore_mismatched_sizes=True,
    )


class SegFormerWrapper(torch.nn.Module):
    """Minimal copy of the baseline wrapper from the training script."""
    def __init__(self, hf_model, output_size):
        super().__init__()
        self.model = hf_model
        self.output_size = output_size

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return torch.nn.functional.interpolate(
            logits, size=self.output_size, mode="bilinear", align_corners=False)


def main():
    ckpt = torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state"]

    # ── 1. Key compatibility ────────────────────────────────────────────────
    dual = DualBranchSegFormer(build_hf(), output_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    missing, unexpected = dual.load_state_dict(state, strict=False)

    assert len(unexpected) == 0, (
        f"UNEXPECTED KEYS ({len(unexpected)}) — baseline weights not fully "
        f"restored. First few: {unexpected[:5]}")
    bad_missing = [k for k in missing
                   if not k.startswith(("detail.", "fusion.", "aux_head."))]
    assert len(bad_missing) == 0, (
        f"Missing keys outside the new modules: {bad_missing[:5]}")
    print(f"[1/3] PASS  checkpoint keys: {len(state)} restored, "
          f"{len(missing)} new-module keys initialised fresh")

    # ── 2. Zero-init output parity vs the baseline wrapper ─────────────────
    base = SegFormerWrapper(build_hf(), output_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    base.load_state_dict(state, strict=True)

    dual.eval(); base.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        lo_d = dual(x)
        lo_b = base(x)
    max_diff = (lo_d - lo_b).abs().max().item()
    assert max_diff < 1e-4, f"Parity FAILED: max logit diff = {max_diff:.2e}"
    print(f"[2/3] PASS  zero-init parity: max |Δlogit| = {max_diff:.2e} "
          f"(dual-branch == baseline at initialisation)")

    # ── 3. Parameter overhead ───────────────────────────────────────────────
    n_pre = sum(p.numel() for p in dual.pretrained_parameters())
    n_new = sum(p.numel() for p in dual.new_parameters())
    print(f"[3/3] INFO  pretrained params: {n_pre/1e6:.2f} M  |  "
          f"new params: {n_new/1e6:.3f} M  "
          f"(+{100*n_new/n_pre:.2f}% overhead)")
    print("\nAll checks passed — safe to start training.")


if __name__ == "__main__":
    main()
