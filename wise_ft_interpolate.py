"""
wise_ft_interpolate.py
======================
Design A, step 1 — WiSE-FT weight-space interpolation (Wortsman et al.,
CVPR 2022) between the two checkpoints you already have:

    ANCHOR  (alpha = 0.0) : segformer_b2_local.pth  — early-stopped baseline,
                            best cross-domain transfer (RipVIS mIoU 0.6523)
    FINETUNED (alpha=1.0) : segformer_b2_ft.pth     — arm 1a, best in-domain
                            fit (val mIoU 0.7954) but degraded transfer

For each alpha in the grid this produces
        theta(alpha) = (1 - alpha) * theta_anchor + alpha * theta_finetuned
and saves it as a checkpoint that evaluate_test_set.py loads unchanged
(--family segformer, NO --detail-branch: both parents are plain
SegFormerWrapper-layout models).

Rules applied per tensor:
  * floating-point tensors  -> linear interpolation
  * integer buffers (e.g. BatchNorm num_batches_tracked) -> copied from the
    fine-tuned parent (interpolating counters is meaningless)
  * key sets must match EXACTLY between parents, else the script aborts —
    this is the guard against accidentally mixing a dual-branch checkpoint in.

No gradients, no GPU needed, ~seconds per alpha.

Usage (from the project root, conda env rip-segment):
    python wise_ft_interpolate.py
    python wise_ft_interpolate.py --alphas 0.3 0.5 0.7        # custom grid
    python wise_ft_interpolate.py --anchor A.pth --finetuned B.pth --outdir D

After it finishes, it prints the exact evaluation + significance commands.
"""

import argparse
from pathlib import Path

import torch

# ── Defaults matched to your project layout ──────────────────────────────────
DEF_ANCHOR    = "./trained_models/segformer_b2_local.pth"
DEF_FINETUNED = "./trained_models/segformer_b2_ft.pth"
DEF_OUTDIR    = "./trained_models/wise_ft"
DEF_ALPHAS    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# alpha = 0.0 and 1.0 are the parents themselves — already evaluated, skipped.


def load_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise KeyError(f"{path}: no 'model_state' key — is this a training "
                       f"checkpoint from the segmentation pipeline?")
    return ckpt["model_state"]


def interpolate(anchor: dict, finetuned: dict, alpha: float) -> dict:
    """theta(alpha) = (1-alpha)*anchor + alpha*finetuned, tensor by tensor."""
    out = {}
    for k in anchor:
        a, f = anchor[k], finetuned[k]
        if a.shape != f.shape:
            raise ValueError(f"Shape mismatch at '{k}': {a.shape} vs {f.shape}")
        if torch.is_floating_point(a):
            out[k] = (1.0 - alpha) * a.to(torch.float32) \
                     + alpha * f.to(torch.float32)
            out[k] = out[k].to(a.dtype)
        else:
            # Integer buffers (BatchNorm counters etc.): take fine-tuned's.
            out[k] = f.clone()
    return out


def main():
    ap = argparse.ArgumentParser(description="WiSE-FT weight interpolation")
    ap.add_argument("--anchor",    default=DEF_ANCHOR,
                    help="OOD-robust parent (alpha=0), default: baseline")
    ap.add_argument("--finetuned", default=DEF_FINETUNED,
                    help="ID-strong parent (alpha=1), default: arm 1a")
    ap.add_argument("--outdir",    default=DEF_OUTDIR)
    ap.add_argument("--alphas",    nargs="+", type=float, default=DEF_ALPHAS)
    args = ap.parse_args()

    anchor    = load_state(args.anchor)
    finetuned = load_state(args.finetuned)

    # ── Guard: identical architectures required ─────────────────────────────
    ka, kf = set(anchor.keys()), set(finetuned.keys())
    if ka != kf:
        only_a, only_f = sorted(ka - kf)[:5], sorted(kf - ka)[:5]
        raise SystemExit(
            f"ABORT: checkpoints have different key sets "
            f"({len(ka)} vs {len(kf)} tensors).\n"
            f"  only in anchor   : {only_a}\n"
            f"  only in finetuned: {only_f}\n"
            f"Both parents must be plain SegFormerWrapper checkpoints "
            f"(no --detail-branch models).")
    print(f"Parents compatible: {len(ka)} tensors each.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = []
    for alpha in args.alphas:
        state = interpolate(anchor, finetuned, alpha)
        tag   = f"{int(round(alpha * 100)):03d}"          # 0.5 -> "050"
        label = f"segformer_b2_wise_a{tag}"
        path  = outdir / f"{label}.pth"
        torch.save({
            "model_state": state,
            "epoch":       -1,                            # not a training ckpt
            "val_iou":     float("nan"),                  # to be measured
            "wise_ft":     {"alpha": alpha,
                            "anchor": str(args.anchor),
                            "finetuned": str(args.finetuned)},
        }, path)
        labels.append((alpha, label, path))
        print(f"  alpha={alpha:.2f}  ->  {path}")

    # ── Print the follow-up commands so nothing is retyped by hand ──────────
    print("\n# ── Evaluate every alpha on the RipVIS test set ─────────────")
    for alpha, label, path in labels:
        print(f"python evaluate_test_set.py --family segformer "
              f"--checkpoint {path} "
              f"--segformer-path ./segformer-b2-local --label {label}")

    print("\n# ── Evaluate every alpha on the validation split ────────────")
    for alpha, label, path in labels:
        print(f"EVAL_IMAGES=data_local/val_local/images "
              f"EVAL_MASKS=data_local/val_local/masks "
              f"EVAL_RESULTS=results_val "
              f"python evaluate_test_set.py --family segformer "
              f"--checkpoint {path} "
              f"--segformer-path ./segformer-b2-local --label {label}")

    print("\n# ── Primary pre-registered comparison (alpha = 0.5) ─────────")
    print("python significance_test.py --best segformer_b2_wise_a050 "
          "--baseline segformer_b2 --metric miou --alternative two-sided")
    print("python significance_test.py --best segformer_b2_wise_a050 "
          "--baseline segformer_b2 --metric recall --alternative two-sided")
    print("python significance_test.py --best segformer_b2_wise_a050 "
          "--baseline segformer_b2_ft --metric miou --alternative two-sided "
          "--results-dir results_val")


if __name__ == "__main__":
    main()
